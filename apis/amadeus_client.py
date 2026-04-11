"""
Amadeus Flight Search API Client.

This module provides a clean client for searching flights using the Amadeus API.
Features include token caching, request retry logic, and response caching.
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class AmadeusClient:
    """Client for Amadeus Flight Search API."""

    def __init__(self, client_id: str, client_secret: str, sandbox: bool = True):
        """
        Initialize AmadeusClient.

        Args:
            client_id: Amadeus API client ID
            client_secret: Amadeus API client secret
            sandbox: Use sandbox environment if True (default), production if False
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = (
            "https://test.api.amadeus.com"
            if sandbox
            else "https://api.amadeus.com"
        )
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._cache: Dict[tuple, tuple[datetime, List[dict]]] = {}
        self._cache_ttl = 600  # 10 minutes in seconds

    def _get_token(self) -> str:
        """
        Get or refresh OAuth2 token.

        Returns cached token if valid, otherwise requests a new one from Amadeus.
        Token is cached for the duration specified by expires_in (typically 1799 seconds).

        Returns:
            Access token string

        Raises:
            RuntimeError: If authentication fails
        """
        # Return cached token if still valid
        if self._token is not None and self._token_expiry is not None:
            if datetime.now() < self._token_expiry:
                return self._token

        # Request new token
        url = f"{self.base_url}/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            token_data = response.json()

            self._token = token_data["access_token"]
            # Token lasts 1799 seconds, use 10s buffer for safety
            expires_in = token_data.get("expires_in", 1799)
            self._token_expiry = datetime.now() + timedelta(
                seconds=expires_in - 10
            )

            return self._token
        except RequestException as e:
            raise RuntimeError(f"Authentication failed: {str(e)}")

    def _retry_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """
        Perform HTTP request with retry logic.

        Retries on 429 (rate limit) or 5xx errors using exponential backoff.
        Retry delay: 2^attempt seconds (1s, 2s, 4s).

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response object

        Raises:
            RequestException: If all retries fail
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, timeout=10, **kwargs)

                # Retry on 429 (rate limit) or 5xx errors
                if response.status_code == 429 or (500 <= response.status_code < 600):
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Request failed with status {response.status_code}, "
                            f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()
                return response

            except RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Request error: {str(e)}, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # This shouldn't be reached, but just in case
        raise RequestException("Max retries exceeded")

    def _is_cache_valid(
        self, cache_entry: tuple[datetime, List[dict]]
    ) -> bool:
        """Check if cache entry is still valid based on TTL."""
        timestamp, _ = cache_entry
        return datetime.now() - timestamp < timedelta(seconds=self._cache_ttl)

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        adults: int = 1,
        cabin: str = "ECONOMY",
        preferred_airline: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for flight offers.

        Results are cached for 10 minutes. If the same search parameters are used
        within the cache window, cached results are returned immediately.

        Args:
            origin: IATA code of origin airport (e.g., "DEL")
            destination: IATA code of destination airport (e.g., "BOM")
            date: Departure date in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            cabin: Cabin class - "ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"
                   (default: "ECONOMY")
            preferred_airline: Optional IATA airline code to filter results

        Returns:
            List of flight offer dictionaries from API response "data" field.
            Returns empty list on error (with error logged to stderr).
        """
        # Check cache
        cache_key = (origin, destination, date, cabin)
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(
                    f"Returning cached results for {origin}-{destination} on {date}"
                )
                _, cached_result = cache_entry
                return cached_result
            else:
                # Remove expired cache entry
                del self._cache[cache_key]

        try:
            # Get auth token
            token = self._get_token()

            # Prepare request
            url = f"{self.base_url}/v2/shopping/flight-offers"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": date,
                "adults": adults,
                "travelClass": cabin,
                "currencyCode": "INR",
                "max": 10,
            }

            # Add airline filter if specified
            if preferred_airline is not None:
                params["includedAirlineCodes"] = preferred_airline

            # Make request with retry logic
            response = self._retry_request("GET", url, headers=headers, params=params)
            response.raise_for_status()

            result = response.json().get("data", [])

            # Cache the result
            self._cache[cache_key] = (datetime.now(), result)

            return result

        except RequestException as e:
            print(
                f"Error: Flight search failed for {origin}-{destination}: {str(e)}",
                file=sys.stderr,
            )
            return []
        except RuntimeError as e:
            print(f"Error: Authentication error: {str(e)}", file=sys.stderr)
            return []
        except Exception as e:
            print(
                f"Error: Unexpected error during flight search: {str(e)}",
                file=sys.stderr,
            )
            return []
