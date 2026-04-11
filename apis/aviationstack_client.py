"""
Aviationstack Real-Time Flight Status API Client.

This module provides a client for querying real-time flight status and route
information using the Aviationstack API. Features include automatic field parsing,
delay calculation, and TTL-based caching.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dateutil import parser as date_parser

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class AviationstackClient:
    """Client for Aviationstack Flight Status API."""

    def __init__(self, api_key: str):
        """
        Initialize AviationstackClient.

        Args:
            api_key: Aviationstack API key for authentication
        """
        self.api_key = api_key
        self.base_url = "http://api.aviationstack.com/v1"
        self._cache: Dict[str, tuple[datetime, Any]] = {}
        self._status_ttl = 300  # 5 minutes in seconds
        self._routes_ttl = 3600  # 60 minutes in seconds

    def _normalize_flight_iata(self, flight_iata: str) -> str:
        """
        Normalize flight IATA code by removing hyphens and spaces.

        Args:
            flight_iata: Flight IATA code (e.g., "6E-204" or "AI101")

        Returns:
            Normalized flight IATA code
        """
        return flight_iata.replace("-", "").replace(" ", "").upper()

    def _is_cache_valid(self, cache_entry: tuple[datetime, Any], ttl: int) -> bool:
        """Check if cache entry is still valid based on TTL."""
        timestamp, _ = cache_entry
        return datetime.now() - timestamp < timedelta(seconds=ttl)

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[str]:
        """
        Parse datetime string and return ISO format or None.

        Args:
            dt_str: Datetime string from API

        Returns:
            ISO format datetime string or None if parsing fails
        """
        if not dt_str:
            return None
        try:
            # Parse flexible datetime formats and return ISO format
            dt = date_parser.parse(dt_str)
            return dt.isoformat()
        except (ValueError, TypeError, date_parser.ParserError):
            return None

    def _calculate_delay_minutes(
        self, scheduled: Optional[str], actual: Optional[str]
    ) -> Optional[int]:
        """
        Calculate delay in minutes between scheduled and actual departure.

        Args:
            scheduled: Scheduled departure time (ISO format or any datetime string)
            actual: Actual departure time (ISO format or any datetime string)

        Returns:
            Delay in minutes, or None if cannot calculate
        """
        if not scheduled or not actual:
            return None

        try:
            scheduled_dt = date_parser.parse(scheduled)
            actual_dt = date_parser.parse(actual)
            delay = (actual_dt - scheduled_dt).total_seconds() / 60
            return int(delay)
        except (ValueError, TypeError, date_parser.ParserError):
            return None

    def get_flight_status(self, flight_iata: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time flight status for a specific flight.

        Results are cached for 5 minutes.

        Args:
            flight_iata: Flight IATA code (e.g., "6E-204", "AI101")

        Returns:
            Dictionary with standardized flight status fields, or None on error:
            {
                "flight_number": str,
                "airline": str,
                "status": str,  # "scheduled" | "active" | "landed" | "cancelled"
                "departure_airport": str,
                "arrival_airport": str,
                "scheduled_departure": str,  # ISO datetime
                "actual_departure": str | None,
                "scheduled_arrival": str,
                "actual_arrival": str | None,
                "delay_minutes": int | None
            }
        """
        # Normalize flight code
        normalized_flight = self._normalize_flight_iata(flight_iata)

        # Check cache
        cache_key = f"status:{normalized_flight}"
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry, self._status_ttl):
                logger.info(f"Returning cached status for flight {normalized_flight}")
                _, cached_result = cache_entry
                return cached_result
            else:
                del self._cache[cache_key]

        try:
            url = f"{self.base_url}/flights"
            params = {"access_key": self.api_key, "flight_iata": normalized_flight}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json().get("data", [])
            if not data:
                logger.warning(f"No flight data found for {normalized_flight}")
                return None

            flight_data = data[0]

            # Parse and map fields
            result = {
                "flight_number": flight_data.get("flight", {}).get("iata", ""),
                "airline": flight_data.get("airline", {}).get("iata", ""),
                "status": flight_data.get("flight_status", ""),
                "departure_airport": flight_data.get("departure", {}).get("iata", ""),
                "arrival_airport": flight_data.get("arrival", {}).get("iata", ""),
                "scheduled_departure": self._parse_datetime(
                    flight_data.get("departure", {}).get("scheduled")
                ),
                "actual_departure": self._parse_datetime(
                    flight_data.get("departure", {}).get("actual")
                ),
                "scheduled_arrival": self._parse_datetime(
                    flight_data.get("arrival", {}).get("scheduled")
                ),
                "actual_arrival": self._parse_datetime(
                    flight_data.get("arrival", {}).get("actual")
                ),
                "delay_minutes": None,
            }

            # Calculate delay
            result["delay_minutes"] = self._calculate_delay_minutes(
                result["scheduled_departure"], result["actual_departure"]
            )

            # Cache the result
            self._cache[cache_key] = (datetime.now(), result)

            return result

        except RequestException as e:
            print(
                f"Error: Failed to fetch flight status for {normalized_flight}: {str(e)}",
                file=sys.stderr,
            )
            return None
        except (KeyError, ValueError, TypeError) as e:
            print(
                f"Error: Failed to parse flight status data: {str(e)}", file=sys.stderr
            )
            return None
        except Exception as e:
            print(
                f"Error: Unexpected error fetching flight status: {str(e)}",
                file=sys.stderr,
            )
            return None

    def get_routes(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        """
        Get available routes between two airports.

        Results are cached for 60 minutes.

        Args:
            origin: IATA code of origin airport (e.g., "DEL")
            destination: IATA code of destination airport (e.g., "BOM")

        Returns:
            List of up to 10 route dictionaries with structure:
            {
                "airline": str,
                "flight_number": str,
                "frequency": str
            }
            Returns empty list on error.
        """
        # Check cache
        cache_key = f"routes:{origin.upper()}:{destination.upper()}"
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry, self._routes_ttl):
                logger.info(f"Returning cached routes for {origin}-{destination}")
                _, cached_result = cache_entry
                return cached_result
            else:
                del self._cache[cache_key]

        try:
            url = f"{self.base_url}/routes"
            params = {
                "access_key": self.api_key,
                "dep_iata": origin.upper(),
                "arr_iata": destination.upper(),
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json().get("data", [])

            # Parse routes and cap at 10 results
            routes = []
            for route in data[:10]:
                route_dict = {
                    "airline": route.get("airline_iata", ""),
                    "flight_number": route.get("flight_number", ""),
                    "frequency": route.get("codeshare", ""),  # or relevant frequency field
                }
                routes.append(route_dict)

            # Cache the result
            self._cache[cache_key] = (datetime.now(), routes)

            return routes

        except RequestException as e:
            print(
                f"Error: Failed to fetch routes {origin}-{destination}: {str(e)}",
                file=sys.stderr,
            )
            return []
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error: Failed to parse routes data: {str(e)}", file=sys.stderr)
            return []
        except Exception as e:
            print(
                f"Error: Unexpected error fetching routes: {str(e)}", file=sys.stderr
            )
            return []
