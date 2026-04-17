"""FlightAPI.io client for one-way and round-trip flight search."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from json import JSONDecodeError
from typing import Any

import requests


class FlightAPIClient:
    """Client for FlightAPI.io endpoints."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict[tuple[str, str, str, str, str, str | None], tuple[datetime, Any]] = {}
        self.base_url = "https://api.flightapi.io"

    def _build_url(
        self,
        origin: str,
        destination: str,
        date: str,
        adults: int,
        children: int,
        cabin: str,
        currency: str,
        return_date: str | None,
    ) -> str:
        if return_date is None:
            return (
                f"{self.base_url}/onewaytrip/{self.api_key}/{origin}/{destination}/"
                f"{date}/{adults}/{children}/{cabin}/{currency}"
            )

        return (
            f"{self.base_url}/roundtrip/{self.api_key}/{origin}/{destination}/"
            f"{date}/{return_date}/{adults}/{children}/{cabin}/{currency}"
        )

    @staticmethod
    def _log_request(method: str, url: str, status_code: int, elapsed_ms: int) -> None:
        print(
            f"[FlightAPI] {method} {url} → {status_code} ({elapsed_ms}ms)",
            file=sys.stderr,
        )

    @staticmethod
    def _to_price(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        adults: int = 1,
        children: int = 0,
        cabin: str = "Economy",
        currency: str = "INR",
        return_date: str = None,
    ) -> list[dict]:
        if cabin not in {"Economy", "Business", "First"}:
            raise ValueError("cabin must be one of: Economy, Business, First")

        cache_key = (origin, destination, date, cabin, currency, return_date)
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if (datetime.now() - ts).seconds < 600:
                return cached

        url = self._build_url(
            origin=origin,
            destination=destination,
            date=date,
            adults=adults,
            children=children,
            cabin=cabin,
            currency=currency,
            return_date=return_date,
        )

        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                response = requests.get(url, timeout=20)
                elapsed_ms = int(response.elapsed.total_seconds() * 1000)
                self._log_request("GET", url, response.status_code, elapsed_ms)

                if response.status_code == 429:
                    if attempts < 2:
                        time.sleep(5)
                        continue
                    return []

                if 500 <= response.status_code < 600:
                    if attempts < 2:
                        time.sleep(2)
                        continue
                    return []

                response.raise_for_status()

                try:
                    payload = response.json()
                except (ValueError, JSONDecodeError) as exc:
                    print(f"[FlightAPI ERROR] {exc}", file=sys.stderr)
                    return []

                self._cache[cache_key] = (datetime.now(), payload)
                return payload
            except requests.exceptions.Timeout:
                print("FlightAPI timeout", file=sys.stderr)
                return []
            except requests.exceptions.RequestException as exc:
                print(f"[FlightAPI ERROR] {exc}", file=sys.stderr)
                return []

        return []

    def get_cheapest(
        self,
        origin: str,
        destination: str,
        date: str,
        adults: int = 1,
    ) -> list[dict]:
        result = self.search_flights(
            origin=origin,
            destination=destination,
            date=date,
            adults=adults,
            children=0,
            cabin="Economy",
        )

        if isinstance(result, list) and result and all(
            isinstance(item, dict) for item in result
        ):
            if any("price" in item for item in result):
                return sorted(result, key=lambda item: self._to_price(item.get("price")))

        return result