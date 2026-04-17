"""FlightAPI response parser utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FlightSegment:
    airline_code: str
    airline_name: str
    flight_number: str
    departure_iata: str
    arrival_iata: str
    departure_time: str
    arrival_time: str
    duration_minutes: int


@dataclass
class FlightOffer:
    offer_id: str
    total_price: float
    currency: str
    base_price: float
    taxes: float
    cabin_class: str
    segments: list[FlightSegment]
    total_duration_minutes: int
    num_stops: int
    validating_airline: str
    seats_remaining: int | None
    value_score: float = 0.0
    tags: list = field(default_factory=list)


AIRLINE_LOOKUP = {
    "6E": "IndiGo",
    "AI": "Air India",
    "SG": "SpiceJet",
    "UK": "Vistara",
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "BA": "British Airways",
    "LH": "Lufthansa",
    "SQ": "Singapore Airlines",
    "G8": "Go First",
    "IX": "Air India Express",
    "G9": "Air Arabia",
}


def airline_name_lookup(iata: str) -> str:
    """Resolve common airline name from IATA code."""
    code = (iata or "").upper()
    return AIRLINE_LOOKUP.get(code, iata or "")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_price(raw_price: Any, formatted_price: Any) -> float | None:
    parsed_raw = _to_float(raw_price)
    if parsed_raw is not None:
        return parsed_raw

    formatted = str(formatted_price or "")
    if not formatted:
        return None

    # Keep only digits and decimal point for robust fallback parsing.
    normalized = re.sub(r"[^0-9.]", "", formatted.replace(",", ""))
    if not normalized:
        return None

    return _to_float(normalized)


def _extract_itineraries_and_legs(raw: dict | list) -> tuple[list[dict], list[dict]]:
    itineraries: list[dict] = []
    legs: list[dict] = []

    if isinstance(raw, dict):
        itineraries_val = raw.get("itineraries", [])
        legs_val = raw.get("legs", [])

        if isinstance(itineraries_val, list):
            itineraries.extend([x for x in itineraries_val if isinstance(x, dict)])
        if isinstance(legs_val, list):
            legs.extend([x for x in legs_val if isinstance(x, dict)])

        return itineraries, legs

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue

            nested_itins = item.get("itineraries")
            nested_legs = item.get("legs")

            if isinstance(nested_itins, list):
                itineraries.extend([x for x in nested_itins if isinstance(x, dict)])
            elif "price" in item or "id" in item:
                itineraries.append(item)

            if isinstance(nested_legs, list):
                legs.extend([x for x in nested_legs if isinstance(x, dict)])

        return itineraries, legs

    return itineraries, legs


def _build_segment_from_leg(leg: dict) -> FlightSegment:
    carriers = leg.get("carriers", [])
    carrier = carriers[0] if isinstance(carriers, list) and carriers else {}
    if not isinstance(carrier, dict):
        carrier = {}

    airline_code = str(carrier.get("alternateId") or "")
    airline_name = str(carrier.get("name") or airline_name_lookup(airline_code) or "")
    flight_number = str(
        leg.get("flight_number")
        or leg.get("flightNumber")
        or f"{airline_code}- unknown"
    )

    origin = leg.get("origin", {})
    destination = leg.get("destination", {})
    if not isinstance(origin, dict):
        origin = {}
    if not isinstance(destination, dict):
        destination = {}

    return FlightSegment(
        airline_code=airline_code,
        airline_name=airline_name,
        flight_number=flight_number,
        departure_iata=str(origin.get("display_code") or ""),
        arrival_iata=str(destination.get("display_code") or ""),
        departure_time=str(leg.get("departure") or ""),
        arrival_time=str(leg.get("arrival") or ""),
        duration_minutes=_to_int(leg.get("duration_in_minutes"), 0),
    )


def parse_flightapi_response(raw: dict | list, cabin: str = "Economy") -> list[FlightOffer]:
    """Parse FlightAPI payload to normalized FlightOffer list."""
    try:
        itineraries, legs = _extract_itineraries_and_legs(raw)
        leg_map = {str(leg.get("id")): leg for leg in legs if leg.get("id") is not None}

        offers: list[FlightOffer] = []

        for itinerary in itineraries:
            if not isinstance(itinerary, dict):
                continue

            price_info = itinerary.get("price", {})
            if not isinstance(price_info, dict):
                price_info = {}

            total_price = _parse_price(price_info.get("raw"), price_info.get("formatted"))
            if total_price is None:
                continue

            formatted_price = str(price_info.get("formatted") or "")
            currency = "INR" if "\u20b9" in formatted_price else "USD"

            leg_refs = itinerary.get("legs", [])
            if not isinstance(leg_refs, list):
                leg_refs = []

            resolved_legs: list[dict] = []
            for leg_ref in leg_refs:
                leg_obj: dict | None = None

                if isinstance(leg_ref, dict):
                    leg_obj = leg_ref
                else:
                    leg_obj = leg_map.get(str(leg_ref))

                if isinstance(leg_obj, dict):
                    resolved_legs.append(leg_obj)

            segments = [_build_segment_from_leg(leg) for leg in resolved_legs]
            total_duration_minutes = sum(seg.duration_minutes for seg in segments)
            num_stops = sum(_to_int(leg.get("stop_count"), 0) for leg in resolved_legs)
            validating_airline = segments[0].airline_name if segments else ""

            score_val = _to_float(itinerary.get("score"))
            value_score = score_val if score_val is not None else 0.0

            base_price = total_price * 0.85
            taxes = total_price - base_price

            offers.append(
                FlightOffer(
                    offer_id=str(itinerary.get("id") or ""),
                    total_price=total_price,
                    currency=currency,
                    base_price=base_price,
                    taxes=taxes,
                    cabin_class=cabin,
                    segments=segments,
                    total_duration_minutes=total_duration_minutes,
                    num_stops=num_stops,
                    validating_airline=validating_airline,
                    seats_remaining=None,
                    value_score=value_score,
                )
            )

        offers.sort(key=lambda offer: offer.total_price)
        return offers

    except Exception as exc:
        logger.error("Failed to parse FlightAPI response: %s", exc)
        return []


def debug_print_raw(raw: dict | list) -> None:
    """Pretty-print first 2 itineraries and first 2 legs for debugging."""
    try:
        itineraries, legs = _extract_itineraries_and_legs(raw)
        preview = {
            "itineraries": itineraries[:2],
            "legs": legs[:2],
        }
        print(json.dumps(preview, indent=2, default=str))
    except Exception as exc:
        logger.error("Failed to pretty-print FlightAPI raw payload: %s", exc)