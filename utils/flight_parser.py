"""
Amadeus Flight Offer Parser.

Converts raw Amadeus API flight offer objects into clean typed dataclasses.
Handles nested JSON parsing, duration format conversion, and price normalization.
"""

from dataclasses import dataclass, field
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FlightSegment:
    """Represents a single flight segment (leg) of a journey."""

    airline_code: str
    airline_name: str
    flight_number: str
    departure_iata: str
    arrival_iata: str
    departure_time: str  # ISO datetime string
    arrival_time: str  # ISO datetime string
    duration_minutes: int


@dataclass
class FlightOffer:
    """Represents a complete flight offer from Amadeus."""

    offer_id: str
    total_price: float
    currency: str  # always INR after normalisation
    base_price: float
    taxes: float
    cabin_class: str
    segments: List[FlightSegment]
    total_duration_minutes: int
    num_stops: int  # len(segments) - 1
    validating_airline: str
    seats_remaining: Optional[int]
    value_score: float = 0.0
    tags: List[str] = field(default_factory=list)


# Airline IATA code to name mapping
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
}


def airline_name_lookup(iata_code: str) -> str:
    """
    Look up airline full name from IATA code.

    Args:
        iata_code: Airline IATA code (e.g., "6E", "AI")

    Returns:
        Full airline name if found, or iata_code unchanged if not found
    """
    return AIRLINE_LOOKUP.get(iata_code.upper(), iata_code)


def _parse_duration(duration_str: str) -> int:
    """
    Parse ISO 8601 duration string to minutes.

    Args:
        duration_str: Duration in ISO 8601 format (e.g., "PT2H30M", "PT1H15M")

    Returns:
        Duration in minutes

    Raises:
        ValueError: If format is invalid
    """
    # ISO 8601 duration format: PT[n]H[n]M[n]S
    # Examples: PT2H30M, PT1H15M, PT45M, PT2H

    if not duration_str or not duration_str.startswith("PT"):
        raise ValueError(f"Invalid duration format: {duration_str}")

    # Remove 'PT' prefix
    duration_str = duration_str[2:]

    hours = 0
    minutes = 0
    seconds = 0

    # Extract hours
    hours_match = re.search(r"(\d+)H", duration_str)
    if hours_match:
        hours = int(hours_match.group(1))

    # Extract minutes
    minutes_match = re.search(r"(\d+)M", duration_str)
    if minutes_match:
        minutes = int(minutes_match.group(1))

    # Extract seconds (if present)
    seconds_match = re.search(r"(\d+)S", duration_str)
    if seconds_match:
        seconds = int(seconds_match.group(1))

    # Convert to total minutes
    total_minutes = hours * 60 + minutes + (seconds // 60)

    return total_minutes


def _parse_segment(segment: Dict[str, Any]) -> FlightSegment:
    """
    Parse a single flight segment from Amadeus raw data.

    Args:
        segment: Raw segment dictionary from Amadeus

    Returns:
        FlightSegment object

    Raises:
        KeyError: If required fields are missing
    """
    departure_time = segment.get("departure", {}).get("at", "")
    arrival_time = segment.get("arrival", {}).get("at", "")
    duration_str = segment.get("duration", "PT0M")

    duration_minutes = _parse_duration(duration_str)

    airline_code = segment.get("operating", {}).get("carrierCode") or segment.get(
        "carrierCode", ""
    )
    airline_name = airline_name_lookup(airline_code)

    return FlightSegment(
        airline_code=airline_code,
        airline_name=airline_name,
        flight_number=segment.get("number", ""),
        departure_iata=segment.get("departure", {}).get("iataCode", ""),
        arrival_iata=segment.get("arrival", {}).get("iataCode", ""),
        departure_time=departure_time,
        arrival_time=arrival_time,
        duration_minutes=duration_minutes,
    )


def _parse_cabin_class(offer: Dict[str, Any]) -> str:
    """
    Extract cabin class from offer traveler pricing.

    Args:
        offer: Raw offer dictionary from Amadeus

    Returns:
        Cabin class string (e.g., "ECONOMY", "BUSINESS"), or "ECONOMY" as default
    """
    try:
        traveler_pricings = offer.get("travelerPricings", [])
        if not traveler_pricings:
            return "ECONOMY"

        fare_details = traveler_pricings[0].get("fareDetailsBySegment", [])
        if not fare_details:
            return "ECONOMY"

        cabin = fare_details[0].get("cabin", "ECONOMY")
        return cabin
    except (KeyError, IndexError, TypeError):
        return "ECONOMY"


def parse_offers(raw_data: List[Dict[str, Any]]) -> List[FlightOffer]:
    """
    Parse raw Amadeus flight offers into typed FlightOffer objects.

    Args:
        raw_data: List of raw offer dictionaries from Amadeus API response["data"]

    Returns:
        List of FlightOffer objects sorted by total_price ascending

    Raises:
        No exceptions raised - logs errors for individual offers and continues
    """
    offers = []

    for offer_data in raw_data:
        try:
            # Extract IDs and prices
            offer_id = offer_data.get("id", "")

            # Parse price (comes as strings)
            price_info = offer_data.get("price", {})
            total_price_str = price_info.get("grandTotal", "0")
            base_price_str = price_info.get("base", "0")

            try:
                total_price = float(total_price_str)
                base_price = float(base_price_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid price format for offer {offer_id}")
                total_price = 0.0
                base_price = 0.0

            # Calculate taxes
            taxes = total_price - base_price

            # Currency (should already be INR after Amadeus request params)
            currency = price_info.get("currency", "INR")

            # Parse segments from first itinerary
            segments = []
            itineraries = offer_data.get("itineraries", [])
            total_duration_minutes = 0

            if itineraries:
                segment_list = itineraries[0].get("segments", [])
                for segment in segment_list:
                    try:
                        parsed_segment = _parse_segment(segment)
                        segments.append(parsed_segment)
                        total_duration_minutes += parsed_segment.duration_minutes
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse segment: {str(e)}")
                        continue

            # Parse cabin class
            cabin_class = _parse_cabin_class(offer_data)

            # Get validating airline
            validating_airlines = offer_data.get("validatingAirlineCodes", [])
            validating_airline = validating_airlines[0] if validating_airlines else ""

            # Get available seats
            seats_remaining = offer_data.get("numberOfBookableSeats")
            if seats_remaining is not None:
                try:
                    seats_remaining = int(seats_remaining)
                except (ValueError, TypeError):
                    seats_remaining = None

            # Calculate number of stops
            num_stops = len(segments) - 1

            offer = FlightOffer(
                offer_id=offer_id,
                total_price=total_price,
                currency=currency,
                base_price=base_price,
                taxes=taxes,
                cabin_class=cabin_class,
                segments=segments,
                total_duration_minutes=total_duration_minutes,
                num_stops=num_stops,
                validating_airline=validating_airline,
                seats_remaining=seats_remaining,
            )

            offers.append(offer)

        except Exception as e:
            logger.error(f"Failed to parse offer: {str(e)}")
            continue

    # Sort by total price ascending
    offers.sort(key=lambda x: x.total_price)

    return offers
