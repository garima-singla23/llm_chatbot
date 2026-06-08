from __future__ import annotations

import re
from datetime import date as date_cls
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from tools.live_flights import CITY_TO_IATA, search_live_flights
from .tool_definitions import get_booking_page_url, lookup_booking_by_pnr

BOOKING_COMPLETE_TRIGGERS = (
    "i completed the booking",
    "i completed my booking",
    "i paid",
    "payment done",
    "booking completed",
)

FLIGHT_SEARCH_TRIGGERS = (
    "search flights",
    "find flights",
    "show flights",
    "flights from",
    "fly from",
    "book a flight",
    "flight to",
    "flights to",
    "search for flights",
    "available flights",
    "what flights are available",
)


def normalize_text(text: str) -> str:
  return re.sub(r"\s+", " ", text.strip().lower())


def is_booking_completion_message(text: str) -> bool:
  normalized = normalize_text(text)
  return any(trigger in normalized for trigger in BOOKING_COMPLETE_TRIGGERS)


def is_flight_search_message(text: str) -> bool:
  normalized = normalize_text(text)
  return any(trigger in normalized for trigger in FLIGHT_SEARCH_TRIGGERS)


def extract_route_and_date(message: str) -> Tuple[Optional[str], Optional[str], str]:
  lowered = message.lower()
  matched_cities: List[str] = []
  for city in sorted(CITY_TO_IATA.keys(), key=len, reverse=True):
    if city in lowered:
      matched_cities.append(city.title())

  origin = matched_cities[0] if len(matched_cities) >= 1 else None
  destination = matched_cities[1] if len(matched_cities) >= 2 else None

  date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", message)
  date = date_match.group(1) if date_match else date_cls.today().isoformat()
  return origin, destination, date


def extract_pnr(text: str) -> str:
  match = re.search(r"\bPNR[A-Z0-9]{4,10}\b", text, flags=re.IGNORECASE)
  if match:
    return match.group(0).upper()
  return text.strip().upper()


def build_live_flight_options(origin: str, destination: str, date: str = "") -> Dict[str, Any]:
  flights = search_live_flights(origin, destination, date or None)
  flight_cards = []
  for flight in flights[:3]:
    card = (
      f"✈ {flight['airline']} {flight['flight_no']} | "
      f"{flight['departure']} → {flight['arrival']} | "
      f"₹{flight['price']:,} | {flight['seats_available']} seats"
    )
    booking = get_booking_page_url(flight, origin, destination, date)
    flight_cards.append({"card": card, "booking_url": booking["booking_url"], "flight": flight})

  return {
    "origin": origin,
    "destination": destination,
    "date": date,
    "flights": flights,
    "flight_cards": flight_cards,
    "booking_url": flight_cards[0]["booking_url"] if flight_cards else "",
    "count": len(flights),
  }


def handle_booking_completion(message: str) -> Dict[str, Any]:
  pnr = extract_pnr(message)
  booking = lookup_booking_by_pnr(pnr)
  return {
    "awaiting_pnr": False,
    "pnr": pnr,
    "booking": booking,
    "message": "" if booking.get("error") else f"Booking found for {pnr}",
  }
