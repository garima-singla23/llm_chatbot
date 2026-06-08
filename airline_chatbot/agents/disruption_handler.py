"""Smart flight disruption recovery helpers for the airline chatbot."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .tool_definitions import (
    calculate_refund,
    check_flight_status,
    get_alternative_flights,
    lookup_booking,
    rebook_flight,
)


_CITY_TO_CODE = {
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Chennai": "MAA",
    "Kolkata": "CCU",
    "Hyderabad": "HYD",
    "Pune": "PNQ",
    "Ahmedabad": "AMD",
}


def _infer_airline(flight_no: str) -> str:
    """Infer a readable airline name from a flight number prefix."""
    flight_no = (flight_no or "").upper()
    if flight_no.startswith("6E"):
        return "IndiGo"
    if flight_no.startswith("AI"):
        return "Air India"
    if flight_no.startswith("SG"):
        return "SpiceJet"
    if flight_no.startswith("UK"):
        return "Vistara"
    if flight_no.startswith("IX"):
        return "Air India Express"
    return "your airline"


def _normalize_origin_destination(value: Any) -> str:
    """Convert city names to airport-style codes when possible."""
    if not value:
        return ""
    text = str(value)
    return _CITY_TO_CODE.get(text, text)


def _format_departure(value: Any) -> str:
    """Return a compact departure string suitable for user-facing messages."""
    if not value:
        return "your scheduled departure"
    return str(value)


def _format_arrival(value: Any) -> str:
    """Return a compact arrival string suitable for user-facing messages."""
    if not value:
        return "scheduled arrival"
    return str(value)


def handle_disruption(pnr: str, memory) -> str:
    """Handle a booking disruption and present rebooking or refund options."""
    booking = lookup_booking(pnr)
    flight_no = booking.get("flight_no", "")
    origin = booking.get("origin", "")
    destination = booking.get("destination", "")
    date = booking.get("date", "")
    amount_paid = booking.get("amount_paid", 0)
    airline = _infer_airline(flight_no)

    flight_status = check_flight_status(flight_no)
    status = flight_status.get("status", "on_time")
    delay_minutes = int(flight_status.get("delay_minutes", 0) or 0)

    active_booking = getattr(memory, "active_booking", None)
    if active_booking is None:
        memory.active_booking = {}
        active_booking = memory.active_booking

    active_booking["pnr"] = pnr
    active_booking["booking"] = booking
    active_booking["flight_status"] = flight_status

    departure = _format_departure(flight_status.get("departure") or booking.get("date"))

    if status == "cancelled" or delay_minutes >= 120:
        alternatives = get_alternative_flights(origin, destination, date, n=3)
        refund_info = calculate_refund(pnr)
        active_booking["disruption_options"] = alternatives

        delay_text = "cancelled" if status == "cancelled" else f"delayed by {delay_minutes // 60} hours"
        compensation_text = ""
        if refund_info.get("compensation_amount", 0) > 0:
            compensation_text = "\n+₹1,000 compensation (airline-caused cancellation)"

        lines: List[str] = [
            f"I can see your {airline} {flight_no} ({_normalize_origin_destination(origin)} → {_normalize_origin_destination(destination)}, {departure})",
            f"has been {delay_text}.",
            "",
            "✈ REBOOK — 3 alternatives:",
        ]

        for idx, option in enumerate(alternatives[:3], start=1):
            fee = "₹0 fee" if status == "cancelled" else "₹500 fee"
            lines.append(
                f"{idx}. {option.get('airline', 'Airline')} {option.get('flight_no', '')} | "
                f"{_format_departure(option.get('departure'))} → {_format_arrival(option.get('arrival'))} | "
                f"₹{option.get('price', 0)} | {fee}"
            )

        lines.extend([
            "",
            f"💰 REFUND — ₹{refund_info.get('refund_amount', 0)} back to your account",
        ])
        if compensation_text:
            lines.append(compensation_text)
        lines.extend([
            f"Processing time: {refund_info.get('timeline_days', 0)} business days",
            "",
            "Reply 1, 2, or 3 to rebook instantly, or 'refund' to get your money back.",
        ])

        return "\n".join(lines)

    return (
        f"Your flight {flight_no} is on time — departing at {departure} from "
        f"{flight_status.get('gate', 'your gate')}."
    )


def detect_disruption_query(message: str) -> bool:
    """Detect whether a message is about disruption recovery."""
    keywords = [
        "cancelled",
        "delayed",
        "what happened",
        "disrupted",
        "alternative",
        "options",
        "rebook",
        "missed flight",
        "flight status",
        "is my flight",
    ]
    lowered = message.lower()
    return any(keyword in lowered for keyword in keywords)


def handle_disruption_choice(choice: str, memory) -> str:
    """Handle a user's rebooking or refund choice after a disruption."""
    options = getattr(memory, "active_booking", {}).get("disruption_options", [])
    pnr = getattr(memory, "active_booking", {}).get("pnr", "")

    if choice in ["1", "2", "3"] and options:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(options):
            return "I couldn't find that option. Please reply 1, 2, or 3."

        selected = options[idx]
        result = rebook_flight(pnr, selected["flight_no"])
        return (
            f"Rebooked! New PNR: {result['new_pnr']} on "
            f"{selected['airline']} {selected['flight_no']}"
        )

    if choice.lower() == "refund":
        result = calculate_refund(pnr)
        return (
            f"Refund of ₹{result['refund_amount']} initiated. Expected in "
            f"{result['timeline_days']} business days."
        )

    return "Please reply 1, 2, 3, or 'refund'."


__all__ = [
    "handle_disruption",
    "detect_disruption_query",
    "handle_disruption_choice",
]
