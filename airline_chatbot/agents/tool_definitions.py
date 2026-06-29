"""Tool definitions for Phase 2 airline chatbot.

Each tool is a small, deterministic (where requested) helper that operates on
the in-memory `BOOKINGS_DB` from `booking.booking_engine` or generates realistic
synthetic responses.
"""
from __future__ import annotations

import datetime
import hashlib
import random
import re
import uuid
from urllib.parse import urlencode
from typing import Dict, List, Optional
from airline_chatbot.config import BOOKING_BASE_URL
from proactive.delay_predictor import predict_delay

from booking.booking_engine import BOOKINGS_DB


def _deterministic_rng(seed_value: str) -> random.Random:
    """Return a Random instance seeded deterministically from a string.

    Uses SHA256 to ensure consistent results across processes and runs.
    """
    digest = hashlib.sha256(seed_value.encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16)
    return random.Random(seed_int)


def search_flights(origin: str, destination: str, date: str = None) -> Dict:
    from tools.live_flights import search_live_flights

    flights = search_live_flights(origin, destination, date)
    return {
        "flights": flights[:4],
        "count": len(flights),
        "source": flights[0].get("source", "mock") if flights else "mock",
    }


def check_flight_status(flight_number: str) -> Dict:
    from tools.live_flights import check_live_status

    return check_live_status(flight_number)


def get_booking_page_url(flight: Dict, origin: str, destination: str, date: str = "") -> Dict:
    """Generate the booking page URL to open in browser."""
    params = {
        "flight_no": flight.get("flight_no", ""),
        "airline": flight.get("airline", ""),
        "origin": origin,
        "destination": destination,
        "departure": flight.get("departure", ""),
        "arrival": flight.get("arrival", ""),
        "price": flight.get("price", ""),
        "date": date,
        "gate": flight.get("gate", "TBA"),
        "terminal": flight.get("terminal", "T1"),
    }
    url = f"{BOOKING_BASE_URL}/book?{urlencode(params)}"
    return {"booking_url": url, "message": f"Click this link to complete booking: {url}"}


def lookup_booking_by_pnr(pnr: str) -> Dict:
    """Look up a confirmed booking directly from SQLite."""
    import sqlite3
    from pathlib import Path

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "data" / "bookings.db",
        here.parent.parent / "data" / "bookings.db",
        here.parent.parent.parent / "data" / "bookings.db",
    ]
    db_path = next((p for p in candidates if p.exists()), candidates[1])

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        row = cur.execute(
            "SELECT * FROM bookings WHERE UPPER(TRIM(pnr)) = UPPER(TRIM(?))",
            (pnr,),
        ).fetchone()
        conn.close()
        if row:
            return dict(row)
        return {"error": "Booking not found", "pnr": pnr}
    except Exception as e:
        return {"error": f"DB lookup failed: {type(e).__name__}: {e}", "pnr": pnr}


def lookup_booking(pnr: str) -> Dict:
    """Lookup a booking by `pnr`. If not found, generate a realistic booking.

    Returns a dict with pnr, passenger, flight_no, origin, destination, date,
    seat, status, amount_paid, fare_type.
    """
    booking = BOOKINGS_DB.get(pnr)
    if booking:
        # Enrich: flatten some nested fields for convenience
        flight = booking.get("flight", {})
        return {
            "pnr": booking.get("pnr"),
            "passenger": booking.get("passenger", "Guest"),
            "flight_no": flight.get("flight_no") if isinstance(flight, dict) else flight,
            "origin": flight.get("origin") if isinstance(flight, dict) else None,
            "destination": flight.get("destination") if isinstance(flight, dict) else None,
            "date": flight.get("date") if isinstance(flight, dict) else None,
            "seat": booking.get("seat"),
            "status": booking.get("status", "confirmed"),
            "amount_paid": booking.get("amount_paid", 0),
            "fare_type": booking.get("fare_type", "non_refundable"),
        }

    rng = _deterministic_rng(pnr)
    flight_no = f"AI{rng.randint(100,999)}"
    origin = rng.choice(["DEL", "BOM", "BLR", "MAA", "HYD"]) 
    destination = rng.choice([c for c in ["DEL", "BOM", "BLR", "MAA", "HYD"] if c != origin])
    date = (datetime.date.today() + datetime.timedelta(days=rng.randint(1, 90))).isoformat()
    seat = f"{rng.randint(1,30)}{rng.choice(['A','B','C','D','E','F'])}"
    amount_paid = int(rng.uniform(3000, 25000))
    fare_type = rng.choice(["refundable", "non_refundable", "semi_refundable"]) 

    generated = {
        "pnr": pnr,
        "passenger": "Guest",
        "flight_no": flight_no,
        "origin": origin,
        "destination": destination,
        "date": date,
        "seat": seat,
        "status": "confirmed",
        "amount_paid": amount_paid,
        "fare_type": fare_type,
    }

    return generated


def get_alternative_flights(origin: str, destination: str, date: str, n: int = 3) -> List[Dict]:
    """Return `n` alternative flights between `origin` and `destination` on `date`.

    Ensures at least one IndiGo and one Air India in the results.
    Each result contains flight_no, airline, departure, arrival, price,
    seats_available, duration_minutes.
    """
    seed_string = f"{origin}-{destination}-{date}"
    rng = _deterministic_rng(seed_string)

    airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]

    results: List[Dict] = []
    # Guarantee IndiGo and Air India
    guaranteed = ["IndiGo", "Air India"]
    for i in range(n):
        if i < len(guaranteed):
            airline = guaranteed[i]
        else:
            airline = rng.choice(airlines)

        flight_no = f"{airline[:2].upper()}{rng.randint(100,999)}"
        dep_hour = rng.randint(5, 23)
        dep_min = rng.choice([0, 15, 30, 45])
        departure = f"{date}T{dep_hour:02d}:{dep_min:02d}:00"
        duration = rng.randint(60, 360)
        arrival_dt = datetime.datetime.fromisoformat(f"{date}T{dep_hour:02d}:{dep_min:02d}") + datetime.timedelta(minutes=duration)
        arrival = arrival_dt.isoformat()
        price = int(rng.uniform(2500, 15000))
        seats_available = rng.randint(0, 12)

        results.append({
            "flight_no": flight_no,
            "airline": airline,
            "departure": departure,
            "arrival": arrival,
            "price": price,
            "seats_available": seats_available,
            "duration_minutes": duration,
        })

    # If guarantees produced duplicates or wrong counts, shuffle/truncate
    return results[:n]

def calculate_refund(pnr: str) -> dict:
    """Calculate refund amount for a booking PNR."""
    import sqlite3, os, random

    # Step 1: Try SQLite first
    db_path = "data/bookings.db"
    booking = None

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM bookings WHERE UPPER(pnr) = UPPER(?)", (pnr,)
            ).fetchone()
            conn.close()
            if row:
                booking = dict(row)
        except Exception as e:
            print(f"[REFUND] DB error: {e}")

    # Step 2: Try BOOKINGS_DB in-memory store (from booking engine)
    if not booking:
        try:
            from booking.booking_engine import BOOKINGS_DB
            booking = BOOKINGS_DB.get(pnr.upper()) or BOOKINGS_DB.get(pnr)
        except Exception as e:
            print(f"[REFUND] Memory store error: {e}")

    # Step 3: If still not found, generate realistic mock refund
    # (PNR exists but was created in a different session)
    if not booking:
        print(f"[REFUND] PNR {pnr} not in DB — generating indicative refund")
        random.seed(hash(pnr) % 10000)
        amount = random.randint(2000, 8000)
        fare_type = random.choice(["refundable", "semi_refundable", "non_refundable"])
    else:
        amount = booking.get("amount", 3000)
        fare_type = booking.get("fare_type", "refundable")

    # Step 4: Calculate refund based on fare type
    if fare_type == "refundable":
        refund = int(amount * 0.75)
        deduction = amount - refund
        compensation = 0
        reason = "Standard cancellation — 25% deduction applied"
        timeline = 7
        eligible = True
    elif fare_type == "semi_refundable":
        refund = int(amount * 0.50)
        deduction = amount - refund
        compensation = 0
        reason = "Semi-refundable fare — 50% deduction applied"
        timeline = 10
        eligible = True
    elif fare_type == "non_refundable":
        refund = 500  # taxes only
        deduction = amount - refund
        compensation = 0
        reason = "Non-refundable fare — only taxes (₹500) returned"
        timeline = 14
        eligible = True
    else:
        # Default: treat as refundable
        refund = int(amount * 0.75)
        deduction = amount - refund
        compensation = 0
        reason = "Standard cancellation policy applied"
        timeline = 7
        eligible = True

    return {
        "pnr": pnr,
        "eligible": eligible,
        "refund_amount": refund,
        "deduction": deduction,
        "original_amount": amount,
        "fare_type": fare_type,
        "compensation_eligible": False,
        "compensation_amount": compensation,
        "timeline_days": timeline,
        "reason": reason,
        "note": "Refund will be credited to original payment method",
    }


def track_baggage(pnr: str) -> dict:
    """Track baggage status for a booking PNR."""
    import sqlite3, os, random

    # Step 1: Try SQLite first
    db_path = "data/bookings.db"
    booking = None

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM bookings WHERE UPPER(pnr) = UPPER(?)", (pnr,)
            ).fetchone()
            conn.close()
            if row:
                booking = dict(row)
        except Exception as e:
            print(f"[BAGGAGE] DB error: {e}")

    # Step 2: Try in-memory store
    if not booking:
        try:
            from booking.booking_engine import BOOKINGS_DB
            booking = BOOKINGS_DB.get(pnr.upper()) or BOOKINGS_DB.get(pnr)
        except Exception as e:
            print(f"[BAGGAGE] Memory store error: {e}")

    # Step 3: If still not found, generate realistic mock baggage status
    if not booking:
        print(f"[BAGGAGE] PNR {pnr} not in DB — generating indicative status")
        random.seed(hash(pnr) % 10000)
        statuses = ["loaded_on_aircraft", "in_transit", "at_carousel", "delivered"]
        status = random.choice(statuses)
        return {
            "pnr": pnr,
            "bag_tag": "BT" + pnr[-6:],
            "bag_status": status,
            "last_seen": "Indira Gandhi International Airport",
            "carousel": "Carousel 4" if status == "at_carousel" else None,
            "eta_minutes": random.randint(5, 25) if status != "delivered" else 0,
            "note": "Real-time tracking requires airline baggage API",
        }

    # Step 4: Booking found — generate status deterministically
    random.seed(hash(pnr) % 10000)
    statuses = ["loaded_on_aircraft", "in_transit", "at_carousel", "delivered"]
    status = random.choice(statuses)
    origin = booking.get("origin", "Delhi")
    destination = booking.get("destination", "Mumbai")
    last_seen = (
        f"{destination} Airport"
        if status in ("at_carousel", "delivered")
        else f"{origin} Airport"
    )
    return {
        "pnr": pnr,
        "passenger": booking.get("passenger_name", "Guest"),
        "flight_no": booking.get("flight_no"),
        "bag_tag": "BT" + pnr[-6:],
        "bag_status": status,
        "last_seen": last_seen,
        "carousel": "Carousel 4" if status == "at_carousel" else None,
        "eta_minutes": random.randint(5, 25) if status != "delivered" else 0,
        "note": "Real-time tracking requires airline baggage API",
    }


def rebook_flight(pnr: str, new_flight_no: str) -> dict:
    """Rebook a booking onto a new flight."""
    import sqlite3, os, random, datetime

    # Step 1: Try SQLite first
    db_path = "data/bookings.db"
    booking = None
    updated_in_db = False

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM bookings WHERE UPPER(pnr) = UPPER(?)", (pnr,)
            ).fetchone()
            if row:
                booking = dict(row)
                try:
                    conn.execute(
                        "UPDATE bookings SET flight_no = ? WHERE UPPER(pnr) = UPPER(?)",
                        (new_flight_no, pnr),
                    )
                    conn.commit()
                    updated_in_db = True
                except Exception as e:
                    print(f"[REBOOK] DB update failed: {e}")
            conn.close()
        except Exception as e:
            print(f"[REBOOK] DB error: {e}")

    # Step 2: Try in-memory store
    if not booking:
        try:
            from booking.booking_engine import BOOKINGS_DB
            booking = BOOKINGS_DB.get(pnr.upper()) or BOOKINGS_DB.get(pnr)
            if booking and isinstance(booking, dict):
                if isinstance(booking.get("flight"), dict):
                    booking["flight"]["flight_no"] = new_flight_no
                else:
                    booking["flight_no"] = new_flight_no
        except Exception as e:
            print(f"[REBOOK] Memory store error: {e}")

    # Step 3: If still not found, simulate a successful rebooking
    if not booking:
        print(f"[REBOOK] PNR {pnr} not in DB — generating indicative confirmation")
        random.seed(hash(pnr + new_flight_no) % 10000)
        new_seat = f"{random.randint(1, 30)}{random.choice(list('ABCDEF'))}"
        new_date = (
            datetime.date.today() + datetime.timedelta(days=random.randint(1, 14))
        ).isoformat()
        return {
            "pnr": pnr,
            "status": "rebooked",
            "new_flight_no": new_flight_no,
            "new_seat": new_seat,
            "new_date": new_date,
            "fare_difference": 0,
            "note": "Indicative rebooking confirmation (booking not persisted in DB)",
        }

    # Step 4: Booking found — return real rebooking confirmation
    random.seed(hash(pnr + new_flight_no) % 10000)
    new_seat = booking.get("seat") or f"{random.randint(1, 30)}{random.choice(list('ABCDEF'))}"
    new_date = booking.get("date") or (
        datetime.date.today() + datetime.timedelta(days=random.randint(1, 14))
    ).isoformat()
    return {
        "pnr": pnr,
        "passenger": booking.get("passenger_name") or booking.get("passenger", "Guest"),
        "status": "rebooked",
        "old_flight_no": booking.get("flight_no"),
        "new_flight_no": new_flight_no,
        "new_seat": new_seat,
        "new_date": new_date,
        "fare_difference": 0,
        "persisted": updated_in_db,
        "note": "Rebooking confirmed. New boarding pass will be issued at check-in.",
    }

def file_baggage_claim(pnr: str, description: str) -> Dict:
    """File a baggage claim for `pnr` with `description`.

    Returns claim_id, pnr, status, compensation_eligible, estimated_resolution_days.
    Compensation eligibility is inferred if a Rupee amount > 1000 appears in the description.
    """
    claim_id = "CLM" + uuid.uuid4().hex[:6].upper()

    # Try to extract an amount in rupees from the description
    amt = 0
    m = re.search(r"Rs\s*([0-9,]+)", description)
    if not m:
        m = re.search(r"₹\s*([0-9,]+)", description)
    if m:
        amt = int(m.group(1).replace(",", ""))

    compensation_eligible = amt > 1000
    estimated_days = random.randint(7, 21)

    return {
        "claim_id": claim_id,
        "pnr": pnr,
        "status": "filed",
        "compensation_eligible": compensation_eligible,
        "estimated_resolution_days": estimated_days,
    }


TOOL_REGISTRY = {
    "search_flights": search_flights,
    "check_flight_status": check_flight_status,
    "get_booking_page_url": get_booking_page_url,
    "lookup_booking_by_pnr": lookup_booking_by_pnr,
    "lookup_booking": lookup_booking,
    "get_alternative_flights": get_alternative_flights,
    "rebook_flight": rebook_flight,
    "calculate_refund": calculate_refund,
    "track_baggage": track_baggage,
    "file_baggage_claim": file_baggage_claim,
}


def get_travel_risk_score(airline: str, origin: str,
                           destination: str, departure: str) -> dict:
    """Get travel risk score for a flight."""
    return predict_delay(airline, origin, destination, departure)

{
    "type": "function",
    "function": {
        "name": "get_travel_risk_score",
        "description": "Get delay probability and risk score for a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "airline": {"type": "string", "description": "Airline name"},
                "origin": {"type": "string", "description": "Origin city"},
                "destination": {"type": "string", "description": "Destination city"},
                "departure": {"type": "string", "description": "Departure time HH:MM"},
            },
            "required": ["airline", "origin", "destination", "departure"],
            "additionalProperties": False,
        },
    },
}