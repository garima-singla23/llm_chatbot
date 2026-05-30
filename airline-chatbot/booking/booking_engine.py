import uuid
import datetime
import random

from . import mock_data


# In-memory bookings store
BOOKINGS_DB = {}


def search_flights(origin, destination, date):
    return mock_data.generate_flights(origin, destination, date)


def _generate_pnr():
    # 6 hex chars uppercase
    return "PNR" + uuid.uuid4().hex[:6].upper()


def _random_seat():
    row = random.randint(1, 30)
    seat = random.choice(["A", "B", "C", "D", "E", "F"])
    return f"{row}{seat}"


def book_flight(flight: dict, passenger_name: str, seat_pref: str = "window") -> dict:
    pnr = _generate_pnr()
    seat = _random_seat()
    booked_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    booking = {
        "pnr": pnr,
        "passenger": passenger_name,
        "flight": flight,
        "seat": seat,
        "seat_pref": seat_pref,
        "status": "confirmed",
        "booked_at": booked_at,
        "meal": "veg",
    }

    BOOKINGS_DB[pnr] = booking
    return booking


def get_booking(pnr: str):
    return BOOKINGS_DB.get(pnr)


def cancel_booking(pnr: str) -> dict:
    booking = BOOKINGS_DB.get(pnr)
    if not booking:
        return {"error": "Booking not found"}

    price = booking.get("flight", {}).get("price", 0)
    refund = int(round(price * 0.75))

    booking["status"] = "cancelled"
    booking["cancelled_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "pnr": pnr,
        "refund_amount": refund,
        "status": "cancelled",
        "processing_days": 7,
    }


__all__ = ["BOOKINGS_DB", "search_flights", "book_flight", "get_booking", "cancel_booking"]
