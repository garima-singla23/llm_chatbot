import pytest
import datetime
from booking import booking_engine
from booking.booking_engine import BOOKINGS_DB


@pytest.fixture(autouse=True)
def clear_bookings_db():
    """Clear BOOKINGS_DB before each test to avoid cross-test state pollution."""
    BOOKINGS_DB.clear()
    yield
    BOOKINGS_DB.clear()


class TestSearchFlights:
    """Unit tests for search_flights function."""

    def test_search_flights_returns_list(self):
        """Test that search_flights returns a list."""
        result = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        assert isinstance(result, list)

    def test_search_flights_returns_dicts(self):
        """Test that search_flights returns a list of dicts."""
        result = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        assert len(result) > 0
        for flight in result:
            assert isinstance(flight, dict)

    def test_search_flights_has_required_keys(self):
        """Test that each flight dict has all required keys."""
        required_keys = {
            "flight_no", "airline", "origin", "destination",
            "date", "departure", "arrival", "price", "seats_available"
        }
        
        result = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        
        for flight in result:
            actual_keys = set(flight.keys())
            assert required_keys.issubset(actual_keys), \
                f"Missing keys: {required_keys - actual_keys}"

    def test_search_flights_sorted_by_price_ascending(self):
        """Test that search_flights returns results sorted by price ascending."""
        result = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        
        prices = [flight["price"] for flight in result]
        assert prices == sorted(prices), "Flights should be sorted by price ascending"

    def test_search_flights_multiple_flights(self):
        """Test that search_flights returns multiple flight options."""
        result = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        
        # Default is n=5, so we should get at least 5 flights
        assert len(result) >= 5

    def test_search_flights_correct_route(self):
        """Test that all returned flights have correct origin and destination."""
        origin = "Mumbai"
        destination = "Delhi"
        result = booking_engine.search_flights(origin, destination, "2026-05-30")
        
        for flight in result:
            assert flight["origin"] == origin
            assert flight["destination"] == destination

    def test_search_flights_different_dates(self):
        """Test search_flights with different date formats."""
        # ISO string
        result1 = booking_engine.search_flights("Mumbai", "Delhi", "2026-06-15")
        assert len(result1) > 0
        
        # datetime.date object
        date_obj = datetime.date(2026, 6, 16)
        result2 = booking_engine.search_flights("Mumbai", "Delhi", date_obj)
        assert len(result2) > 0


class TestBookFlight:
    """Unit tests for book_flight function."""

    def test_book_flight_returns_dict(self):
        """Test that book_flight returns a dict."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        
        assert isinstance(booking, dict)

    def test_book_flight_pnr_format(self):
        """Test that book_flight returns PNR starting with 'PNR' (6 chars after)."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        assert pnr.startswith("PNR"), "PNR should start with 'PNR'"
        assert len(pnr) > 3, "PNR should be longer than 3 characters"

    def test_book_flight_pnr_characters(self):
        """Test that PNR after 'PNR' consists of hex characters."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        # PNR format: "PNR" + 6 hex chars
        after_prefix = pnr[3:]
        assert all(c in "0123456789ABCDEF" for c in after_prefix), \
            "Characters after 'PNR' should be hex"

    def test_book_flight_has_required_fields(self):
        """Test that booking dict has required fields."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe", seat_pref="aisle")
        
        required_fields = {
            "pnr", "passenger", "flight", "seat", "seat_pref",
            "status", "booked_at", "meal"
        }
        assert set(booking.keys()) >= required_fields

    def test_book_flight_stores_in_bookings_db(self):
        """Test that book_flight stores booking in BOOKINGS_DB."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "Jane Smith")
        pnr = booking["pnr"]
        
        # Should be retrievable from BOOKINGS_DB
        assert pnr in BOOKINGS_DB
        assert BOOKINGS_DB[pnr] == booking

    def test_book_flight_status_confirmed(self):
        """Test that book_flight sets status to 'confirmed'."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        
        assert booking["status"] == "confirmed"

    def test_book_flight_passenger_name(self):
        """Test that book_flight stores correct passenger name."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        passenger_name = "Alice Johnson"
        booking = booking_engine.book_flight(flight, passenger_name)
        
        assert booking["passenger"] == passenger_name

    def test_book_flight_seat_assigned(self):
        """Test that book_flight assigns a seat."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        
        assert "seat" in booking
        assert booking["seat"] is not None
        # Seat should be in format like "12A"
        assert len(booking["seat"]) >= 2

    def test_book_flight_multiple_bookings_unique_pnr(self):
        """Test that multiple bookings get different PNRs."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking1 = booking_engine.book_flight(flight, "John Doe")
        booking2 = booking_engine.book_flight(flight, "Jane Smith")
        
        assert booking1["pnr"] != booking2["pnr"]


class TestGetBooking:
    """Unit tests for get_booking function."""

    def test_get_booking_retrieves_correct_booking(self):
        """Test that get_booking retrieves the correct booking from BOOKINGS_DB."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        retrieved = booking_engine.get_booking(pnr)
        
        assert retrieved == booking

    def test_get_booking_returns_none_for_unknown_pnr(self):
        """Test that get_booking returns None for unknown PNR."""
        result = booking_engine.get_booking("PNR000000")
        
        assert result is None

    def test_get_booking_multiple_bookings(self):
        """Test get_booking with multiple bookings in DB."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking1 = booking_engine.book_flight(flight, "John Doe")
        booking2 = booking_engine.book_flight(flight, "Jane Smith")
        
        retrieved1 = booking_engine.get_booking(booking1["pnr"])
        retrieved2 = booking_engine.get_booking(booking2["pnr"])
        
        assert retrieved1 == booking1
        assert retrieved2 == booking2
        assert retrieved1 != retrieved2


class TestCancelBooking:
    """Unit tests for cancel_booking function."""

    def test_cancel_booking_sets_status_to_cancelled(self):
        """Test that cancel_booking sets status to 'cancelled'."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        # Check that the booking's status is updated in BOOKINGS_DB
        assert BOOKINGS_DB[pnr]["status"] == "cancelled"

    def test_cancel_booking_returns_pnr(self):
        """Test that cancel_booking returns dict with correct PNR."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        assert result["pnr"] == pnr

    def test_cancel_booking_refund_75_percent(self):
        """Test that cancel_booking returns refund_amount = 75% of price."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        price = flight["price"]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        expected_refund = int(round(price * 0.75))
        assert result["refund_amount"] == expected_refund

    def test_cancel_booking_returns_status_cancelled(self):
        """Test that cancel_booking return dict has status='cancelled'."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        assert result["status"] == "cancelled"

    def test_cancel_booking_returns_processing_days(self):
        """Test that cancel_booking return dict includes processing_days."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        assert "processing_days" in result
        assert result["processing_days"] == 7

    def test_cancel_booking_unknown_pnr_returns_error(self):
        """Test that cancel_booking returns error dict for unknown PNR."""
        result = booking_engine.cancel_booking("PNR000000")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Booking not found"

    def test_cancel_booking_adds_cancelled_at_timestamp(self):
        """Test that cancel_booking adds cancelled_at timestamp to booking."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        booking_engine.cancel_booking(pnr)
        
        assert "cancelled_at" in BOOKINGS_DB[pnr]
        # Should be ISO format string
        assert isinstance(BOOKINGS_DB[pnr]["cancelled_at"], str)

    def test_cancel_booking_multiple_bookings(self):
        """Test cancelling one booking doesn't affect others."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        flight = flights[0]
        
        booking1 = booking_engine.book_flight(flight, "John Doe")
        booking2 = booking_engine.book_flight(flight, "Jane Smith")
        pnr1 = booking1["pnr"]
        pnr2 = booking2["pnr"]
        
        booking_engine.cancel_booking(pnr1)
        
        assert BOOKINGS_DB[pnr1]["status"] == "cancelled"
        assert BOOKINGS_DB[pnr2]["status"] == "confirmed"

    def test_cancel_booking_refund_calculation_edge_case(self):
        """Test refund calculation with specific price."""
        flights = booking_engine.search_flights("Mumbai", "Delhi", "2026-05-30")
        # Find a flight with a specific price or use the first one
        flight = flights[0]
        
        booking = booking_engine.book_flight(flight, "John Doe")
        pnr = booking["pnr"]
        
        result = booking_engine.cancel_booking(pnr)
        
        # Verify 75% calculation
        price = flight["price"]
        expected_refund = int(round(price * 0.75))
        actual_refund = result["refund_amount"]
        
        assert actual_refund == expected_refund
