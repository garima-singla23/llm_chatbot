import random
import datetime
import string


AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Vistara"]

# Eight major Indian cities
CITIES = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Ahmedabad",
    "Pune",
]

# Base prices (INR) - IndiGo cheapest, Vistara most expensive
FLIGHT_PRICES = {
    "IndiGo": 3000,
    "SpiceJet": 3200,
    "Air India": 3800,
    "Vistara": 5200,
}

# Airline IATA-like codes for nicer flight numbers
AIRLINE_CODES = {"IndiGo": "6E", "Air India": "AI", "SpiceJet": "SG", "Vistara": "UK"}


def _parse_date(date):
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, datetime.datetime):
        return date.date()
    # expect ISO date string
    return datetime.date.fromisoformat(date)


def _duration_hours_by_distance(origin, destination):
    # simple heuristic: use index distance in CITIES list to estimate short/med/long
    try:
        i = CITIES.index(origin)
        j = CITIES.index(destination)
    except ValueError:
        # default medium
        return 2
    diff = abs(i - j)
    if diff <= 1:
        return 1
    if diff <= 3:
        return 2
    return 3


def _random_flight_no(airline_code: str) -> str:
    digits = "".join(random.choices(string.digits, k=4))
    return f"{airline_code}{digits}"


def generate_flights(origin, destination, date, n=5):
    """Generate n mock flights between origin and destination on the given date.

    date may be a datetime.date, datetime.datetime, or ISO date string (YYYY-MM-DD).
    Returns a list of flight dicts sorted by price ascending.
    """
    if origin == destination:
        raise ValueError("origin and destination must differ")

    if origin not in CITIES or destination not in CITIES:
        raise ValueError("origin and destination must be one of predefined CITIES")

    travel_date = _parse_date(date)

    flights = []
    start_time = datetime.time(hour=6, minute=0)
    for i in range(n):
        # pick an airline (round-robin to get variety)
        airline = AIRLINES[i % len(AIRLINES)]
        code = AIRLINE_CODES.get(airline, airline[:2].upper())

        dep_hour = (6 + 2 * i) % 24
        departure_dt = datetime.datetime.combine(travel_date, datetime.time(hour=dep_hour, minute=0))

        base_hours = _duration_hours_by_distance(origin, destination)
        extra_minutes = random.randint(0, 50)
        duration_td = datetime.timedelta(hours=base_hours, minutes=extra_minutes)

        arrival_dt = departure_dt + duration_td

        # price variation ±20%
        base_price = FLIGHT_PRICES.get(airline, 3500)
        price = int(round(base_price * random.uniform(0.8, 1.2) / 10.0) * 10)

        # seats and class
        seat_choice = random.choices(["Economy", "Premium Economy", "Business"], weights=[85, 10, 5], k=1)[0]
        seats_available = random.randint(0, 150)

        flight = {
            "flight_no": _random_flight_no(code),
            "airline": airline,
            "origin": origin,
            "destination": destination,
            "date": travel_date.isoformat(),
            "departure": departure_dt.strftime("%H:%M"),
            "arrival": arrival_dt.strftime("%H:%M"),
            "duration": f"{duration_td.seconds//3600}h {(duration_td.seconds%3600)//60}m",
            "price": price,
            "seats_available": seats_available,
            "class": seat_choice,
        }

        flights.append(flight)

    # sort by price ascending
    flights.sort(key=lambda f: f["price"])
    return flights


if __name__ == "__main__":
    # quick demo
    demo = generate_flights("Mumbai", "Delhi", datetime.date.today().isoformat(), n=5)
    for f in demo:
        print(f)
