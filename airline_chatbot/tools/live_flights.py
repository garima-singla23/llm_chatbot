import requests
import os
import random
from dotenv import load_dotenv
from datetime import datetime, timedelta

from tools.price_scraper import get_indicative_price

load_dotenv()

BASE = "http://api.aviationstack.com/v1"
KEY = os.getenv("AVIATIONSTACK_KEY")

CITY_TO_IATA = {
  "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR",
  "bengaluru": "BLR", "chennai": "MAA", "kolkata": "CCU", "hyderabad": "HYD",
  "pune": "PNQ", "ahmedabad": "AMD", "goa": "GOI", "kochi": "COK",
  "jaipur": "JAI", "lucknow": "LKO", "chandigarh": "IXC"
}

INDIAN_AIRLINES = {
  "6E": "IndiGo",
  "AI": "Air India",
  "SG": "SpiceJet",
  "UK": "Vistara",
  "QP": "Akasa Air",
}


def city_to_iata(city: str):
  return CITY_TO_IATA.get(city.lower().strip())


def _price_for(flight_no: str) -> int:
  rng = random.Random(flight_no)
  return rng.randint(2200, 8500)


def search_live_flights(origin: str, destination: str, date=None) -> list[dict]:
  origin_iata = city_to_iata(origin) or origin.upper()
  destination_iata = city_to_iata(destination) or destination.upper()

  if not KEY:
    return _mock_flights(origin, destination, date)

  indicative_price = get_indicative_price(origin_iata, destination_iata, date)
  price = indicative_price or random.randint(2200, 8500)
  price_source = "google_flights" if indicative_price else "estimated"

  params = {
    "access_key": KEY,
    "dep_iata": origin_iata,
    "arr_iata": destination_iata,
    "limit": 10,
  }
  if date:
    params["flight_date"] = str(date)

  try:
    resp = requests.get(f"{BASE}/flights", params=params, timeout=8)
    data = resp.json().get("data", [])
    flights = []
    for item in data:
      airline = item.get("airline", {})
      airline_code = airline.get("iata", "")
      if airline_code not in INDIAN_AIRLINES:
        continue
      departure = item.get("departure", {})
      arrival = item.get("arrival", {})
      flight_no = item.get("flight", {}).get("iata", "")
      dep_time = (departure.get("scheduled") or "")
      arr_time = (arrival.get("scheduled") or "")
      dep_display = dep_time[11:16] if len(dep_time) >= 16 else dep_time[:5] or "TBD"
      arr_display = arr_time[11:16] if len(arr_time) >= 16 else arr_time[:5] or "TBD"
      flights.append({
        "flight_no": flight_no,
        "airline": INDIAN_AIRLINES.get(airline_code, airline_code),
        "airline_code": airline_code,
        "origin": origin,
        "destination": destination,
        "departure": dep_display,
        "arrival": arr_display,
        "price": price,
        "seats_available": random.randint(3, 40),
        "terminal": departure.get("terminal", "T1") or "T1",
        "gate": departure.get("gate", "TBA") or "TBA",
        "status": item.get("flight_status", "scheduled"),
        "source": price_source,
      })
    if not flights:
      return _mock_flights(origin, destination, date)
    print(f"[LIVE] Found {len(flights)} flights {origin}->{destination}")
    return sorted(flights, key=lambda x: x["price"])
  except Exception as exc:
    print(f"[LIVE] API error: {exc}")
    return _mock_flights(origin, destination, date)


def check_live_status(flight_no):
  try:
    if not KEY:
      raise ValueError("No AviationStack key")
    resp = requests.get(f"{BASE}/flights", params={"access_key": KEY, "flight_iata": flight_no, "limit": 1}, timeout=8)
    data = resp.json().get("data", [])
    if data:
      item = data[0]
      departure = item.get("departure", {})
      arrival = item.get("arrival", {})
      return {
        "flight_no": flight_no,
        "status": item.get("flight_status", "unknown"),
        "delay_minutes": int(departure.get("delay") or 0),
        "gate": departure.get("gate", "TBA") or "TBA",
        "terminal": departure.get("terminal", "T1") or "T1",
        "departure": (departure.get("scheduled") or "")[:16],
        "arrival": (arrival.get("scheduled") or "")[:16],
        "source": "live",
      }
  except Exception as exc:
    print(f"[LIVE] Status API error: {exc}")

  rng = random.Random(flight_no)
  status = rng.choice(["scheduled", "scheduled", "scheduled", "delayed", "cancelled"])
  return {
    "flight_no": flight_no,
    "status": status,
    "delay_minutes": rng.randint(10, 180) if status == "delayed" else 0,
    "gate": "B14",
    "terminal": "T2",
    "departure": "06:00",
    "arrival": "08:30",
    "source": "mock",
  }


def _mock_flights(origin: str, destination: str, date=None) -> list[dict]:
  print(f"[MOCK] Using fallback flights for {origin}->{destination}")
  airlines = [
    ("IndiGo", "6E"),
    ("Air India", "AI"),
    ("SpiceJet", "SG"),
  ]
  flights = []
  for idx, (airline, code) in enumerate(airlines):
    rng = random.Random(f"{origin}-{destination}-{idx}")
    dep_hour = 6 + idx * 3
    arr_hour = dep_hour + rng.randint(1, 3)
    flight_no = f"{code}{rng.randint(100,999)}"
    flights.append({
      "flight_no": flight_no,
      "airline": airline,
      "airline_code": code,
      "origin": origin,
      "destination": destination,
      "departure": f"{dep_hour:02d}:00",
      "arrival": f"{arr_hour:02d}:00",
      "price": _price_for(flight_no),
      "seats_available": rng.randint(5, 40),
      "terminal": "T2",
      "gate": "B12",
      "status": "scheduled",
      "source": "estimated",
    })
  return sorted(flights, key=lambda x: x["price"])
