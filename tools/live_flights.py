import requests, os
from dotenv import load_dotenv
from datetime import datetime, timedelta
load_dotenv()

BASE = "http://api.aviationstack.com/v1"
KEY = os.getenv("AVIATIONSTACK_KEY")

CITY_TO_IATA = {
  "delhi":"DEL","new delhi":"DEL","mumbai":"BOM","bangalore":"BLR",
  "bengaluru":"BLR","chennai":"MAA","kolkata":"CCU","hyderabad":"HYD",
  "pune":"PNQ","ahmedabad":"AMD","goa":"GOI","kochi":"COK",
  "jaipur":"JAI","lucknow":"LKO","chandigarh":"IXC"
}

IATA_TO_CITY = {v:k.title() for k,v in CITY_TO_IATA.items()}

INDIAN_AIRLINES = {"6E":"IndiGo","AI":"Air India","SG":"SpiceJet",
                   "UK":"Vistara","G8":"Go First","QP":"Akasa Air"}

def city_to_iata(city:str)->str|None:
  return CITY_TO_IATA.get(city.lower().strip())

def search_live_flights(origin:str, destination:str, date:str=None)->list[dict]:
  """Search real flights via AviationStack. Falls back to mock if API fails."""
  origin_iata = city_to_iata(origin) or origin.upper()
  dest_iata = city_to_iata(destination) or destination.upper()
  
  params = {
    "access_key": KEY,
    "dep_iata": origin_iata,
    "arr_iata": dest_iata,
    "airline_iata": ",".join(INDIAN_AIRLINES.keys()),
    "flight_status": "scheduled",
    "limit": 10
  }
  
  try:
    if not KEY:
      raise ValueError("No API key")
    r = requests.get(f"{BASE}/flights", params=params, timeout=8)
    data = r.json()
    
    if "data" not in data or not data["data"]:
      print(f"[LIVE] No flights found, using mock fallback")
      return _mock_flights(origin, destination, date)
    
    flights = []
    for f in data["data"][:6]:
      dep = f.get("departure",{})
      arr = f.get("arrival",{})
      airline_code = f.get("airline",{}).get("iata","??")
      flight_num = f.get("flight",{}).get("iata","??")
      
      dep_time = dep.get("scheduled","")[:16] if dep.get("scheduled") else "TBD"
      arr_time = arr.get("scheduled","")[:16] if arr.get("scheduled") else "TBD"
      dep_display = dep_time[11:16] if len(dep_time)>11 else dep_time
      arr_display = arr_time[11:16] if len(arr_time)>11 else arr_time
      
      import random
      random.seed(flight_num)
      price = random.randint(2200, 8500)
      
      flights.append({
        "flight_no": flight_num,
        "airline": INDIAN_AIRLINES.get(airline_code, airline_code),
        "airline_code": airline_code,
        "origin": origin,
        "origin_iata": origin_iata,
        "destination": destination,
        "destination_iata": dest_iata,
        "departure": dep_display,
        "arrival": arr_display,
        "departure_full": dep_time,
        "arrival_full": arr_time,
        "status": f.get("flight_status","scheduled"),
        "terminal": dep.get("terminal","T1"),
        "gate": dep.get("gate","TBA"),
        "price": price,
        "seats_available": random.randint(3,40),
        "source": "live"
      })
    
    print(f"[LIVE] Found {len(flights)} real flights {origin}→{destination}")
    return sorted(flights, key=lambda x:x["price"])
    
  except Exception as e:
    print(f"[LIVE] API error: {e} — using mock fallback")
    return _mock_flights(origin, destination, date)

def check_live_status(flight_no:str)->dict:
  """Check real-time status of a specific flight."""
  try:
    if not KEY: raise ValueError("No key")
    r = requests.get(f"{BASE}/flights",
      params={"access_key":KEY,"flight_iata":flight_no,"limit":1}, timeout=8)
    data = r.json()
    if "data" in data and data["data"]:
      f = data["data"][0]
      dep = f.get("departure",{})
      arr = f.get("arrival",{})
      delay = dep.get("delay",0) or 0
      return {
        "flight_no": flight_no,
        "status": f.get("flight_status","unknown"),
        "delay_minutes": int(delay),
        "gate": dep.get("gate","TBA"),
        "terminal": dep.get("terminal","T1"),
        "departure": dep.get("scheduled","")[:16],
        "arrival": arr.get("scheduled","")[:16],
        "source": "live"
      }
  except Exception as e:
    print(f"[STATUS] API error: {e}")
  
  import random
  random.seed(hash(flight_no) % 1000)
  status = random.choice(["scheduled","scheduled","scheduled","delayed","cancelled"])
  delay = random.randint(30,180) if status=="delayed" else 0
  return {
    "flight_no":flight_no,"status":status,"delay_minutes":delay,
    "gate":"B14","terminal":"T2","source":"mock"
  }

def _mock_flights(origin:str, destination:str, date:str=None)->list[dict]:
  """Realistic mock fallback when API is unavailable."""
  import random, uuid
  airlines = [
    ("IndiGo","6E",2500,5500),("Air India","AI",3200,7000),
    ("SpiceJet","SG",2200,5000),("Vistara","UK",4000,9000)
  ]
  flights = []
  for i,(name,code,min_p,max_p) in enumerate(airlines[:3]):
    random.seed(hash(f"{origin}{destination}{i}"))
    dep_h = 6 + i*3
    arr_h = dep_h + random.randint(1,3)
    flights.append({
      "flight_no":f"{code}{random.randint(100,999)}",
      "airline":name,"airline_code":code,
      "origin":origin,"destination":destination,
      "departure":f"{dep_h:02d}:00","arrival":f"{arr_h:02d}:00",
      "price":random.randint(min_p,max_p),
      "seats_available":random.randint(5,40),
      "terminal":"T2","gate":"B12","source":"mock"
    })
  return sorted(flights, key=lambda x:x["price"])
