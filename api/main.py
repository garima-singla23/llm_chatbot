from fastapi import FastAPI, HTTPException
import sqlite3, uuid, os, random, string
from datetime import datetime
from dotenv import load_dotenv

from api.models import BookingRequest, BookingResponse

load_dotenv()

app = FastAPI(title="AirAssist Confirmation Engine")
DB_PATH = "data/bookings.db"


def init_db():
  os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.execute("""CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pnr TEXT UNIQUE NOT NULL,
    passenger_name TEXT,
    passenger_email TEXT,
    passenger_phone TEXT,
    flight_no TEXT,
    airline TEXT,
    origin TEXT,
    destination TEXT,
    departure TEXT,
    arrival TEXT,
    date TEXT,
    seat TEXT,
    meal TEXT,
    amount INTEGER,
    payment_id TEXT,
    razorpay_order_id TEXT,
    status TEXT DEFAULT 'confirmed',
    booked_at TEXT,
    gate TEXT,
    terminal TEXT
  )""")
  conn.commit()
  conn.close()


init_db()


def generate_pnr()->str:
  chars = string.ascii_uppercase + string.digits
  return "PNR" + "".join(random.choices(chars, k=6))


def assign_seat(seat_pref:str)->str:
  rows = list(range(10,40))
  random.shuffle(rows)
  row = rows[0]
  if seat_pref == "window": col = random.choice(["A","F"])
  elif seat_pref == "aisle": col = random.choice(["C","D"])
  else: col = random.choice(["B","E"])
  return f"{row}{col}"


@app.post("/confirm-booking", response_model=BookingResponse)
def confirm_booking(req: BookingRequest):
  pnr = generate_pnr()
  seat = assign_seat(req.passenger.get("seat_pref","window"))
  dep = req.flight.get("departure","06:00")
  hour = int(dep.split(":")[0]) if ":" in dep else 6
  boarding_time = f"{hour-1:02d}:30"
  booked_at = datetime.now().isoformat()
  
  conn = sqlite3.connect(DB_PATH)
  conn.execute("""INSERT INTO bookings (
    pnr,passenger_name,passenger_email,passenger_phone,
    flight_no,airline,origin,destination,departure,arrival,
    date,seat,meal,amount,payment_id,razorpay_order_id,
    status,booked_at,gate,terminal
  ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
    pnr,
    req.passenger.get("name",""),
    req.passenger.get("email",""),
    req.passenger.get("phone","") ,
    req.flight.get("flight_no",""),
    req.flight.get("airline",""),
    req.flight.get("origin",""),
    req.flight.get("destination",""),
    req.flight.get("departure",""),
    req.flight.get("arrival",""),
    req.flight.get("date",""),
    seat,
    req.passenger.get("meal_pref","veg"),
    req.amount,
    req.payment_id,
    req.order_id,
    "confirmed",
    booked_at,
    req.flight.get("gate","TBA"),
    req.flight.get("terminal","T1")
  ))
  conn.commit()
  conn.close()
  
  # Fire notifications asynchronously
  from tools.notifications import send_booking_confirmation
  try:
    send_booking_confirmation(
      email=req.passenger.get("email",""),
      phone=req.passenger.get("phone",""),
      name=req.passenger.get("name",""),
      pnr=pnr,
      flight=req.flight,
      seat=seat,
      boarding_time=boarding_time,
      amount=req.amount
    )
  except Exception as e:
    print(f"[NOTIFY] Error: {e}")
  
  return BookingResponse(
    pnr=pnr,
    seat=seat,
    status="confirmed",
    passenger=req.passenger.get("name",""),
    flight_no=req.flight.get("flight_no",""),
    boarding_time=boarding_time,
    message=f"Booking confirmed! Your PNR is {pnr}. Check your email and SMS."
  )


@app.get("/booking/{pnr}")
def get_booking(pnr:str):
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  row = conn.execute("SELECT * FROM bookings WHERE pnr=?", (pnr,)).fetchone()
  conn.close()
  if not row: raise HTTPException(404, "Booking not found")
  return dict(row)


@app.get("/bookings/{email}")
def get_bookings_by_email(email:str):
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  rows = conn.execute(
    "SELECT * FROM bookings WHERE passenger_email=? ORDER BY booked_at DESC LIMIT 10",
    (email,)).fetchall()
  conn.close()
  return [dict(r) for r in rows]


@app.get("/health")
def health(): return {"status":"ok","service":"AirAssist Confirmation Engine"}
