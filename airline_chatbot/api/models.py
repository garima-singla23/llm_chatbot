from pydantic import BaseModel
from typing import Optional


class PassengerDetails(BaseModel):
  name: str
  email: str
  phone: str
  age: Optional[int] = None
  gender: Optional[str] = None
  seat_pref: Optional[str] = "window"
  meal_pref: Optional[str] = "veg"


class FlightDetails(BaseModel):
  flight_no: str
  airline: str
  origin: str
  destination: str
  departure: str
  arrival: str
  date: Optional[str] = ""
  gate: Optional[str] = "TBA"
  terminal: Optional[str] = "T1"
  price: Optional[int] = 3000


class BookingRequest(BaseModel):
  flight: dict
  passenger: dict
  payment_id: str
  order_id: str
  amount: int


class BookingResponse(BaseModel):
  pnr: str
  seat: str
  status: str
  passenger: str
  flight_no: str
  boarding_time: str
  gate: str
  terminal: str
  message: str


class HealthResponse(BaseModel):
  status: str
  service: str
