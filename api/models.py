from pydantic import BaseModel


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
  message: str
