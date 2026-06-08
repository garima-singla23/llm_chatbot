import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()


def get_client():
  import razorpay
  return razorpay.Client(auth=(
    os.getenv("RAZORPAY_KEY_ID"),
    os.getenv("RAZORPAY_KEY_SECRET")
  ))


def create_order(amount_inr, flight_no, passenger_name) -> dict:
  client = get_client()
  order = client.order.create({
    "amount": int(amount_inr) * 100,
    "currency": "INR",
    "payment_capture": 1,
    "receipt": f"rcpt_{flight_no}",
    "notes": {"flight": flight_no, "passenger": passenger_name}
  })
  return {"order_id": order["id"], "amount_paise": int(amount_inr) * 100}


def verify_payment(order_id, payment_id, signature) -> bool:
  secret = os.getenv("RAZORPAY_KEY_SECRET", "").encode()
  msg = f"{order_id}|{payment_id}".encode()
  expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
  return hmac.compare_digest(expected, signature)


def fetch_payment(payment_id) -> dict:
  try:
    client = get_client()
    return client.payment.fetch(payment_id)
  except Exception as exc:
    return {"error": str(exc)}


TEST_CARD = "4111 1111 1111 1111"
TEST_EXPIRY = "12/28"
TEST_CVV = "123"
TEST_UPI = "success@razorpay"
