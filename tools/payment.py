import razorpay, os, hmac, hashlib
from dotenv import load_dotenv

load_dotenv()


def get_client():
  return razorpay.Client(auth=(
    os.getenv("RAZORPAY_KEY_ID"),
    os.getenv("RAZORPAY_KEY_SECRET")
  ))


def create_order(amount_inr:int, flight_no:str, passenger_name:str)->dict:
  """Create a Razorpay order. amount_inr is in rupees."""
  client = get_client()
  order = client.order.create({
    "amount": amount_inr * 100,  # convert to paise
    "currency": "INR",
    "payment_capture": 1,
    "receipt": f"rcpt_{flight_no}",
    "notes": {"flight": flight_no, "passenger": passenger_name}
  })
  return {"order_id": order["id"], "amount_paise": amount_inr*100}


def verify_payment(order_id:str, payment_id:str, signature:str)->bool:
  """Verify Razorpay payment signature."""
  secret = os.getenv("RAZORPAY_KEY_SECRET","").encode()
  msg = f"{order_id}|{payment_id}".encode()
  expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
  return hmac.compare_digest(expected, signature)


def fetch_payment(payment_id:str)->dict:
  """Fetch payment details from Razorpay."""
  try:
    client = get_client()
    return client.payment.fetch(payment_id)
  except Exception as e:
    return {"error":str(e)}


RAZORPAY_CHECKOUT_JS = "https://checkout.razorpay.com/v1/checkout.js"
TEST_CARD = "4111 1111 1111 1111"
TEST_CARD_EXPIRY = "12/28"
TEST_CARD_CVV = "123"
TEST_UPI = "success@razorpay"
TEST_NETBANKING = "Use any bank, credentials: any"