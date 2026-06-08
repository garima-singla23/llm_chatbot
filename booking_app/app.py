from flask import Flask, render_template, request, redirect, session, jsonify
import os, requests
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY","dev_secret")

@app.route("/book", methods=["GET"])
def booking_page():
  """Pre-filled booking form. Flight details passed as query params."""
  flight = {
    "flight_no": request.args.get("flight_no",""),
    "airline": request.args.get("airline",""),
    "origin": request.args.get("origin",""),
    "destination": request.args.get("destination",""),
    "departure": request.args.get("departure",""),
    "arrival": request.args.get("arrival",""),
    "price": request.args.get("price",""),
    "date": request.args.get("date",""),
    "gate": request.args.get("gate","TBA"),
    "terminal": request.args.get("terminal","T1"),
  }
  session["pending_flight"] = flight
  razorpay_key = os.getenv("RAZORPAY_KEY_ID","")
  return render_template("booking.html", flight=flight, razorpay_key=razorpay_key)

@app.route("/initiate-payment", methods=["POST"])
def initiate_payment():
  """Create Razorpay order and return order_id to frontend."""
  import razorpay
  client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
  )
  flight = session.get("pending_flight",{})
  passenger = {
    "name": request.form.get("name",""),
    "email": request.form.get("email",""),
    "phone": request.form.get("phone",""),
    "age": request.form.get("age",""),
    "gender": request.form.get("gender",""),
    "seat_pref": request.form.get("seat_pref","window"),
    "meal_pref": request.form.get("meal_pref","veg"),
  }
  session["passenger"] = passenger
  
  amount = int(float(flight.get("price",3000))) * 100  # paise
  order = client.order.create({
    "amount": amount,
    "currency": "INR",
    "payment_capture": 1,
    "notes": {
      "flight_no": flight.get("flight_no",""),
      "passenger": passenger["name"]
    }
  })
  session["razorpay_order_id"] = order["id"]
  return jsonify({
    "order_id": order["id"],
    "amount": amount,
    "key": os.getenv("RAZORPAY_KEY_ID"),
    "name": passenger["name"],
    "email": passenger["email"],
    "phone": passenger["phone"],
    "flight_no": flight.get("flight_no",""),
  })

@app.route("/payment-success", methods=["POST"])
def payment_success():
  """Razorpay calls this after successful payment."""
  import razorpay, hmac, hashlib
  client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
  )
  payment_id = request.form.get("razorpay_payment_id")
  order_id = request.form.get("razorpay_order_id")
  signature = request.form.get("razorpay_signature")
  
  # Verify signature
  msg = f"{order_id}|{payment_id}"
  secret = os.getenv("RAZORPAY_KEY_SECRET","").encode()
  expected = hmac.new(secret, msg.encode(), hashlib.sha256).hexdigest()
  
  if signature != expected:
    return "Payment verification failed", 400
  
  flight = session.get("pending_flight",{})
  passenger = session.get("passenger",{})
  
  # Call FastAPI confirmation engine
  try:
    resp = requests.post("http://localhost:8001/confirm-booking", json={
      "flight": flight,
      "passenger": passenger,
      "payment_id": payment_id,
      "order_id": order_id,
      "amount": int(flight.get("price",3000))
    }, timeout=10)
    booking = resp.json()
  except Exception as e:
    print(f"Confirmation API error: {e}")
    booking = {"pnr": f"PNR{payment_id[:6].upper()}", "status":"confirmed"}
  
  return render_template("success.html",
    flight=flight, passenger=passenger, booking=booking)

if __name__ == "__main__":
  app.run(port=int(os.getenv("FLASK_PORT",5000)), debug=True)
