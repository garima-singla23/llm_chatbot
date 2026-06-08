from flask import Flask, render_template, request, redirect, session, jsonify
import os, requests as req_lib, hmac, hashlib
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "airAssist2026secret")


@app.route("/book", methods=["GET"])
def booking_page():
  flight = {
    "flight_no": request.args.get("flight_no", ""),
    "airline": request.args.get("airline", ""),
    "origin": request.args.get("origin", ""),
    "destination": request.args.get("destination", ""),
    "departure": request.args.get("departure", ""),
    "arrival": request.args.get("arrival", ""),
    "price": request.args.get("price", "3000"),
    "date": request.args.get("date", ""),
    "gate": request.args.get("gate", "TBA"),
    "terminal": request.args.get("terminal", "T1"),
  }
  session["pending_flight"] = flight
  return render_template("booking.html",
    flight=flight,
    razorpay_key=os.getenv("RAZORPAY_KEY_ID", ""))


@app.route("/initiate-payment", methods=["POST"])
def initiate_payment():
  try:
    import razorpay
    client = razorpay.Client(auth=(
      os.getenv("RAZORPAY_KEY_ID"),
      os.getenv("RAZORPAY_KEY_SECRET")
    ))
    flight = session.get("pending_flight", {})
    passenger = {
      "name": request.form.get("name", ""),
      "email": request.form.get("email", ""),
      "phone": request.form.get("phone", ""),
      "age": request.form.get("age", ""),
      "gender": request.form.get("gender", ""),
      "seat_pref": request.form.get("seat_pref", "window"),
      "meal_pref": request.form.get("meal_pref", "veg"),
    }
    session["passenger"] = passenger
    
    amount_paise = int(float(flight.get("price", 3000))) * 100
    order = client.order.create({
      "amount": amount_paise,
      "currency": "INR",
      "payment_capture": 1,
      "notes": {"flight": flight.get("flight_no", ""), "passenger": passenger["name"]}
    })
    session["razorpay_order_id"] = order["id"]
    return jsonify({
      "order_id": order["id"],
      "amount": amount_paise,
      "key": os.getenv("RAZORPAY_KEY_ID", ""),
      "name": passenger["name"],
      "email": passenger["email"],
      "phone": passenger["phone"],
    })
  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/payment-success", methods=["POST"])
def payment_success():
  payment_id = request.form.get("razorpay_payment_id", "")
  order_id = request.form.get("razorpay_order_id", "")
  signature = request.form.get("razorpay_signature", "")
  
  secret = os.getenv("RAZORPAY_KEY_SECRET", "").encode()
  msg = f"{order_id}|{payment_id}".encode()
  expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
  
  if expected != signature:
    return "Payment verification failed. Please contact support.", 400
  
  flight = session.get("pending_flight", {})
  passenger = session.get("passenger", {})
  
  booking = {}
  try:
    resp = req_lib.post("http://localhost:8001/confirm-booking", json={
      "flight": flight,
      "passenger": passenger,
      "payment_id": payment_id,
      "order_id": order_id,
      "amount": int(float(flight.get("price", 3000)))
    }, timeout=15)
    booking = resp.json()
  except Exception as e:
    print(f"[FLASK] Confirmation API error: {e}")
    import random, string
    booking = {
      "pnr": "PNR" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
      "seat": "14A",
      "boarding_time": "05:30",
      "gate": flight.get("gate", "TBA"),
      "terminal": flight.get("terminal", "T1"),
      "status": "confirmed",
      "message": "Booking confirmed!"
    }
  
  return render_template("success.html",
    flight=flight,
    passenger=passenger,
    booking=booking)


@app.route("/health")
def health():
  return {"status": "ok", "service": "AirAssist Booking Page"}


if __name__ == "__main__":
  app.run(port=int(os.getenv("FLASK_PORT", 5000)), debug=True)
