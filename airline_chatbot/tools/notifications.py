import os
from dotenv import load_dotenv

load_dotenv()


def send_email_confirmation(email, name, pnr, flight, seat, boarding_time, amount) -> bool:
  try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    subject = f"Booking Confirmed — {pnr} | {flight.get('flight_no','')} {flight.get('origin','')}→{flight.get('destination','')}"
    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#0b0f13;font-family:Arial,sans-serif;color:#e6eef6;">
        <div style="max-width:640px;margin:0 auto;padding:24px;">
          <div style="background:linear-gradient(90deg,#2196f3,#ff7a18);padding:18px 20px;border-radius:16px 16px 0 0;color:#fff;font-size:20px;font-weight:700;">
            ✈ AirAssist
          </div>
          <div style="background:#111826;border:1px solid rgba(255,255,255,.06);border-top:none;border-radius:0 0 16px 16px;padding:24px;">
            <h2 style="margin:0 0 8px;font-size:22px;">Your booking is confirmed</h2>
            <p style="margin:0 0 18px;color:#9fb0c8;">Please keep this message for check-in.</p>
            <div style="background:#0b111a;border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:18px;line-height:1.7;">
              <div><strong>PNR NUMBER</strong><br>{pnr}</div>
              <div style="margin-top:12px;"><strong>Passenger</strong><br>{name}</div>
              <div style="margin-top:12px;"><strong>Flight</strong><br>{flight.get('flight_no','')} · {flight.get('airline','')}</div>
              <div style="margin-top:12px;"><strong>Route</strong><br>{flight.get('origin','')} → {flight.get('destination','')}</div>
              <div style="margin-top:12px;"><strong>Departure</strong><br>{flight.get('departure','')}</div>
              <div style="margin-top:12px;"><strong>Boarding</strong><br>{boarding_time}</div>
              <div style="margin-top:12px;"><strong>Seat</strong><br>{seat}</div>
              <div style="margin-top:12px;"><strong>Gate/Terminal</strong><br>{flight.get('gate','TBA')} · {flight.get('terminal','T1')}</div>
              <div style="margin-top:12px;"><strong>Amount Paid</strong><br>₹{amount:,}</div>
            </div>
            <p style="margin:18px 0 0;color:#9fb0c8;">Please arrive at {flight.get('terminal','T1')} at least 2 hours before departure. Carry a valid government-issued photo ID.</p>
          </div>
        </div>
      </body>
    </html>
    """
    message = Mail(
      from_email=os.getenv("SENDGRID_FROM_EMAIL"),
      to_emails=email,
      subject=subject,
      html_content=html
    )
    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
    sg.send(message)
    print(f"[EMAIL] Sent to {email}")
    return True
  except Exception as exc:
    print(f"[EMAIL] Failed: {exc}")
    return False


def send_sms_confirmation(phone, name, pnr, flight, seat, boarding_time) -> bool:
  try:
    from twilio.rest import Client
    if not phone.startswith("+"):
      phone = "+91" + phone.lstrip("0")

    msg = (
      f"AirAssist: Booking Confirmed!\n"
      f"PNR: {pnr}\n"
      f"Passenger: {name}\n"
      f"Flight: {flight.get('flight_no','')} ({flight.get('airline','')})\n"
      f"{flight.get('origin','')}→{flight.get('destination','')}\n"
      f"Dep: {flight.get('departure','')} | Seat: {seat}\n"
      f"Board by: {boarding_time} at {flight.get('gate','TBA')},{flight.get('terminal','T1')}"
    )
    client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    client.messages.create(body=msg, from_=os.getenv("TWILIO_PHONE"), to=phone)
    print(f"[SMS] Sent to {phone}")
    return True
  except Exception as exc:
    print(f"[SMS] Failed: {exc}")
    return False


def send_booking_confirmation(email, phone, name, pnr, flight, seat, boarding_time, amount):
  email_ok = send_email_confirmation(email, name, pnr, flight, seat, boarding_time, amount)
  sms_ok = send_sms_confirmation(phone, name, pnr, flight, seat, boarding_time)
  return {"email": email_ok, "sms": sms_ok}


def send_delay_alert(phone, email, name, pnr, flight_no, delay_minutes):
  try:
    from twilio.rest import Client
    if not phone.startswith("+"):
      phone = "+91" + phone.lstrip("0")
    msg = f"AirAssist Alert: Your flight {flight_no} is delayed by {delay_minutes} minutes. We're checking alternatives. Reply REBOOK for options."
    client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    client.messages.create(body=msg, from_=os.getenv("TWILIO_PHONE"), to=phone)
    print(f"[SMS ALERT] Delay alert sent to {phone}")
  except Exception as exc:
    print(f"[SMS ALERT] Failed: {exc}")
