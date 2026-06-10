# proactive/checkin_alerts.py

from datetime import datetime, timedelta
from tools.notifications import send_sms_confirmation

def should_send_checkin_alert(departure_datetime: datetime) -> bool:
    now = datetime.now()
    hours_until = (departure_datetime - now).total_seconds() / 3600
    return 47 <= hours_until <= 49

def should_send_boarding_reminder(departure_datetime: datetime) -> bool:
    now = datetime.now()
    hours_until = (departure_datetime - now).total_seconds() / 3600
    return 2.5 <= hours_until <= 3.5

def send_checkin_alert(phone: str, name: str, pnr: str,
                        flight_no: str, departure: str):
    try:
        from twilio.rest import Client
        import os
        if not phone.startswith("+"): phone = "+91" + phone.lstrip("0")
        msg = (f"AirAssist: Check-in is now open for your flight!\n"
               f"Flight: {flight_no} | PNR: {pnr}\n"
               f"Departure: {departure}\n"
               f"Check in now to get your preferred seat.")
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"),
                        os.getenv("TWILIO_AUTH_TOKEN"))
        client.messages.create(body=msg, from_=os.getenv("TWILIO_PHONE"), to=phone)
        print(f"[CHECKIN ALERT] Sent to {phone} for {pnr}")
    except Exception as e:
        print(f"[CHECKIN ALERT] Failed: {e}")


