# proactive/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3, os
from datetime import datetime
from proactive.checkin_alerts import (
    should_send_checkin_alert, send_checkin_alert
)

scheduler = BackgroundScheduler()

def check_upcoming_flights():
    """Runs every hour — sends check-in alerts for flights 48hrs away."""
    db = "data/bookings.db"
    if not os.path.exists(db):
        return
    
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM bookings 
        WHERE status='confirmed' AND passenger_phone != ''
        ORDER BY booked_at DESC LIMIT 100
    """).fetchall()
    conn.close()
    
    for row in rows:
        try:
            dep_str = f"{row['date']} {row['departure']}"
            dep_dt = datetime.strptime(dep_str, "%Y-%m-%d %H:%M")
            
            if should_send_checkin_alert(dep_dt):
                send_checkin_alert(
                    phone=row["passenger_phone"],
                    name=row["passenger_name"],
                    pnr=row["pnr"],
                    flight_no=row["flight_no"],
                    departure=dep_str
                )
        except Exception as e:
            print(f"[SCHEDULER] Error for {row['pnr']}: {e}")

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(check_upcoming_flights, "interval",
                          hours=1, id="checkin_checker",
                          replace_existing=True)
        scheduler.start()
        print("[SCHEDULER] Background jobs started")

def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()