import sqlite3

conn = sqlite3.connect("data/bookings.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
SELECT pnr, passenger_name, flight_no, status, booked_at
FROM bookings
ORDER BY booked_at DESC
LIMIT 5
""").fetchall()

print("Total bookings:",
      conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0])

print()

for r in rows:
    print(
        f"PNR: {r['pnr']} | "
        f"Passenger: {r['passenger_name']} | "
        f"Flight: {r['flight_no']} | "
        f"Status: {r['status']}"
    )

conn.close()