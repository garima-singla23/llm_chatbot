import sqlite3, os

db_path = 'data/bookings.db'
print('CWD:', os.getcwd())
print('Abs DB path:', os.path.abspath(db_path))
print('Exists:', os.path.exists(db_path))

conn = sqlite3.connect(db_path)
cur = conn.cursor()

print('Tables:', cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print('Row count:', cur.execute("SELECT COUNT(*) FROM bookings").fetchone())
print('Exact match:', cur.execute("SELECT pnr, passenger_name FROM bookings WHERE pnr = ?", ('PNR0FO1G9',)).fetchall())
print('Sample 3 rows:', cur.execute("SELECT pnr FROM bookings LIMIT 3").fetchall())

conn.close()