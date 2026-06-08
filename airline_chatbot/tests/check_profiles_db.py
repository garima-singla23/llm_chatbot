import sqlite3

conn = sqlite3.connect("data/user_profiles.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT * FROM user_profiles LIMIT 5"
).fetchall()

print(f"User profiles: {len(rows)}")

for r in rows:
    print(dict(r))

conn.close()