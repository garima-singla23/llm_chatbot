import sqlite3
import os

for db in ["data/bookings.db", "data/user_profiles.db"]:
    if os.path.exists(db):
        print(f"\n{db}")

        conn = sqlite3.connect(db)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        print("Tables:", [t[0] for t in tables])

        for t in tables:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {t[0]}"
            ).fetchone()[0]

            print(f"  {t[0]}: {count} rows")

        conn.close()
    else:
        print(f"{db}: NOT FOUND")