from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("data/user_profiles.db")


def _get_conn():
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return sqlite3.connect(DB_PATH)


def init_db():
  conn = _get_conn()
  try:
    conn.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
      user_id TEXT PRIMARY KEY,
      name TEXT,
      email TEXT,
      phone TEXT,
      seat_pref TEXT DEFAULT 'window',
      meal_pref TEXT DEFAULT 'veg',
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS booking_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      pnr TEXT,
      flight_no TEXT,
      origin TEXT,
      destination TEXT,
      seat TEXT,
      amount INTEGER,
      booked_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
  finally:
    conn.close()


init_db()


class UserProfile:
  def __init__(self, user_id: str):
    self.user_id = user_id

  def get_or_create(self) -> Dict[str, Any]:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    try:
      row = conn.execute("SELECT * FROM user_profiles WHERE user_id=?", (self.user_id,)).fetchone()
      if row:
        return dict(row)
      conn.execute("INSERT INTO user_profiles (user_id) VALUES (?)", (self.user_id,))
      conn.commit()
      return self.get_or_create()
    finally:
      conn.close()

  def update_pref(self, **fields: Any) -> Dict[str, Any]:
    allowed = {"name", "email", "phone", "seat_pref", "meal_pref"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
      return self.get_or_create()

    profile = self.get_or_create()
    profile.update(updates)

    conn = _get_conn()
    try:
      conn.execute("""UPDATE user_profiles SET
        name=?, email=?, phone=?, seat_pref=?, meal_pref=?, updated_at=CURRENT_TIMESTAMP
        WHERE user_id=?""", (
        profile.get("name"), profile.get("email"), profile.get("phone"),
        profile.get("seat_pref", "window"), profile.get("meal_pref", "veg"), self.user_id
      ))
      conn.commit()
    finally:
      conn.close()
    return self.get_or_create()

  def get_pref(self, key: str, default: Any = None) -> Any:
    return self.get_or_create().get(key, default)

  def add_booking(self, pnr: str, flight_no: str, origin: str, destination: str, seat: str, amount: int) -> None:
    conn = _get_conn()
    try:
      conn.execute("""INSERT INTO booking_history
        (user_id, pnr, flight_no, origin, destination, seat, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", (
        self.user_id, pnr, flight_no, origin, destination, seat, amount
      ))
      conn.commit()
    finally:
      conn.close()

  def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    try:
      rows = conn.execute(
        "SELECT * FROM booking_history WHERE user_id=? ORDER BY booked_at DESC LIMIT ?",
        (self.user_id, limit)
      ).fetchall()
      return [dict(r) for r in rows]
    finally:
      conn.close()

  def get_summary(self) -> Dict[str, Any]:
    profile = self.get_or_create()
    return {
      "user_id": self.user_id,
      "name": profile.get("name"),
      "email": profile.get("email"),
      "phone": profile.get("phone"),
      "seat_pref": profile.get("seat_pref", "window"),
      "meal_pref": profile.get("meal_pref", "veg"),
      "booking_count": len(self.get_history(limit=100)),
    }
