from __future__ import annotations

import re
import time
import threading
from datetime import date as date_type
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright

_MIN_CALL_INTERVAL_SECONDS = 3.0
_LAST_CALL_AT = 0.0
_CALL_LOCK = threading.Lock()


def _throttle_calls() -> None:
  global _LAST_CALL_AT

  with _CALL_LOCK:
    elapsed = time.monotonic() - _LAST_CALL_AT
    if _LAST_CALL_AT and elapsed < _MIN_CALL_INTERVAL_SECONDS:
      time.sleep(_MIN_CALL_INTERVAL_SECONDS - elapsed)
    _LAST_CALL_AT = time.monotonic()


def _normalize_date(value) -> str:
  if value is None:
    return ""
  if isinstance(value, date_type):
    return value.isoformat()
  return str(value).strip()


def _build_google_flights_url(origin_iata: str, destination_iata: str, travel_date) -> str:
  route = f"{origin_iata.strip().upper()}-{destination_iata.strip().upper()}-{_normalize_date(travel_date)}"
  return f"https://www.google.com/travel/flights/search?tfs={quote_plus(route)}"


def _parse_price(text: str) -> int | None:
  digits = re.sub(r"[^0-9]", "", text or "")
  if not digits:
    return None
  try:
    return int(digits)
  except ValueError:
    return None


def _collect_prices(page) -> list[int]:
  selectors = [
    ".YMlIz",
    "span.YMlIz",
    "div.YMlIz",
    "[aria-label*='Price']",
    "[data-testid*='price']",
  ]
  prices: list[int] = []
  for selector in selectors:
    try:
      texts = page.locator(selector).all_inner_texts()
    except Exception:
      continue
    for text in texts[:3]:
      price = _parse_price(text)
      if price is not None:
        prices.append(price)
    if prices:
      break
  return prices


def get_indicative_price(origin_iata: str, destination_iata: str, date) -> int | None:
  try:
    _throttle_calls()
    url = _build_google_flights_url(origin_iata, destination_iata, date)

    with sync_playwright() as playwright:
      browser = playwright.chromium.launch(headless=True)
      page = browser.new_page(viewport={"width": 1440, "height": 1024})
      try:
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
        try:
          page.wait_for_selector(".YMlIz", timeout=15000)
        except Exception:
          pass
        prices = _collect_prices(page)
        if not prices:
          return None
        return min(prices[:3])
      finally:
        browser.close()
  except Exception:
    return None