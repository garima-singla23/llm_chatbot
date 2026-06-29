# config.py
import os

CONFIRMATION_API_URL = os.getenv(
    "CONFIRMATION_API_URL",
    "http://localhost:8001"
)

BOOKING_BASE_URL = os.getenv(
    "BOOKING_BASE_URL",
    "http://localhost:5000"
)