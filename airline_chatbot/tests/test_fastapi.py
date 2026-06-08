import requests
import json

url = "http://localhost:8001/confirm-booking"

payload = {
    "flight": {
        "flight_no": "6E204",
        "airline": "IndiGo",
        "origin": "Delhi",
        "destination": "Mumbai",
        "departure": "06:00",
        "arrival": "07:31",
        "date": "2026-06-10",
        "gate": "B14",
        "terminal": "T2",
        "price": 2520
    },
    "passenger": {
        "name": "Test User",
        "email": "test@example.com",
        "phone": "9876543210",
        "seat_pref": "window",
        "meal_pref": "veg"
    },
    "payment_id": "pay_test123",
    "order_id": "order_test123",
    "amount": 2520
}

try:
    response = requests.post(url, json=payload)

    print("Status Code:", response.status_code)
    print("\nResponse:")

    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

except Exception as e:
    print("Error:", e)