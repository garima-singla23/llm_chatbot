import requests
import json

results = {}

# ----------------------------
# Health Checks
# ----------------------------
services = [
    ("FastAPI", "http://localhost:8001/health"),
    ("Flask", "http://localhost:5000/health"),
]

for name, url in services:
    try:
        r = requests.get(url, timeout=5)

        if r.status_code == 200:
            results[name] = "PASS"
            print(f"[PASS] {name} Health Check")
        else:
            results[name] = "FAIL"
            print(f"[FAIL] {name} -> HTTP {r.status_code}")

    except Exception as e:
        results[name] = "FAIL"
        print(f"[FAIL] {name} -> {e}")

print()

# ----------------------------
# Booking API Test
# ----------------------------
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
    r = requests.post(
        "http://localhost:8001/confirm-booking",
        json=payload,
        timeout=10
    )

    if r.status_code == 200:
        data = r.json()

        if "pnr" in data:
            results["Booking API"] = "PASS"
            print(f"[PASS] Booking API")
            print(f"PNR: {data.get('pnr')}")
        else:
            results["Booking API"] = "FAIL"
            print("[FAIL] Booking API - PNR missing")

    else:
        results["Booking API"] = "FAIL"
        print(f"[FAIL] Booking API -> HTTP {r.status_code}")

except Exception as e:
    results["Booking API"] = "FAIL"
    print(f"[FAIL] Booking API -> {e}")

# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 40)
passed = sum(1 for v in results.values() if v == "PASS")

print("SUMMARY")
for test, status in results.items():
    print(f"{status:5} - {test}")

print(f"\nPassed: {passed}/{len(results)}")
print("=" * 40)