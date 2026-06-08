from tools.notifications import send_booking_confirmation

result = send_booking_confirmation(
    email="garimasingla732@gmail.com",
    phone="9815393628",
    name="Test Passenger",
    pnr="PNRTEST1",
    flight={
        "flight_no": "6E204",
        "airline": "IndiGo",
        "origin": "Delhi",
        "destination": "Mumbai",
        "departure": "06:00",
        "arrival": "07:31",
        "gate": "B14",
        "terminal": "T2",
    },
    seat="14A",
    boarding_time="05:00",
    amount=2520,
)

print("Email sent:", result["email"])
print("SMS sent:", result["sms"])
print(result)