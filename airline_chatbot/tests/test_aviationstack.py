from tools.live_flights import search_live_flights, check_live_status

print('--- Flight Search ---')

flights = search_live_flights('Delhi', 'Mumbai')

for f in flights[:3]:
    print(
        f'  {f["airline"]} {f["flight_no"]} | '
        f'{f["departure"]}→{f["arrival"]} | '
        f'INR {f["price"]} | [{f.get("source","?")}]'
    )

print()

print('--- Flight Status ---')

status = check_live_status('6E204')

print(
    f'  Status: {status["status"]} | '
    f'Delay: {status["delay_minutes"]}min | '
    f'Gate: {status["gate"]} | '
    f'[{status.get("source","?")}]'
)