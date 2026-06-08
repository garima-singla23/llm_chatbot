from agents.tool_definitions import TOOL_REGISTRY

print("=== Tool Tests ===")

# Test 1: flight status
r = TOOL_REGISTRY["check_flight_status"]("6E204")
print("check_flight_status:", r.get("status"), "| gate:", r.get("gate"))

# Test 2: calculate refund
r = TOOL_REGISTRY["calculate_refund"]("PNRTEST1")
print("calculate_refund: eligible=", r.get("eligible"), "| amount=", r.get("refund_amount"))

# Test 3: track baggage
r = TOOL_REGISTRY["track_baggage"]("PNRTEST1")
print("track_baggage: status=", r.get("bag_status"), "| carousel=", r.get("carousel"))

# Test 4: booking page URL
r = TOOL_REGISTRY["get_booking_page_url"](
    {
        "flight_no": "6E204",
        "airline": "IndiGo",
        "departure": "06:00",
        "arrival": "07:31",
        "price": 2520,
    },
    "Delhi",
    "Mumbai",
    "2026-06-10",
)

print("booking_url:", r.get("booking_url"))

print()
print("All", len(TOOL_REGISTRY), "tools loaded:", list(TOOL_REGISTRY.keys()))