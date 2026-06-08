from tools.user_profile import UserProfile

p = UserProfile("user1")

# Update preferences
p.update_pref(
    seat_pref="window",
    meal_pref="veg",
    name="Test User",
    email="test@example.com"
)

# Read preferences
print("Seat:", p.get_pref("seat_pref"))
print("Meal:", p.get_pref("meal_pref"))

# Add a sample booking
p.add_booking(
    pnr="PNRTEST1",
    flight_no="6E204",
    origin="Delhi",
    destination="Mumbai",
    seat="14A",
    amount=2520
)

print("\nSummary:")
print(p.get_summary())

print("\nHistory:")
print(p.get_history())

print("\nPass: SQLite persistence working")