# Reverse IATA code to city name mapping for natural language generation
IATA_TO_CITY = {
    "DEL": "Delhi",
    "BOM": "Mumbai",
    "BLR": "Bangalore",
    "MAA": "Chennai",
    "HYD": "Hyderabad",
    "CCU": "Kolkata",
    "LHR": "London",
    "DXB": "Dubai",
    "SIN": "Singapore",
    "JFK": "New York",
    "GOI": "Goa",
    "COK": "Kochi",
}


def needs_clarification(entities: dict) -> tuple[bool, str]:
    """Check if required flight fields are missing and generate a clarification question.

    Args:
        entities: Dict with at least origin_iata, destination_iata, departure_date.

    Returns:
        (False, "") if all required fields present.
        (True, question_str) if any required field is null.
        The question naturally asks for ALL missing fields at once.
    """
    origin = entities.get("origin_iata")
    destination = entities.get("destination_iata")
    date = entities.get("departure_date")

    missing_origin = origin is None
    missing_destination = destination is None
    missing_date = date is None

    # If all required fields present, no clarification needed
    if not (missing_origin or missing_destination or missing_date):
        return (False, "")

    # Helper to get city name from IATA code for natural language
    def city_name(iata: str) -> str:
        return IATA_TO_CITY.get(iata, iata) if iata else None

    # Generate context-aware questions for all missing-field combinations

    if missing_origin and missing_destination and missing_date:
        return (True, "Where would you like to fly from, to, and when?")

    if missing_origin and missing_destination:
        return (True, "Where would you like to fly from and to?")

    if missing_origin and missing_date:
        dest_name = city_name(destination)
        return (
            True,
            f"Where are you flying from, heading to {dest_name}?",
        )

    if missing_destination and missing_date:
        origin_name = city_name(origin)
        return (
            True,
            f"Great, flying from {origin_name} — where are you heading and when?",
        )

    if missing_origin:
        dest_name = city_name(destination)
        return (True, f"Where are you flying from to {dest_name}?")

    if missing_destination:
        origin_name = city_name(origin)
        return (True, f"Great, flying from {origin_name} — where are you heading?")

    if missing_date:
        origin_name = city_name(origin)
        return (True, f"And when would you like to travel from {origin_name}?")

    # Should never reach here if logic is correct
    return (False, "")


def merge_entities(old: dict, new: dict) -> dict:
    """Merge a follow-up extraction into a prior one.

    For each field: if new value is not None, use it; otherwise keep old value.
    This allows follow-up queries like "what about business class?" to inherit
    origin and destination from the previous turn.

    Args:
        old: Previous entity extraction.
        new: New entity extraction from follow-up query.

    Returns:
        Merged entities dict with non-null values from new, falling back to old.
    """
    merged = old.copy()

    for key, new_value in new.items():
        if new_value is not None:
            merged[key] = new_value
        # If new_value is None, keep old value (retained via copy)

    return merged
