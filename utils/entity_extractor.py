import json
from datetime import date


def extract_entities(query: str, llm, chat_history: list = []) -> dict:
    """Extract structured flight parameters from natural language query.

    Args:
        query: The user's input query.
        llm: LLM instance with chat(messages, stream) method.
        chat_history: List of (user, assistant) tuples for context.

    Returns:
        {
            "origin_iata": str or null,
            "destination_iata": str or null,
            "departure_date": "YYYY-MM-DD" or null,
            "return_date": "YYYY-MM-DD" or null,
            "cabin_class": "ECONOMY" | "BUSINESS" | "FIRST" | "PREMIUM_ECONOMY",
            "adults": int (minimum 1),
            "children": int,
            "preferred_airline_iata": str or null
        }

    On any failure, returns all fields as null except adults=1.
    """
    today = date.today().isoformat()

    city_mapping = (
        "Delhi=DEL, Mumbai=BOM, Bangalore=BLR, Chennai=MAA, Hyderabad=HYD, "
        "Kolkata=CCU, London=LHR, Dubai=DXB, Singapore=SIN, New York=JFK, "
        "Goa=GOI, Kochi=COK"
    )

    airline_mapping = (
        "IndiGo=6E, Air India=AI, SpiceJet=SG, Emirates=EK, Qatar=QR, "
        "British Airways=BA"
    )

    system_prompt = (
        "You are a flight search entity extractor. Parse the user query and extract flight parameters.\n\n"
        "CITY TO IATA CODE MAPPINGS:\n"
        f"{city_mapping}\n\n"
        "AIRLINE NAME TO IATA CODE MAPPINGS:\n"
        f"{airline_mapping}\n\n"
        f"TODAY'S DATE (use for relative date resolution): {today}\n\n"
        "INSTRUCTIONS:\n"
        "1. Convert city names to IATA codes using the mappings above.\n"
        "2. Resolve dates to ISO 8601 format (YYYY-MM-DD):\n"
        "   - 'tomorrow' -> tomorrow's date\n"
        "   - 'next Friday' -> next Friday's date\n"
        "   - 'April 15' -> 2026-04-15 (assume current year if no year specified)\n"
        "   - 'Dec 25' with year ambiguity: assume current/next year as context dictates\n"
        "3. Default cabin_class to 'ECONOMY' if not mentioned.\n"
        "4. Default adults to 1 if not mentioned.\n"
        "5. Children and preferred_airline_iata: set to null/0 if not mentioned.\n"
        "6. For any field you cannot confidently determine, use null (except adults/children which use defaults).\n\n"
        "Return ONLY valid JSON in this exact format, no explanation or markdown:\n"
        "{\n"
        '  "origin_iata": "DEL" or null,\n'
        '  "destination_iata": "BOM" or null,\n'
        '  "departure_date": "2026-04-15" or null,\n'
        '  "return_date": "2026-04-20" or null,\n'
        '  "cabin_class": "ECONOMY" or "BUSINESS" or "FIRST" or "PREMIUM_ECONOMY",\n'
        '  "adults": 1,\n'
        '  "children": 0,\n'
        '  "preferred_airline_iata": "6E" or null\n'
        "}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include last 2 turns of chat history for context
    recent_history = chat_history[-2:] if chat_history else []
    for user_turn, assistant_turn in recent_history:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})

    messages.append({"role": "user", "content": query})

    try:
        result = llm.chat(messages, stream=False)
        response_text = str(result).strip()

        # Try to parse JSON, stripping common markdown wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        parsed = json.loads(response_text)

        # Validate required fields exist
        required_fields = {
            "origin_iata",
            "destination_iata",
            "departure_date",
            "return_date",
            "cabin_class",
            "adults",
            "children",
            "preferred_airline_iata",
        }
        if not all(field in parsed for field in required_fields):
            return _default_entities()

        # Validate cabin_class enum
        valid_cabins = {"ECONOMY", "BUSINESS", "FIRST", "PREMIUM_ECONOMY"}
        if parsed["cabin_class"] not in valid_cabins:
            return _default_entities()

        # Validate IATA codes (3 letters if not null)
        origin = parsed["origin_iata"]
        destination = parsed["destination_iata"]
        if origin is not None and (not isinstance(origin, str) or len(origin) != 3):
            origin = None
        if destination is not None and (
            not isinstance(destination, str) or len(destination) != 3
        ):
            destination = None

        # Validate adults/children are non-negative integers
        adults = int(parsed["adults"]) if parsed["adults"] else 1
        children = int(parsed["children"]) if parsed["children"] else 0
        if adults < 1:
            adults = 1
        if children < 0:
            children = 0

        # Validate preferred_airline_iata (2 letters if not null)
        preferred_airline = parsed["preferred_airline_iata"]
        if preferred_airline is not None and (
            not isinstance(preferred_airline, str) or len(preferred_airline) != 2
        ):
            preferred_airline = None

        # Post-parse validation: origin == destination means destination is uncertain
        if origin is not None and destination is not None and origin == destination:
            destination = None

        return {
            "origin_iata": origin,
            "destination_iata": destination,
            "departure_date": parsed["departure_date"],
            "return_date": parsed["return_date"],
            "cabin_class": parsed["cabin_class"],
            "adults": adults,
            "children": children,
            "preferred_airline_iata": preferred_airline,
        }

    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return _default_entities()
    except Exception:
        return _default_entities()


def _default_entities() -> dict:
    """Return default/null entity structure on extraction failure."""
    return {
        "origin_iata": None,
        "destination_iata": None,
        "departure_date": None,
        "return_date": None,
        "cabin_class": "ECONOMY",
        "adults": 1,
        "children": 0,
        "preferred_airline_iata": None,
    }
