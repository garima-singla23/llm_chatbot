import json


def classify_intent(query: str, llm, chat_history: list = []) -> dict:
    """Classify user query intent into one of five categories with confidence score.

    Args:
        query: The user's input query.
        llm: LLM instance with chat(messages, stream) method.
        chat_history: List of (user, assistant) tuples for context.

    Returns:
        {"type": str, "confidence": float}
        - type: one of "flight_search", "price_compare", "policy", "flight_status", "general"
        - confidence: 0.0-1.0 floating point score

    On any failure, defaults to {"type": "policy", "confidence": 0.0}
    """
    system_prompt = (
        "You are an intent classifier for an airline customer support chatbot. "
        "Classify the user's query into exactly one of these categories:\n"
        '- "flight_search": User wants to find, book, or search for flights\n'
        '- "price_compare": User wants to compare prices across airlines or fare options\n'
        '- "policy": Questions about baggage, cancellation, check-in, refund rules, or airline policies\n'
        '- "flight_status": User asking if a specific flight is on time, delayed, or cancelled\n'
        '- "general": Greetings, unrelated questions, or out-of-scope queries\n\n'
        "Return ONLY valid JSON in this exact format, no explanation or markdown:\n"
        '{"type": "<intent_type>", "confidence": <0.0-1.0>}'
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

        # Validate shape
        if "type" not in parsed or "confidence" not in parsed:
            return {"type": "policy", "confidence": 0.0}

        intent_type = parsed["type"]
        confidence = float(parsed["confidence"])

        # Validate type is one of the allowed values
        valid_types = {
            "flight_search",
            "price_compare",
            "policy",
            "flight_status",
            "general",
        }
        if intent_type not in valid_types:
            return {"type": "policy", "confidence": 0.0}

        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        return {"type": intent_type, "confidence": confidence}

    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return {"type": "policy", "confidence": 0.0}
    except Exception:
        return {"type": "policy", "confidence": 0.0}


def needs_retrieval(query: str, llm) -> bool:
    """Return True when the query likely needs airline policy lookup.

    Falls back to True on model errors or ambiguous outputs to avoid missing context.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier for an airline assistant. "
                "Decide whether the user query requires looking up airline policy information. "
                "Reply with exactly one word: yes or no. "
                "Do not include any explanation or extra text."
            ),
        },
        {"role": "user", "content": query},
    ]

    try:
        result = llm.chat(messages, stream=False)
    except Exception:
        return True

    answer = str(result).strip().lower()
    if answer == "yes":
        return True
    if answer == "no":
        return False

    return True
