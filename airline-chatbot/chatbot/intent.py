from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

INTENT_SYSTEM = (
    "You are an intent classifier for an airline chatbot. "
    "Classify the user's message into EXACTLY ONE of the following labels: "
    "flight_search, flight_booking, booking_modify, baggage_faq, refund_faq, "
    "check_in_faq, flight_status, general_faq, greeting. "
    "Respond with ONLY the single label (no explanations, no punctuation, no extra text)."
)


INTENT_ROUTES = {
    "flight_search": "User is looking for available flights or searching routes/times",
    "flight_booking": "User intends to book a flight or request booking assistance",
    "booking_modify": "User wants to change or cancel an existing booking",
    "baggage_faq": "Questions about baggage allowances, fees, and rules",
    "refund_faq": "Questions about refunds, cancellations, and refund status",
    "check_in_faq": "Questions about check-in procedures and boarding passes",
    "flight_status": "Requests about flight status, delays, or cancellations",
    "general_faq": "General informational questions or unclear intent",
    "greeting": "Simple greeting or salutation from the user",
}


def classify_intent(message: str) -> str:
    # Quick local rules to avoid LLM misclassification for obvious booking/search commands
    msg_lower = message.lower()
    FLIGHT_SEARCH_KEYWORDS = [
        "search flights", "find flights", "show flights", "flights from",
        "fly from", "book a flight", "flight to", "flights to",
        "search for flights", "available flights"
    ]
    FLIGHT_BOOK_KEYWORDS = [
        "book the", "book flight", "confirm booking", "book cheapest",
        "book first", "i'll take", "reserve the"
    ]
    if any(kw in msg_lower for kw in FLIGHT_SEARCH_KEYWORDS):
        return "flight_search"
    if any(kw in msg_lower for kw in FLIGHT_BOOK_KEYWORDS):
        return "flight_booking"
    try:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        resp = client.chat.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=20,
        )

        # Try a few ways to extract the assistant text to be robust across SDK versions
        content = ""
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                content = resp.choices[0].message["content"]
            except Exception:
                try:
                    content = resp["choices"][0]["message"]["content"]
                except Exception:
                    try:
                        content = resp.choices[0].text
                    except Exception:
                        content = ""

        return content.strip()
    except Exception:
        return "general_faq"


__all__ = ["INTENT_SYSTEM", "INTENT_ROUTES", "classify_intent"]
