"""Chain module: routes user messages to the right handler."""
from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
import time

# ── Third-party ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Local imports ─────────────────────────────────────────────────────────────
from . import memory as memory_module
from . import intent as intent_module
from agents import booking_agent
from rag import retriever as retriever_module
from booking import booking_engine

# ── Phase 2 components (optional) ─────────────────────────────────────────────
try:
    from agents.agent_loop import run_agent, extract_entities
    from agents.disruption_handler import handle_disruption, detect_disruption_query
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Agent loop not available: {e}")
    AGENT_AVAILABLE = False

# ── Keyword lists ──────────────────────────────────────────────────────────────
FLIGHT_SEARCH_KEYWORDS = [
    "search flights", "find flights", "show flights", "flights from",
    "fly from", "book a flight", "flight to", "flights to",
    "search for flights", "available flights",
]

FLIGHT_BOOK_KEYWORDS = [
    "book the", "book flight", "confirm booking", "book cheapest",
    "book first", "i'll take", "reserve the",
]

FLIGHT_STATUS_KEYWORDS = [
    "status of flight", "check flight", "flight status",
    "is my flight on time", "gate for flight", "delayed flight",
    "what gate", "which terminal",
]

REFUND_KEYWORDS = [
    "how much refund", "what is my refund", "refund for pnr",
    "cancel my booking", "refund amount", "how much will i get",
]

BAGGAGE_KEYWORDS = [
    "track my baggage", "where is my bag", "baggage status",
    "track baggage", "my luggage",
]

DISRUPTION_KEYWORDS = [
    "cancelled", "canceled", "delayed", "disrupted",
    "what are my options", "alternative flight",
    "flight was cancelled", "flight got cancelled",
]

AGENTIC_INTENTS = [
    "flight_status",
    "refund_request",
    "baggage_track",
    "booking_modify",
]

AIRLINES_LIST_TRIGGERS = [
    "which airlines do you know",
    "which airlines can you",
    "what airlines do you cover",
    "which flights do you know",
    "what can you help with",
    "what airlines are you",
]

INDIAN_AIRLINES = ["indigo", "air_india", "spicejet", "vistara", "general"]

RAG_SYSTEM = (
    "You are a helpful airline policy assistant. Answer ONLY using the provided context passages. "
    "When you cite policy, mention which airline's policy you are using. Be concise and friendly. "
    "When citing which airline a policy belongs to, say 'According to IndiGo's policy...' not '[indigo — baggage]'. "
    "Never include source tags, chunk labels, or metadata references in your answer. "
    "Do not write things like [airline — topic]. Just answer naturally and mention the airline name in plain English. "
    "When answering questions about airline policies, focus on Indian airlines — IndiGo, Air India, SpiceJet, and Vistara. "
    "Do not reference foreign airlines like Air Vanuatu, Lufthansa, Porter Airlines, Air Algerie, or other non-Indian carriers unless specifically asked. "
    "User preferences: {memory_summary}"
)


# ── Main chat function (single definition) ────────────────────────────────────
def chat(
    user_message: str,
    memory: memory_module.ConversationMemory,
    retriever,
    airline_filter=None,
    session_id: str = "default",          # ← added here, not in a second function
) -> str:
    """Route a user message to the correct handler and return a reply."""

    start_time = time.time()
    intent = "general"                     # ← default so it's always in scope

    # 1. Add user message to memory
    memory.add("user", user_message)
    msg_lower = user_message.lower().strip()

    # ── Booking completion ─────────────────────────────────────────────────────
    if booking_agent.is_booking_completion_message(user_message):
        result = booking_agent.handle_booking_completion(user_message)
        if result.get("booking") and not result["booking"].get("error"):
            reply = (
                f"Booking found for PNR {result['pnr']}: "
                f"{result['booking'].get('passenger_name', result['booking'].get('passenger', 'Guest'))} on "
                f"flight {result['booking'].get('flight_no', '')}, "
                f"seat {result['booking'].get('seat', 'TBA')}, "
                f"status {result['booking'].get('status', 'confirmed')}."
            )
        else:
            reply = (
                f"I couldn't find a booking for {result.get('pnr', '')}. "
                "Please check the PNR and try again."
            )
        memory.add("assistant", reply)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    # ── Flight search / booking short-circuits ────────────────────────────────
    if booking_agent.is_flight_search_message(user_message) or any(
        kw in msg_lower for kw in FLIGHT_SEARCH_KEYWORDS
    ):
        intent = "flight_search"
        reply = handle_flight_search(user_message, memory)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    if any(kw in msg_lower for kw in FLIGHT_BOOK_KEYWORDS):
        intent = "flight_booking"
        reply = handle_booking(user_message, memory)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    # ── Agentic keyword short-circuits ────────────────────────────────────────
    if AGENT_AVAILABLE:
        if any(kw in msg_lower for kw in FLIGHT_STATUS_KEYWORDS):
            intent = "flight_status"
            print("[CHAIN] Flight status keyword match → agent loop")
            reply = run_agent(user_message, memory, retriever)
            return _log_and_return(
                reply, start_time, user_message, intent, memory, airline_filter, session_id
            )

        if any(kw in msg_lower for kw in REFUND_KEYWORDS):
            intent = "refund_request"
            print("[CHAIN] Refund keyword match → agent loop")
            reply = run_agent(user_message, memory, retriever)
            return _log_and_return(
                reply, start_time, user_message, intent, memory, airline_filter, session_id
            )

        if any(kw in msg_lower for kw in BAGGAGE_KEYWORDS):
            intent = "baggage_track"
            print("[CHAIN] Baggage keyword match → agent loop")
            reply = run_agent(user_message, memory, retriever)
            return _log_and_return(
                reply, start_time, user_message, intent, memory, airline_filter, session_id
            )

    # ── Airline list trigger ───────────────────────────────────────────────────
    if any(msg_lower.startswith(phrase) for phrase in AIRLINES_LIST_TRIGGERS):
        intent = "airlines_list"
        reply = (
            "I can help you with flight information for these 4 airlines:\n\n"
            "1. IndiGo (6E)\n"
            "2. Air India (AI)\n"
            "3. SpiceJet (SG)\n"
            "4. Vistara (UK)\n\n"
            "Ask me about baggage, refunds, check-in, or seat policies "
            "for any of these airlines. You can also use the airline filter "
            "in the sidebar to focus on one airline."
        )
        memory.add("assistant", reply)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    # ── Intent classification ──────────────────────────────────────────────────
    intent = intent_module.classify_intent(user_message)   # real value now

    # ── Phase 2 disruption / agentic routing ──────────────────────────────────
    if AGENT_AVAILABLE:
        is_disruption = intent == "disruption" or any(
            kw in msg_lower for kw in DISRUPTION_KEYWORDS
        )
        if detect_disruption_query(user_message):
            is_disruption = True

        if is_disruption:
            intent = "disruption"
            entities = extract_entities(user_message)
            pnr = entities.get("pnr")
            flight_no = entities.get("flight_no")

            if pnr:
                reply = handle_disruption(pnr, memory)
            elif flight_no:
                reply = (
                    f"I can see you're asking about flight {flight_no}. "
                    "Could you share your PNR number so I can check your specific booking "
                    "and find the best alternatives for you?"
                )
                memory.add("assistant", reply)
            else:
                reply = run_agent(user_message, memory, retriever)

            return _log_and_return(
                reply, start_time, user_message, intent, memory, airline_filter, session_id
            )

        if intent in AGENTIC_INTENTS:
            print(f"[CHAIN] Routing to agent loop: intent={intent}")
            reply = run_agent(user_message, memory, retriever)
            return _log_and_return(
                reply, start_time, user_message, intent, memory, airline_filter, session_id
            )

    # ── Legacy intent routing ─────────────────────────────────────────────────
    if intent == "flight_search":
        reply = handle_flight_search(user_message, memory)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    if intent == "flight_booking":
        reply = handle_booking(user_message, memory)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    if intent == "booking_modify":
        reply = "Please provide your PNR and what you would like to change."
        memory.add("assistant", reply)
        return _log_and_return(
            reply, start_time, user_message, intent, memory, airline_filter, session_id
        )

    # ── RAG fallback ───────────────────────────────────────────────────────────
    docs = retriever_module.retrieve(
        user_message, retriever, airline_filter=airline_filter, top_k=8
    )

    if intent in ("general_faq", "general"):
        preferred = [
            d for d in docs
            if d.metadata.get("airline", "unknown").lower() in INDIAN_AIRLINES
        ]
        docs = preferred[:4] if len(preferred) >= 3 else docs[:4]

    context_parts = []
    for d in docs:
        airline = d.metadata.get("airline", "unknown")
        topic   = d.metadata.get("topic", "general")
        context_parts.append(
            f"Source ({airline} airline, {topic} policy):\n{d.page_content}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = RAG_SYSTEM.format(memory_summary=memory.summary())
    prior_messages = memory.to_messages()[:-1]
    user_with_context = {
        "role": "user",
        "content": f"{user_message}\n\nContext:\n{context_text}",
    }
    messages = (
        [{"role": "system", "content": system_prompt}]
        + prior_messages
        + [user_with_context]
    )

    llm_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    # ── LLM call with fallback chain ──────────────────────────────────────────
    resp = None
    try:
        resp = llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
    except Exception as e:
        print(f"[WARN] Chat API call failed: {e}")
        fallback_parts = [
            f"[{d.metadata.get('airline', 'Unknown')}] {d.page_content}"
            for d in docs[:6]
        ]
        fallback = "\n\n".join(fallback_parts).strip()
        fallback = (fallback[:2000] + "...") if len(fallback) > 2000 else fallback
        memory.add("assistant", fallback)
        return _log_and_return(
            fallback, start_time, user_message, intent, memory, airline_filter, session_id
        )

    # ── Extract reply robustly ────────────────────────────────────────────────
    reply = ""
    try:
        reply = resp.choices[0].message.content
    except Exception:
        try:
            reply = resp["choices"][0]["message"]["content"]
        except Exception:
            try:
                reply = resp.choices[0].text
            except Exception:
                reply = ""

    reply = reply.strip()
    memory.add("assistant", reply)
    return _log_and_return(
        reply, start_time, user_message, intent, memory, airline_filter, session_id
    )


# ── Helper: log analytics and return reply ─────────────────────────────────────
def _log_and_return(
    reply: str,
    start_time: float,
    user_message: str,
    intent: str,
    memory,
    airline_filter,
    session_id: str,
) -> str:
    """Log the query to analytics and return the reply unchanged."""
    elapsed_ms   = int((time.time() - start_time) * 1000)
    tools_used   = getattr(memory, "last_tools_used", [])
    was_agentic  = len(tools_used) > 0

    try:
        from analytics.tracker import log_query
        log_query(
            session_id=session_id,
            user_message=user_message,
            intent=intent,
            tools_called=tools_used,
            response_time_ms=elapsed_ms,
            was_agentic=was_agentic,
            airline_filter=airline_filter,
        )
    except Exception as e:
        print(f"[ANALYTICS] Log failed: {e}")

    return reply


# ── Flight search / booking handlers ─────────────────────────────────────────
def handle_flight_search(
    message: str, memory: memory_module.ConversationMemory
) -> str:
    origin, destination, date = booking_agent.extract_route_and_date(message)
    if not origin or not destination:
        return "Please specify origin and destination cities (e.g. 'Mumbai to Delhi')."

    live = booking_agent.build_live_flight_options(origin, destination, date)
    memory.active_booking["search_results"]  = live["flights"]
    memory.active_booking["booking_url"]     = live["booking_url"]
    memory.active_booking["flight_cards"]    = live["flight_cards"]

    if not live["flights"]:
        return "No flights found for the requested route/date."

    lines = []
    for item in live["flight_cards"]:
        lines.append(item["card"])
        lines.append(f"Book: {item['booking_url']}")

    return "Available flights:\n" + "\n".join(lines)


def handle_booking(
    message: str, memory: memory_module.ConversationMemory
) -> str:
    results = memory.active_booking.get("search_results") or []
    if not results:
        return "No active search results — please search for flights first."

    flight    = results[0]
    passenger = memory.user_prefs.get("name", "Guest")
    booking   = booking_engine.book_flight(flight, passenger)

    memory.active_booking["last_booking"] = booking
    return (
        f"Booking confirmed: PNR {booking['pnr']}, "
        f"Seat {booking['seat']}, "
        f"Amount INR {booking['flight']['price']}"
    )


__all__ = ["chat", "handle_flight_search", "handle_booking", "RAG_SYSTEM"]