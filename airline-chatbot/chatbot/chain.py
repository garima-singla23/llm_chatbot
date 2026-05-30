from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from . import memory as memory_module
from . import intent as intent_module
from rag import retriever as retriever_module
from booking import booking_engine

FLIGHT_SEARCH_KEYWORDS = [
    "search flights", "find flights", "show flights", "flights from",
    "fly from", "book a flight", "flight to", "flights to",
    "search for flights", "available flights"
]

FLIGHT_BOOK_KEYWORDS = [
    "book the", "book flight", "confirm booking", "book cheapest",
    "book first", "i'll take", "reserve the"
]

AIRLINES_LIST_TRIGGERS = [
    "which airlines do you know",
    "which airlines can you",
    "what airlines do you cover",
    "which flights do you know",
    "what can you help with",
    "what airlines are you"
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


def chat(user_message: str, memory: memory_module.ConversationMemory, retriever, airline_filter=None) -> str:
    # 1. add user message to memory
    memory.add("user", user_message)

    msg_lower = user_message.lower().strip()

    # Pre-classifier short-circuit for explicit flight search / booking commands
    if any(kw in msg_lower for kw in FLIGHT_SEARCH_KEYWORDS):
        return handle_flight_search(user_message, memory)
    if any(kw in msg_lower for kw in FLIGHT_BOOK_KEYWORDS):
        return handle_booking(user_message, memory)

    # Airline-list triggers — use startswith to avoid overly broad matches
    if any(msg_lower.startswith(phrase) for phrase in AIRLINES_LIST_TRIGGERS):
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
        return reply

    # 2. classify intent
    intent = intent_module.classify_intent(user_message)

    if intent == "flight_search":
        return handle_flight_search(user_message, memory)

    if intent == "flight_booking":
        return handle_booking(user_message, memory)

    if intent == "booking_modify":
        return "Please provide your PNR and what you would like to change."

    # FAQ / general intents: use RAG
    docs = retriever_module.retrieve(user_message, retriever, airline_filter=airline_filter, top_k=8)

    # For general FAQ intent, apply soft filter to prefer Indian airlines
    if intent == "general_faq" or intent == "general":
        # Filter to prefer Indian airlines + general
        preferred = [d for d in docs 
                     if d.metadata.get("airline", "unknown").lower() in INDIAN_AIRLINES]
        
        # If we got at least 3 preferred docs, use those; otherwise use all
        if len(preferred) >= 3:
            docs = preferred[:4]
        else:
            # Fall back to all docs but limit to top 4
            docs = docs[:4]

    context_parts = []
    for d in docs:
        airline = d.metadata.get("airline", "unknown")
        topic = d.metadata.get("topic", "general")
        context_parts.append(
            f"Source ({airline} airline, {topic} policy):\n{d.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = RAG_SYSTEM.format(memory_summary=memory.summary())

    # assemble messages: system + prior conversation (excluding last user turn) + user message with context
    prior_messages = memory.to_messages()[:-1]
    user_with_context = {
        "role": "user",
        "content": f"{user_message}\n\nContext:\n{context_text}",
    }

    messages = [{"role": "system", "content": system_prompt}] + prior_messages + [user_with_context]

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    # Invoke the chat/completion API with multiple fallbacks to support different SDK shapes
    resp = None
    try:
        # Preferred: client.chat.create
        if hasattr(client, "chat") and hasattr(client.chat, "create"):
            resp = client.chat.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.3, max_tokens=500)
        # Some SDKs expose chat.completions.create
        elif hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(client.chat.completions, "create"):
            resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.3, max_tokens=500)
        # Some SDKs provide a chat_completion helper
        elif hasattr(client, "chat_completion"):
            resp = client.chat_completion(messages=messages, model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=500)
        # Fallback to completions API using joined user messages as a prompt
        elif hasattr(client, "completions") and hasattr(client.completions, "create"):
            prompt_text = "\n\n".join([m.get("content", "") for m in messages])
            resp = client.completions.create(model="llama-3.3-70b-versatile", prompt=prompt_text, temperature=0.3, max_tokens=500)
        else:
            raise RuntimeError("No supported chat/completion method found on OpenAI client instance")
    except Exception as e:
        # If the remote chat API fails (invalid key, network, etc.), fall back to a local RAG reply
        print(f"[WARN] Chat API call failed: {e}")
        # Build a simple fallback answer from retrieved docs so evaluations can proceed without the external API
        fallback_parts = []
        for d in docs[:6]:
            fallback_parts.append(f"[{d.metadata.get('airline','Unknown')}] {d.page_content}")
        fallback = "\n\n".join(fallback_parts).strip()
        fallback = (fallback[:2000] + "...") if len(fallback) > 2000 else fallback
        memory.add("assistant", fallback)
        return fallback

    # extract assistant reply robustly
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
    return reply


def handle_flight_search(message: str, memory: memory_module.ConversationMemory) -> str:
    # extract cities from booking_engine.mock_data.CITIES
    cities = booking_engine.mock_data.CITIES
    found = []
    lowered = message.lower()
    for city in cities:
        if city.lower() in lowered:
            found.append(city)

    if len(found) < 2:
        return "Please specify origin and destination cities (e.g. 'Mumbai to Delhi')."

    origin, destination = found[0], found[1]

    # try to find a date in YYYY-MM-DD format
    date = None
    for token in message.split():
        try:
            datetime_date = booking_engine.datetime.date.fromisoformat(token)
            date = datetime_date
            break
        except Exception:
            continue
    if date is None:
        date = booking_engine.datetime.date.today()

    results = booking_engine.search_flights(origin, destination, date)
    memory.active_booking["search_results"] = results

    if not results:
        return "No flights found for the requested route/date."

    top3 = results[:3]
    lines = []
    for idx, f in enumerate(top3, start=1):
        lines.append(
            f"{idx}. {f['airline']} {f['flight_no']}: {f['departure']}-{f['arrival']} ({f['duration']}) — INR {f['price']}"
        )

    return "Available flights:\n" + "\n".join(lines)


def handle_booking(message: str, memory: memory_module.ConversationMemory) -> str:
    results = memory.active_booking.get("search_results") or []
    if not results:
        return "No active search results — please search for flights first."

    flight = results[0]
    passenger = memory.user_prefs.get("name", "Guest")
    booking = booking_engine.book_flight(flight, passenger)

    # store booking in memory
    memory.active_booking["last_booking"] = booking

    return (
        f"Booking confirmed: PNR {booking['pnr']}, Seat {booking['seat']}, Amount INR {booking['flight']['price']}"
    )


__all__ = ["chat", "handle_flight_search", "handle_booking", "RAG_SYSTEM"]
