import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import re
import sys
import pickle

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
	sys.path.insert(0, str(APP_DIR))

from chatbot import chain as chain_module
from chatbot import memory as memory_module
from rag import embedder as embedder_module
from rag import chunker as chunker_module
from rag import retriever as retriever_module
from tools.live_flights import CITY_TO_IATA, search_live_flights
from agents.tool_definitions import get_booking_page_url, lookup_booking_by_pnr


load_dotenv()

CHUNK_CACHE_PATH = Path("data/processed/chunked_documents.pkl")

st.set_page_config(page_title="AirAssist", layout="wide")

CSS = """
<style>
.user-bubble{background:#0b69ff;color:white;padding:12px;border-radius:18px 18px 4px 18px;display:inline-block;float:right;max-width:80%;}
.bot-bubble{background:#e5e7eb;color:#111;padding:12px;border-radius:18px 18px 18px 4px;display:inline-block;float:left;max-width:80%;}
.clearfix{clear:both}
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

QUICK_QUESTIONS = [
	"What is carry-on baggage allowance?",
	"How do I cancel and get a refund?",
	"How early to check in?",
	"Where is my PNR?",
	"Check status of my flight 6E204",
	"I completed my booking — check my PNR",
	"Track my baggage",
	"What flights are available Delhi to Mumbai?",
]

BOOKING_COMPLETE_TRIGGERS = (
	"i completed the booking",
	"i completed my booking",
	"i paid",
	"payment done",
	"booking completed",
)


@st.cache_resource(show_spinner=False)
def load_knowledge_base():
	load_dotenv()
	vector_store = embedder_module.load_vector_store()
	if CHUNK_CACHE_PATH.exists():
		with CHUNK_CACHE_PATH.open("rb") as f:
			documents = pickle.load(f)
	else:
		documents = chunker_module.chunk_all("data/raw")
		CHUNK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
		with CHUNK_CACHE_PATH.open("wb") as f:
			pickle.dump(documents, f)
	return retriever_module.build_hybrid_retriever(vector_store, documents)


st.title("Your AI airline assistant")
st.caption("Loading the knowledge base can take a few moments on first launch.")


def init_session():
	if "memory" not in st.session_state:
		st.session_state.memory = memory_module.ConversationMemory()
	if "messages" not in st.session_state:
		st.session_state.messages = []
	if "retriever" not in st.session_state:
		st.session_state.retriever = None
		st.session_state.load_error = None

	if "pending_input" not in st.session_state:
		st.session_state.pending_input = ""
	if "flight_options" not in st.session_state:
		st.session_state.flight_options = []
	if "booking_url" not in st.session_state:
		st.session_state.booking_url = ""
	if "awaiting_pnr" not in st.session_state:
		st.session_state.awaiting_pnr = False
	if "booking_lookup_pending" not in st.session_state:
		st.session_state.booking_lookup_pending = False


init_session()


def _normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip().lower())


def _is_booking_completion_message(text: str) -> bool:
	normalized = _normalize_text(text)
	return any(trigger in normalized for trigger in BOOKING_COMPLETE_TRIGGERS)


def _extract_date_token(message: str) -> str:
	match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", message)
	return match.group(0) if match else ""


def _extract_route(message: str):
	lowered = message.lower()
	city_matches = []
	for city in sorted(CITY_TO_IATA.keys(), key=len, reverse=True):
		if city in lowered:
			city_matches.append(city.title())

	if len(city_matches) < 2:
		return None, None
	return city_matches[0], city_matches[1]


def _extract_pnr(text: str) -> str:
	match = re.search(r"\bPNR[A-Z0-9]{4,10}\b", text, flags=re.IGNORECASE)
	if match:
		return match.group(0).upper()
	match = re.search(r"\b[A-Z0-9]{6,12}\b", text.strip(), flags=re.IGNORECASE)
	return match.group(0).upper() if match else text.strip().upper()


def _render_booking_url_callout():
	booking_url = st.session_state.get("booking_url", "")
	if booking_url:
		st.markdown(f"[Click here to complete your booking]({booking_url})")
		st.info("Complete your payment on the booking page. Your PNR will be sent via email and SMS.")


def _render_flight_cards():
	flight_options = st.session_state.get("flight_options", [])
	if not flight_options:
		return

	st.subheader("Live flight options")
	for idx, item in enumerate(flight_options):
		card_text, booking_url = item
		parts = [piece.strip() for piece in card_text.split("|")]
		left = parts[0] if len(parts) > 0 else card_text
		middle = parts[1] if len(parts) > 1 else ""
		right = parts[2] if len(parts) > 2 else ""
		seats = parts[3] if len(parts) > 3 else ""

		with st.container(border=True):
			col1, col2 = st.columns([4, 1])
			with col1:
				st.markdown(f"**{left}**")
				if middle:
					st.write(middle)
				if right:
					st.write(right)
				if seats:
					st.caption(seats)
			with col2:
				if hasattr(st, "link_button"):
					st.link_button("Book this flight", booking_url, use_container_width=True)
				else:
					st.markdown(f"[Book this flight]({booking_url})")


def handle_flight_search(message: str) -> str:
	origin, destination = _extract_route(message)
	if not origin or not destination:
		return "Please specify origin and destination cities (for example, 'Delhi to Mumbai')."

	date = _extract_date_token(message)
	flights = search_live_flights(origin, destination, date or None)

	flight_cards = []
	for f in flights[:3]:
		card = (
			f"✈ {f['airline']} {f['flight_no']} | "
			f"{f['departure']} → {f['arrival']} | "
			f"₹{f['price']:,} | {f['seats_available']} seats"
		)
		booking = get_booking_page_url(f, origin, destination, date)
		flight_cards.append((card, booking["booking_url"]))

	st.session_state.flight_options = flight_cards
	st.session_state.awaiting_pnr = False
	st.session_state.booking_lookup_pending = False
	st.session_state.booking_url = flight_cards[0][1] if flight_cards else ""

	if not flight_cards:
		return f"I couldn't find live flights for {origin} to {destination}."

	return f"I found {len(flights)} live flights for {origin} to {destination}. Choose a card below to book."


def _format_booking_result(booking: dict) -> str:
	if booking.get("error"):
		return f"I couldn't find that booking. {booking['error']}"

	return (
		f"PNR {booking.get('pnr', '')} is confirmed for {booking.get('passenger_name', booking.get('passenger', 'Guest'))}. "
		f"Flight {booking.get('flight_no', '')} from {booking.get('origin', '')} to {booking.get('destination', '')}, "
		f"seat {booking.get('seat', 'TBA')}, status {booking.get('status', 'confirmed')}."
	)


if st.session_state.retriever is None and st.session_state.load_error is None:
	with st.spinner("Loading knowledge base..."):
		try:
			st.session_state.retriever = load_knowledge_base()
			st.success("Knowledge base loaded!")
		except Exception as e:
			st.session_state.load_error = str(e)
			st.error(f"Failed to load knowledge base: {e}")
			st.info("Run: python build_knowledge_base.py first")
			st.stop()


with st.sidebar:
	st.title("AirAssist")
	st.write("Your AI airline assistant — policy-aware answers")

	airline_filter = st.selectbox("Airline filter", ["All", "IndiGo", "Air India", "SpiceJet", "Vistara"])

	st.divider()
	st.caption("Quick questions")
	for idx, question in enumerate(QUICK_QUESTIONS):
		if st.button(question, key=f"quick_q_{idx}"):
			st.session_state.pending_input = question

	st.divider()
	if st.button("Clear chat"):
		st.session_state.messages = []
		st.session_state.memory = memory_module.ConversationMemory()
		st.session_state.flight_options = []
		st.session_state.booking_url = ""
		st.session_state.awaiting_pnr = False
		st.session_state.booking_lookup_pending = False
		st.session_state.booking_lookup_result = None

if st.session_state.retriever is None:
	st.warning("Retriever not available. Run build_knowledge_base.py to create the vector store, then reload this app.")
	if st.session_state.load_error:
		st.info(f"Last error: {st.session_state.load_error}")

_render_booking_url_callout()

# render messages
for msg in st.session_state.messages:
	if msg["role"] == "user":
		st.markdown(f"<div class='user-bubble'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)
	else:
		st.markdown(f"<div class='bot-bubble'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)

_render_flight_cards()

if st.session_state.awaiting_pnr:
	st.info("Please enter your PNR so I can look up your confirmed booking.")


def submit_message(text):
	user_text = text.strip()
	st.session_state.messages.append({"role": "user", "content": text})

	if st.session_state.awaiting_pnr:
		pnr = _extract_pnr(user_text)
		with st.spinner("Looking up your booking..."):
			booking = lookup_booking_by_pnr(pnr)
		assistant_reply = _format_booking_result(booking)
		st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
		st.session_state.awaiting_pnr = False
		st.session_state.booking_lookup_pending = False
		st.session_state.booking_url = ""
		if hasattr(st, "rerun"):
			st.rerun()
		else:
			st.experimental_rerun()

	if _is_booking_completion_message(user_text):
		st.session_state.awaiting_pnr = True
		st.session_state.booking_lookup_pending = True
		st.session_state.booking_url = ""
		assistant_reply = "Please share your PNR and I’ll look up your confirmed booking."
		st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
		if hasattr(st, "rerun"):
			st.rerun()
		else:
			st.experimental_rerun()

	if any(phrase in _normalize_text(user_text) for phrase in ["what flights are available", "search flights", "find flights", "flights from", "flight to", "available flights"]):
		with st.spinner("Searching live flights..."):
			resp = handle_flight_search(user_text)
		st.session_state.messages.append({"role": "assistant", "content": resp})
		if hasattr(st, "rerun"):
			st.rerun()
		else:
			st.experimental_rerun()

	with st.spinner("Thinking..."):
		resp = chain_module.chat(user_text, st.session_state.memory, st.session_state.retriever, None if airline_filter == "All" else airline_filter)
	st.session_state.messages.append({"role": "assistant", "content": resp})
	if hasattr(st, "rerun"):
		st.rerun()
	else:
		st.experimental_rerun()


# handle pending quick question
pending = st.session_state.get("pending_input", "")
if pending:
	text = pending
	st.session_state.pending_input = ""
	submit_message(text)

# chat input
user_input = st.chat_input("Ask me about flights, baggage, refunds...")
if user_input:
	submit_message(user_input)

# Add to app.py — Phase 3 additions (do NOT remove Phase 1/2 code)

# 1. Start the background scheduler when app loads
from proactive.scheduler import start_scheduler
if "scheduler_started" not in st.session_state:
    start_scheduler()
    st.session_state.scheduler_started = True

# 2. Add risk score to flight search results
# After showing flight results, add:
if st.session_state.get("last_flights"):
    with st.expander("⚡ Travel Risk Score"):
        for f in st.session_state.last_flights[:1]:  # show for cheapest
            from proactive.delay_predictor import predict_delay
            risk = predict_delay(
                f["airline"], f["origin"], f["destination"], f["departure"]
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("On-time Probability",
                        f"{risk['on_time_probability']}%")
            col2.metric("Risk Level", risk["risk_level"])
            col3.metric("Delay Risk", f"{risk['delay_probability']}%")
            st.info(f"💡 {risk['recommendation']}")

# 3. Add quick links to new pages in sidebar
st.sidebar.divider()
st.sidebar.page_link("pages/1_Analytics.py", label="Analytics Dashboard")
st.sidebar.page_link("pages/2_My_Bookings.py", label="My Bookings")