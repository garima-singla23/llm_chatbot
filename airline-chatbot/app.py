import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import pickle

from chatbot import chain as chain_module
from chatbot import memory as memory_module
from rag import embedder as embedder_module
from rag import chunker as chunker_module
from rag import retriever as retriever_module


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


init_session()


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
	if st.button("What is carry-on baggage allowance?"):
		st.session_state.pending_input = "What is the carry-on baggage allowance for my flight?"
	if st.button("How do I cancel and get a refund?"):
		st.session_state.pending_input = "How can I cancel my booking and what is the refund policy?"
	if st.button("How early to check in?"):
		st.session_state.pending_input = "How early should I check in for a domestic flight?"
	if st.button("Where is my PNR?"):
		st.session_state.pending_input = "How do I find my PNR and booking details?"

	st.divider()
	if st.button("Clear chat"):
		st.session_state.messages = []
		st.session_state.memory = memory_module.ConversationMemory()

if st.session_state.retriever is None:
	st.warning("Retriever not available. Run build_knowledge_base.py to create the vector store, then reload this app.")
	if st.session_state.load_error:
		st.info(f"Last error: {st.session_state.load_error}")

# render messages
for msg in st.session_state.messages:
	if msg["role"] == "user":
		st.markdown(f"<div class='user-bubble'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)
	else:
		st.markdown(f"<div class='bot-bubble'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)


def submit_message(text):
	st.session_state.messages.append({"role": "user", "content": text})
	with st.spinner("Thinking..."):
		resp = chain_module.chat(text, st.session_state.memory, st.session_state.retriever, None if airline_filter == "All" else airline_filter)
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

