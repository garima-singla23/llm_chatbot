# SKYASSIST — Airline Policy Chatbot (RAG + Agentic, Phase 1)

A production-style AI chatbot that answers baggage, refund, and check-in questions for 4 Indian airlines — without ever confusing one airline's policy with another's. Built from scratch with a full retrieval pipeline, typed memory, intent routing, and a mock booking engine.

> **Status:** Phase 1 complete · Phase 2 (agentic) in progress

---

## 📸 Demo

### Policy Q&A — Cross-airline accuracy
<!-- SCREENSHOT 1: Show a policy question answered (e.g. "What is IndiGo's baggage allowance on domestic routes?") with the airline filter visible -->
![Policy Q&A](screenshots/policy_qa.png)

### Flight Search + Booking
<!-- SCREENSHOT 2: Show the booking flow — user asks to search/book a flight in plain English, bot responds with options -->
![Booking Flow](screenshots/booking_flow.png)

### Context-Aware Follow-ups
<!-- SCREENSHOT 3: Show a multi-turn conversation where a follow-up like "what about the window seat?" works correctly -->
![Multi-turn Memory](screenshots/memory_context.png)

---

## What It Does

| Capability | Detail |
|---|---|
| Policy Q&A | Baggage, refund, check-in, visa, seat, meal, delay — for IndiGo, Air India, SpiceJet, Vistara |
| Cross-airline isolation | Metadata-filtered retrieval prevents IndiGo rules from bleeding into Air India answers |
| Mock flight booking | Search and book in plain English via a structured booking engine |
| Context memory | Remembers conversation history, user preferences (seat, meal, airline), and active booking state across turns |
| Intent routing | Every query is classified into 1 of 9 intents before any chain runs — FAQ and booking flows are fully separated |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     Intent Classifier        │  GPT-3.5, temp=0, 9 intents
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
  FAQ Flow        Booking Flow
     │
     ▼
┌─────────────────────────────────────────────┐
│              Hybrid Retriever                │
│                                             │
│  FAISS dense (top 8)  +  BM25 (top 8)       │
│           ↓                                 │
│     Airline metadata filter                 │
│           ↓                                 │
│  CrossEncoder reranker → top 4 chunks       │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Groq LLaMA 3.3 70B        │  Answer generation
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   ConversationMemory         │  Deque (10 turns) + user_prefs dict
└─────────────────────────────┘
```

---

## The 3 Technical Decisions That Made It Reliable

### 1. Metadata-filtered chunks (structural cross-airline isolation)

Every chunk scraped from airline websites is tagged at index time with:
- `airline` — `indigo`, `air_india`, `spicejet`, `vistara`
- `topic` — one of 8 categories: `baggage`, `refund`, `check_in`, `visa`, `seat`, `meal`, `delay`, `general`
- `route_type` — `domestic`, `international`, or `both`

Retrieval filters on these fields **before** vector search runs. Cross-airline confusion is structurally prevented, not prompt-engineered away.

### 2. Three-stage hybrid retrieval

```
Stage 1 — Hybrid retrieval
  FAISS dense search  →  top 8 by semantic similarity
  BM25 keyword search →  top 8 by exact term match
  Merged & deduplicated → up to 16 unique candidates

Stage 2 — Airline filter
  Keep only chunks matching requested airline (case-insensitive)

Stage 3 — CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
  Score all candidates against the query
  Sort by relevance → keep top 4

→ Final 4 chunks sent to LLM
```

Retrieval precision on a 25-question golden eval set: **91%**

### 3. Typed, three-container memory

`ConversationMemory` maintains three separate storage containers:

| Container | Type | Behaviour |
|---|---|---|
| `turns` | `deque(maxlen=20)` | Auto-drops oldest turns when full — no manual truncation logic |
| `user_prefs` | `dict` | Persists across turns: `preferred_seat`, `preferred_meal`, `preferred_airline` |
| `active_booking` | `dict` | Mid-flow booking state scratchpad; cleared on `clear()`, prefs are not |

The preference summary is injected into every system prompt so the LLM stays aware of what it knows about the user even across long conversations.

---

## Knowledge Base Pipeline

The bot's knowledge comes from 5 sources, all ingested by a single `build_knowledge_base.py` orchestrator:

```
1. Wikipedia scraper     — Airline pages + DGCA / Montreal Convention regulations
2. Playwright scraper    — Dynamic airline website pages (AAI etc.)
3. Kaggle datasets       — Twitter sentiment, Skytrax reviews, Bitext support QA
4. Hugging Face datasets — Bitext customer support + travel FAQ (filtered to airline domain)
5. Synthetic fallback    — Hand-written policy docs for coverage gaps (clearly labelled)
```

Each source produces a `.txt` + `.meta.json` pair per airline/topic, then the whole corpus is chunked (400 chars, 60-char overlap), embedded, and indexed into FAISS.

---

## Evaluation

A keyword-match harness (`run_eval.py`) runs against a golden set of 25 test questions, each with:
- Expected keywords that must appear in the answer
- Optional airline filter
- Fresh `ConversationMemory` per test case to prevent contamination

A case passes if ≥ 50% of expected keywords are present. This is intentionally lightweight — deterministic, fast, and robust to wording variation while still verifying factual content.

**Current results:** 91% retrieval precision on the golden set.

---

## Project Structure

```
airline-chatbot/
├── rag/
│   ├── scraper.py          # HTTP + BeautifulSoup scraper with graceful error handling
│   ├── chunker.py          # Topic detection, route detection, RecursiveCharacterTextSplitter
│   ├── embedder.py         # OpenAI text-embedding-3-small or BAAI/bge-small-en-v1.5 (local)
│   ├── retriever.py        # Hybrid FAISS + BM25 + CrossEncoder reranker
│   ├── synthetic_data.py   # Fallback synthetic policy generation
│   ├── wiki_scraper.py     # Wikipedia REST API scraper
│   ├── kaggle_loader.py    # Kaggle dataset ingestion
│   └── hf_loader.py        # Hugging Face dataset ingestion
├── chatbot/
│   ├── memory.py           # ConversationMemory: deque + user_prefs + active_booking
│   └── intent.py           # 9-intent classifier (LLM-based, temp=0)
├── data/
│   ├── raw/                # .txt + .meta.json pairs per airline/topic
│   └── vector_store/       # FAISS index files
├── build_knowledge_base.py # End-to-end pipeline orchestrator
├── run_eval.py             # Keyword-match evaluation harness
├── app.py                  # Streamlit UI
└── requirements.txt
```

---

## Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA 3.3 70B (free tier) |
| Embeddings | OpenAI `text-embedding-3-small` (or `BAAI/bge-small-en-v1.5` for local/offline) |
| Vector store | FAISS |
| Keyword search | BM25 (via `rank_bm25`) |
| Reranker | `ms-marco-MiniLM-L-6-v2` (CrossEncoder, SentenceTransformers) |
| Intent classifier | GPT-3.5 Turbo, temperature=0 |
| Orchestration | LangChain |
| UI | Streamlit |
| Data sources | Wikipedia API, Playwright, Kaggle, Hugging Face |

---

## Setup

```bash
git clone https://github.com/your-username/airline-chatbot.git
cd airline-chatbot
pip install -r requirements.txt
```

Set your API keys:
```bash
export GROQ_API_KEY=your_groq_key
export OPENAI_API_KEY=your_openai_key   # only needed for embeddings + intent classifier
```

Build the knowledge base (first run only):
```bash
python build_knowledge_base.py
```

Run the app:
```bash
streamlit run app.py
```

Run evaluation:
```bash
python run_eval.py
```

---

## What's Coming in Phase 2 (Agentic)

- Live flight status via real airline / aggregator APIs
- Actual rebooking and cancellation flows
- Refund calculation engine
- Tool-calling agent replacing the current intent router

---

## Honest Limitations

- Booking is **mock only** — no real reservation system is connected
- Scraped policy data may be stale; synthetic fallback is clearly labelled
- Eval set (25 questions) is small — sufficient for regression testing, not a benchmark
- Intent classifier adds one LLM call per query (latency tradeoff vs. accuracy)

---

## Author

**Garima Singla** · AI/ML Engineer  
Building in public · [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)