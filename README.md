<div align="center">
  AirAssist — AI Airline Travel Concierge

**A 3-phase production-style AI system for airline travel — answers policy questions, executes real bookings, and proactively assists travellers.**

[Architecture](#architecture) · [Evaluation](#evaluation) · [Quick Start](#quick-start) · [Tech Stack](#tech-stack)

</div>

---

## What This Is

Most airline chatbots break the moment you ask anything unexpected. AirAssist is built across three phases that transform a passive FAQ responder into a proactive travel concierge — covering IndiGo, Air India, SpiceJet, and Vistara.

| Phase | Capability |
|-------|-----------|
| **Phase 1 — RAG Foundation** | Policy Q&A with hybrid retrieval, metadata-filtered FAISS + BM25 + CrossEncoder reranker, 9-intent classifier |
| **Phase 2 — Agentic Layer** | 10 tools, real API integrations — live flights, Razorpay payments, SendGrid email, Twilio SMS |
| **Phase 3 — Proactive AI** | Delay prediction, check-in alerts, PDF boarding passes, analytics dashboard |

---

## Architecture

![Architecture Diagram](docs/airAssist_architecture_diagram.png)

### How a query flows through the system

```
User Query
    │
    ▼
Streamlit UI (localhost:8501)
    │
    ▼
Intent Classifier — Groq LLaMA 3.3 70B · temp=0 · 14 classes
    │
    ├── FAQ intent ──────────► RAG Chain
    │                           FAISS (dense) + BM25 (sparse) · 50/50 ensemble
    │                           CrossEncoder reranker (ms-marco-MiniLM)
    │                           Airline metadata filter → zero cross-airline contamination
    │                           15,000 chunks · 5 data sources
    │
    ├── Booking ─────────────► Booking Engine
    │                           Flask :5000 · Razorpay checkout
    │                           FastAPI :8001 · PNR · SQLite · PDF boarding pass
    │
    └── Agentic ─────────────► Agent Loop (observe → plan → act · max 5 iterations)
                                10 tools: check_status · search_flights · rebook
                                calculate_refund · track_baggage · file_claim · …
                                External: AviationStack · Razorpay · SendGrid · Twilio
```

### Data pipeline (Phase 1)

```
Wikipedia API ──┐
Kaggle 128k   ──┤──► Chunker (400 tok) ──► Metadata tagger ──► FAISS vector store
HuggingFace   ──┤                           airline/topic/route   15,000 chunks
Playwright    ──┤
Synthetic     ──┘
```
Every chunk is tagged at ingest. At query time, `airline="indigo"` filter runs *before* retrieval — Air India content cannot appear in an IndiGo answer by design.

### Full booking flow (Phase 2)
```
Chat: "Search DEL→BOM"
  → AviationStack API (real flight data)
  → Flask booking page — passenger form + Razorpay checkout
  → Payment verified via HMAC signature
  → FastAPI: PNR generated + stored in SQLite
  → SendGrid email dispatched
  → Twilio SMS dispatched
  → PDF boarding pass (ReportLab + QR code)
```

---
## Evaluation

### Retrieval precision

| Method | Precision |
|--------|-----------|
| Dense only (FAISS) | 60% |
| Hybrid (FAISS + BM25) | 68% |
| **Hybrid + CrossEncoder rerank** | **91%** |

### Golden eval set — 25 hand-crafted Q&A pairs

```
Passed:                   23 / 25  (92%)
Keyword hit rate:         75.0%    (69 / 92 keywords)
Cross-airline contamination:  0 cases
```
The 2 failures are data gaps (SpiceJet pet policy, Vistara fare conditions) — not retrieval errors.

### Latency

```
RAG (FAQ query):     ~678 ms
Agentic (tool call): ~714–2652 ms
Average:             ~1348 ms
```

### Test coverage
```
67 tests collected · 64 passed · 3 failed (intent edge cases, not core pipeline)
```

---
## Quick Start
### Prerequisites

- Python 3.12+
- [Groq API key](https://console.groq.com) — free
- [OpenAI API key](https://platform.openai.com) — embeddings only (~₹1 total)
- [Razorpay test account](https://razorpay.com) — free
- [AviationStack key](https://aviationstack.com) — free 500 calls/month
- [SendGrid key](https://sendgrid.com) — free 100 emails/day
- [Twilio account](https://twilio.com) — free trial

### Setup

```bash
git clone https://github.com/yourusername/airAssist
cd airAssist

python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

### Build knowledge base (one-time, ~15 min)

```bash
python build_knowledge_base.py
```

### Run (3 terminals)

```bash
# Terminal 1 — FastAPI confirmation engine
uvicorn api.main:app --port 8001 --reload

# Terminal 2 — Flask booking + Razorpay
python booking_app/app.py

# Terminal 3 — Streamlit chatbot
streamlit run app.py --server.fileWatcherType none
```

Open [localhost:8501](http://localhost:8501)

### Test payment flow
```
Card:    4111 1111 1111 1111
Expiry:  12/28 · CVV: 123 · OTP: 1234
```
---

## Project Structure

```
airline_chatbot/
├── agents/          Agent loop, disruption handler, 10 tool definitions
├── api/             FastAPI :8001 — confirmation, PNR, booking lookup
├── booking/
├── booking_app/     Flask :5000 — passenger form + Razorpay checkout
├── boarding_pass/   PDF generator with QR code (ReportLab)
├── chatbot/         Intent classifier, RAG chain, conversation memory
├── rag/             Scraper, chunker, embedder, hybrid retriever
├── tools/           Live flights, payment, notifications, user profiles
├── guards/          PII redactor (Presidio), faithfulness checker
├── proactive/       Delay predictor, check-in scheduler (APScheduler)
├── analytics/       Query tracker + Plotly dashboard
├── evals/           Golden eval set (25 Q&A), eval runner
├── tests/           Unit + integration tests (pytest · 67 tests)
├── pages/           Streamlit multi-page (Analytics, My Bookings)
├── data/            Vector store, SQLite databases, boarding passes
└── app.py           Entry point
```

---
## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq LLaMA 3.3 70B |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS (local) |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain EnsembleRetriever |
| Agent framework | Custom observe-plan-act loop (Groq function calling) |
| Payment | Razorpay |
| Email | SendGrid |
| SMS | Twilio |
| Flight data | AviationStack REST API |
| Backend | FastAPI + uvicorn |
| Booking UI | Flask + Bootstrap 5 |
| Frontend | Streamlit (multi-page) |
| PDF | ReportLab + qrcode |
| Scheduling | APScheduler |
| PII | Microsoft Presidio |
| Database | SQLite |
| Charts | Plotly |
| Testing | pytest |

---
## Pages

| URL | Description |
|-----|-------------|
| `localhost:8501` | Main chatbot |
| `localhost:8501/Analytics` | Metrics dashboard — intent distribution, latency, tool usage |
| `localhost:8501/My_Bookings` | Booking history + boarding pass download |
| `localhost:5000/book?...` | Passenger form + Razorpay checkout |
| `localhost:8001/booking/{pnr}` | Booking lookup API |

---

