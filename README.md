# FlightAI: Airline Customer Support Chatbot

FlightAI is a Python chatbot for airline support workflows.
It combines intent routing, entity extraction, flight APIs, and retrieval-augmented generation (RAG) over policy documents.

## Features

- Intent routing for:
  - flight search
  - price comparison
  - policy questions
  - flight status
  - general chat
- Multi-turn clarification for missing flight entities (origin, destination, date)
- Live flight offer search via FlightAPI.io
- Flight status lookup via Aviationstack client
- Hybrid RAG retrieval (FAISS semantic + BM25 lexical)
- Metadata-aware doc-type filtering for policy chunks
- Scheduled live policy updater for airline policy web pages
- Multiple LLM backends:
  - OpenAI
  - OpenRouter
  - Ollama (local)
- Gradio web interface with model selection and streaming responses
- Retrieval evaluation utilities with MRR, Precision@K, Recall@K

## Repository Structure

- app.py: Gradio app entrypoint
- build_vectorstore.py: Build FAISS and BM25 artifacts
- run_evaluation.py: Run retrieval evaluation and log results
- test_llm.py: Simple retrieval metric smoke test
- settings.py: Default provider setting
- apis/: External API clients
- llm/: LLM provider abstractions and factory
- pipeline/: Routing and context-building pipeline
- rag/: Retriever and vectorstore artifacts
- rag_evaluation/: Evaluation datasets and metrics
- utils/: Intent/entity parsing, formatting, normalization helpers
- prompts/system.txt: System prompt reference

## Prerequisites

- Python 3.10+
- Internet access for APIs and model downloads
- API keys for the providers you use
- Optional: Ollama installed and running for local models

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Flight API setup

This project uses FlightAPI.io for live flight data.

1. Sign up at https://www.flightapi.io — instant free key
2. Add to .env: FLIGHTAPI_KEY=your_key_here
3. Free tier: 20 calls. Use caching in development to preserve quota.

## Environment Variables

Set these before running the app:

- FLIGHTAPI_KEY
- AVIATIONSTACK_API_KEY (required for flight status)
- OPENAI_API_KEY (required only for OpenAI models)
- OPENROUTER_API_KEY (required only for OpenRouter models)
- POLICY_AUTO_UPDATE (`true` or `false`, default `true`)
- POLICY_UPDATE_DAY (weekday for scheduled updater, default `monday`)

On PowerShell, example:

```powershell
$env:FLIGHTAPI_KEY="your_flightapi_key"
$env:AVIATIONSTACK_API_KEY="your_aviationstack_key"
$env:OPENAI_API_KEY="your_openai_key"
$env:OPENROUTER_API_KEY="your_openrouter_key"
```

## Build Retrieval Artifacts

Place your airline policy PDFs in data/ and run:

```bash
python build_vectorstore.py
```

This creates:

- rag/vectorstore/
- rag/bm25.pkl
- rag/chunk_metadata.json
- all_chunks.csv

## Live Policy Updater

Run the updater to fetch live airline policy pages, re-chunk content, and refresh retrieval artifacts:

```bash
python scripts/policy_updater.py
```

What it does:

- fetches configured airline policy URLs
- cleans boilerplate-heavy HTML text
- chunks text using 500 chars with 50 overlap
- replaces old chunks from the same source in FAISS/BM25
- updates `rag/chunk_metadata.json` with source/doc_type/airline/timestamp metadata
- appends run logs to `logs/policy_updates.csv`

## Run the App

```bash
python app.py
```

Then open the local Gradio URL shown in terminal (typically http://127.0.0.1:7860).

## Model Selection

The UI dropdown in app.py supports:

- GPT-4o Mini (OpenRouter)
- Claude Haiku (OpenRouter)
- Phi-3 (Ollama)
- Mistral (Ollama)
- gpt-4o
- gpt-3.5-turbo

## Evaluation

Run retrieval evaluation:

```bash
python run_evaluation.py
```

This uses:

- rag_evaluation/evaluation.py for labeled queries
- rag_evaluation/metrics.py for MRR, Precision@K, Recall@K
- rag_evaluation/experiment_logger.py to append results to experiments.csv

Quick smoke metric run:

```bash
python test_llm.py
```

## High-Level Flow

1. User query enters app.py
2. pipeline/router.py classifies intent
3. For flight flows, utils/entity_extractor.py extracts entities
4. utils/clarifier.py requests missing fields when needed
5. API and/or retrieval data is gathered
6. pipeline/context_builder.py merges context for LLM response
7. Final response is rendered in Gradio chat

## Troubleshooting

- Vectorstore not found:
  - Ensure PDFs exist in data/
  - Run python build_vectorstore.py
- Ollama errors:
  - Start Ollama server and ensure model is pulled
- API errors:
  - Verify keys and quota limits
- Empty or weak retrieval:
  - Rebuild artifacts after changing data/

## Notes

- Current settings.py defaults LLM_PROVIDER to ollama.
- The app is designed to keep conversational state in memory during a session.
- Retrieval quality depends heavily on document quality and chunk coverage.
