import csv
import json
import os
import pickle
import re
import time
from datetime import datetime

from dotenv import load_dotenv

import requests
import schedule
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from rag.retriever import _get_embeddings

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - fallback for older langchain installs
    from langchain.schema import Document


POLICY_SOURCES = {
    "indigo_baggage": {
        "url": "https://www.goindigo.in/information/baggage.html",
        "doc_type": "baggage",
        "airline": "IndiGo",
    },
    "airindia_baggage": {
        "url": "https://www.airindia.com/in/en/baggage.html",
        "doc_type": "baggage",
        "airline": "Air India",
    },
    "indigo_cancellation": {
        "url": "https://www.goindigo.in/information/cancellation-policy.html",
        "doc_type": "cancellation",
        "airline": "IndiGo",
    },
    "spicejet_baggage": {
        "url": "https://www.spicejet.com/baggage-info",
        "doc_type": "baggage",
        "airline": "SpiceJet",
    },
}

USER_AGENT = {"User-Agent": "Mozilla/5.0"}
NAV_WORDS = {
    "menu",
    "home",
    "about",
    "contact",
    "privacy",
    "cookie",
    "terms",
    "sign in",
    "login",
    "register",
    "book",
    "flights",
    "offers",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DIR = os.path.join(BASE_DIR, "rag")
VECTORSTORE_PATH = os.path.join(RAG_DIR, "vectorstore")
BM25_PATH = os.path.join(RAG_DIR, "bm25.pkl")
METADATA_PATH = os.path.join(RAG_DIR, "chunk_metadata.json")
ALL_CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "policy_updates.csv")

load_dotenv(os.path.join(BASE_DIR, ".env"))


def _chunk_id_sort_key(record: dict) -> int:
    try:
        return int(record.get("chunk_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def fetch_policy_text(url: str, timeout: int = 15) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout, headers=USER_AGENT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[Updater] Fetch failed for {url}: {exc}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    raw_text = soup.get_text(separator=" ", strip=True)

    fragments = re.split(r"[|•·\\n\\r\\t]+", raw_text)
    cleaned_fragments = []
    for fragment in fragments:
        normalized = re.sub(r"\\s+", " ", fragment).strip()
        if len(normalized) < 30:
            continue

        lowered = normalized.lower()
        if any(nav_word in lowered for nav_word in NAV_WORDS):
            continue

        cleaned_fragments.append(normalized)

    cleaned_text = "\n".join(cleaned_fragments)
    if len(cleaned_text) < 100:
        return None

    return cleaned_text


def chunk_text(
    text: str,
    source_key: str,
    doc_type: str,
    airline: str,
    fetched_at: str,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "source": source_key,
            "doc_type": doc_type,
            "airline": airline,
            "fetched_at": fetched_at,
            "chunk_id": None,
        }
        for chunk in chunks
        if chunk.strip()
    ]


def _load_existing_chunks(vectorstore: FAISS) -> list[dict]:
    existing_chunks: list[dict] = []
    docstore = getattr(vectorstore, "docstore", None)
    docs_map = getattr(docstore, "_dict", {}) if docstore else {}

    for doc in docs_map.values():
        if not isinstance(doc, Document):
            continue

        metadata = dict(getattr(doc, "metadata", {}) or {})
        chunk_text_value = str(getattr(doc, "page_content", "") or "").strip()
        if not chunk_text_value:
            continue

        existing_chunks.append(
            {
                "text": chunk_text_value,
                "source": metadata.get("source", "legacy"),
                "doc_type": metadata.get("doc_type", "general"),
                "airline": metadata.get("airline", "Unknown"),
                "fetched_at": metadata.get("fetched_at", ""),
                "chunk_id": metadata.get("chunk_id"),
                "page": metadata.get("page", 0),
            }
        )

    return existing_chunks


def _load_metadata(metadata_path: str) -> list[dict]:
    if not os.path.exists(metadata_path):
        return []

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[Updater] Failed to read metadata file: {exc}")
        return []

    return data if isinstance(data, list) else []


def merge_into_index(
    new_chunks: list[dict],
    vectorstore_path: str,
    bm25_path: str,
    metadata_path: str,
):
    if not new_chunks:
        return

    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}")

    vectorstore = FAISS.load_local(
        vectorstore_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    existing_metadata = _load_metadata(metadata_path)
    existing_chunks = _load_existing_chunks(vectorstore)

    source_key = str(new_chunks[0].get("source", ""))
    kept_metadata = [m for m in existing_metadata if m.get("source") != source_key]
    kept_chunks = [c for c in existing_chunks if c.get("source") != source_key]

    max_existing_id = -1
    for meta in kept_metadata:
        try:
            max_existing_id = max(max_existing_id, int(meta.get("chunk_id", -1)))
        except (TypeError, ValueError):
            continue

    for offset, chunk in enumerate(new_chunks, start=1):
        chunk["chunk_id"] = max_existing_id + offset

    all_chunks = kept_chunks + new_chunks

    rebuilt_texts = [chunk["text"] for chunk in kept_chunks]
    rebuilt_metas = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "doc_type": chunk.get("doc_type", "general"),
            "airline": chunk.get("airline", "Unknown"),
            "fetched_at": chunk.get("fetched_at", ""),
            "page": chunk.get("page", 0),
        }
        for chunk in kept_chunks
    ]

    if rebuilt_texts:
        vectorstore = FAISS.from_texts(
            texts=rebuilt_texts,
            embedding=_get_embeddings(),
            metadatas=rebuilt_metas,
            distance_strategy=DistanceStrategy.COSINE,
        )
    else:
        seed_text = [new_chunks[0]["text"]]
        seed_meta = [
            {
                "chunk_id": new_chunks[0]["chunk_id"],
                "source": new_chunks[0]["source"],
                "doc_type": new_chunks[0]["doc_type"],
                "airline": new_chunks[0]["airline"],
                "fetched_at": new_chunks[0]["fetched_at"],
                "page": 0,
            }
        ]
        vectorstore = FAISS.from_texts(
            texts=seed_text,
            embedding=_get_embeddings(),
            metadatas=seed_meta,
            distance_strategy=DistanceStrategy.COSINE,
        )
        new_chunks = new_chunks[1:]

    if new_chunks:
        new_texts = [chunk["text"] for chunk in new_chunks]
        new_metas = [
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "doc_type": chunk["doc_type"],
                "airline": chunk["airline"],
                "fetched_at": chunk["fetched_at"],
                "page": 0,
            }
            for chunk in new_chunks
        ]
        vectorstore.add_texts(new_texts, metadatas=new_metas)

    vectorstore.save_local(vectorstore_path)

    metadata_records = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "doc_type": chunk.get("doc_type", "general"),
            "source": chunk.get("source", ""),
            "page": int(chunk.get("page", 0) or 0),
            "airline": chunk.get("airline", "Unknown"),
            "fetched_at": chunk.get("fetched_at", ""),
        }
        for chunk in all_chunks
    ]

    metadata_records.sort(key=_chunk_id_sort_key)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    tokenized_corpus = [re.sub(r"[^\\w\\s]", "", c["text"].lower()).split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    split_docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "doc_type": chunk.get("doc_type", "general"),
                "airline": chunk.get("airline", "Unknown"),
                "fetched_at": chunk.get("fetched_at", ""),
                "page": int(chunk.get("page", 0) or 0),
            },
        )
        for chunk in all_chunks
    ]

    with open(bm25_path, "wb") as f:
        pickle.dump((bm25, split_docs), f)

    try:
        import pandas as pd

        chunk_rows = [{"chunk_id": c.get("chunk_id"), "text": c.get("text", "")} for c in all_chunks]
        pd.DataFrame(chunk_rows).to_csv(ALL_CHUNKS_PATH, index=False)
    except Exception as exc:
        print(f"[Updater] Could not refresh all_chunks.csv: {exc}")


def _append_log_row(timestamp: str, source_key: str, chunks_added: int, status: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "chunks_added", "status"])
        writer.writerow([timestamp, source_key, chunks_added, status])


def run_update(dry_run: bool = False):
    fetched_at = datetime.now().isoformat()
    updated_count = 0

    for source_key, source in POLICY_SOURCES.items():
        text = fetch_policy_text(source["url"])
        if text is None:
            _append_log_row(fetched_at, source_key, 0, "fetch_failed")
            print(f"[Updater] {source_key}: skipped (fetch failed)")
            continue

        chunks = chunk_text(
            text=text,
            source_key=source_key,
            doc_type=source["doc_type"],
            airline=source["airline"],
            fetched_at=fetched_at,
        )

        try:
            if not dry_run:
                merge_into_index(
                    new_chunks=chunks,
                    vectorstore_path=VECTORSTORE_PATH,
                    bm25_path=BM25_PATH,
                    metadata_path=METADATA_PATH,
                )
                status = "indexed"
            else:
                status = "dry_run"

            updated_count += 1
            _append_log_row(fetched_at, source_key, len(chunks), status)
            print(f"[Updater] {source_key}: {len(chunks)} chunks indexed")
        except Exception as exc:
            _append_log_row(fetched_at, source_key, 0, f"merge_failed: {exc}")
            print(f"[Updater] {source_key}: merge failed ({exc})")

    print(f"Policy update complete. {updated_count} sources updated.")


if __name__ == "__main__":
    run_update()

    auto_update_enabled = os.getenv("POLICY_AUTO_UPDATE", "true").strip().lower() == "true"
    update_day = os.getenv("POLICY_UPDATE_DAY", "monday").strip().lower()

    if auto_update_enabled:
        day_scheduler = getattr(schedule.every(), update_day, None)
        if day_scheduler is None:
            day_scheduler = schedule.every().monday
        day_scheduler.at("02:00").do(run_update)

        while True:
            schedule.run_pending()
            time.sleep(3600)
