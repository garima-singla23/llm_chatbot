import os
import hashlib
import pickle
import logging
import socket
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import re
from retriever.reranker import get_reranker
from retriever.window_expander import expand_with_metadata
from retriever.metadata_filter import MetadataFilter, detect_doc_type

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - fallback for older langchain installs
    from langchain.schema import Document

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RAG_DIR = os.path.join(PROJECT_ROOT, "rag")
VECTOR_PATH = os.path.join(RAG_DIR, "vectorstore")
BM25_PATH = os.path.join(RAG_DIR, "bm25.pkl")
CHUNK_METADATA_PATH = os.path.join(RAG_DIR, "chunk_metadata.json")
CHUNKS_CSV_PATH = os.path.join(PROJECT_ROOT, "all_chunks.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_VECTORSTORE: Optional[FAISS] = None
_BM25: Optional[BM25Okapi] = None
_SPLIT_DOCS = None
_ALL_CHUNKS: Optional[list[str]] = None
_METADATA_FILTER: Optional[MetadataFilter] = None


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def generate_chunk_id(text):
    text = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(text.encode()).hexdigest()


def _doc_type_from_source(source: str) -> str:
    filename = os.path.basename(str(source or "")).lower()
    mapping = {
        "baggage_policy.pdf": "baggage",
        "cancellation_policy.pdf": "cancellation",
        "checkin_rules.pdf": "checkin",
    }
    return mapping.get(filename, "general")


def _load_chunk_metadata() -> list[dict]:
    if not os.path.exists(CHUNK_METADATA_PATH):
        return []

    try:
        with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load chunk metadata: %s", exc)
        return []

    if isinstance(data, list):
        return data

    return []

def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS


def _host_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443)
        return True
    except socket.gaierror:
        return False


def _load_docs_from_chunks_csv() -> list:
    if not os.path.exists(CHUNKS_CSV_PATH):
        return []

    df = pd.read_csv(CHUNKS_CSV_PATH)
    if "text" not in df.columns:
        return []

    docs = []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue

        chunk_id = row.get("chunk_id")
        if pd.isna(chunk_id) or not str(chunk_id).strip():
            chunk_id = generate_chunk_id(text)

        docs.append(Document(page_content=text, metadata={"chunk_id": str(chunk_id)}))

    return docs

def _load_retrieval_resources() -> Tuple[FAISS, Optional[BM25Okapi], Optional[list]]:
    global _VECTORSTORE, _BM25, _SPLIT_DOCS, _ALL_CHUNKS, _METADATA_FILTER

    if _VECTORSTORE is None:
        if os.path.exists(VECTOR_PATH):
            if _host_resolves("huggingface.co"):
                try:
                    _VECTORSTORE = FAISS.load_local(
                        VECTOR_PATH,
                        _get_embeddings(),
                        allow_dangerous_deserialization=True,
                    )
                except Exception as exc:
                    logger.warning("Semantic retriever disabled: %s", exc)
                    _VECTORSTORE = None
            else:
                logger.warning("Semantic retriever disabled: huggingface.co is not reachable.")
                _VECTORSTORE = None

    if (_BM25 is None or _SPLIT_DOCS is None) and os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            _BM25, _SPLIT_DOCS = pickle.load(f)

    if _SPLIT_DOCS is None:
        _SPLIT_DOCS = _load_docs_from_chunks_csv()

    if _BM25 is None and _SPLIT_DOCS:
        tokenized_corpus = [tokenize(doc.page_content) for doc in _SPLIT_DOCS]
        _BM25 = BM25Okapi(tokenized_corpus)

    if _SPLIT_DOCS:
        for idx, doc in enumerate(_SPLIT_DOCS):
            doc.metadata["chunk_id"] = idx
        _ALL_CHUNKS = [doc.page_content for doc in _SPLIT_DOCS]
    else:
        _ALL_CHUNKS = []

    if _METADATA_FILTER is None:
        _METADATA_FILTER = MetadataFilter(_load_chunk_metadata())

    return _VECTORSTORE, _BM25, _SPLIT_DOCS


def _source_doc_name(doc) -> str:
    source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
    if source:
        return os.path.basename(str(source))

    filename = doc.metadata.get("file") if hasattr(doc, "metadata") else None
    if filename:
        return os.path.basename(str(filename))

    return "policy"


def _build_sources(docs: list, score_lookup: Optional[dict[str, float]] = None) -> list[dict]:
    sources = []
    for idx, doc in enumerate(docs, start=1):
        text = str(getattr(doc, "page_content", "") or "")
        snippet = text[:120]

        chunk_id = idx
        if hasattr(doc, "metadata"):
            chunk_id = doc.metadata.get("chunk_id", idx)

        score = 0.0
        if score_lookup is not None and hasattr(doc, "metadata"):
            score = float(score_lookup.get(doc.metadata.get("chunk_id"), 0.0))

        sources.append(
            {
                "doc": _source_doc_name(doc),
                "chunk_id": chunk_id,
                "score": score,
                "snippet": snippet,
            }
        )

    return sources


def _docs_to_context_and_sources(
    docs: list,
    score_lookup: Optional[dict[str, float]] = None,
) -> tuple[str, list[dict]]:
    context_string = "\n\n".join(
        str(getattr(doc, "page_content", "") or "") for doc in docs if getattr(doc, "page_content", None)
    )
    sources = _build_sources(docs, score_lookup=score_lookup)
    return context_string, sources

def build_vectorstore():

    os.makedirs(RAG_DIR, exist_ok=True)

    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["doc_type"] = _doc_type_from_source(doc.metadata.get("source", file))
            docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    metadata_list = []
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = idx
        doc.metadata["doc_type"] = _doc_type_from_source(doc.metadata.get("source", ""))

        source = os.path.basename(str(doc.metadata.get("source", "")))
        page = doc.metadata.get("page", 0)
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0

        metadata_list.append(
            {
                "chunk_id": idx,
                "doc_type": str(doc.metadata.get("doc_type", "general")),
                "source": source,
                "page": page,
            }
        )

    chunk_data = [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "text": doc.page_content
        }
        for doc in split_docs
    ]

    pd.DataFrame(chunk_data).to_csv(CHUNKS_CSV_PATH, index=False)
    print(f"Chunks saved to {CHUNKS_CSV_PATH}")

    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)
    print(f"Chunk metadata saved to {CHUNK_METADATA_PATH}")

    # ---- Build FAISS ----
    embeddings = _get_embeddings()

    vectorstore = FAISS.from_documents(split_docs, embeddings,distance_strategy=DistanceStrategy.COSINE)
    vectorstore.save_local(VECTOR_PATH)

    # ---- Build BM25 ----
    texts = [doc.page_content for doc in split_docs]
    tokenized_corpus = [tokenize(text) for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, split_docs), f)

    print("Vectorstore + BM25 index built successfully.")

def retrieve(
    query,
    k=3,
    alpha=0.7,
    rerank: bool = True,
    expand_window: bool = True,
) -> tuple[str, list[dict]]:
    """
    alpha = weight for semantic score
    (1 - alpha) = weight for BM25 score
    """

    global _ALL_CHUNKS
    vectorstore, bm25, split_docs = _load_retrieval_resources()
    metadata_filter = _METADATA_FILTER or MetadataFilter([])

    if not vectorstore and (not bm25 or not split_docs):
        raise FileNotFoundError(
            "No retrieval artifacts available. Ensure rag/vectorstore or all_chunks.csv exists."
        )

    semantic_results = []
    if vectorstore:
        semantic_k = 20
        if split_docs:
            semantic_k = min(semantic_k, len(split_docs))

        try:
            semantic_results = vectorstore.similarity_search_with_score(query, k=semantic_k)
        except Exception as exc:
            logger.warning("Semantic search failed; using lexical fallback: %s", exc)
            semantic_results = []

    doc_type = detect_doc_type(query)
    semantic_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc, _ in semantic_results
        if hasattr(doc, "metadata") and doc.metadata.get("chunk_id") is not None
    ]
    filtered_ids = metadata_filter.filter_results(semantic_chunk_ids, doc_type)
    filtered_id_set = set(filtered_ids)

    if not bm25 or not split_docs:
        if filtered_id_set:
            top_docs = [
                doc
                for doc, _ in sorted(semantic_results, key=lambda item: item[1])
                if doc.metadata.get("chunk_id") in filtered_id_set
            ][:k]
        else:
            top_docs = [doc for doc, _ in sorted(semantic_results, key=lambda item: item[1])[:k]]
        semantic_score_lookup = {
            doc.metadata.get("chunk_id"): float(score)
            for doc, score in semantic_results
            if hasattr(doc, "metadata")
        }
        return _docs_to_context_and_sources(top_docs, score_lookup=semantic_score_lookup)

    distances = np.array([score for _, score in semantic_results])

    if len(distances) > 0:
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        normalized = (max_dist - distances) / (max_dist - min_dist + 1e-8)
    else:
        normalized = distances

    semantic_scores = {
        doc.metadata["chunk_id"]: score
        for (doc, _), score in zip(semantic_results, normalized)
    }

    tokenized_query = tokenize(query)
    bm25_raw_scores = bm25.get_scores(tokenized_query)

    max_score = np.max(bm25_raw_scores)
    if max_score > 0:
        bm25_scores = bm25_raw_scores / max_score
    else:
        bm25_scores = bm25_raw_scores

    bm25_dict = {
        doc.metadata["chunk_id"]: score
        for doc, score in zip(split_docs, bm25_scores)
    }

    candidate_docs = split_docs
    if filtered_id_set:
        candidate_docs = [doc for doc in split_docs if doc.metadata.get("chunk_id") in filtered_id_set]

    if not candidate_docs:
        candidate_docs = split_docs

    combined_scores = {}
    for doc in candidate_docs:
        cid = doc.metadata["chunk_id"]
        semantic = semantic_scores.get(cid, 0)
        lexical = bm25_dict.get(cid, 0)
        combined_scores[cid] = alpha * semantic + (1 - alpha) * lexical

    ranked_docs = sorted(
        candidate_docs,
        key=lambda d: combined_scores[d.metadata["chunk_id"]],
        reverse=True,
    )

    hybrid_top_docs = ranked_docs[:10]

    if not rerank:
        return _docs_to_context_and_sources(hybrid_top_docs[:k], score_lookup=combined_scores)

    raw_chunks = [doc.page_content for doc in hybrid_top_docs]

    try:
        reranker = get_reranker()
        reranked_chunks = reranker.rerank(query, raw_chunks, top_k=3)
    except Exception as exc:
        logger.warning("Reranker failed; returning hybrid results unchanged: %s", exc)
        return _docs_to_context_and_sources(hybrid_top_docs[:k], score_lookup=combined_scores)

    if not reranked_chunks:
        return _docs_to_context_and_sources(hybrid_top_docs[:k], score_lookup=combined_scores)

    text_to_docs = {}
    for doc in hybrid_top_docs:
        text_to_docs.setdefault(doc.page_content, []).append(doc)

    reranked_docs = []
    for chunk_text in reranked_chunks:
        candidates = text_to_docs.get(chunk_text, [])
        if candidates:
            reranked_docs.append(candidates.pop(0))

    final_docs = reranked_docs if reranked_docs else hybrid_top_docs[:k]
    context_string, sources = _docs_to_context_and_sources(final_docs, score_lookup=combined_scores)

    if not expand_window:
        return context_string, sources

    expanded_texts, expanded_sources = expand_with_metadata(
        sources=sources,
        all_chunks=_ALL_CHUNKS or [],
        window=1,
    )

    if not expanded_texts:
        return context_string, sources

    expanded_context = "\n\n".join(expanded_texts)
    return expanded_context, expanded_sources


def retrieve_docs(
    query,
    k=3,
    alpha=0.7,
    rerank: bool = True,
    expand_window: bool = True,
) -> tuple[str, list[dict]]:
    return retrieve(query=query, k=k, alpha=alpha, rerank=rerank, expand_window=expand_window)

if __name__ == "__main__":
    from rag_evaluation.metrics import compute_metrics
    from rag_evaluation.evaluation import get_evaluation_data

    build_vectorstore()

    results = compute_metrics(
        get_evaluation_data(),
        retriever_func=retrieve_docs,
        k=3
    )

    print("\nFinal Metrics:", results)
