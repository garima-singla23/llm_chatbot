import os
import hashlib
import pickle
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RAG_DIR = os.path.join(PROJECT_ROOT, "rag")
VECTOR_PATH = os.path.join(RAG_DIR, "vectorstore")
BM25_PATH = os.path.join(RAG_DIR, "bm25.pkl")
CHUNKS_CSV_PATH = os.path.join(PROJECT_ROOT, "all_chunks.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_VECTORSTORE: Optional[FAISS] = None
_BM25: Optional[BM25Okapi] = None
_SPLIT_DOCS = None


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def generate_chunk_id(text):
    text = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(text.encode()).hexdigest()

def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS

def _load_retrieval_resources() -> Tuple[FAISS, Optional[BM25Okapi], Optional[list]]:
    global _VECTORSTORE, _BM25, _SPLIT_DOCS

    if _VECTORSTORE is None:
        if not os.path.exists(VECTOR_PATH):
            raise FileNotFoundError(
                f"Vectorstore not found at {VECTOR_PATH}. Run build_vectorstore.py first."
            )
        _VECTORSTORE = FAISS.load_local(
            VECTOR_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    if (_BM25 is None or _SPLIT_DOCS is None) and os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            _BM25, _SPLIT_DOCS = pickle.load(f)

    return _VECTORSTORE, _BM25, _SPLIT_DOCS

def build_vectorstore():

    os.makedirs(RAG_DIR, exist_ok=True)

    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    for doc in split_docs:
      doc.metadata["chunk_id"] = generate_chunk_id(doc.page_content)

    chunk_data = [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "text": doc.page_content
        }
        for doc in split_docs
    ]

    pd.DataFrame(chunk_data).to_csv(CHUNKS_CSV_PATH, index=False)
    print(f"Chunks saved to {CHUNKS_CSV_PATH}")

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

def retrieve_docs(query, k=3, alpha=0.7):
    """
    alpha = weight for semantic score
    (1 - alpha) = weight for BM25 score
    """

    vectorstore, bm25, split_docs = _load_retrieval_resources()

    semantic_k = max(k, 1)
    if split_docs:
        semantic_k = min(50, len(split_docs))

    semantic_results = vectorstore.similarity_search_with_score(query, k=semantic_k)

    if not bm25 or not split_docs:
        return [doc for doc, _ in sorted(semantic_results, key=lambda item: item[1])[:k]]

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

    combined_scores = {}

    for doc in split_docs:
        cid = doc.metadata["chunk_id"]
        semantic = semantic_scores.get(cid, 0)
        lexical = bm25_dict.get(cid, 0)
        combined_scores[cid] = alpha * semantic + (1 - alpha) * lexical

    ranked_docs = sorted(
        split_docs,
        key=lambda d: combined_scores[d.metadata["chunk_id"]],
        reverse=True,
    )

    return ranked_docs[:k]

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
