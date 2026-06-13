import json
import gc
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


BATCH_SIZE = 16


def get_embeddings(use_openai=False):
    if use_openai:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 16},
    )


def build_vector_store(documents, save_path="data/vector_store"):
    """Build FAISS vector store with guaranteed unique IDs and resume support."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    checkpoint_path = Path(save_path) / "checkpoint.json"

    # ---- ASSIGN GLOBALLY UNIQUE IDS (overwrite metadata chunk_id) ----
    # This is the critical fix: never trust metadata['chunk_id'] for uniqueness.
    for i, d in enumerate(documents):
        if isinstance(d.metadata, dict):
            d.metadata["chunk_id"] = f"chunk_{i:06d}"

    total = len(documents)

    # ---- Resume logic ----
    start_idx = 0
    vectorstore = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("embedded_count", 0)
            vectorstore = FAISS.load_local(
                save_path, embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[RESUME] Loaded existing index, resuming from chunk {start_idx}/{total}")
        except Exception as e:
            print(f"[RESUME] Failed to load — starting fresh: {e}")
            start_idx = 0
            vectorstore = None

    remaining = documents[start_idx:]
    if not remaining:
        print(f"[INFO] All {total} chunks already embedded.")
        return vectorstore

    print(f"[INFO] Embedding {len(remaining)} remaining chunks in batches of {BATCH_SIZE}...")

    # Track IDs already added (in case of resume)
    seen_ids = set()
    if vectorstore is not None:
        seen_ids = set(vectorstore.docstore._dict.keys())

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            # Build per-batch unique IDs, skipping any duplicates
            batch_docs = []
            batch_ids = []
            for j, d in enumerate(batch):
                global_idx = start_idx + batch_start + j
                doc_id = f"chunk_{global_idx:06d}"
                if doc_id in seen_ids:
                    continue  # Skip duplicates defensively
                seen_ids.add(doc_id)
                batch_docs.append(d)
                batch_ids.append(doc_id)

            if not batch_docs:
                print(f"[INFO] Batch {batch_num}/{total_batches} skipped (all duplicates)")
                continue

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_docs, embeddings, ids=batch_ids)
            else:
                batch_store = FAISS.from_documents(batch_docs, embeddings, ids=batch_ids)
                vectorstore.merge_from(batch_store)
                del batch_store

            vectorstore.save_local(save_path)
            embedded_so_far = start_idx + batch_start + len(batch)
            with open(checkpoint_path, "w") as f:
                json.dump({"embedded_count": embedded_so_far, "total": total}, f)

            print(f"[INFO] Batch {batch_num}/{total_batches} done — {embedded_so_far}/{total} chunks saved")
            gc.collect()

        except MemoryError:
            print(f"[ERROR] Memory error at batch {batch_num}. Re-run to resume.")
            raise

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"[DONE] All {total} chunks embedded and saved to {save_path}")
    return vectorstore


def load_vector_store(path="data/vector_store"):
    embeddings = get_embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def verify_vector_store(path="data/vector_store"):
    from collections import Counter
    embeddings = get_embeddings()
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    total = len(vs.docstore._dict)
    print(f"\n[VERIFY] Total chunks in store: {total}")

    docs = vs.similarity_search("IndiGo baggage", k=10)
    print("\n[VERIFY] Metadata check on 10 retrieved docs:")
    for doc in docs:
        airline = doc.metadata.get('airline', 'MISSING')
        topic = doc.metadata.get('topic', 'MISSING')
        print(f"  airline={airline} topic={topic} | {doc.page_content[:60]}...")

    all_docs = vs.similarity_search("flight airline", k=100)
    airlines = Counter(d.metadata.get('airline', 'MISSING') for d in all_docs)
    print(f"\n[VERIFY] Airline distribution in sample: {dict(airlines)}")
    return vs


if __name__ == "__main__":
    verify_vector_store()