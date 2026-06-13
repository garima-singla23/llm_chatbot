"""
evals/run_ablation.py
Retrieval ablation: Dense-only (FAISS) vs Hybrid+Rerank on full 25-question golden set.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embedder import load_vector_store
from rag.retriever import build_hybrid_retriever, retrieve, rerank


# ---------- Helpers ----------

def chunk_contains_keywords(chunks, keywords, threshold=0.5):
    if not keywords:
        return False
    combined = " ".join(
        (c.page_content if hasattr(c, "page_content") else str(c)).lower()
        for c in chunks
    )
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return (hits / len(keywords)) >= threshold


def dense_only_retrieve(vector_store, query, k=4, airline=None):
    """Pure FAISS dense retrieval, NO reranker, NO BM25."""
    if airline:
        try:
            return vector_store.similarity_search(query, k=k, filter={"airline": airline})
        except Exception:
            return vector_store.similarity_search(query, k=k)
    return vector_store.similarity_search(query, k=k)


def evaluate(name, retrieve_fn, golden_set):
    print(f"\n>>> {name}")
    passed = 0
    results = []
    for i, case in enumerate(golden_set, 1):
        q = case["question"]
        kws = case.get("expected_keywords", [])
        airline = case.get("airline")
        try:
            chunks = retrieve_fn(q, airline)
        except Exception as e:
            chunks = []
            print(f"  [ERR] Q{i}: {e}")
        hit = chunk_contains_keywords(chunks, kws)
        passed += int(hit)
        results.append((i, "PASS" if hit else "FAIL", q[:55]))
        print(f"  [{'PASS' if hit else 'FAIL'}] Q{i}: {q[:55]}")
    return passed, results


def main():
    # 1. Load golden set
    golden_path = Path(__file__).parent / "golden_set.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    total = len(golden_set)
    print(f"Loaded {total} golden questions")

    # 2. Load vector store
    print("\nLoading vector store...")
    vector_store = load_vector_store()

    # 3. Extract chunks for BM25
    print("Extracting chunks from vector store for BM25...")
    chunks_all = list(vector_store.docstore._dict.values())
    print(f"  -> {len(chunks_all)} chunks loaded")

    if len(chunks_all) < 5000:
        print(f"\n[WARNING] Only {len(chunks_all)} chunks in vector store.")
        print("Expected ~15,000. Run: python build_knowledge_base.py\n")

    # 4. Build hybrid retriever
    print("Building hybrid retriever...")
    hybrid_retriever = build_hybrid_retriever(vector_store, chunks_all, k=8)

    # 5. Define retrieval functions
    def dense_fn(q, airline):
        return dense_only_retrieve(vector_store, q, k=4, airline=airline)

    def hybrid_fn(q, airline):
        # CORRECT signature: retrieve(query, retriever, airline_filter, top_k)
        return retrieve(q, hybrid_retriever, airline_filter=airline, top_k=4)

    # 6. Run both
    dense_passed, dense_res = evaluate("Dense only (FAISS)", dense_fn, golden_set)
    hybrid_passed, hybrid_res = evaluate("Hybrid + Rerank", hybrid_fn, golden_set)

    # 7. Report
    print("\n" + "=" * 90)
    print(f"{'#':<4}{'Question':<60}{'Dense':<12}{'Hybrid':<10}")
    print("-" * 90)
    for (i, d, q), (_, h, _) in zip(dense_res, hybrid_res):
        print(f"{i:<4}{q:<60}{d:<12}{h:<10}")
    print("=" * 90)
    print(f"\nDense-only precision:    {dense_passed}/{total} = {dense_passed/total*100:.1f}%")
    print(f"Hybrid+rerank precision: {hybrid_passed}/{total} = {hybrid_passed/total*100:.1f}%")
    print(f"Improvement:             {hybrid_passed - dense_passed:+d} questions ({(hybrid_passed-dense_passed)/total*100:+.1f} pp)\n")


if __name__ == "__main__":
    main()