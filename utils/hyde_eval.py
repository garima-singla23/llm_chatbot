from __future__ import annotations

from utils.hyde import hyde_retrieve

SHORT_QUERIES = [
    "baggage?",
    "cancel fee",
    "check in time",
    "refund policy",
    "hand luggage size",
]


def compare_hyde_vs_direct(queries, llm, retriever):
    for q in queries:
        direct_ctx, direct_src = retriever.retrieve(q)
        hyde_ctx, hyde_src = hyde_retrieve(q, llm, retriever)

        direct_score = max((s["score"] for s in direct_src), default=0)
        hyde_score = max((s["score"] for s in hyde_src), default=0)

        print(f"Query: {q}")
        print(f"  Direct score: {direct_score:.3f}")
        print(f"  HyDE score:   {hyde_score:.3f}")
        print(f"  Winner: {'HyDE' if hyde_score > direct_score else 'Direct'}")
