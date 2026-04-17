from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME: Optional[str] = None
_MODEL_LOAD_FAILED = False


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        global _MODEL, _MODEL_NAME, _MODEL_LOAD_FAILED

        self.model = None
        self.model_name = model_name

        if _MODEL is not None and _MODEL_NAME == model_name:
            self.model = _MODEL
            return

        if _MODEL_LOAD_FAILED:
            logger.warning("[Reranker] Previous model load failed; using passthrough mode.")
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            _MODEL_LOAD_FAILED = True
            logger.warning(
                "[Reranker] sentence-transformers is not installed. "
                "Reranking is disabled and original chunks will be used."
            )
            return

        try:
            print("[Reranker] Loading cross-encoder model...")
            _MODEL = CrossEncoder(model_name)
            _MODEL_NAME = model_name
            self.model = _MODEL
        except Exception as exc:
            _MODEL_LOAD_FAILED = True
            logger.warning(
                "[Reranker] Failed to load model '%s'. Reranking disabled: %s",
                model_name,
                exc,
            )

    def rerank(self, query: str, chunks: list[str], top_k: int = 3) -> list[str]:
        if not chunks or len(chunks) <= top_k:
            return chunks

        if self.model is None:
            return chunks

        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, _ in ranked[:top_k]]

        print(
            f"[Reranker] {len(chunks)} → {top_k} chunks, "
            f"top score: {max(scores):.3f}"
        )
        return top_chunks

    def rerank_with_scores(self, query: str, chunks: list[str]) -> list[tuple[str, float]]:
        if not chunks:
            return []

        if self.model is None:
            return [(chunk, 0.0) for chunk in chunks]

        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked]


_reranker = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
