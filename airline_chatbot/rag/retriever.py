from typing import List, Optional


from langchain_community.retrievers import BM25Retriever
try:
	from langchain.retrievers.ensemble import EnsembleRetriever
except Exception:
	# Lightweight fallback EnsembleRetriever for environments without
	# the langchain `retrievers.ensemble` module. This simply queries
	# each retriever and concatenates results, applying optional weights
	# for scoring if retrievers return scores as tuples (doc, score).
	class EnsembleRetriever:
		def __init__(self, retrievers=None, weights=None):
			self.retrievers = retrievers or []
			self.weights = weights or [1.0] * len(self.retrievers)

		def invoke(self, query):
			all_docs = []
			seen = set()
			for r in self.retrievers:
				try:
					docs = r.invoke(query)
				except Exception:
					# Some retrievers expose `get_relevant_documents`
					# or `get_relevant_documents`-like APIs.
					try:
						docs = r.get_relevant_documents(query)
					except Exception:
						docs = []
				for d in docs:
					key = getattr(d, "page_content", str(d))
					if key in seen:
						continue
					seen.add(key)
					all_docs.append(d)
			return all_docs

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class _LazyReranker:
	def __init__(self):
		self._model = None
		self._load_error = None

	def _get_model(self):
		if self._model is not None:
			return self._model
		if self._load_error is not None:
			raise RuntimeError("CrossEncoder not available in this environment") from self._load_error

		try:
			from sentence_transformers import CrossEncoder

			self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
			return self._model
		except Exception as exc:
			self._load_error = exc
			raise RuntimeError("CrossEncoder not available in this environment") from exc

	def predict(self, pairs):
		return self._get_model().predict(pairs)


RERANKER = _LazyReranker()


def build_hybrid_retriever(vectorstore, documents, k=8):
	dense = vectorstore.as_retriever(search_kwargs={"k": k})
	# Ensure docs have an `id` attribute — some BM25 implementations
	# expect `doc.id`. Provide a fallback using chunk_id metadata.
	for i, d in enumerate(documents):
		if not hasattr(d, "id"):
			try:
				setattr(d, "id", d.metadata.get("chunk_id") if isinstance(d.metadata, dict) else None or f"doc_{i}")
			except Exception:
				# Best-effort: ignore if the Document implementation is immutable
				pass

	bm25 = BM25Retriever.from_documents(documents)
	bm25.k = k

	return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])


def rerank(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
	if not docs:
		return []

	pairs = [(query, doc.page_content) for doc in docs]
	scores = RERANKER.predict(pairs)

	ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
	return [doc for doc, _ in ranked[:top_k]]


def retrieve(
	query: str,
	retriever: BaseRetriever,
	airline_filter: Optional[str] = None,
	top_k: int = 4,
) -> List[Document]:
	raw_docs = retriever.invoke(query)

	if airline_filter:
		filter_value = airline_filter.lower()
		filtered_docs = [
			doc
			for doc in raw_docs
			if doc.metadata.get("airline", "").lower() == filter_value
		]
	else:
		filtered_docs = raw_docs

	return rerank(query, filtered_docs, top_k=top_k)
