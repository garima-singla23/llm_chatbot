from typing import List, Optional


from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

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
