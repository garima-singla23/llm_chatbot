import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

import rag.retriever
from rag.retriever import rerank, retrieve


class TestRerank:
    """Unit tests for rerank function."""

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_returns_at_most_top_k_docs(self, mock_predict):
        """Test that rerank returns at most top_k docs."""
        # Create 5 fake documents
        docs = [
            Document(page_content=f"content {i}", metadata={"airline": "indigo"})
            for i in range(5)
        ]
        
        # Mock RERANKER.predict to return scores
        mock_predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = rerank("test query", docs, top_k=3)
        
        assert len(result) == 3

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_returns_docs_sorted_by_score_descending(self, mock_predict):
        """Test that rerank returns docs sorted by cross-encoder score descending."""
        # Create 3 fake documents
        doc1 = Document(page_content="content 1", metadata={"airline": "indigo"})
        doc2 = Document(page_content="content 2", metadata={"airline": "air_india"})
        doc3 = Document(page_content="content 3", metadata={"airline": "spicejet"})
        docs = [doc1, doc2, doc3]
        
        # Mock RERANKER.predict to return scores [0.2, 0.9, 0.5]
        # So doc2 should be first (0.9), then doc3 (0.5), then doc1 (0.2)
        mock_predict.return_value = [0.2, 0.9, 0.5]
        
        result = rerank("test query", docs, top_k=3)
        
        # Check order: highest score first
        assert len(result) == 3
        assert result[0].page_content == "content 2"  # score 0.9
        assert result[1].page_content == "content 3"  # score 0.5
        assert result[2].page_content == "content 1"  # score 0.2

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_with_top_k_smaller_than_num_docs(self, mock_predict):
        """Test rerank with top_k smaller than number of docs."""
        docs = [
            Document(page_content=f"content {i}", metadata={"airline": "indigo"})
            for i in range(10)
        ]
        
        mock_predict.return_value = list(range(10, 0, -1))  # scores 10 down to 1
        
        result = rerank("test query", docs, top_k=4)
        
        assert len(result) == 4

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_empty_docs(self, mock_predict):
        """Test rerank with empty document list."""
        result = rerank("test query", [], top_k=4)
        
        assert result == []
        mock_predict.assert_not_called()

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_top_k_equals_num_docs(self, mock_predict):
        """Test rerank when top_k equals number of docs."""
        docs = [
            Document(page_content=f"content {i}", metadata={"airline": "indigo"})
            for i in range(3)
        ]
        
        mock_predict.return_value = [0.5, 0.8, 0.3]
        
        result = rerank("test query", docs, top_k=3)
        
        assert len(result) == 3

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_rerank_preserves_document_metadata(self, mock_predict):
        """Test that rerank preserves document metadata after reordering."""
        docs = [
            Document(page_content="content 1", metadata={"airline": "indigo", "topic": "baggage"}),
            Document(page_content="content 2", metadata={"airline": "air_india", "topic": "refund"}),
        ]
        
        mock_predict.return_value = [0.3, 0.9]
        
        result = rerank("test query", docs, top_k=2)
        
        # Second doc should be first (higher score)
        assert result[0].metadata["airline"] == "air_india"
        assert result[0].metadata["topic"] == "refund"
        assert result[1].metadata["airline"] == "indigo"
        assert result[1].metadata["topic"] == "baggage"


class TestRetrieve:
    """Unit tests for retrieve function."""

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_with_airline_filter_excludes_mismatched(self, mock_predict):
        """Test that retrieve with airline_filter excludes docs where metadata['airline'] != filter."""
        # Create docs with different airlines
        docs = [
            Document(page_content="indigo 1", metadata={"airline": "indigo"}),
            Document(page_content="air india 1", metadata={"airline": "air_india"}),
            Document(page_content="indigo 2", metadata={"airline": "indigo"}),
            Document(page_content="spicejet 1", metadata={"airline": "spicejet"}),
        ]
        
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        # Mock RERANKER.predict - we expect only 2 docs (the indigo ones)
        mock_predict.return_value = [0.8, 0.9]
        
        result = retrieve("test query", mock_retriever, airline_filter="indigo", top_k=4)
        
        # Only indigo docs should be in result
        for doc in result:
            assert doc.metadata["airline"].lower() == "indigo"

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_with_no_airline_filter_returns_all(self, mock_predict):
        """Test that retrieve with airline_filter=None returns all docs (no filtering)."""
        # Create docs with different airlines
        docs = [
            Document(page_content="indigo 1", metadata={"airline": "indigo"}),
            Document(page_content="air india 1", metadata={"airline": "air_india"}),
            Document(page_content="spicejet 1", metadata={"airline": "spicejet"}),
        ]
        
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        # Mock RERANKER.predict
        mock_predict.return_value = [0.5, 0.8, 0.7]
        
        result = retrieve("test query", mock_retriever, airline_filter=None, top_k=4)
        
        # All docs should be present (after reranking)
        assert len(result) == 3

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_returns_empty_when_no_docs_pass_airline_filter(self, mock_predict):
        """Test that retrieve returns empty list when no docs pass the airline filter."""
        # Create docs with different airlines (no indigo)
        docs = [
            Document(page_content="air india 1", metadata={"airline": "air_india"}),
            Document(page_content="spicejet 1", metadata={"airline": "spicejet"}),
        ]
        
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        result = retrieve("test query", mock_retriever, airline_filter="indigo", top_k=4)
        
        # Should be empty after filtering
        assert result == []
        # RERANKER.predict should not be called if no docs pass filter
        mock_predict.assert_not_called()

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_airline_filter_case_insensitive(self, mock_predict):
        """Test that airline filter matching is case-insensitive."""
        docs = [
            Document(page_content="indigo 1", metadata={"airline": "IndiGo"}),
            Document(page_content="indigo 2", metadata={"airline": "INDIGO"}),
            Document(page_content="air india", metadata={"airline": "Air India"}),
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        mock_predict.return_value = [0.8, 0.9]
        
        result = retrieve("test query", mock_retriever, airline_filter="indigo", top_k=4)
        
        # Should match both "IndiGo" and "INDIGO" (case-insensitive)
        assert len(result) == 2

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_calls_retriever_get_relevant_documents(self, mock_predict):
        """Test that retrieve calls retriever.invoke() with the query."""
        docs = [Document(page_content="test", metadata={"airline": "indigo"})]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        mock_predict.return_value = [0.5]
        
        query = "baggage allowance"
        retrieve(query, mock_retriever, airline_filter=None, top_k=4)
        
        mock_retriever.invoke.assert_called_once_with(query)

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_respects_top_k_parameter(self, mock_predict):
        """Test that retrieve respects the top_k parameter."""
        docs = [
            Document(page_content=f"content {i}", metadata={"airline": "indigo"})
            for i in range(10)
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = docs
        
        mock_predict.return_value = list(range(10, 0, -1))
        
        result = retrieve("test query", mock_retriever, airline_filter=None, top_k=3)
        
        assert len(result) <= 3

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_filters_then_reranks(self, mock_predict):
        """Test that retrieve applies airline filter before reranking."""
        docs = [
            Document(page_content="indigo 1", metadata={"airline": "indigo"}),
            Document(page_content="air india 1", metadata={"airline": "air_india"}),
            Document(page_content="indigo 2", metadata={"airline": "indigo"}),
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        # Should receive only 2 docs (the indigo ones) for reranking
        mock_predict.return_value = [0.8, 0.9]
        
        result = retrieve("test query", mock_retriever, airline_filter="indigo", top_k=4)
        
        # Check that RERANKER.predict was called with 2 pairs (only indigo docs)
        mock_predict.assert_called_once()
        call_args = mock_predict.call_args[0][0]
        assert len(call_args) == 2  # Only 2 indigo docs

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_with_missing_airline_metadata(self, mock_predict):
        """Test that retrieve handles docs missing 'airline' metadata."""
        docs = [
            Document(page_content="indigo 1", metadata={"airline": "indigo"}),
            Document(page_content="no airline", metadata={}),  # Missing airline
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        
        mock_predict.return_value = [0.9]
        
        result = retrieve("test query", mock_retriever, airline_filter="indigo", top_k=4)
        
        # Only the indigo doc should pass filter (missing airline treated as "")
        assert len(result) == 1
        assert result[0].metadata.get("airline") == "indigo"

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_with_default_top_k(self, mock_predict):
        """Test retrieve uses default top_k=4 when not specified."""
        docs = [
            Document(page_content=f"content {i}", metadata={"airline": "indigo"})
            for i in range(10)
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = docs
        
        mock_predict.return_value = list(range(10, 0, -1))
        
        # Call without top_k parameter
        result = retrieve("test query", mock_retriever)
        
        # Default top_k=4
        assert len(result) <= 4

    @patch.object(rag.retriever.RERANKER, "predict")
    def test_retrieve_empty_retriever_results(self, mock_predict):
        """Test retrieve when retriever returns no documents."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        
        result = retrieve("test query", mock_retriever, airline_filter=None, top_k=4)
        
        assert result == []
        mock_predict.assert_not_called()
