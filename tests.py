"""
Integration tests for RAG system.
"""
import pytest
from app.rag_system import RAGSystem
from app.models.schemas import Query, RAGResponse


@pytest.fixture
def rag_system():
    """Initialize RAG system for testing."""
    return RAGSystem(use_mlflow=False)


class TestDocumentIngestion:
    """Test document ingestion."""

    def test_ingest_text(self, rag_system):
        """Test ingesting plain text."""
        text = "Machine learning is a subset of AI."
        result = rag_system.ingest_document(
            source=text,
            source_type="text",
            metadata={"test": "true"},
        )

        assert result["document_id"] is not None
        assert result["chunks_created"] > 0
        assert result["embeddings_created"] > 0
        assert result["processing_time"] > 0

    def test_ingest_multiple_documents(self, rag_system):
        """Test ingesting multiple documents."""
        docs = [
            "Document 1: Information about Python",
            "Document 2: Information about JavaScript",
            "Document 3: Information about Rust",
        ]

        results = []
        for i, doc in enumerate(docs):
            result = rag_system.ingest_document(
                source=doc,
                source_type="text",
                metadata={"index": i},
            )
            results.append(result)

        assert len(results) == 3
        assert all(r["document_id"] for r in results)


class TestRetrieval:
    """Test retrieval functionality."""

    def test_retrieve_documents(self, rag_system):
        """Test document retrieval."""
        # Ingest a document first
        rag_system.ingest_document(
            source="Machine learning enables computers to learn from data.",
            source_type="text",
        )

        # Retrieve documents
        results = rag_system.retrieve("machine learning", k=5)

        assert len(results) > 0
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)


class TestReranking:
    """Test reranking functionality."""

    def test_rerank_documents(self, rag_system):
        """Test document reranking."""
        documents = [
            {
                "chunk_id": "1",
                "content": "Machine learning is a type of AI",
                "score": 0.8,
            },
            {
                "chunk_id": "2",
                "content": "Deep learning is a subset of ML",
                "score": 0.7,
            },
        ]

        results = rag_system.rerank("machine learning", documents, k=2)

        assert len(results) == 2
        assert all("reranked_score" in r for r in results)
        assert all("rank" in r for r in results)


class TestGeneration:
    """Test answer generation."""

    def test_generate_answer(self, rag_system):
        """Test answer generation."""
        context = ["Machine learning is a subset of artificial intelligence"]

        result = rag_system.generate_answer(
            query="What is machine learning?",
            context=context,
        )

        assert "answer" in result
        assert "generation_time" in result
        assert len(result["answer"]) > 0


class TestRAGPipeline:
    """Test complete RAG pipeline."""

    def test_end_to_end_rag(self, rag_system):
        """Test end-to-end RAG pipeline."""
        # Ingest documents
        docs = [
            "Python is a high-level programming language",
            "Machine learning uses algorithms to learn from data",
            "Deep learning is a neural network approach",
        ]

        for doc in docs:
            rag_system.ingest_document(
                source=doc,
                source_type="text",
            )

        # Query the system
        response = rag_system.answer_query(
            query="What is machine learning?",
            k_retrieve=5,
            k_rerank=3,
        )

        assert isinstance(response, RAGResponse)
        assert response.query == "What is machine learning?"
        assert len(response.answer) > 0
        assert len(response.retrieved_documents) > 0
        assert response.total_time > 0


class TestCollectionStats:
    """Test collection statistics."""

    def test_get_stats(self, rag_system):
        """Test getting collection statistics."""
        # Ingest a document
        rag_system.ingest_document(
            source="Test document",
            source_type="text",
        )

        stats = rag_system.get_collection_stats()

        assert "total_chunks" in stats
        assert "collection_name" in stats
        assert stats["total_chunks"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
