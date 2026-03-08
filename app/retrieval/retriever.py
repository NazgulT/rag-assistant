"""
Retrieval module with hybrid search (BM25 + semantic).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from app.logging.logger import get_logger
from app.config.settings import settings
from app.embeddings.embedding import EmbeddingManager
from app.storage.chroma_store import ChromaVectorStore
from app.models.schemas import RetrievalResult

logger = get_logger(__name__)


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        pass


class SemanticRetriever(Retriever):
    """Semantic retriever using vector similarity."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: ChromaVectorStore,
    ):
        """
        Initialize semantic retriever.

        Args:
            embedding_manager: Embedding manager
            vector_store: Vector store
        """
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        logger.info("SemanticRetriever initialized")

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using semantic similarity.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        try:
            # Encode query
            query_embedding = self.embedding_manager.encode_query(query)

            # Search in vector store
            results = self.vector_store.search(query_embedding[0], k=k)

            # Convert to RetrievalResult format
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "chunk_id": result["chunk_id"],
                        "content": result["content"],
                        "document_id": result["metadata"].get("document_id", "unknown"),
                        "score": result["similarity"],
                        "metadata": result["metadata"],
                    }
                )

            logger.debug(f"Retrieved {len(formatted_results)} documents semantically")
            return formatted_results
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            raise


class BM25Retriever(Retriever):
    """BM25 retriever for keyword-based search."""

    def __init__(self, documents: List[Dict[str, Any]] = None):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of documents (chunks with content)
        """
        self.documents = documents or []
        self.bm25 = None
        self.corpus = []
        self._build_bm25()
        logger.info("BM25Retriever initialized")

    def _build_bm25(self):
        """Build BM25 index from documents."""
        if not self.documents:
            logger.warning("No documents provided for BM25 indexing")
            return

        try:
            # Tokenize documents
            self.corpus = [
                doc["content"].lower().split() for doc in self.documents
            ]
            self.bm25 = BM25Okapi(self.corpus)
            logger.debug(f"Built BM25 index for {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            raise

    def update_documents(self, documents: List[Dict[str, Any]]):
        """
        Update documents for BM25 indexing.

        Args:
            documents: List of documents
        """
        self.documents = documents
        self._build_bm25()

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 scoring.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        if self.bm25 is None:
            logger.warning("BM25 not initialized, returning empty results")
            return []

        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:k]

            formatted_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    formatted_results.append(
                        {
                            "chunk_id": self.documents[idx].get("chunk_id", f"chunk_{idx}"),
                            "content": self.documents[idx]["content"],
                            "document_id": self.documents[idx].get(
                                "document_id", "unknown"
                            ),
                            "score": float(scores[idx]),
                            "metadata": self.documents[idx].get("metadata", {}),
                        }
                    )

            logger.debug(f"Retrieved {len(formatted_results)} documents using BM25")
            return formatted_results
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            raise


class HybridRetriever(Retriever):
    """Hybrid retriever combining semantic and BM25 search."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: ChromaVectorStore,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_manager: Embedding manager
            vector_store: Vector store
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
        """
        self.semantic_retriever = SemanticRetriever(embedding_manager, vector_store)
        self.bm25_retriever = BM25Retriever()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # Normalize weights
        total = semantic_weight + bm25_weight
        self.semantic_weight = semantic_weight / total
        self.bm25_weight = bm25_weight / total

        logger.info(
            f"HybridRetriever initialized with weights: semantic={self.semantic_weight:.2f}, "
            f"bm25={self.bm25_weight:.2f}"
        )

    def set_documents_for_bm25(self, documents: List[Dict[str, Any]]):
        """Set documents for BM25 indexing."""
        self.bm25_retriever.update_documents(documents)

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for result in results:
                result["score"] = 1.0
        else:
            for result in results:
                result["score"] = (result["score"] - min_score) / (max_score - min_score)

        return results

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        try:
            # Get results from both retrievers
            semantic_results = self.semantic_retriever.retrieve(query, k=k)
            bm25_results = self.bm25_retriever.retrieve(query, k=k)

            # Normalize scores
            semantic_results = self._normalize_scores(semantic_results)
            bm25_results = self._normalize_scores(bm25_results)

            # Combine results
            combined = {}
            for result in semantic_results:
                combined[result["chunk_id"]] = {
                    **result,
                    "semantic_score": result["score"],
                    "bm25_score": 0.0,
                }

            for result in bm25_results:
                if result["chunk_id"] in combined:
                    combined[result["chunk_id"]]["bm25_score"] = result["score"]
                else:
                    combined[result["chunk_id"]] = {
                        **result,
                        "semantic_score": 0.0,
                        "bm25_score": result["score"],
                    }

            # Calculate final scores
            for chunk_id in combined:
                combined[chunk_id]["score"] = (
                    combined[chunk_id]["semantic_score"] * self.semantic_weight
                    + combined[chunk_id]["bm25_score"] * self.bm25_weight
                )

            # Sort by final score and return top k
            sorted_results = sorted(
                combined.values(), key=lambda x: x["score"], reverse=True
            )[:k]

            logger.debug(
                f"Retrieved {len(sorted_results)} documents using hybrid search"
            )
            return sorted_results
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
