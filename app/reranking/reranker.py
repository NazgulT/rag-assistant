"""
Reranking module for improving retrieval quality.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
from app.logging.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query."""
        pass


class CrossEncoderReranker(Reranker):
    """Reranker using Cross-Encoder models."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        """
        Initialize Cross-Encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self.device = "mps" if settings.LLM_DEVICE == "mps" else "cpu"

        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"Loaded reranker model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading reranker model {model_name}: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query text
            documents: List of documents to rerank
            k: Number of top documents to return

        Returns:
            Reranked documents
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        try:
            # Prepare document texts
            doc_texts = [doc["content"] for doc in documents]

            # Create pairs of (query, document)
            pairs = [[query, doc_text] for doc_text in doc_texts]

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Add scores to documents
            reranked_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["original_score"] = doc.get("score", 0.0)
                doc_copy["reranked_score"] = float(scores[i])
                reranked_docs.append(doc_copy)

            # Sort by reranked score
            reranked_docs = sorted(
                reranked_docs, key=lambda x: x["reranked_score"], reverse=True
            )

            # Add rank information
            for idx, doc in enumerate(reranked_docs[:k], 1):
                doc["rank"] = idx

            logger.debug(f"Reranked {len(documents)} documents, returning top {k}")
            return reranked_docs[:k]
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            raise


class NoOpReranker(Reranker):
    """No-op reranker that returns documents as-is."""

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return documents without reranking.

        Args:
            query: Query text (unused)
            documents: List of documents
            k: Number of documents to return

        Returns:
            Top k documents with rank information
        """
        reranked_docs = documents[:k].copy()
        for idx, doc in enumerate(reranked_docs, 1):
            doc_copy = doc.copy()
            doc_copy["original_score"] = doc.get("score", 0.0)
            doc_copy["reranked_score"] = doc.get("score", 0.0)
            doc_copy["rank"] = idx
            reranked_docs[idx - 1] = doc_copy

        return reranked_docs
