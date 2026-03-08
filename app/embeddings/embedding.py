"""
Embedding module for RAG system using Sentence Transformers.
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.logging.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using Sentence Transformers."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize Sentence Transformer embedding model.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.device = "mps" if settings.LLM_DEVICE == "mps" else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded embedding model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            raise

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=settings.BATCH_SIZE,
                show_progress_bar=False,
            )
            logger.debug(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class EmbeddingManager:
    """Manages embeddings for documents and queries."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize embedding manager.

        Args:
            model_name: Name of the embedding model
        """
        self.embedding_model = SentenceTransformerEmbedding(model_name)
        self.embedding_dim = self.embedding_model.get_dimension()
        logger.info(
            f"EmbeddingManager initialized with dimension: {self.embedding_dim}"
        )

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode document texts.

        Args:
            texts: List of document texts

        Returns:
            Embeddings array (N, D)
        """
        logger.debug(f"Encoding {len(texts)} documents")
        return self.embedding_model.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query.

        Args:
            query: Query text

        Returns:
            Query embedding (1, D)
        """
        embedding = self.embedding_model.encode(query)
        return embedding.reshape(1, -1) if embedding.ndim == 1 else embedding

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get model name."""
        return self.embedding_model.get_model_name()
