"""
Vector storage module using ChromaDB.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.logging.logger import get_logger
from app.config.settings import settings
from app.models.schemas import DocumentChunk, Embedding

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete document from store."""
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from store."""
        pass


class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""

    def __init__(
        self,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        persist_directory: str = str(settings.CHROMA_DB_PATH),
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        try:
            '''
            # Initialize ChromaDB client with persistence
            chroma_settings = ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
            '''
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": settings.CHROMA_DISTANCE_METRIC},
            )

            logger.info(
                f"ChromaVectorStore initialized with collection: {collection_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document chunks with embeddings to the store.

        Args:
            chunks: List of document chunks
            embeddings: Embedding vectors (N, D)
        """
        try:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
                for chunk in chunks
            ]

            # Convert embeddings to lists for ChromaDB
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of search results with metadata
        """
        try:
            # Convert to list for ChromaDB
            query_embedding_list = (
                query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else query_embedding
            )

            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
            )

            # Format results
            formatted_results = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "chunk_id": chunk_id,
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "similarity": 1
                            - results["distances"][0][i],  # Convert distance to similarity
                        }
                    )

            logger.debug(f"Found {len(formatted_results)} similar documents")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks of a document.

        Args:
            document_id: ID of the document to delete
        """
        try:
            # Get all chunks of the document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}},
            )

            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks of document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the collection."""
        try:
            results = self.collection.get()
            formatted_results = []

            if results and results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    formatted_results.append(
                        {
                            "chunk_id": chunk_id,
                            "content": results["documents"][i],
                            "metadata": results["metadatas"][i],
                        }
                    )

            logger.info(f"Retrieved {len(formatted_results)} documents from ChromaDB")
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving documents from ChromaDB: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": settings.CHROMA_DISTANCE_METRIC},
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
