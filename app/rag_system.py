"""
Main RAG system orchestrator.
"""
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from app.logging.logger import get_logger
from app.config.settings import settings
from app.ingestion.loaders import DocumentIngestionManager
from app.embeddings.embedding import EmbeddingManager
from app.storage.chroma_store import ChromaVectorStore
from app.retrieval.retriever import HybridRetriever
from app.reranking.reranker import CrossEncoderReranker
from app.generation.generator import RAGGenerator, HuggingFaceGenerator
from app.evaluation.evaluator import RAGEvaluator
from app.logging.mlflow_tracker import MLFlowTracker
from app.utils.helpers import chunk_text, generate_id
from app.models.schemas import (
    DocumentChunk,
    RAGResponse,
    RetrievalResult,
    RerankedResult,
)

logger = get_logger(__name__)


class RAGSystem:
    """Main RAG system orchestrator."""

    def __init__(self, use_mlflow: bool = True):
        """
        Initialize RAG system.

        Args:
            use_mlflow: Whether to use MLFlow tracking
        """
        logger.info("Initializing RAG System...")

        # Initialize components
        self.ingestion_manager = DocumentIngestionManager()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = ChromaVectorStore()
        self.retriever = HybridRetriever(
            self.embedding_manager,
            self.vector_store,
        )
        self.reranker = CrossEncoderReranker()
        self.generator = RAGGenerator(HuggingFaceGenerator())
        self.evaluator = RAGEvaluator()

        # MLFlow tracking
        self.use_mlflow = use_mlflow
        if use_mlflow:
            self.mlflow_tracker = MLFlowTracker()
        else:
            self.mlflow_tracker = None

        # Document cache
        self._documents_cache: List[Dict[str, Any]] = []
        self._load_documents_to_cache()

        logger.info("RAG System initialized successfully")

    def _load_documents_to_cache(self):
        """Load all documents from vector store to cache for BM25."""
        try:
            docs = self.vector_store.get_all_documents()
            self._documents_cache = docs
            self.retriever.set_documents_for_bm25(docs)
            logger.debug(f"Loaded {len(docs)} documents to cache")
        except Exception as e:
            logger.warning(f"Could not load documents to cache: {str(e)}")

    def ingest_document(
        self,
        source: str,
        source_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> Dict[str, Any]:
        """
        Ingest a document.

        Args:
            source: Source of document (file path, URL, etc.)
            source_type: Type of source (file, url, dataframe, text)
            metadata: Additional metadata
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Ingestion result
        """
        start_time = time.time()

        try:
            # Load document based on source type
            if source_type == "file":
                doc = self.ingestion_manager.ingest_file(
                    source, metadata=metadata
                )
            elif source_type == "url":
                doc = self.ingestion_manager.ingest_url(
                    source, metadata=metadata
                )
            elif source_type == "dataframe" and isinstance(source, pd.DataFrame):
                doc = self.ingestion_manager.ingest_dataframe(
                    source, metadata=metadata
                )
            elif source_type == "text":
                doc = self.ingestion_manager.ingest_text(
                    source, metadata=metadata
                )
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Chunk document
            chunks = chunk_text(
                doc["content"],
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            # Create document chunks
            doc_chunks = []
            for i, chunk_content in enumerate(chunks):
                chunk = DocumentChunk(
                    id=generate_id(f"{doc['id']}_{i}", "chunk"),
                    document_id=doc["id"],
                    content=chunk_content,
                    chunk_index=i,
                    metadata=doc["metadata"],
                )
                doc_chunks.append(chunk)

            # Encode chunks
            chunk_texts = [chunk.content for chunk in doc_chunks]
            embeddings = self.embedding_manager.encode_documents(chunk_texts)

            # Store in vector database
            self.vector_store.add_documents(doc_chunks, embeddings)

            # Update cache and BM25
            for chunk in doc_chunks:
                self._documents_cache.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                })
            self.retriever.set_documents_for_bm25(self._documents_cache)

            processing_time = time.time() - start_time

            result = {
                "document_id": doc["id"],
                "chunks_created": len(doc_chunks),
                "embeddings_created": len(embeddings),
                "processing_time": processing_time,
            }

            logger.info(f"Ingested document {doc['id']} with {len(doc_chunks)} chunks")

            if self.mlflow_tracker:
                self.mlflow_tracker.log_params({
                    "document_id": doc["id"],
                    "source_type": source_type,
                    "chunks_created": len(doc_chunks),
                })

            return result
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise

    def retrieve(
        self,
        query: str,
        k: int = settings.N_RETRIEVE,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of retrieved documents
        """
        try:
            results = self.retriever.retrieve(query, k=k)
            logger.debug(f"Retrieved {len(results)} documents for query")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = settings.N_RERANK,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents.

        Args:
            query: Query text
            documents: List of retrieved documents
            k: Number of results to return

        Returns:
            List of reranked documents
        """
        try:
            results = self.reranker.rerank(query, documents, k=k)
            logger.debug(f"Reranked documents, returned {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise

    def generate_answer(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer based on context.

        Args:
            query: Query text
            context: List of context passages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Generation result
        """
        try:
            result = self.generator.generate(
                query,
                context,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug(f"Generated answer in {result['generation_time']:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def answer_query(
        self,
        query: str,
        k_retrieve: int = settings.N_RETRIEVE,
        k_rerank: int = settings.N_RERANK,
        use_reranking: bool = True,
    ) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve, rerank, and generate.

        Args:
            query: User query
            k_retrieve: Number of documents to retrieve
            k_rerank: Number of documents to rerank
            use_reranking: Whether to apply reranking

        Returns:
            Complete RAG response
        """
        total_start_time = time.time()

        try:
            # Step 1: Retrieve
            retrieval_start = time.time()
            retrieved_docs = self.retrieve(query, k=k_retrieve)
            retrieval_time = time.time() - retrieval_start

            # Step 2: Rerank
            if use_reranking and retrieved_docs:
                reranked_docs = self.rerank(query, retrieved_docs, k=k_rerank)
            else:
                reranked_docs = retrieved_docs[:k_rerank]

            # Step 3: Generate
            context = [doc["content"] for doc in reranked_docs]
            gen_result = self.generate_answer(query, context)

            # Step 4: Format response
            total_time = time.time() - total_start_time

            response = RAGResponse(
                query=query,
                answer=gen_result["answer"],
                retrieved_documents=[
                    RetrievalResult(
                        chunk_id=doc["chunk_id"],
                        content=doc["content"],
                        document_id=doc.get("document_id", "unknown"),
                        score=doc.get("score", 0.0),
                        metadata=doc.get("metadata", {}),
                    )
                    for doc in retrieved_docs
                ],
                reranked_documents=[
                    RerankedResult(
                        chunk_id=doc["chunk_id"],
                        content=doc["content"],
                        document_id=doc.get("document_id", "unknown"),
                        original_score=doc.get("original_score", 0.0),
                        reranked_score=doc.get("reranked_score", 0.0),
                        rank=doc.get("rank", 0),
                    )
                    for doc in reranked_docs
                ],
                generation_time=gen_result["generation_time"],
                retrieval_time=retrieval_time,
                total_time=total_time,
            )

            logger.info(
                f"Completed RAG pipeline in {total_time:.2f}s "
                f"(retrieval: {retrieval_time:.2f}s, generation: {gen_result['generation_time']:.2f}s)"
            )

            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics({
                    "retrieval_time": retrieval_time,
                    "generation_time": gen_result["generation_time"],
                    "total_time": total_time,
                    "retrieved_documents": len(retrieved_docs),
                    "reranked_documents": len(reranked_docs),
                })

            return response
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a generated answer.

        Args:
            query: Query text
            answer: Generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer

        Returns:
            Evaluation metrics
        """
        try:
            result = self.evaluator.evaluate_response(
                query, answer, contexts, ground_truth=ground_truth
            )
            logger.info(f"Evaluated answer, average score: {result.get_average_score():.4f}")
            return result.to_dict()
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise
