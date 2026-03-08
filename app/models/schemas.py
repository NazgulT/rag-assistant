"""
Pydantic data models for RAG system.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Document(BaseModel):
    """Document data model."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source of document (file path, URL, etc.)")
    document_type: str = Field(
        default="text", description="Type of document (pdf, csv, doc, etc.)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "content": "This is a sample document content.",
                "source": "/path/to/document.pdf",
                "document_type": "pdf",
                "metadata": {"author": "John Doe", "pages": 10},
            }
        }


class DocumentChunk(BaseModel):
    """Document chunk after splitting."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Index of chunk within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_001",
                "document_id": "doc_001",
                "content": "Sample chunk content...",
                "chunk_index": 0,
                "metadata": {},
            }
        }


class Embedding(BaseModel):
    """Embedding data model."""

    chunk_id: str = Field(..., description="Chunk ID for this embedding")
    embedding: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Model used to generate embedding")
    embedding_dim: int = Field(..., description="Dimension of embedding")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "embedding": [0.1, 0.2, 0.3],
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384,
            }
        }


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    chunk_id: str = Field(..., description="Retrieved chunk ID")
    content: str = Field(..., description="Chunk content")
    document_id: str = Field(..., description="Source document ID")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "content": "Retrieved content...",
                "document_id": "doc_001",
                "score": 0.95,
                "metadata": {},
            }
        }


class RerankedResult(BaseModel):
    """Reranked retrieval result."""

    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    document_id: str = Field(..., description="Document ID")
    original_score: float = Field(..., description="Original retrieval score")
    reranked_score: float = Field(..., description="Score after reranking")
    rank: int = Field(..., description="Final rank after reranking")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "content": "Retrieved content...",
                "document_id": "doc_001",
                "original_score": 0.85,
                "reranked_score": 0.95,
                "rank": 1,
            }
        }


class Query(BaseModel):
    """Query data model."""

    text: str = Field(..., description="Query text")
    query_id: Optional[str] = Field(None, description="Optional query ID")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is machine learning?",
                "query_id": "query_001",
                "filters": None,
            }
        }


class GenerationRequest(BaseModel):
    """LLM generation request."""

    query: str = Field(..., description="Query text")
    context: List[str] = Field(..., description="Context passages for generation")
    temperature: Optional[float] = Field(None, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "context": ["Machine learning is..."],
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }


class GenerationResponse(BaseModel):
    """LLM generation response."""

    answer: str = Field(..., description="Generated answer")
    context_used: List[str] = Field(..., description="Context passages used")
    generation_time: float = Field(..., description="Time taken to generate")
    model_name: str = Field(..., description="Model used for generation")
    tokens_generated: int = Field(..., description="Number of tokens generated")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of AI...",
                "context_used": ["Machine learning is..."],
                "generation_time": 2.5,
                "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "tokens_generated": 150,
            }
        }


class RAGResponse(BaseModel):
    """Complete RAG response."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    retrieved_documents: List[RetrievalResult] = Field(..., description="Retrieved documents")
    reranked_documents: List[RerankedResult] = Field(..., description="Reranked documents")
    generation_time: float = Field(..., description="Generation time in seconds")
    retrieval_time: float = Field(..., description="Retrieval time in seconds")
    total_time: float = Field(..., description="Total processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI...",
                "retrieved_documents": [],
                "reranked_documents": [],
                "generation_time": 2.5,
                "retrieval_time": 0.5,
                "total_time": 3.0,
            }
        }


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for RAGAS."""

    answer_relevancy: Optional[float] = Field(None, description="Answer relevancy score")
    faithfulness: Optional[float] = Field(None, description="Faithfulness score")
    context_precision: Optional[float] = Field(None, description="Context precision score")
    context_recall: Optional[float] = Field(None, description="Context recall score")
    average_score: Optional[float] = Field(None, description="Average of all metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "answer_relevancy": 0.9,
                "faithfulness": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.88,
                "average_score": 0.86,
            }
        }


class DocumentIngestRequest(BaseModel):
    """Request to ingest documents."""

    file_path: Optional[str] = Field(None, description="File path for local documents")
    url: Optional[str] = Field(None, description="URL for web documents")
    dataframe_data: Optional[Dict[str, Any]] = Field(None, description="DataFrame data")
    document_type: str = Field(..., description="Type of document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "document_type": "pdf",
                "metadata": {"author": "John Doe"},
            }
        }


class DocumentIngestResponse(BaseModel):
    """Response from document ingestion."""

    document_id: str = Field(..., description="Ingested document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    embeddings_created: int = Field(..., description="Number of embeddings created")
    processing_time: float = Field(..., description="Processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_001",
                "chunks_created": 10,
                "embeddings_created": 10,
                "processing_time": 5.2,
            }
        }
