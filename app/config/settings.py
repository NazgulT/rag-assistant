"""
Central configuration management for RAG system.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """Central configuration for the RAG system."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"
    MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Document processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SUPPORTED_FILE_TYPES: list = ["pdf", "csv", "docx", "doc", "txt"]

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    BATCH_SIZE: int = 32

    # Retrieval
    N_RETRIEVE: int = 10
    RETRIEVAL_TYPE: str = "hybrid"  # "hybrid", "semantic", "bm25"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # Reranking
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_RERANK: int = 5
    RERANK_THRESHOLD: float = 0.5

    # LLM Configuration
    LLM_MODEL: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    LLM_DEVICE: str = "cpu"  # "cuda", "cpu", "mps" for Mac
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512
    LLM_TOP_P: float = 0.95

    # Vector Database (ChromaDB)
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    CHROMA_DISTANCE_METRIC: str = "cosine"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "rag_system.log"

    # MLFlow
    MLFLOW_TRACKING_URI: str = "mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "RAG_Experiments"
    MLFLOW_RUN_NAME: str = "default_run"

    # RAGAS Evaluation
    RAGAS_METRICS: list = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Cache settings
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = RAGSettings()
