"""
FastAPI application for RAG system.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pathlib import Path
import tempfile
from pydantic import BaseModel

from app.logging.logger import get_logger
from app.config.settings import settings
from app.rag_system import RAGSystem
from app.models.schemas import (
    RAGResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    Query,
    EvaluationMetrics,
)

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Modular Retrieval-Augmented Generation System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent.parent / "front-end"), name="static")

# Initialize RAG system
rag_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        logger.info("Starting RAG System...")
        rag_system = RAGSystem(use_mlflow=True)
        logger.info("RAG System started successfully")
    except Exception as e:
        logger.error(f"Error starting RAG system: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG System...")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_system": "initialized" if rag_system else "not initialized",
    }


@app.get("/api/v1/info")
async def get_system_info():
    """Get system information."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        return {
            "embedding_model": rag_system.embedding_manager.get_model_name(),
            "embedding_dimension": rag_system.embedding_manager.get_embedding_dimension(),
            "llm_model": rag_system.generator.llm_generator.model_name,
            "retrieval_type": settings.RETRIEVAL_TYPE,
            "reranker_model": rag_system.reranker.model_name,
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collections/stats")
async def get_collection_stats():
    """Get collection statistics."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        return rag_system.get_collection_stats()
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = None,
):
    """Ingest a file document."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest document
        result = rag_system.ingest_document(
            source=tmp_path,
            source_type="file",
            metadata=eval(metadata) if metadata else None,
        )

        # Clean up
        Path(tmp_path).unlink()

        return DocumentIngestResponse(**result)
    except Exception as e:
        logger.error(f"Error ingesting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/ingest-url")
async def ingest_url(request: DocumentIngestRequest):
    """Ingest a URL document."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        result = rag_system.ingest_document(
            source=request.url,
            source_type="url",
            metadata=request.metadata,
        )
        return DocumentIngestResponse(**result)
    except Exception as e:
        logger.error(f"Error ingesting URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/ingest-text")
async def ingest_text(request: DocumentIngestRequest):
    """Ingest plain text document."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not request.file_path:
        raise HTTPException(status_code=400, detail="Text content is required")

    try:
        result = rag_system.ingest_document(
            source=request.file_path,
            source_type="text",
            metadata=request.metadata,
        )
        return DocumentIngestResponse(**result)
    except Exception as e:
        logger.error(f"Error ingesting text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query")
async def query_rag(
    query: Query,
    k_retrieve: int = settings.N_RETRIEVE,
    k_rerank: int = settings.N_RERANK,
    use_reranking: bool = True,
) -> RAGResponse:
    """
    Execute RAG query.

    Args:
        query: Query object
        k_retrieve: Number of documents to retrieve
        k_rerank: Number of documents to rerank
        use_reranking: Whether to apply reranking

    Returns:
        RAG response with answer and retrieved documents
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        response = rag_system.answer_query(
            query=query.text,
            k_retrieve=k_retrieve,
            k_rerank=k_rerank,
            use_reranking=use_reranking,
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve")
async def retrieve(
    query: Query,
    k: int = settings.N_RETRIEVE,
):
    """
    Retrieve documents for a query.

    Returns:
        List of retrieved documents
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        results = rag_system.retrieve(query.text, k=k)
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rerank")
async def rerank(
    query: Query,
    documents: List[dict],
    k: int = settings.N_RERANK,
):
    """
    Rerank documents for a query.

    Returns:
        Reranked documents
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        results = rag_system.rerank(query.text, documents, k=k)
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate")
async def evaluate(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Evaluate a generated answer.

    Returns:
        Evaluation metrics
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_system.evaluate_answer(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        return EvaluationMetrics(
            answer_relevancy=result.get("answer_relevancy"),
            faithfulness=result.get("faithfulness"),
            context_precision=result.get("context_precision"),
            context_recall=result.get("context_recall"),
            average_score=calculate_average_score(result),
        )
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_average_score(result: dict) -> Optional[float]:
    """Calculate average score from evaluation result."""
    scores = [
        result.get("answer_relevancy"),
        result.get("faithfulness"),
        result.get("context_precision"),
        result.get("context_recall"),
    ]
    scores = [s for s in scores if s is not None]
    if scores:
        return sum(scores) / len(scores)
    return None


@app.get("/frontend")
async def get_frontend():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent.parent.parent / "front-end" / "index.html"
    return FileResponse(frontend_path, media_type="text/html")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "health": "/health",
        "frontend": "/frontend",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )
