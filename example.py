def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

"""Example usage of the RAG system."""
from app.rag_system import RAGSystem
from app.logging.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run RAG system example."""
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = RAGSystem(use_mlflow=True)

    # Example 1: Ingest a text document
    logger.info("\n=== Example 1: Ingesting text ===")
    text_content = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed.
    It focuses on the development of algorithms and statistical models that allow 
    computers to identify patterns in data and make decisions with minimal human intervention.
    """
    result = rag.ingest_document(
        source=text_content,
        source_type="text",
        metadata={"source": "example", "topic": "machine_learning"},
    )
    print(f"Ingested document: {result}")

    # Example 2: Query the RAG system
    logger.info("\n=== Example 2: Querying RAG system ===")
    query = "What is machine learning?"
    response = rag.answer_query(query)
    print(f"\nQuery: {query}")
    print(f"Answer: {response.answer}")
    print(f"Total time: {response.total_time:.2f}s")
    print(f"Retrieved documents: {len(response.retrieved_documents)}")
    print(f"Reranked documents: {len(response.reranked_documents)}")

    # Example 3: Get collection statistics
    logger.info("\n=== Example 3: Collection statistics ===")
    stats = rag.get_collection_stats()
    print(f"Collection stats: {stats}")

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
