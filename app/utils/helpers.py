"""
Utility functions for RAG system.
"""
import hashlib
import time
from typing import List, Dict, Any
from functools import wraps
from app.logging.logger import get_logger

logger = get_logger(__name__)


def generate_id(content: str, prefix: str = "") -> str:
    """
    Generate a unique ID based on content hash.

    Args:
        content: Content to hash
        prefix: Prefix for the ID

    Returns:
        Generated ID
    """
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_value}" if prefix else hash_value


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [chunk for chunk in chunks if chunk.strip()]


def timing_decorator(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.time() - start_time
            logger.debug(f"{func.__name__} took {elapsed_time:.4f} seconds")

    return wrapper


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func,
) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: Items to process
        batch_size: Size of each batch
        process_func: Function to process each batch

    Returns:
        List of processed results
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        results.extend(process_func(batch))
    return results


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    return " ".join(text.split())
