#!/usr/bin/env python3
"""
Main entry point for RAG System API.
"""
import os
import sys
from pathlib import Path

# Disable tokenizers parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

if __name__ == "__main__":
    import uvicorn
    from app.config.settings import settings
    from app.logging.logger import get_logger

    logger = get_logger(__name__)

    logger.info(f"Starting RAG System API on {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
