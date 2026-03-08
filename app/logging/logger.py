"""
Logging configuration for RAG system.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.config.settings import settings


class RAGLogger:
    """Centralized logger for RAG system."""

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._logger = self._setup_logger()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger("RAG_System")
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

        # File handler with rotation
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger


def get_logger(name: str = "RAG_System") -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    rag_logger = RAGLogger()
    base_logger = rag_logger.get_logger()
    return logging.getLogger(f"{base_logger.name}.{name}")
