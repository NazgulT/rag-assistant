"""
Document ingestion module for RAG system.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
from app.logging.logger import get_logger
from app.utils.helpers import generate_id, normalize_text

logger = get_logger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str) -> str:
        """Load document from source."""
        pass

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """Check if loader supports file type."""
        pass


class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["pdf"]

    def load(self, source: str) -> str:
        """Load PDF document."""
        try:
            import PyPDF2

            text = []
            with open(source, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            logger.info(f"Successfully loaded PDF: {source}")
            return "\n".join(text)
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Error loading PDF {source}: {str(e)}")
            raise


class CSVLoader(DocumentLoader):
    """Loader for CSV documents."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["csv"]

    def load(self, source: str) -> str:
        """Load CSV document."""
        try:
            df = pd.read_csv(source)
            # Convert DataFrame to text format
            text = df.to_string()
            logger.info(f"Successfully loaded CSV: {source}")
            return text
        except Exception as e:
            logger.error(f"Error loading CSV {source}: {str(e)}")
            raise


class DocxLoader(DocumentLoader):
    """Loader for DOCX documents."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["docx", "doc"]

    def load(self, source: str) -> str:
        """Load DOCX document."""
        try:
            from docx import Document

            doc = Document(source)
            text = [para.text for para in doc.paragraphs]
            logger.info(f"Successfully loaded DOCX: {source}")
            return "\n".join(text)
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            raise
        except Exception as e:
            logger.error(f"Error loading DOCX {source}: {str(e)}")
            raise


class TextLoader(DocumentLoader):
    """Loader for plain text documents."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["txt", "text"]

    def load(self, source: str) -> str:
        """Load text document."""
        try:
            with open(source, "r", encoding="utf-8") as file:
                text = file.read()
            logger.info(f"Successfully loaded TXT: {source}")
            return text
        except Exception as e:
            logger.error(f"Error loading TXT {source}: {str(e)}")
            raise


class URLLoader(DocumentLoader):
    """Loader for web documents."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["url", "web", "html"]

    def load(self, source: str) -> str:
        """Load web document."""
        try:
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(source, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            logger.info(f"Successfully loaded URL: {source}")
            return normalize_text(text)
        except ImportError:
            logger.error("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            raise
        except Exception as e:
            logger.error(f"Error loading URL {source}: {str(e)}")
            raise


class DataFrameLoader(DocumentLoader):
    """Loader for pandas DataFrames."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in ["dataframe", "df"]

    def load(self, source: str) -> str:
        """Load DataFrame. Source should be the DataFrame itself."""
        try:
            if isinstance(source, pd.DataFrame):
                text = source.to_string()
                logger.info("Successfully loaded DataFrame")
                return text
            else:
                raise ValueError("Source must be a pandas DataFrame")
        except Exception as e:
            logger.error(f"Error loading DataFrame: {str(e)}")
            raise


class DocumentIngestionManager:
    """Manages document ingestion with multiple loaders."""

    def __init__(self):
        """Initialize document ingestion manager."""
        self.loaders: Dict[str, DocumentLoader] = {
            "pdf": PDFLoader(),
            "csv": CSVLoader(),
            "docx": DocxLoader(),
            "doc": DocxLoader(),
            "txt": TextLoader(),
            "text": TextLoader(),
            "url": URLLoader(),
            "web": URLLoader(),
            "html": URLLoader(),
            "dataframe": DataFrameLoader(),
            "df": DataFrameLoader(),
        }
        logger.info("DocumentIngestionManager initialized")

    def ingest_file(
        self,
        file_path: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a file document.

        Args:
            file_path: Path to the file
            document_type: Type of document (auto-detected if None)
            metadata: Additional metadata

        Returns:
            Document dictionary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect document type
        if document_type is None:
            document_type = file_path.suffix.lower().lstrip(".")

        loader = self.loaders.get(document_type.lower())
        if not loader:
            raise ValueError(f"Unsupported document type: {document_type}")

        content = loader.load(str(file_path))
        doc_id = generate_id(content, "doc")

        return {
            "id": doc_id,
            "content": content,
            "source": str(file_path),
            "document_type": document_type,
            "metadata": metadata or {},
        }

    def ingest_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a web document.

        Args:
            url: URL of the web document
            metadata: Additional metadata

        Returns:
            Document dictionary
        """
        loader = self.loaders.get("url")
        content = loader.load(url)
        doc_id = generate_id(content, "doc")

        return {
            "id": doc_id,
            "content": content,
            "source": url,
            "document_type": "url",
            "metadata": metadata or {},
        }

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a pandas DataFrame.

        Args:
            df: DataFrame to ingest
            metadata: Additional metadata

        Returns:
            Document dictionary
        """
        loader = self.loaders.get("dataframe")
        content = loader.load(df)
        doc_id = generate_id(content, "doc")

        return {
            "id": doc_id,
            "content": content,
            "source": "dataframe",
            "document_type": "dataframe",
            "metadata": metadata or {},
        }

    def ingest_text(
        self,
        text: str,
        source: str = "text_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest plain text.

        Args:
            text: Text content
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Document dictionary
        """
        doc_id = generate_id(text, "doc")
        return {
            "id": doc_id,
            "content": text,
            "source": source,
            "document_type": "text",
            "metadata": metadata or {},
        }
