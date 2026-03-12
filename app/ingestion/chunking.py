"""
Advanced text chunking module for RAG system.
Implements recursive chunking strategy for better semantic preservation.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from app.logging.logger import get_logger
from app.models.schemas import DocumentChunk
from app.utils.helpers import generate_id

logger = get_logger(__name__)


class RecursiveChunker:
    """
    Recursive text chunker that splits text hierarchically to preserve semantic boundaries.
    
    Uses separators in order of preference to split text:
    1. Paragraphs (double newlines)
    2. Sentences (. ! ?)
    3. Words (spaces)
    4. Characters (fallback)
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to use in priority order
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraph boundary
            "\n",    # Line boundary
            ".",     # Sentence boundary
            " ",     # Word boundary
            "",      # Character fallback
        ]

        logger.info(
            f"Initialized RecursiveChunker with size={chunk_size}, overlap={chunk_overlap}"
        )


    def chunk_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk document content and return DocumentChunk objects.

        Args:
            content: Document content
            document_id: ID of parent document
            metadata: Optional document metadata

        Returns:
            List of DocumentChunk objects
        """

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        
        chunks = text_splitter.split_text(content)
        
        document_chunks = []
        existing_ids = set()
        
        for idx, chunk_content in enumerate(chunks):
            # Generate unique ID using document_id + index + content hash
            unique_content = f"{document_id}_{idx}_{chunk_content[:50]}"  # Include first 50 chars for uniqueness
            chunk_id = generate_id(unique_content, "chunk")
            
            # Ensure ID is unique within this batch
            counter = 0
            original_id = chunk_id
            while chunk_id in existing_ids:
                chunk_id = f"{original_id}_{counter}"
                counter += 1
            
            existing_ids.add(chunk_id)
            
            document_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=idx,
                    metadata=metadata or {},
                )
            )
        
        logger.info(
            f"Created {len(document_chunks)} chunks from document {document_id}"
        )
        return document_chunks


