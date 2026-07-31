"""
Domain enumerations for document processing and indexing states.
"""

from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    DISCOVERED = "discovered"
    HASHING = "hashing"
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STAGING = "staging"
    ACTIVE = "active"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"
    EXCLUDED = "excluded"
    OCR_REQUIRED = "ocr_required"


class IndexStatus(str, Enum):
    """Index job status."""
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types."""
    PDF = ".pdf"
    DOCX = ".docx"
    TXT = ".txt"
    MD = ".md"
    XLSX = ".xlsx"
    PPTX = ".pptx"


class BlockType(str, Enum):
    """Parsed document block types."""
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    HEADING = "heading"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"


class ChunkStrategy(str, Enum):
    """Chunking strategy."""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"


class RetrievalMode(str, Enum):
    """Retrieval search mode."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RerankerFallback(str, Enum):
    """Reranker fallback policy on failure."""
    DENSE_ONLY = "dense_only"
    ERROR = "error"


class OCRStatus(str, Enum):
    """OCR processing status."""
    NOT_NEEDED = "not_needed"
    REQUIRED = "required"
    PARTIAL = "partial"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"