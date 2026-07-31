"""
Custom exception classes for the Document RAG system.

Provides structured exception hierarchy for different error categories
with context information for debugging and user feedback.
"""

from pathlib import Path
from typing import Optional
from uuid import UUID


class RAGException(Exception):
    """Base exception for all RAG system errors."""
    
    def __init__(self, message: str, *, cause: Optional[Exception] = None, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context = context or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


# Configuration Errors
class ConfigurationError(RAGException):
    """Raised when configuration loading or validation fails."""
    pass


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


class PathTraversalError(ConfigurationError):
    """Raised when a path attempts to escape allowed root."""
    
    def __init__(self, path: Path, root: Path, path_name: str = "path"):
        message = f"Path traversal detected: {path_name} ({path}) is outside allowed root ({root})"
        super().__init__(message, context={"path": str(path), "root": str(root), "path_name": path_name})
        self.path = path
        self.root = root
        self.path_name = path_name


# File Processing Errors
class FileProcessingError(RAGException):
    """Base exception for file processing errors."""
    pass


class FileNotFoundError(FileProcessingError):
    """Raised when a file to process is not found."""
    pass


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when file type is not supported."""
    
    def __init__(self, file_path: Path, supported_types: list[str]):
        message = f"Unsupported file type: {file_path.suffix}. Supported: {supported_types}"
        super().__init__(message, context={"file_path": str(file_path), "supported_types": supported_types})
        self.file_path = file_path
        self.supported_types = supported_types


class FileTooLargeError(FileProcessingError):
    """Raised when file exceeds maximum size limit."""
    
    def __init__(self, file_path: Path, size_mb: float, max_mb: int):
        message = f"File too large: {file_path} ({size_mb:.1f} MB > {max_mb} MB limit)"
        super().__init__(message, context={"file_path": str(file_path), "size_mb": size_mb, "max_mb": max_mb})
        self.file_path = file_path
        self.size_mb = size_mb
        self.max_mb = max_mb


class FileEncryptedError(FileProcessingError):
    """Raised when file is encrypted/password protected."""
    
    def __init__(self, file_path: Path):
        message = f"File is encrypted or password protected: {file_path}"
        super().__init__(message, context={"file_path": str(file_path)})
        self.file_path = file_path


class FileCorruptedError(FileProcessingError):
    """Raised when file is corrupted or unreadable."""
    
    def __init__(self, file_path: Path, cause: Optional[Exception] = None):
        message = f"File appears corrupted or unreadable: {file_path}"
        super().__init__(message, cause=cause, context={"file_path": str(file_path)})
        self.file_path = file_path


class FileStabilityError(FileProcessingError):
    """Raised when file changes during processing (size/mtime mismatch)."""
    
    def __init__(self, file_path: Path):
        message = f"File changed during processing: {file_path}"
        super().__init__(message, context={"file_path": str(file_path)})
        self.file_path = file_path


# Parser Errors
class ParsingError(RAGException):
    """Base exception for document parsing errors."""
    pass


class PDFParseError(ParsingError):
    """Raised when PDF parsing fails."""
    
    def __init__(self, file_path: Path, cause: Optional[Exception] = None):
        message = f"Failed to parse PDF: {file_path}"
        super().__init__(message, cause=cause, context={"file_path": str(file_path)})
        self.file_path = file_path


class DOCXParseError(ParsingError):
    """Raised when DOCX parsing fails."""
    
    def __init__(self, file_path: Path, cause: Optional[Exception] = None):
        message = f"Failed to parse DOCX: {file_path}"
        super().__init__(message, cause=cause, context={"file_path": str(file_path)})
        self.file_path = file_path


class TextParseError(ParsingError):
    """Raised when text file parsing fails (encoding, etc.)."""
    
    def __init__(self, file_path: Path, cause: Optional[Exception] = None):
        message = f"Failed to parse text file: {file_path}"
        super().__init__(message, cause=cause, context={"file_path": str(file_path)})
        self.file_path = file_path


class MarkdownParseError(ParsingError):
    """Raised when Markdown parsing fails."""
    
    def __init__(self, file_path: Path, cause: Optional[Exception] = None):
        message = f"Failed to parse Markdown: {file_path}"
        super().__init__(message, cause=cause, context={"file_path": str(file_path)})
        self.file_path = file_path


class ZipBombError(ParsingError):
    """Raised when ZIP-based document (DOCX/XLSX/PPTX) appears to be a zip bomb."""
    
    def __init__(self, file_path: Path, uncompressed_size: Optional[int] = None, file_count: Optional[int] = None):
        message = f"Potential zip bomb detected: {file_path}"
        context = {"file_path": str(file_path)}
        if uncompressed_size:
            context["uncompressed_size_bytes"] = uncompressed_size
        if file_count:
            context["file_count"] = file_count
        super().__init__(message, context=context)
        self.file_path = file_path
        self.uncompressed_size = uncompressed_size
        self.file_count = file_count


class XXEAttackError(ParsingError):
    """Raised when XML External Entity attack is detected."""
    
    def __init__(self, file_path: Path):
        message = f"Potential XXE attack detected in XML document: {file_path}"
        super().__init__(message, context={"file_path": str(file_path)})
        self.file_path = file_path


class OCRProcessingError(ParsingError):
    """Raised when OCR processing fails."""
    
    def __init__(self, file_path: Path, page_numbers: Optional[list[int]] = None, cause: Optional[Exception] = None):
        message = f"OCR processing failed: {file_path}"
        context = {"file_path": str(file_path)}
        if page_numbers:
            context["failed_pages"] = page_numbers
        super().__init__(message, cause=cause, context=context)
        self.file_path = file_path
        self.page_numbers = page_numbers


class OCRRequiredError(ParsingError):
    """Raised when document requires OCR but it's disabled."""
    
    def __init__(self, file_path: Path, page_numbers: Optional[list[int]] = None):
        message = f"Document requires OCR but it's disabled: {file_path}"
        context = {"file_path": str(file_path)}
        if page_numbers:
            context["pages_requiring_ocr"] = page_numbers
        super().__init__(message, context=context)
        self.file_path = file_path
        self.page_numbers = page_numbers


# Chunking Errors
class ChunkingError(RAGException):
    """Base exception for chunking errors."""
    pass


class EmptyChunkError(ChunkingError):
    """Raised when chunking produces empty chunks."""
    pass


class ChunkSizeError(ChunkingError):
    """Raised when chunk exceeds maximum token limit."""
    
    def __init__(self, token_count: int, max_tokens: int):
        message = f"Chunk exceeds maximum token limit: {token_count} > {max_tokens}"
        super().__init__(message, context={"token_count": token_count, "max_tokens": max_tokens})
        self.token_count = token_count
        self.max_tokens = max_tokens


# Embedding Errors
class EmbeddingError(RAGException):
    """Base exception for embedding generation errors."""
    pass


class EmbeddingModelError(EmbeddingError):
    """Raised when embedding model fails to load or process."""
    
    def __init__(self, model_id: str, cause: Optional[Exception] = None):
        message = f"Embedding model error: {model_id}"
        super().__init__(message, cause=cause, context={"model_id": model_id})
        self.model_id = model_id


class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding dimension doesn't match expected size."""
    
    def __init__(self, expected: int, actual: int, model_id: str):
        message = f"Embedding dimension mismatch for {model_id}: expected {expected}, got {actual}"
        super().__init__(message, context={"expected": expected, "actual": actual, "model_id": model_id})
        self.expected = expected
        self.actual = actual
        self.model_id = model_id


class EmptyTextEmbeddingError(EmbeddingError):
    """Raised when attempting to embed empty or whitespace-only text."""
    pass


# Vector Store Errors
class VectorStoreError(RAGException):
    """Base exception for vector store (Qdrant) errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""
    
    def __init__(self, url: str, cause: Optional[Exception] = None):
        message = f"Failed to connect to vector store: {url}"
        super().__init__(message, cause=cause, context={"url": url})
        self.url = url


class VectorStoreCollectionError(VectorStoreError):
    """Raised when collection operations fail."""
    
    def __init__(self, collection: str, operation: str, cause: Optional[Exception] = None):
        message = f"Vector store collection '{collection}' {operation} failed"
        super().__init__(message, cause=cause, context={"collection": collection, "operation": operation})
        self.collection = collection
        self.operation = operation


class VectorStoreIndexError(VectorStoreError):
    """Raised when indexing/upserting vectors fails."""
    
    def __init__(self, count: int, cause: Optional[Exception] = None):
        message = f"Failed to index {count} vectors"
        super().__init__(message, cause=cause, context={"vector_count": count})
        self.count = count


class VectorStoreSearchError(VectorStoreError):
    """Raised when vector search fails."""
    
    def __init__(self, query_id: UUID, cause: Optional[Exception] = None):
        message = f"Vector search failed for query {query_id}"
        super().__init__(message, cause=cause, context={"query_id": str(query_id)})
        self.query_id = query_id


# Retrieval Errors
class RetrievalError(RAGException):
    """Base exception for retrieval errors."""
    pass


class NoResultsError(RetrievalError):
    """Raised when search returns no results."""
    
    def __init__(self, query: str):
        message = f"No results found for query: {query}"
        super().__init__(message, context={"query": query})
        self.query = query


class LowScoreError(RetrievalError):
    """Raised when all search results are below minimum score threshold."""
    
    def __init__(self, max_score: float, threshold: float):
        message = f"All search results below minimum score: max={max_score:.4f} < threshold={threshold:.4f}"
        super().__init__(message, context={"max_score": max_score, "threshold": threshold})
        self.max_score = max_score
        self.threshold = threshold


# Reranker Errors
class RerankerError(RAGException):
    """Base exception for reranker errors."""
    pass


class RerankerModelError(RerankerError):
    """Raised when reranker model fails to load or process."""
    
    def __init__(self, model_id: str, cause: Optional[Exception] = None):
        message = f"Reranker model error: {model_id}"
        super().__init__(message, cause=cause, context={"model_id": model_id})
        self.model_id = model_id


# LLM Errors
class LLMError(RAGException):
    """Base exception for LLM errors."""
    pass


class LLMModelError(LLMError):
    """Raised when LLM model fails to load or process."""
    
    def __init__(self, model_id: str, cause: Optional[Exception] = None):
        message = f"LLM model error: {model_id}"
        super().__init__(message, cause=cause, context={"model_id": model_id})
        self.model_id = model_id


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    
    def __init__(self, timeout_seconds: float, cause: Optional[Exception] = None):
        message = f"LLM request timed out after {timeout_seconds}s"
        super().__init__(message, cause=cause, context={"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""
    
    def __init__(self, retry_after_seconds: Optional[float] = None, cause: Optional[Exception] = None):
        message = "LLM API rate limit exceeded"
        context = {}
        if retry_after_seconds:
            context["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, cause=cause, context=context)
        self.retry_after_seconds = retry_after_seconds


class LLMContentFilterError(LLMError):
    """Raised when LLM content filter triggers."""
    
    def __init__(self, cause: Optional[Exception] = None):
        message = "LLM content filter triggered"
        super().__init__(message, cause=cause)


class PromptInjectionDetectedError(LLMError):
    """Raised when prompt injection attempt is detected in user query."""
    
    def __init__(self, query: str, patterns_matched: list[str]):
        # Don't log full query for security
        message = f"Prompt injection attempt detected (patterns: {patterns_matched})"
        super().__init__(message, context={"patterns_matched": patterns_matched, "query_length": len(query)})
        self.patterns_matched = patterns_matched


class InvalidCitationError(LLMError):
    """Raised when LLM generates invalid citation references."""
    
    def __init__(self, invalid_citations: list[int], available_citations: list[int]):
        message = f"LLM generated invalid citations: {invalid_citations}. Available: {available_citations}"
        super().__init__(message, context={"invalid_citations": invalid_citations, "available_citations": available_citations})
        self.invalid_citations = invalid_citations
        self.available_citations = available_citations


# Generation Errors
class GenerationError(RAGException):
    """Base exception for answer generation errors."""
    pass


class InsufficientContextError(GenerationError):
    """Raised when retrieved context is insufficient to answer."""
    
    def __init__(self, question: str, num_sources: int):
        message = f"Insufficient context to answer: {question} (found {num_sources} sources)"
        super().__init__(message, context={"question": question, "num_sources": num_sources})
        self.question = question
        self.num_sources = num_sources


# Database Errors
class DatabaseError(RAGException):
    """Base exception for database (SQLite) errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, db_path: Path, cause: Optional[Exception] = None):
        message = f"Failed to connect to database: {db_path}"
        super().__init__(message, cause=cause, context={"db_path": str(db_path)})
        self.db_path = db_path


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails."""
    
    def __init__(self, version: int, cause: Optional[Exception] = None):
        message = f"Database migration to version {version} failed"
        super().__init__(message, cause=cause, context={"version": version})
        self.version = version


class DatabaseConstraintError(DatabaseError):
    """Raised when database constraint is violated (e.g., unique constraint)."""
    
    def __init__(self, constraint: str, cause: Optional[Exception] = None):
        message = f"Database constraint violation: {constraint}"
        super().__init__(message, cause=cause, context={"constraint": constraint})
        self.constraint = constraint


# Indexing Errors
class IndexingError(RAGException):
    """Base exception for indexing pipeline errors."""
    pass


class VersionActivationError(IndexingError):
    """Raised when activating a new document version fails."""
    
    def __init__(self, version_id: UUID, cause: Optional[Exception] = None):
        message = f"Failed to activate version: {version_id}"
        super().__init__(message, cause=cause, context={"version_id": str(version_id)})
        self.version_id = version_id


class CleanupError(IndexingError):
    """Raised when cleaning up old versions fails."""
    
    def __init__(self, version_ids: list[UUID], cause: Optional[Exception] = None):
        message = f"Failed to cleanup old versions: {version_ids}"
        super().__init__(message, cause=cause, context={"version_ids": [str(v) for v in version_ids]})
        self.version_ids = version_ids


# Scheduler Errors
class SchedulerError(RAGException):
    """Base exception for index scheduler errors."""
    pass


class SchedulerLockError(SchedulerError):
    """Raised when scheduler cannot acquire lock."""
    pass


# Validation Errors
class ValidationError(RAGException):
    """Base exception for validation errors."""
    pass


class CitationValidationError(ValidationError):
    """Raised when citation validation fails."""
    
    def __init__(self, message: str, invalid_citations: Optional[list] = None):
        super().__init__(message, context={"invalid_citations": invalid_citations or []})
        self.invalid_citations = invalid_citations or []


# Security Errors
class SecurityError(RAGException):
    """Base exception for security-related errors."""
    pass


class PathTraversalAttemptError(SecurityError):
    """Raised when path traversal attempt is detected and blocked."""
    
    def __init__(self, attempted_path: Path, root: Path):
        message = f"Path traversal attempt blocked: {attempted_path} (root: {root})"
        super().__init__(message, context={"attempted_path": str(attempted_path), "root": str(root)})
        self.attempted_path = attempted_path
        self.root = root


class SecretExposureError(SecurityError):
    """Raised when secret/credential exposure is detected."""
    
    def __init__(self, location: str):
        message = f"Potential secret exposure detected at: {location}"
        super().__init__(message, context={"location": location})
        self.location = location


# Exception Aliases for Codebase Compatibility
RAGSystemError = RAGException
ParserError = ParsingError
ModelError = EmbeddingModelError
PersistenceError = DatabaseError