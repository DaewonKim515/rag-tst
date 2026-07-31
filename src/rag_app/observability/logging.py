"""
Structured JSON Lines logging with correlation ID tracking.

Provides structured logging for the Document RAG system with support for
job/request correlation IDs, consistent log schemas, and configurable output.
"""

import json
import logging
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from ..config.schema import LoggingConfig, AppConfig


# Context variables for correlation tracking
job_id_var: ContextVar[Optional[UUID]] = ContextVar("job_id", default=None)
request_id_var: ContextVar[Optional[UUID]] = ContextVar("request_id", default=None)
document_id_var: ContextVar[Optional[UUID]] = ContextVar("document_id", default=None)
version_id_var: ContextVar[Optional[UUID]] = ContextVar("version_id", default=None)


class JSONLinesFormatter(logging.Formatter):
    """Format log records as JSON Lines for structured logging."""
    
    def __init__(self, include_query_text: bool = False):
        super().__init__()
        self.include_query_text = include_query_text
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line."""
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation IDs if present
        job_id = job_id_var.get()
        if job_id:
            log_entry["job_id"] = str(job_id)
        
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = str(request_id)
        
        document_id = document_id_var.get()
        if document_id:
            log_entry["document_id"] = str(document_id)
        
        version_id = version_id_var.get()
        if version_id:
            log_entry["version_id"] = str(version_id)
        
        # Add extra fields from record
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        
        if hasattr(record, "chunk_count"):
            log_entry["chunk_count"] = record.chunk_count
        
        if hasattr(record, "status"):
            log_entry["status"] = record.status
        
        if hasattr(record, "error_code"):
            log_entry["error_code"] = record.error_code
        
        if hasattr(record, "source_path"):
            log_entry["source_path"] = record.source_path
        
        if hasattr(record, "file_name"):
            log_entry["file_name"] = record.file_name
        
        if hasattr(record, "model_id"):
            log_entry["model_id"] = record.model_id
        
        if hasattr(record, "tokens"):
            log_entry["tokens"] = record.tokens
        
        # Include query text only if explicitly enabled (privacy)
        if self.include_query_text and hasattr(record, "query_text"):
            log_entry["query_text"] = record.query_text
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry["stack_trace"] = record.stack_info
        
        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Wrapper for structured logging with convenience methods."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log(self, level: int, event: str, message: str, **kwargs):
        """Internal log method with extra fields."""
        extra = {"event": event, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, event: str, message: str, **kwargs):
        self._log(logging.DEBUG, event, message, **kwargs)
    
    def info(self, event: str, message: str, **kwargs):
        self._log(logging.INFO, event, message, **kwargs)
    
    def warning(self, event: str, message: str, **kwargs):
        self._log(logging.WARNING, event, message, **kwargs)
    
    def error(self, event: str, message: str, **kwargs):
        self._log(logging.ERROR, event, message, **kwargs)
    
    def critical(self, event: str, message: str, **kwargs):
        self._log(logging.CRITICAL, event, message, **kwargs)
    
    # Convenience methods for common events
    
    def document_indexed(self, document_id: UUID, version_id: UUID, source_path: str, 
                         duration_ms: float, chunk_count: int, status: str, error_code: Optional[str] = None):
        """Log document indexing completion."""
        self._log(
            logging.INFO if status == "ACTIVE" else logging.ERROR,
            "document_indexed",
            f"Document indexed: {source_path} (chunks: {chunk_count})",
            document_id=document_id,
            version_id=version_id,
            source_path=source_path,
            duration_ms=duration_ms,
            chunk_count=chunk_count,
            status=status,
            error_code=error_code,
        )
    
    def query_processed(self, request_id: UUID, query_text: str, 
                        retrieval_time_ms: float, rerank_time_ms: float,
                        generation_time_ms: float, total_time_ms: float,
                        sources_count: int, status: str, error_code: Optional[str] = None):
        """Log query processing completion."""
        self._log(
            logging.INFO if status == "success" else logging.ERROR,
            "query_processed",
            f"Query processed: {status} (sources: {sources_count})",
            request_id=request_id,
            query_text=query_text,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            sources_count=sources_count,
            status=status,
            error_code=error_code,
        )
    
    def index_job_started(self, job_id: UUID, job_type: str, discovered_count: int = 0):
        """Log index job start."""
        self._log(
            logging.INFO,
            "index_job_started",
            f"Index job started: {job_type}",
            job_id=job_id,
            job_type=job_type,
            discovered_count=discovered_count,
        )
    
    def index_job_completed(self, job_id: UUID, job_type: str, status: str,
                            new_count: int, changed_count: int, deleted_count: int,
                            skipped_count: int, failed_count: int, duration_ms: float):
        """Log index job completion."""
        self._log(
            logging.INFO if status in ("completed", "partial_failed") else logging.ERROR,
            "index_job_completed",
            f"Index job completed: {job_type} ({status})",
            job_id=job_id,
            job_type=job_type,
            status=status,
            new_count=new_count,
            changed_count=changed_count,
            deleted_count=deleted_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            duration_ms=duration_ms,
        )
    
    def file_processed(self, job_id: UUID, source_path: str, file_name: str,
                       status: str, duration_ms: float, chunk_count: Optional[int] = None,
                       error_code: Optional[str] = None):
        """Log individual file processing result."""
        self._log(
            logging.INFO if status == "completed" else logging.ERROR,
            "file_processed",
            f"File processed: {file_name} ({status})",
            job_id=job_id,
            source_path=source_path,
            file_name=file_name,
            status=status,
            duration_ms=duration_ms,
            chunk_count=chunk_count,
            error_code=error_code,
        )
    
    def model_loaded(self, model_type: str, model_id: str, load_time_ms: float):
        """Log model loading."""
        self._log(
            logging.INFO,
            "model_loaded",
            f"Model loaded: {model_type} ({model_id})",
            model_type=model_type,
            model_id=model_id,
            load_time_ms=load_time_ms,
        )
    
    def model_error(self, model_type: str, model_id: str, error_code: str, 
                    message: str, retryable: bool = False):
        """Log model error."""
        self._log(
            logging.ERROR,
            "model_error",
            f"Model error: {model_type} ({model_id}) - {message}",
            model_type=model_type,
            model_id=model_id,
            error_code=error_code,
            retryable=retryable,
        )
    
    def security_event(self, event_type: str, message: str, **context):
        """Log security-related event."""
        self._log(
            logging.WARNING,
            "security_event",
            f"Security event: {event_type} - {message}",
            security_event_type=event_type,
            **context,
        )


def setup_logging(config: LoggingConfig, log_dir: Path) -> None:
    """
    Configure application logging with JSON Lines format.
    
    Args:
        config: Logging configuration from AppConfig.
        log_dir: Directory for log files.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, config.level))
    
    # Console handler (text format for readability)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    from logging.handlers import TimedRotatingFileHandler
    
    # File handler with daily rotation and retention policy (retention_days)
    log_file = log_dir / "rag_app.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=config.retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, config.level))
    file_handler.setFormatter(JSONLinesFormatter(include_query_text=config.include_query_text))
    root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Log startup
    logger = StructuredLogger("rag_app.startup")
    logger.info("logging_configured", "Logging configured", 
                log_file=str(log_file), log_level=config.level, format=config.format)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


# Context managers for correlation ID tracking

class JobContext:
    """Context manager for job ID correlation."""
    
    def __init__(self, job_id: Optional[UUID] = None):
        self.job_id = job_id or uuid4()
        self.token = None
    
    def __enter__(self) -> UUID:
        self.token = job_id_var.set(self.job_id)
        return self.job_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            job_id_var.reset(self.token)


class RequestContext:
    """Context manager for request ID correlation."""
    
    def __init__(self, request_id: Optional[UUID] = None):
        self.request_id = request_id or uuid4()
        self.token = None
    
    def __enter__(self) -> UUID:
        self.token = request_id_var.set(self.request_id)
        return self.request_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            request_id_var.reset(self.token)


class DocumentContext:
    """Context manager for document/version ID correlation."""
    
    def __init__(self, document_id: Optional[UUID] = None, version_id: Optional[UUID] = None):
        self.document_id = document_id
        self.version_id = version_id
        self.doc_token = None
        self.ver_token = None
    
    def __enter__(self):
        if self.document_id:
            self.doc_token = document_id_var.set(self.document_id)
        if self.version_id:
            self.ver_token = version_id_var.set(self.version_id)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.doc_token:
            document_id_var.reset(self.doc_token)
        if self.ver_token:
            version_id_var.reset(self.ver_token)


def get_current_job_id() -> Optional[UUID]:
    """Get current job ID from context."""
    return job_id_var.get()


def get_current_request_id() -> Optional[UUID]:
    """Get current request ID from context."""
    return request_id_var.get()


def get_current_document_id() -> Optional[UUID]:
    """Get current document ID from context."""
    return document_id_var.get()


def get_current_version_id() -> Optional[UUID]:
    """Get current version ID from context."""
    return version_id_var.get()


# Alias for correlation context
new_job_context = JobContext