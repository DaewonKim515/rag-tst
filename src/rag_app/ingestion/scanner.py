"""
File scanner with security hardening for path traversal and malicious file detection.

Provides recursive directory traversal with extension filtering, size limits,
symlink protection, and path containment validation.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass
from uuid import UUID, uuid4

from ..config.schema import AppConfig
from ..domain.models import DocumentRecord
from ..domain.enums import FileType, DocumentStatus
from ..domain.exceptions import (
    PathTraversalAttemptError, FileTooLargeError, 
    UnsupportedFileTypeError, FileProcessingError
)
from ..observability.logging import get_logger, get_current_job_id


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """Information about a discovered file."""
    file_path: Path
    relative_path: Path
    file_name: str
    file_size: int
    file_type: FileType
    modified_at: float  # Unix timestamp


class FileScanner:
    """Secure file scanner with path traversal and symlink protection."""
    
    # Maximum path length for safety
    MAX_PATH_LENGTH = 4096
    
    # Suspicious filename patterns
    SUSPICIOUS_PATTERNS = [
        "..",           # Path traversal
        "~",            # Home directory expansion
        "$",            # Variable expansion
        "`",            # Command substitution
        "|",            # Pipe
        ";",            # Command separator
        "&",            # Background/command separator
        "\x00",         # Null byte
    ]
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.document_root = config.paths.document_root.resolve()
        self.supported_extensions = set(config.indexing.supported_extensions)
        self.max_file_size_bytes = config.indexing.max_file_size_mb * 1024 * 1024
        
        # Validate document root exists
        self.document_root.mkdir(parents=True, exist_ok=True)
    
    def scan(self) -> List[DiscoveredFile]:
        """
        Scan document root for supported files.
        
        Returns:
            List of DiscoveredFile objects for supported files.
            
        Raises:
            PathTraversalAttemptError: If path escapes document root.
        """
        discovered = []
        job_id = get_current_job_id()
        
        try:
            for file_path in self.document_root.rglob("*"):
                # Skip directories
                if not file_path.is_file():
                    continue
                
                try:
                    discovered_file = self._process_file(file_path)
                    if discovered_file:
                        discovered.append(discovered_file)
                except PathTraversalAttemptError:
                    # Re-raise path traversal immediately
                    raise
                except (FileTooLargeError, UnsupportedFileTypeError) as e:
                    # Log and continue for expected filtering errors
                    logger.warning("file_filtered", str(e), 
                                 job_id=job_id, file_path=str(file_path),
                                 error_code=type(e).__name__)
                    continue
                except Exception as e:
                    # Log unexpected errors but continue scanning
                    logger.error("file_scan_error", f"Error scanning {file_path}: {e}",
                               job_id=job_id, file_path=str(file_path),
                               error_code=type(e).__name__)
                    continue
            
            logger.info("scan_completed", f"Discovered {len(discovered)} files",
                       job_id=job_id, file_count=len(discovered))
            return discovered
            
        except Exception as e:
            logger.error("scan_failed", f"Scan failed: {e}", 
                        job_id=job_id, error_code=type(e).__name__)
            raise
    
    def _process_file(self, file_path: Path) -> Optional[DiscoveredFile]:
        """Process a single file with all security checks."""
        # 1. Validate path is within document root (path traversal defense)
        self._validate_path_containment(file_path)
        
        # 2. Validate filename for suspicious patterns
        self._validate_filename(file_path.name)
        
        # 3. Check path length
        if len(str(file_path)) > self.MAX_PATH_LENGTH:
            raise FileProcessingError(f"Path too long: {file_path}")
        
        # 4. Get file stats (handles symlinks safely)
        try:
            stat = file_path.stat(follow_symlinks=False)
        except OSError as e:
            raise FileProcessingError(f"Cannot stat file {file_path}: {e}") from e
        
        # 5. Skip symlinks and junctions (security)
        if file_path.is_symlink():
            logger.warning("symlink_skipped", f"Symlink skipped: {file_path}",
                         job_id=get_current_job_id(), file_path=str(file_path))
            return None
        
        # 6. Check file size
        file_size = stat.st_size
        if file_size > self.max_file_size_bytes:
            raise FileTooLargeError(file_path, file_size / (1024*1024), 
                                  self.config.indexing.max_file_size_mb)
        
        # 7. Check extension
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise UnsupportedFileTypeError(file_path, list(self.supported_extensions))
        
        # 8. Determine file type
        file_type = FileType(suffix)
        
        # 9. Compute relative path
        try:
            relative_path = file_path.relative_to(self.document_root)
        except ValueError:
            # This shouldn't happen if containment check passed
            raise PathTraversalAttemptError(file_path, self.document_root, "relative_path")
        
        return DiscoveredFile(
            file_path=file_path,
            relative_path=relative_path,
            file_name=file_path.name,
            file_size=file_size,
            file_type=file_type,
            modified_at=stat.st_mtime,
        )
    
    def _validate_path_containment(self, file_path: Path) -> None:
        """
        Validate that file_path is contained within document_root.
        
        Uses realpath to resolve symlinks and ensure containment.
        """
        try:
            # Resolve both paths to absolute, following symlinks for root but not for file
            real_root = self.document_root.resolve()
            real_file = file_path.resolve()
            
            # Check containment
            real_file.relative_to(real_root)
        except ValueError:
            # Path is outside root
            raise PathTraversalAttemptError(file_path, self.document_root, "containment_check")
    
    def _validate_filename(self, filename: str) -> None:
        """
        Validate filename for suspicious patterns.
        
        Raises:
            PathTraversalAttemptError: If suspicious pattern detected.
        """
        if not filename or filename.strip() != filename:
            raise PathTraversalAttemptError(
                Path(filename), self.document_root, "whitespace_filename"
            )
        
        # Check for null bytes
        if "\x00" in filename:
            raise PathTraversalAttemptError(
                Path(filename), self.document_root, "null_byte"
            )
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in filename:
                raise PathTraversalAttemptError(
                    Path(filename), self.document_root, f"suspicious_pattern:{pattern}"
                )
        
        # Check for absolute paths
        if os.path.isabs(filename):
            raise PathTraversalAttemptError(
                Path(filename), self.document_root, "absolute_path"
            )
        
        # Check for reserved names (Windows)
        reserved_names = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }
        name_upper = filename.split(".")[0].upper()
        if name_upper in reserved_names:
            raise PathTraversalAttemptError(
                Path(filename), self.document_root, "reserved_name"
            )


def create_document_record(discovered: DiscoveredFile, md5_hash: str) -> DocumentRecord:
    """Create a DocumentRecord from a discovered file and its MD5 hash."""
    from datetime import datetime
    
    return DocumentRecord(
        document_id=uuid4(),
        source_path=str(discovered.relative_path),
        file_name=discovered.file_name,
        file_type=discovered.file_type.value,
        file_size=discovered.file_size,
        md5_hash=md5_hash,
        modified_at=datetime.fromtimestamp(discovered.modified_at),
        status=DocumentStatus.DISCOVERED,
        parser_version="1",
    )