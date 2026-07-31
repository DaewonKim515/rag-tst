"""
Streaming MD5 hasher with file stability checks and zip bomb protection.

Provides memory-efficient file hashing and decompression bomb detection
for ZIP-based formats (DOCX, XLSX, PPTX).
"""

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config.schema import AppConfig
from ..domain.exceptions import FileStabilityError, ZipBombError
from ..observability.logging import get_logger, get_current_job_id


logger = get_logger(__name__)


# Streaming hash constants
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB chunks


# Zip bomb protection limits
MAX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_UNCOMPRESSED_RATIO = 100  # 100:1 compression ratio
MAX_FILE_COUNT = 1000


@dataclass(frozen=True, slots=True)
class HashResult:
    """Result of file hashing operation."""
    md5_hash: str
    file_size: int
    stable: bool


@dataclass(frozen=True, slots=True)
class ZipBombCheckResult:
    """Result of zip bomb detection."""
    is_safe: bool
    uncompressed_size: int
    file_count: int
    compression_ratio: float


class StreamHasher:
    """Memory-efficient streaming file hasher with stability checks."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.hash_algorithm = config.indexing.hash_algorithm  # "md5"
    
    def calculate_md5(self, file_path: Path) -> HashResult:
        """
        Calculate MD5 hash using streaming (constant memory).
        
        Also performs file stability check by comparing size/mtime before and after.
        
        Args:
            file_path: Path to file to hash.
            
        Returns:
            HashResult with hash, size, and stability flag.
            
        Raises:
            FileStabilityError: If file changes during hashing.
        """
        job_id = get_current_job_id()
        
        # Initial stat for stability check
        try:
            stat_before = file_path.stat()
        except OSError as e:
            raise FileStabilityError(file_path) from e
        
        size_before = stat_before.st_size
        mtime_before = stat_before.st_mtime
        
        # Calculate hash
        hasher = hashlib.new(self.hash_algorithm)
        
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError as e:
            raise FileStabilityError(file_path) from e
        
        md5_hash = hasher.hexdigest()
        
        # Final stat for stability check
        try:
            stat_after = file_path.stat()
        except OSError as e:
            raise FileStabilityError(file_path) from e
        
        size_after = stat_after.st_size
        mtime_after = stat_after.st_mtime
        
        # Check stability
        stable = (size_before == size_after) and (mtime_before == mtime_after)
        
        if not stable:
            logger.warning("file_unstable", f"File changed during hashing: {file_path}",
                         job_id=job_id, file_path=str(file_path),
                         size_before=size_before, size_after=size_after,
                         mtime_before=mtime_before, mtime_after=mtime_after)
            raise FileStabilityError(file_path)
        
        return HashResult(
            md5_hash=md5_hash,
            file_size=size_after,
            stable=True,
        )
    
    def calculate_md5_with_check(self, file_path: Path, expected_size: Optional[int] = None) -> HashResult:
        """
        Calculate MD5 with optional expected size validation.
        
        Args:
            file_path: Path to file.
            expected_size: Expected file size for quick validation.
            
        Returns:
            HashResult.
        """
        if expected_size is not None:
            try:
                actual_size = file_path.stat().st_size
                if actual_size != expected_size:
                    raise FileStabilityError(file_path)
            except OSError as e:
                raise FileStabilityError(file_path) from e
        
        return self.calculate_md5(file_path)


class ZipBombDetector:
    """Detects zip bombs in ZIP-based document formats (DOCX, XLSX, PPTX)."""
    
    def __init__(
        self,
        max_uncompressed_size: int = MAX_UNCOMPRESSED_SIZE,
        max_uncompressed_ratio: int = MAX_UNCOMPRESSED_RATIO,
        max_file_count: int = MAX_FILE_COUNT,
    ):
        self.max_uncompressed_size = max_uncompressed_size
        self.max_uncompressed_ratio = max_uncompressed_ratio
        self.max_file_count = max_file_count
    
    def check(self, file_path: Path) -> ZipBombCheckResult:
        """
        Check a ZIP-based file for decompression bombs.
        
        Args:
            file_path: Path to ZIP-based file (DOCX, XLSX, PPTX).
            
        Returns:
            ZipBombCheckResult with safety assessment.
            
        Raises:
            ZipBombError: If zip bomb detected.
        """
        job_id = get_current_job_id()
        
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                file_count = len(zf.infolist())
                
                if file_count > self.max_file_count:
                    result = ZipBombCheckResult(
                        is_safe=False,
                        uncompressed_size=0,
                        file_count=file_count,
                        compression_ratio=0.0,
                    )
                    raise ZipBombError(
                        file_path, 
                        uncompressed_size=0, 
                        file_count=file_count
                    )
                
                total_uncompressed = 0
                for info in zf.infolist():
                    total_uncompressed += info.file_size
                    
                    if total_uncompressed > self.max_uncompressed_size:
                        result = ZipBombCheckResult(
                            is_safe=False,
                            uncompressed_size=total_uncompressed,
                            file_count=file_count,
                            compression_ratio=0.0,
                        )
                        raise ZipBombError(
                            file_path,
                            uncompressed_size=total_uncompressed,
                            file_count=file_count,
                        )
                
                # Calculate compression ratio
                compressed_size = file_path.stat().st_size
                if compressed_size > 0:
                    ratio = total_uncompressed / compressed_size
                    if ratio > self.max_uncompressed_ratio:
                        result = ZipBombCheckResult(
                            is_safe=False,
                            uncompressed_size=total_uncompressed,
                            file_count=file_count,
                            compression_ratio=ratio,
                        )
                        raise ZipBombError(
                            file_path,
                            uncompressed_size=total_uncompressed,
                            file_count=file_count,
                        )
                else:
                    ratio = 0.0
                
                return ZipBombCheckResult(
                    is_safe=True,
                    uncompressed_size=total_uncompressed,
                    file_count=file_count,
                    compression_ratio=ratio,
                )
                
        except zipfile.BadZipFile as e:
            # Not a valid zip file - let parser handle it
            logger.debug("not_zip_file", f"Not a valid zip file: {file_path}",
                        job_id=job_id, file_path=str(file_path))
            return ZipBombCheckResult(
                is_safe=True,
                uncompressed_size=0,
                file_count=0,
                compression_ratio=0.0,
            )
        except ZipBombError:
            raise
        except Exception as e:
            logger.error("zip_check_failed", f"Zip bomb check failed: {e}",
                        job_id=job_id, file_path=str(file_path),
                        error_code=type(e).__name__)
            # On unexpected error, assume safe and let parser handle
            return ZipBombCheckResult(
                is_safe=True,
                uncompressed_size=0,
                file_count=0,
                compression_ratio=0.0,
            )


def create_hasher(config: AppConfig) -> StreamHasher:
    """Factory function to create StreamHasher."""
    return StreamHasher(config)


def create_zip_bomb_detector(config: AppConfig) -> ZipBombDetector:
    """Factory function to create ZipBombDetector."""
    return ZipBombDetector()


# Alias for codebase compatibility
Hasher = StreamHasher