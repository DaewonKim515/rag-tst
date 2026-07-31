"""
Version manager for document indexing versioning, state transitions, and atomic activation.
"""

from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

from ..domain.enums import DocumentStatus
from ..domain.models import DocumentRecord, FileVersionRecord
from ..domain.exceptions import PersistenceError
from ..persistence.repositories import DocumentRepository, FileVersionRepository
from ..persistence.sqlite import DatabaseManager
from ..observability.logging import get_logger

logger = get_logger(__name__)


class VersionManager:
    """
    Manages file version lifecycle and atomic activation of document versions in SQLite.
    
    State Flow:
    PENDING -> PARSING -> CHUNKING -> EMBEDDING -> STAGING -> ACTIVE
    Or on failure: -> FAILED
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.doc_repo = DocumentRepository(db_manager)
        self.version_repo = FileVersionRepository(db_manager)

    def create_pending_version(
        self,
        document_id: UUID,
        md5_hash: str,
        index_profile_id: str,
        parser_version: str = "1",
    ) -> FileVersionRecord:
        """Create or reuse a file version in PENDING status."""
        existing = self.version_repo.get_by_md5_and_profile(document_id, md5_hash, index_profile_id)
        if existing:
            self.version_repo.update_status(
                version_id=existing.version_id,
                status=DocumentStatus.PENDING,
                error_message=None,
            )
            logger.info("version_reused", f"Reused existing version {existing.version_id} for doc {document_id}")
            return existing

        version_id = uuid4()
        record = FileVersionRecord(
            version_id=version_id,
            document_id=document_id,
            md5_hash=md5_hash,
            index_profile_id=index_profile_id,
            parser_version=parser_version,
            status=DocumentStatus.PENDING,
        )
        self.version_repo.create(record)
        logger.info("version_created", f"Created pending version {version_id} for doc {document_id}")
        return record

    def update_status(
        self,
        version_id: UUID,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        """Update version status and error message."""
        self.version_repo.update_status(
            version_id=version_id,
            status=status,
            error_message=error_message,
            chunk_count=chunk_count,
        )
        logger.info("version_status_updated", f"Version {version_id} status updated to {status.value}")

    def activate_version(self, document_id: UUID, version_id: UUID) -> bool:
        """
        Atomically swap active_version_id in documents table inside a transaction.
        Updates version status to ACTIVE.
        """
        with self.db_manager.transaction() as conn:
            cursor = conn.cursor()
            
            # Fetch current active version to mark for deletion
            cursor.execute(
                "SELECT active_version_id FROM documents WHERE document_id = ?",
                (str(document_id),)
            )
            row = cursor.fetchone()
            old_version_id = row[0] if row and row[0] else None
            
            # Swap active version in documents table
            cursor.execute(
                "UPDATE documents SET active_version_id = ?, status = ?, updated_at = ? WHERE document_id = ?",
                (str(version_id), DocumentStatus.ACTIVE.value, datetime.now().isoformat(), str(document_id))
            )
            
            # Mark new version as ACTIVE in file_versions table
            cursor.execute(
                "UPDATE file_versions SET status = ?, indexed_at = ? WHERE version_id = ?",
                (DocumentStatus.ACTIVE.value, datetime.now().isoformat(), str(version_id))
            )
            
            # If old version exists, mark as DELETED
            if old_version_id and old_version_id != str(version_id):
                cursor.execute(
                    "UPDATE file_versions SET status = ? WHERE version_id = ?",
                    (DocumentStatus.DELETED.value, old_version_id)
                )
                
        logger.info("version_atomically_activated", f"Document {document_id} activated version {version_id}")
        return True

    def mark_failed(self, document_id: UUID, version_id: UUID, error_msg: str) -> None:
        """Mark version as FAILED without changing documents active version."""
        self.update_status(version_id, DocumentStatus.FAILED, error_message=error_msg)
        logger.error("version_failed", f"Version {version_id} failed: {error_msg}")
