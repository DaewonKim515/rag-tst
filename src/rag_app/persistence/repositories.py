"""
Repository pattern implementation for data access layer.

Provides type-safe, parameterized query access to SQLite database
with full SQL injection protection.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from uuid import UUID, uuid4

from .sqlite import (
    DatabaseManager, get_database_manager, 
    datetime_to_iso, iso_to_datetime, uuid_to_str, str_to_uuid
)
from ..domain.models import (
    DocumentRecord, FileVersionRecord, IndexJobRecord
)
from ..domain.enums import DocumentStatus, IndexStatus
from ..observability.logging import get_logger
from ..domain.exceptions import (
    DatabaseError, DatabaseConstraintError, 
    DatabaseConnectionError
)


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentSearchResult:
    """Search result for document queries."""
    document_id: UUID
    source_path: str
    file_name: str
    file_type: str
    file_size: int
    md5_hash: str
    modified_at: datetime
    active_version_id: Optional[UUID]
    status: DocumentStatus
    error_message: Optional[str]
    parser_version: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class VersionSearchResult:
    """Search result for version queries."""
    version_id: UUID
    document_id: UUID
    md5_hash: str
    index_profile_id: str
    parser_version: str
    chunk_count: Optional[int]
    status: DocumentStatus
    indexed_at: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class JobSearchResult:
    """Search result for index job queries."""
    job_id: UUID
    job_type: str
    status: str
    discovered_count: int
    new_count: int
    changed_count: int
    deleted_count: int
    skipped_count: int
    failed_count: int
    started_at: datetime
    finished_at: Optional[datetime]


class DocumentRepository:
    """Repository for document metadata operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self._db = db_manager or get_database_manager()
    
    def create(self, record: DocumentRecord) -> None:
        """Insert a new document record."""
        with self._db.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, source_path, file_name, file_type, file_size,
                        md5_hash, modified_at, active_version_id, status,
                        error_message, parser_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid_to_str(record.document_id),
                        record.source_path,
                        record.file_name,
                        record.file_type,
                        record.file_size,
                        record.md5_hash,
                        datetime_to_iso(record.modified_at),
                        uuid_to_str(record.active_version_id) if record.active_version_id else None,
                        record.status.value,
                        record.error_message,
                        record.parser_version,
                        datetime_to_iso(record.created_at),
                        datetime_to_iso(record.updated_at),
                    )
                )
                logger.debug("document_created", "Document record created",
                           document_id=str(record.document_id), source_path=record.source_path)
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: documents.source_path" in str(e):
                    raise DatabaseConstraintError("documents.source_path", e) from e
                raise DatabaseError(f"Failed to create document: {e}") from e
    
    def get_by_id(self, document_id: UUID) -> Optional[DocumentRecord]:
        """Get document by ID."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (uuid_to_str(document_id),)
            )
            row = cursor.fetchone()
            return self._row_to_document(row) if row else None
    
    def get_by_source_path(self, source_path: str) -> Optional[DocumentRecord]:
        """Get document by source path."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,)
            )
            row = cursor.fetchone()
            return self._row_to_document(row) if row else None
    
    def get_all(self, status: Optional[DocumentStatus] = None) -> List[DocumentRecord]:
        """Get all documents, optionally filtered by status."""
        with self._db.read_only() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM documents WHERE status = ? ORDER BY created_at DESC",
                    (status.value,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                )
            return [self._row_to_document(row) for row in cursor.fetchall()]
    
    def get_active_documents(self) -> List[DocumentRecord]:
        """Get all documents with active status."""
        return self.get_all(DocumentStatus.ACTIVE)
    
    def update_status(
        self, 
        document_id: UUID, 
        status: DocumentStatus, 
        error_message: Optional[str] = None,
        active_version_id: Optional[UUID] = None
    ) -> None:
        """Update document status and optional fields."""
        with self._db.transaction() as conn:
            if active_version_id is not None:
                conn.execute(
                    """
                    UPDATE documents 
                    SET status = ?, error_message = ?, active_version_id = ?, updated_at = ?
                    WHERE document_id = ?
                    """,
                    (
                        status.value,
                        error_message,
                        uuid_to_str(active_version_id),
                        datetime_to_iso(datetime.now()),
                        uuid_to_str(document_id),
                    )
                )
            else:
                conn.execute(
                    """
                    UPDATE documents 
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE document_id = ?
                    """,
                    (
                        status.value,
                        error_message,
                        datetime_to_iso(datetime.now()),
                        uuid_to_str(document_id),
                    )
                )
            logger.debug("document_status_updated", "Document status updated",
                       document_id=str(document_id), status=status.value)
    
    def update_active_version(self, document_id: UUID, version_id: UUID) -> None:
        """Update document's active version."""
        with self._db.transaction() as conn:
            conn.execute(
                """
                UPDATE documents 
                SET active_version_id = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (
                    uuid_to_str(version_id),
                    datetime_to_iso(datetime.now()),
                    uuid_to_str(document_id),
                )
            )
            logger.debug("document_active_version_updated", "Document active version updated",
                       document_id=str(document_id), version_id=str(version_id))
    
    def update_md5_hash(self, document_id: UUID, md5_hash: str) -> None:
        """Update document's MD5 hash."""
        with self._db.transaction() as conn:
            conn.execute(
                """
                UPDATE documents 
                SET md5_hash = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (md5_hash, datetime_to_iso(datetime.now()), uuid_to_str(document_id))
            )
    
    def delete(self, document_id: UUID) -> bool:
        """Delete document (cascades to file_versions via FK)."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (uuid_to_str(document_id),)
            )
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("document_deleted", "Document deleted", 
                           document_id=str(document_id))
            return deleted
    
    def count(self, status: Optional[DocumentStatus] = None) -> int:
        """Count documents."""
        with self._db.read_only() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE status = ?",
                    (status.value,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]
    
    def get_by_md5(self, md5_hash: str) -> List[DocumentRecord]:
        """Get documents by MD5 hash (for duplicate detection)."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE md5_hash = ?",
                (md5_hash,)
            )
            return [self._row_to_document(row) for row in cursor.fetchall()]
    
    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            document_id=str_to_uuid(row["document_id"]),
            source_path=row["source_path"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            md5_hash=row["md5_hash"],
            modified_at=iso_to_datetime(row["modified_at"]),
            active_version_id=str_to_uuid(row["active_version_id"]) if row["active_version_id"] else None,
            status=DocumentStatus(row["status"]),
            error_message=row["error_message"],
            parser_version=row["parser_version"],
            created_at=iso_to_datetime(row["created_at"]),
            updated_at=iso_to_datetime(row["updated_at"]),
        )


class FileVersionRepository:
    """Repository for file version (MD5 table) operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self._db = db_manager or get_database_manager()
    
    def create(self, record: FileVersionRecord) -> None:
        """Insert a new file version record."""
        with self._db.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO file_versions (
                        version_id, document_id, md5_hash, index_profile_id,
                        parser_version, chunk_count, status, indexed_at,
                        error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid_to_str(record.version_id),
                        uuid_to_str(record.document_id),
                        record.md5_hash,
                        record.index_profile_id,
                        record.parser_version,
                        record.chunk_count if record.chunk_count is not None else 0,
                        record.status.value,
                        datetime_to_iso(record.indexed_at) if record.indexed_at else None,
                        record.error_message,
                        datetime_to_iso(record.created_at),
                    )
                )
                logger.debug("version_created", "File version created",
                           version_id=str(record.version_id), document_id=str(record.document_id))
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: file_versions.document_id" in str(e):
                    raise DatabaseConstraintError("file_versions (document_id, md5_hash, index_profile_id)", e) from e
                raise DatabaseError(f"Failed to create file version: {e}") from e
    
    def get_by_id(self, version_id: UUID) -> Optional[FileVersionRecord]:
        """Get version by ID."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM file_versions WHERE version_id = ?",
                (uuid_to_str(version_id),)
            )
            row = cursor.fetchone()
            return self._row_to_version(row) if row else None
    
    def get_by_document_id(self, document_id: UUID) -> List[FileVersionRecord]:
        """Get all versions for a document."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM file_versions WHERE document_id = ? ORDER BY created_at DESC",
                (uuid_to_str(document_id),)
            )
            return [self._row_to_version(row) for row in cursor.fetchall()]
    
    def get_active_version(self, document_id: UUID) -> Optional[FileVersionRecord]:
        """Get active version for a document."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                """
                SELECT fv.* FROM file_versions fv
                JOIN documents d ON fv.version_id = d.active_version_id
                WHERE d.document_id = ?
                """,
                (uuid_to_str(document_id),)
            )
            row = cursor.fetchone()
            return self._row_to_version(row) if row else None
    
    def get_by_md5_and_profile(
        self, 
        document_id: UUID, 
        md5_hash: str, 
        index_profile_id: str
    ) -> Optional[FileVersionRecord]:
        """Get version by document, MD5, and index profile (for change detection)."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM file_versions 
                WHERE document_id = ? AND md5_hash = ? AND index_profile_id = ?
                """,
                (uuid_to_str(document_id), md5_hash, index_profile_id)
            )
            row = cursor.fetchone()
            return self._row_to_version(row) if row else None
    
    def get_pending_versions(self) -> List[FileVersionRecord]:
        """Get all versions with PENDING status."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM file_versions WHERE status = ? ORDER BY created_at",
                (DocumentStatus.PENDING.value,)
            )
            return [self._row_to_version(row) for row in cursor.fetchall()]
    
    def update_status(
        self, 
        version_id: UUID, 
        status: DocumentStatus, 
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        indexed_at: Optional[datetime] = None
    ) -> None:
        """Update version status and optional fields."""
        with self._db.transaction() as conn:
            fields = ["status = ?"]
            params = [status.value]
            
            if error_message is not None:
                fields.append("error_message = ?")
                params.append(error_message)
            
            if chunk_count is not None:
                fields.append("chunk_count = ?")
                params.append(chunk_count)
            
            if indexed_at is not None:
                fields.append("indexed_at = ?")
                params.append(datetime_to_iso(indexed_at))
            
            params.append(uuid_to_str(version_id))
            
            conn.execute(
                f"UPDATE file_versions SET {', '.join(fields)} WHERE version_id = ?",
                params
            )
            logger.debug("version_status_updated", "File version status updated",
                       version_id=str(version_id), status=status.value)
    
    def activate(self, version_id: UUID, chunk_count: int) -> None:
        """Mark version as active with chunk count and index time."""
        self.update_status(
            version_id,
            DocumentStatus.ACTIVE,
            chunk_count=chunk_count,
            indexed_at=datetime.now()
        )
    
    def mark_failed(self, version_id: UUID, error_message: str) -> None:
        """Mark version as failed."""
        self.update_status(version_id, DocumentStatus.FAILED, error_message=error_message)
    
    def delete(self, version_id: UUID) -> bool:
        """Delete a version record."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM file_versions WHERE version_id = ?",
                (uuid_to_str(version_id),)
            )
            return cursor.rowcount > 0
    
    def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all versions for a document."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM file_versions WHERE document_id = ?",
                (uuid_to_str(document_id),)
            )
            return cursor.rowcount
    
    def count_by_document(self, document_id: UUID) -> int:
        """Count versions for a document."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM file_versions WHERE document_id = ?",
                (uuid_to_str(document_id),)
            )
            return cursor.fetchone()[0]
    
    def _row_to_version(self, row: sqlite3.Row) -> FileVersionRecord:
        """Convert database row to FileVersionRecord."""
        return FileVersionRecord(
            version_id=str_to_uuid(row["version_id"]),
            document_id=str_to_uuid(row["document_id"]),
            md5_hash=row["md5_hash"],
            index_profile_id=row["index_profile_id"],
            parser_version=row["parser_version"],
            chunk_count=row["chunk_count"],
            status=DocumentStatus(row["status"]),
            indexed_at=iso_to_datetime(row["indexed_at"]) if row["indexed_at"] else None,
            error_message=row["error_message"],
            created_at=iso_to_datetime(row["created_at"]),
        )


class IndexJobRepository:
    """Repository for index job tracking."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self._db = db_manager or get_database_manager()
    
    def create(self, record: IndexJobRecord) -> None:
        """Insert a new index job record."""
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO index_jobs (
                    job_id, job_type, status, discovered_count, new_count,
                    changed_count, deleted_count, skipped_count, failed_count,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid_to_str(record.job_id),
                    record.job_type,
                    record.status,
                    record.discovered_count,
                    record.new_count,
                    record.changed_count,
                    record.deleted_count,
                    record.skipped_count,
                    record.failed_count,
                    datetime_to_iso(record.started_at),
                    datetime_to_iso(record.finished_at) if record.finished_at else None,
                )
            )
            logger.debug("index_job_created", "Index job created",
                       job_id=str(record.job_id), job_type=record.job_type)
    
    def get_by_id(self, job_id: UUID) -> Optional[IndexJobRecord]:
        """Get job by ID."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM index_jobs WHERE job_id = ?",
                (uuid_to_str(job_id),)
            )
            row = cursor.fetchone()
            return self._row_to_job(row) if row else None
    
    def get_latest(self, limit: int = 10) -> List[IndexJobRecord]:
        """Get latest index jobs."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM index_jobs ORDER BY started_at DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def get_by_type(self, job_type: str, limit: int = 10) -> List[IndexJobRecord]:
        """Get jobs by type."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT * FROM index_jobs WHERE job_type = ? ORDER BY started_at DESC LIMIT ?",
                (job_type, limit)
            )
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def update_status(
        self,
        job_id: UUID,
        status: str,
        discovered_count: Optional[int] = None,
        new_count: Optional[int] = None,
        changed_count: Optional[int] = None,
        deleted_count: Optional[int] = None,
        skipped_count: Optional[int] = None,
        failed_count: Optional[int] = None,
        finished_at: Optional[datetime] = None
    ) -> None:
        """Update job status and counters."""
        with self._db.transaction() as conn:
            fields = ["status = ?"]
            params = [status]
            
            if discovered_count is not None:
                fields.append("discovered_count = ?")
                params.append(discovered_count)
            if new_count is not None:
                fields.append("new_count = ?")
                params.append(new_count)
            if changed_count is not None:
                fields.append("changed_count = ?")
                params.append(changed_count)
            if deleted_count is not None:
                fields.append("deleted_count = ?")
                params.append(deleted_count)
            if skipped_count is not None:
                fields.append("skipped_count = ?")
                params.append(skipped_count)
            if failed_count is not None:
                fields.append("failed_count = ?")
                params.append(failed_count)
            if finished_at is not None:
                fields.append("finished_at = ?")
                params.append(datetime_to_iso(finished_at))
            
            params.append(uuid_to_str(job_id))
            
            conn.execute(
                f"UPDATE index_jobs SET {', '.join(fields)} WHERE job_id = ?",
                params
            )
            logger.debug("index_job_updated", "Index job updated",
                       job_id=str(job_id), status=status)
    
    def complete(self, job_id: UUID, status: str = "completed") -> None:
        """Mark job as completed."""
        self.update_status(job_id, status, finished_at=datetime.now())
    
    def _row_to_job(self, row: sqlite3.Row) -> IndexJobRecord:
        """Convert database row to IndexJobRecord."""
        return IndexJobRecord(
            job_id=str_to_uuid(row["job_id"]),
            job_type=row["job_type"],
            status=row["status"],
            discovered_count=row["discovered_count"],
            new_count=row["new_count"],
            changed_count=row["changed_count"],
            deleted_count=row["deleted_count"],
            skipped_count=row["skipped_count"],
            failed_count=row["failed_count"],
            started_at=iso_to_datetime(row["started_at"]),
            finished_at=iso_to_datetime(row["finished_at"]) if row["finished_at"] else None,
        )


class MetadataRepository:
    """Combined repository for all metadata operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.documents = DocumentRepository(db_manager)
        self.file_versions = FileVersionRepository(db_manager)
        self.index_jobs = IndexJobRepository(db_manager)
        self._db = db_manager or get_database_manager()
    
    @contextmanager
    def transaction(self):
        """Delegate to database manager transaction."""
        with self._db.transaction() as conn:
            yield conn
    
    def get_active_version_mapping(self) -> Dict[UUID, UUID]:
        """Get mapping of document_id -> active_version_id for retrieval filtering."""
        with self._db.read_only() as conn:
            cursor = conn.execute(
                "SELECT document_id, active_version_id FROM documents WHERE active_version_id IS NOT NULL"
            )
            return {
                str_to_uuid(row["document_id"]): str_to_uuid(row["active_version_id"])
                for row in cursor.fetchall()
            }
    
    def get_document_with_version(self, document_id: UUID) -> Optional[Tuple[DocumentRecord, FileVersionRecord]]:
        """Get document with its active version."""
        doc = self.documents.get_by_id(document_id)
        if not doc or not doc.active_version_id:
            return None
        version = self.file_versions.get_by_id(doc.active_version_id)
        if not version:
            return None
        return (doc, version)
    
    def cleanup_old_versions(self, document_id: UUID, keep_version_id: UUID) -> int:
        """Delete old versions for a document, keeping the specified one."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM file_versions WHERE document_id = ? AND version_id != ?",
                (uuid_to_str(document_id), uuid_to_str(keep_version_id))
            )
            return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._db.read_only() as conn:
            stats = {}
            
            # Document counts by status
            cursor = conn.execute(
                "SELECT status, COUNT(*) as count FROM documents GROUP BY status"
            )
            stats["documents_by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Version counts by status
            cursor = conn.execute(
                "SELECT status, COUNT(*) as count FROM file_versions GROUP BY status"
            )
            stats["versions_by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Total chunks
            cursor = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) as total FROM file_versions")
            stats["total_chunks"] = cursor.fetchone()["total"]
            
            # Job counts
            cursor = conn.execute("SELECT COUNT(*) as count FROM index_jobs")
            stats["total_jobs"] = cursor.fetchone()["count"]
            
            return stats