"""
SQLite database connection manager with WAL mode and migrations.

Provides thread-safe database connections with connection pooling,
WAL mode for concurrent access, and migration system.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

from datetime import datetime
from ..config.schema import AppConfig
from ..domain.models import DocumentRecord, FileVersionRecord, IndexJobRecord
from ..domain.enums import DocumentStatus
from ..observability.logging import get_logger
from ..domain.exceptions import DatabaseConnectionError, DatabaseMigrationError


logger = get_logger(__name__)


# Database schema version
SCHEMA_VERSION = 1


# DDL Statements for tables per architecture Section 7.1
DDL_STATEMENTS = [
    # documents table
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source_path TEXT UNIQUE NOT NULL,
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        md5_hash TEXT NOT NULL,
        modified_at TEXT NOT NULL,
        active_version_id TEXT,
        status TEXT NOT NULL DEFAULT 'discovered',
        error_message TEXT,
        parser_version TEXT NOT NULL DEFAULT '1',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    
    # file_versions table (MD5 table per requirements FR-ING-003, FR-ING-004)
    """
    CREATE TABLE IF NOT EXISTS file_versions (
        version_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        md5_hash TEXT NOT NULL,
        index_profile_id TEXT NOT NULL,
        parser_version TEXT NOT NULL DEFAULT '1',
        chunk_count INTEGER,
        status TEXT NOT NULL DEFAULT 'discovered',
        indexed_at TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    )
    """,
    
    # index_jobs table
    """
    CREATE TABLE IF NOT EXISTS index_jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'running',
        discovered_count INTEGER NOT NULL DEFAULT 0,
        new_count INTEGER NOT NULL DEFAULT 0,
        changed_count INTEGER NOT NULL DEFAULT 0,
        deleted_count INTEGER NOT NULL DEFAULT 0,
        skipped_count INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        started_at TEXT NOT NULL,
        finished_at TEXT
    )
    """,
    
    # Indexes for performance
    "CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)",
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
    "CREATE INDEX IF NOT EXISTS idx_documents_active_version ON documents(active_version_id)",
    "CREATE INDEX IF NOT EXISTS idx_file_versions_document_id ON file_versions(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_file_versions_md5_profile ON file_versions(md5_hash, index_profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_file_versions_status ON file_versions(status)",
    "CREATE INDEX IF NOT EXISTS idx_index_jobs_started_at ON index_jobs(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_index_jobs_status ON index_jobs(status)",
    
    # Unique constraint for file_versions per architecture Section 7.1
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_file_versions_doc_md5_profile ON file_versions(document_id, md5_hash, index_profile_id)",
]


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode."""
    
    def __init__(self, db_path: Path, timeout: float = 30.0):
        self.db_path = db_path
        self.timeout = timeout
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection with WAL mode."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.timeout,
                check_same_thread=False,
            )
            # Enable WAL mode for concurrent reads
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Optimize for performance
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-32768")  # 32MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            # Return rows as dictionaries
            conn.row_factory = sqlite3.Row
            
            self._local.connection = conn
        
        return self._local.connection
    
    def initialize(self) -> None:
        """Initialize database schema (create tables and indexes)."""
        with self._init_lock:
            if self._initialized:
                return
            
            conn = self._get_connection()
            
            try:
                from .migrations.runner import MigrationRunner
                runner = MigrationRunner(conn)
                applied = runner.apply_pending()
                
                self._initialized = True
                logger.info("database_initialized", "Database initialized", 
                           db_path=str(self.db_path), applied_migrations=applied)
                
            except sqlite3.Error as e:
                conn.rollback()
                logger.error("database_init_failed", "Database initialization failed",
                           db_path=str(self.db_path), error_code=type(e).__name__)
                raise DatabaseConnectionError(self.db_path, e) from e
    
    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run database migrations from current version to target."""
        migrations = {
            1: self._migration_v1,
        }
        
        for version in range(from_version + 1, SCHEMA_VERSION + 1):
            if version in migrations:
                try:
                    logger.info("running_migration", f"Running migration v{version}", version=version)
                    migrations[version](conn)
                except Exception as e:
                    logger.error("migration_failed", f"Migration v{version} failed", 
                               version=version, error_code=type(e).__name__)
                    raise DatabaseMigrationError(version, e) from e
    
    def _migration_v1(self, conn: sqlite3.Connection) -> None:
        """Initial schema migration - tables already created in DDL."""
        pass  # Tables created in DDL_STATEMENTS
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    @contextmanager
    def read_only(self):
        """Context manager for read-only operations."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            raise
    
    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def close_all(self) -> None:
        """Close all connections (for cleanup)."""
        self.close()
    
    def vacuum(self) -> None:
        """Run VACUUM to reclaim space."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("database_vacuumed", "Database vacuumed", db_path=str(self.db_path))
    
    def backup(self, backup_path: Path) -> None:
        """Create backup of database."""
        conn = self._get_connection()
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            conn.backup(backup_conn)
            logger.info("database_backed_up", "Database backed up", 
                       db_path=str(self.db_path), backup_path=str(backup_path))
        finally:
            backup_conn.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_database_manager(config: Optional[AppConfig] = None) -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:
                if config is None:
                    raise ValueError("Database manager not initialized and no config provided")
                db_path = config.paths.metadata_db
                _db_manager = DatabaseManager(db_path)
                _db_manager.initialize()
    
    return _db_manager


def set_database_manager(manager: DatabaseManager) -> None:
    """Set global database manager (for testing)."""
    global _db_manager
    _db_manager = manager


def reset_database_manager() -> None:
    """Reset global database manager (for testing)."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all()
    _db_manager = None


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string for storage."""
    return dt.isoformat()


def iso_to_datetime(iso_str: str) -> datetime:
    """Convert ISO format string to datetime."""
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def uuid_to_str(u: UUID) -> str:
    """Convert UUID to string for storage."""
    return str(u)


def str_to_uuid(s: str) -> UUID:
    """Convert string to UUID."""
    return UUID(s)