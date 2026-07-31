"""
Migration v001: Initial schema setup.
"""

import sqlite3

VERSION = 1
DESCRIPTION = "Initial schema for documents, file_versions, and index_jobs"


def upgrade(conn: sqlite3.Connection) -> None:
    """Apply v001 migration."""
    cursor = conn.cursor()
    
    # 1. Documents Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source_path TEXT UNIQUE NOT NULL,
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        md5_hash TEXT NOT NULL,
        modified_at TEXT NOT NULL,
        active_version_id TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        error_message TEXT,
        parser_version TEXT DEFAULT '1.0',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)

    # Indexes for documents
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);")

    # 2. File Versions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS file_versions (
        version_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        md5_hash TEXT NOT NULL,
        index_profile_id TEXT NOT NULL,
        parser_version TEXT NOT NULL DEFAULT '1.0',
        chunk_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'staging',
        indexed_at TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    );
    """)

    # Indexes for file_versions
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_versions_doc_id ON file_versions(document_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_versions_status ON file_versions(status);")

    # 3. Index Jobs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS index_jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'running',
        discovered_count INTEGER NOT NULL DEFAULT 0,
        new_count INTEGER NOT NULL DEFAULT 0,
        changed_count INTEGER NOT NULL DEFAULT 0,
        deleted_count INTEGER NOT NULL DEFAULT 0,
        unchanged_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        completed_at TEXT
    );
    """)

    conn.commit()
