"""
Migration runner for applying sequential database migrations.
"""

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

from . import v001_initial_schema
from ...observability.logging import get_logger

logger = get_logger(__name__)

MIGRATIONS = [
    v001_initial_schema,
]


class MigrationRunner:
    """Manages database schema versions and executes pending migrations."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_schema_version_table()

    def _ensure_schema_version_table(self) -> None:
        """Create schema_version table if it does not exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """)
        self.conn.commit()

    def get_current_version(self) -> int:
        """Get highest applied schema version."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version;")
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 0

    def apply_pending(self) -> List[int]:
        """Apply all unapplied migrations in version order."""
        current_v = self.get_current_version()
        applied = []

        for mod in sorted(MIGRATIONS, key=lambda m: m.VERSION):
            if mod.VERSION > current_v:
                logger.info("applying_migration", f"Applying migration v{mod.VERSION}: {mod.DESCRIPTION}")
                mod.upgrade(self.conn)
                
                # Record migration
                cursor = self.conn.cursor()
                cursor.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?);",
                    (mod.VERSION, datetime.now(timezone.utc).isoformat()),
                )
                self.conn.commit()
                applied.append(mod.VERSION)
                logger.info("migration_applied", f"Successfully applied migration v{mod.VERSION}")

        return applied
