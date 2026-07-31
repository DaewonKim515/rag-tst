"""
Index scheduler for background periodic document scanning with singleton lock.
"""

import threading
import time
from typing import Optional
from ..config.schema import AppConfig
from .coordinator import IndexCoordinator
from ..persistence.sqlite import DatabaseManager, get_database_manager
from ..observability.logging import get_logger

logger = get_logger(__name__)


class IndexScheduler:
    """
    Background scheduler for document scanning and indexing.
    Enforces a singleton lock to prevent overlapping index jobs.
    """

    def __init__(
        self,
        config: AppConfig,
        coordinator: Optional[IndexCoordinator] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.config = config
        self.indexing_config = config.indexing
        self.enabled = self.indexing_config.enabled
        self.scan_on_start = self.indexing_config.scan_on_start
        self.scan_interval_seconds = self.indexing_config.scan_interval_seconds

        self.db_manager = db_manager or get_database_manager(config)
        self.coordinator = coordinator or IndexCoordinator(config=config, db_manager=self.db_manager)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background scheduler thread if enabled."""
        if not self.enabled:
            logger.info("scheduler_disabled", "Indexing scheduler is disabled in config")
            return

        if self._thread and self._thread.is_alive():
            logger.warning("scheduler_already_running", "Scheduler is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="IndexSchedulerThread", daemon=True)
        self._thread.start()
        logger.info(
            "scheduler_started",
            f"Index scheduler started (interval: {self.scan_interval_seconds}s, scan_on_start: {self.scan_on_start})",
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background scheduler thread gracefully."""
        if not self._thread or not self._thread.is_alive():
            return

        logger.info("scheduler_stopping", "Stopping index scheduler...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        logger.info("scheduler_stopped", "Index scheduler stopped")

    def run_once(self, job_type: str = "manual") -> Optional[dict]:
        """
        Trigger a single index job immediately with singleton lock protection.

        Args:
            job_type: Label for job type (e.g., 'startup', 'scheduled', 'manual').

        Returns:
            Job summary dict if executed, None if skipped due to active lock.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            logger.warning("scheduler_job_skipped", f"Job skipped ({job_type}): Previous index job is still running")
            return None

        try:
            logger.info("scheduler_job_started", f"Starting index job ({job_type})")
            summary = self.coordinator.run_indexing_job(job_type=job_type)
            return summary
        except Exception as e:
            logger.error("scheduler_job_error", f"Index job error ({job_type}): {e}")
            raise
        finally:
            self._lock.release()

    def _run_loop(self) -> None:
        """Main background thread loop."""
        # 1. Handle scan_on_start
        if self.scan_on_start:
            self.run_once(job_type="startup")

        # 2. Periodic loop
        while not self._stop_event.is_set():
            # Wait for interval or stop event
            stopped = self._stop_event.wait(timeout=float(self.scan_interval_seconds))
            if stopped:
                break

            self.run_once(job_type="scheduled")
