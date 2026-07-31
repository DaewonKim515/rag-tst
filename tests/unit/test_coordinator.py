"""
Unit and integration tests for VersionManager and IndexCoordinator.
"""

from pathlib import Path
from uuid import uuid4
from datetime import datetime
import pytest

from rag_app.config.schema import AppConfig
from rag_app.domain.enums import DocumentStatus
from rag_app.domain.models import DocumentRecord
from rag_app.persistence.sqlite import DatabaseManager
from rag_app.persistence.repositories import DocumentRepository
from rag_app.indexing.version_manager import VersionManager
from rag_app.ingestion.coordinator import IndexCoordinator


@pytest.fixture
def test_db(tmp_path: Path):
    db_path = tmp_path / "test_metadata.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    return manager


def test_version_manager_lifecycle_and_atomic_swap(test_db: DatabaseManager):
    doc_repo = DocumentRepository(test_db)
    vm = VersionManager(test_db)
    
    doc_id = uuid4()
    doc_record = DocumentRecord(
        document_id=doc_id,
        source_path="test_doc.txt",
        file_name="test_doc.txt",
        file_type="txt",
        file_size=100,
        md5_hash="md5v1",
        modified_at=datetime.now(),
    )
    doc_repo.create(doc_record)
    
    # Create version 1
    v1 = vm.create_pending_version(doc_id, "md5v1", "profile_1")
    assert v1.status == DocumentStatus.PENDING
    
    # Activate version 1
    vm.activate_version(doc_id, v1.version_id)
    
    updated_doc = doc_repo.get_by_id(doc_id)
    assert updated_doc.active_version_id == v1.version_id
    assert updated_doc.status == DocumentStatus.ACTIVE
    
    # Create and activate version 2
    v2 = vm.create_pending_version(doc_id, "md5v2", "profile_1")
    vm.activate_version(doc_id, v2.version_id)
    
    updated_doc_v2 = doc_repo.get_by_id(doc_id)
    assert updated_doc_v2.active_version_id == v2.version_id


def test_index_coordinator_job_run(tmp_path: Path, test_db: DatabaseManager):
    config = AppConfig(environment="test")
    doc_dir = tmp_path / "document"
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    (doc_dir / "hello.txt").write_text("안녕하세요 문서 RAG 시스템 인덱싱 테스트입니다.", encoding="utf-8")
    
    config.paths.document_root = doc_dir
    config.paths.metadata_db = tmp_path / "test_metadata.db"
    
    coordinator = IndexCoordinator(config=config, db_manager=test_db)
    summary = coordinator.run_indexing_job(job_type="test")
    
    assert summary["discovered"] == 1
    assert summary["success_count"] == 1
    assert summary["failed_count"] == 0
