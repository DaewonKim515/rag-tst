"""
Integration tests for E2E Document RAG Pipeline (Scan -> Parse -> Chunk -> Swap -> Answer).
"""

from pathlib import Path
from uuid import uuid4
import pytest

from rag_app.config.schema import AppConfig
from rag_app.persistence.sqlite import DatabaseManager
from rag_app.persistence.repositories import DocumentRepository, FileVersionRepository
from rag_app.ingestion.coordinator import IndexCoordinator
from rag_app.ingestion.scheduler import IndexScheduler
from rag_app.generation.answer_service import AnswerService


@pytest.fixture
def integration_env(tmp_path: Path):
    """Setup an isolated temporary integration environment."""
    doc_dir = tmp_path / "document"
    doc_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "integration_metadata.db"

    config = AppConfig(environment="test")
    config.paths.document_root = doc_dir
    config.paths.metadata_db = db_path

    db_manager = DatabaseManager(db_path)
    db_manager.initialize()

    return {
        "config": config,
        "doc_dir": doc_dir,
        "db_manager": db_manager,
    }


def test_full_rag_pipeline_integration(integration_env):
    config = integration_env["config"]
    doc_dir = integration_env["doc_dir"]
    db_manager = integration_env["db_manager"]

    doc_repo = DocumentRepository(db_manager)
    ver_repo = FileVersionRepository(db_manager)

    # 1. Add Initial Files
    file1 = doc_dir / "service_guide.txt"
    file1.write_text("서비스 설치 및 환경 설정 가이드입니다.\n\n기본 포트는 8080을 사용합니다.", encoding="utf-8")

    file2 = doc_dir / "faq.md"
    file2.write_text("# 자주 묻는 질문\n\nQ: 비밀번호를 잊으셨나요?\nA: 관리자에게 문의하세요.", encoding="utf-8")

    coordinator = IndexCoordinator(config=config, db_manager=db_manager)

    # 2. First Indexing Scan
    summary1 = coordinator.run_indexing_job(job_type="initial_test")
    assert summary1["discovered"] == 2
    assert summary1["success_count"] == 2

    # Verify SQLite active versions created
    docs = doc_repo.get_all()
    assert len(docs) == 2
    for d in docs:
        assert d.active_version_id is not None
        active_ver = ver_repo.get_by_id(d.active_version_id)
        assert active_ver is not None
        assert active_ver.status.value == "active"

    # Save initial version ID of file1
    doc1_record = doc_repo.get_by_source_path("service_guide.txt")
    initial_v1_id = doc1_record.active_version_id

    # 3. Update file1 content (Triggers MD5 change & Atomic Version Swap)
    file1.write_text("업데이트된 서비스 가이드입니다.\n\n변경된 포트는 9090입니다.", encoding="utf-8")

    summary2 = coordinator.run_indexing_job(job_type="update_test")
    assert summary2["changed"] == 1
    assert summary2["success_count"] == 1

    # Verify Version Swap
    doc1_updated = doc_repo.get_by_source_path("service_guide.txt")
    new_v1_id = doc1_updated.active_version_id
    assert new_v1_id != initial_v1_id

    # 4. Scheduler Singleton Lock Verification
    scheduler = IndexScheduler(config=config, coordinator=coordinator, db_manager=db_manager)
    
    # Run once manually
    run_res = scheduler.run_once(job_type="test_run")
    assert run_res is not None
    assert run_res["discovered"] == 2
    assert run_res["unchanged"] == 2  # No files changed

    # 5. Answer Service Question Processing
    answer_service = AnswerService(config=config, db_manager=db_manager)
    result = answer_service.answer_question("서비스 포트는 몇 번인가요?")

    assert result.status == "success"
    assert len(result.sources) > 0
    assert result.total_time_ms > 0
