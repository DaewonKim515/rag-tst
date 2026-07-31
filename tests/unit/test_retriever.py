"""
Unit tests for DenseRetriever in retrieval/retriever.py.
"""

from datetime import datetime
from uuid import uuid4
import pytest

from rag_app.config.schema import AppConfig
from rag_app.domain.models import SearchHit, DocumentRecord, FileVersionRecord
from rag_app.domain.enums import DocumentStatus
from rag_app.persistence.sqlite import DatabaseManager
from rag_app.persistence.repositories import DocumentRepository, FileVersionRepository
from rag_app.retrieval.retriever import DenseRetriever


class MockEmbedder:
    def embed_query(self, text: str):
        return [0.1] * 1024


class MockVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query):
        return self._hits


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "retriever_test.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    return manager


def test_retriever_active_version_filtering(test_db: DatabaseManager):
    config = AppConfig(environment="test")
    doc_repo = DocumentRepository(test_db)
    ver_repo = FileVersionRepository(test_db)

    # 1. Create doc1 with active_version_id = v1
    doc1_id = uuid4()
    v1_id = uuid4()
    doc1 = DocumentRecord(
        document_id=doc1_id,
        source_path="doc1.pdf",
        file_name="doc1.pdf",
        file_type=".pdf",
        file_size=100,
        md5_hash="md5_v1",
        modified_at=datetime.now(),
        active_version_id=v1_id,
        status=DocumentStatus.ACTIVE,
    )
    doc_repo.create(doc1)

    v1 = FileVersionRecord(
        version_id=v1_id,
        document_id=doc1_id,
        md5_hash="md5_v1",
        index_profile_id="p1",
        parser_version="1",
        status=DocumentStatus.ACTIVE,
    )
    ver_repo.create(v1)

    # Old version v0 for doc1
    v0_id = uuid4()
    v0 = FileVersionRecord(
        version_id=v0_id,
        document_id=doc1_id,
        md5_hash="md5_v0",
        index_profile_id="p1",
        parser_version="1",
        status=DocumentStatus.DELETED,
    )
    ver_repo.create(v0)

    # Prepare Mock Vector Store Hits (containing v0 hit and v1 hit)
    hit_old = SearchHit(
        chunk_id=uuid4(),
        document_id=doc1_id,
        version_id=v0_id,
        index_profile_id="p1",
        chunk_index=0,
        content="구버전 내용입니다.",
        source_path="doc1.pdf",
        file_name="doc1.pdf",
        file_type=".pdf",
        score=0.95,
    )
    hit_active = SearchHit(
        chunk_id=uuid4(),
        document_id=doc1_id,
        version_id=v1_id,
        index_profile_id="p1",
        chunk_index=0,
        content="신버전(활성) 내용입니다.",
        source_path="doc1.pdf",
        file_name="doc1.pdf",
        file_type=".pdf",
        score=0.90,
    )

    mock_store = MockVectorStore([hit_old, hit_active])
    mock_embedder = MockEmbedder()

    retriever = DenseRetriever(
        config=config,
        embedder=mock_embedder,
        vector_store=mock_store,
        db_manager=test_db,
    )

    results = retriever.retrieve("테스트 질의")
    assert len(results) == 1
    assert results[0].version_id == v1_id
    assert results[0].content == "신버전(활성) 내용입니다."
