"""
Unit tests for AnswerService and CitationVerifier in generation/answer_service.py and citations.py.
"""

from datetime import datetime
from uuid import uuid4
import pytest

from rag_app.config.schema import AppConfig
from rag_app.domain.models import SearchHit, DocumentRecord, FileVersionRecord
from rag_app.domain.enums import DocumentStatus
from rag_app.persistence.sqlite import DatabaseManager
from rag_app.persistence.repositories import DocumentRepository, FileVersionRepository
from rag_app.generation.citations import CitationVerifier
from rag_app.generation.answer_service import AnswerService
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
    db_path = tmp_path / "answer_test.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    return manager


def test_citation_verifier():
    verifier = CitationVerifier()

    hit1 = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        index_profile_id="p1",
        chunk_index=0,
        content="내용1",
        score=0.9,
        source_path="doc1.pdf",
        file_name="doc1.pdf",
        file_type=".pdf",
        page_number=5,
        section_title="개요",
    )

    answer = "이것은 문서에 기반한 답변입니다. [S1] 또한 존재하지 않는 환각 인용 [S99] 입니다."
    cleaned, citations = verifier.verify_and_clean(answer, [hit1])

    assert len(citations) == 1
    assert citations[0].citation_id == "S1"
    assert citations[0].source_path == "doc1.pdf"
    assert "[S99]" not in cleaned


def test_answer_service_full_flow(test_db: DatabaseManager):
    config = AppConfig(environment="test")
    doc_repo = DocumentRepository(test_db)
    ver_repo = FileVersionRepository(test_db)

    doc_id = uuid4()
    v_id = uuid4()

    doc = DocumentRecord(
        document_id=doc_id,
        source_path="guide.pdf",
        file_name="guide.pdf",
        file_type=".pdf",
        file_size=500,
        md5_hash="md5_guide",
        modified_at=datetime.now(),
        active_version_id=v_id,
        status=DocumentStatus.ACTIVE,
    )
    doc_repo.create(doc)

    ver = FileVersionRecord(
        version_id=v_id,
        document_id=doc_id,
        md5_hash="md5_guide",
        index_profile_id="p1",
        parser_version="1",
        status=DocumentStatus.ACTIVE,
    )
    ver_repo.create(ver)

    hit = SearchHit(
        chunk_id=uuid4(),
        document_id=doc_id,
        version_id=v_id,
        index_profile_id="p1",
        chunk_index=0,
        content="서비스 설치 절차 안내입니다.",
        source_path="guide.pdf",
        file_name="guide.pdf",
        file_type=".pdf",
        page_number=1,
        section_title="설치",
        score=0.95,
    )

    mock_retriever = DenseRetriever(
        config=config,
        embedder=MockEmbedder(),
        vector_store=MockVectorStore([hit]),
        db_manager=test_db,
    )

    service = AnswerService(
        config=config,
        retriever=mock_retriever,
        db_manager=test_db,
    )

    result = service.answer_question("설치 절차를 알려주세요.")

    assert result.status == "success"
    assert len(result.sources) == 1
    assert len(result.citations) == 1
    assert result.citations[0].source_path == "guide.pdf"
    assert result.total_time_ms > 0
