"""
Unit tests for BgeReranker in models/reranker.py.
"""

from uuid import uuid4
from rag_app.config.schema import AppConfig
from rag_app.domain.models import SearchHit
from rag_app.models.reranker import BgeReranker


def test_bge_reranker_fallback():
    config = AppConfig(environment="test")
    reranker = BgeReranker(config)

    hit1 = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        index_profile_id="p1",
        chunk_index=0,
        content="낮은 dense 점수의 아주 관련성 높은 본문",
        source_path="doc1.pdf",
        file_name="doc1.pdf",
        file_type=".pdf",
        score=0.6,
    )
    hit2 = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        index_profile_id="p1",
        chunk_index=0,
        content="높은 dense 점수의 덜 관련성 있는 본문",
        source_path="doc2.pdf",
        file_name="doc2.pdf",
        file_type=".pdf",
        score=0.9,
    )

    hits = [hit1, hit2]
    result = reranker.rerank("관련성 테스트 질문", hits, top_k=2)

    assert len(result) == 2
    # In fallback mode, sorted by score descending (hit2 first)
    assert result[0].score == 0.9
