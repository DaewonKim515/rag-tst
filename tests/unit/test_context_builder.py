"""
Unit tests for ContextBuilder in retrieval/context_builder.py.
"""

from uuid import uuid4
from rag_app.config.schema import AppConfig
from rag_app.domain.models import SearchHit
from rag_app.retrieval.context_builder import ContextBuilder


def test_context_builder_formatting():
    config = AppConfig(environment="test")
    builder = ContextBuilder(config)

    hit1 = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        index_profile_id="p1",
        chunk_index=0,
        content="제품 설치 가이드 본문 내용입니다.",
        source_path="manual/product.pdf",
        file_name="product.pdf",
        file_type=".pdf",
        page_number=12,
        section_title="3. 설치",
        score=0.9,
    )

    context_str, included = builder.build_context([hit1])

    assert len(included) == 1
    assert "[SOURCE S1]" in context_str
    assert "file: manual/product.pdf" in context_str
    assert "location: page 12" in context_str
    assert "section: 3. 설치" in context_str
    assert "제품 설치 가이드 본문 내용입니다." in context_str
    assert "[/SOURCE S1]" in context_str
