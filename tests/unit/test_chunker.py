"""
Unit tests for Chunker in processing/chunker.py.
"""

from datetime import datetime
from uuid import uuid4
import pytest

from rag_app.domain.enums import BlockType, FileType
from rag_app.domain.models import ParsedBlock, ParsedDocument
from rag_app.processing.chunker import Chunker, ChunkingConfig, TokenCounter, create_chunker


def test_token_counter_approximation():
    counter = TokenCounter(model_name="non_existent_model_id_for_fallback")
    text_ko = "안녕하세요 서비스 테스트입니다."
    text_en = "Hello world this is a test service."
    
    cnt_ko = counter.count(text_ko)
    cnt_en = counter.count(text_en)
    
    assert cnt_ko > 0
    assert cnt_en > 0


def test_chunker_basic_flow():
    config = ChunkingConfig(
        target_tokens=20,
        overlap_tokens=5,
        max_tokens=30,
        min_chunk_tokens=2,
        preserve_page_boundary=True,
    )
    chunker = Chunker(config=config)
    
    doc_id = uuid4()
    version_id = uuid4()
    
    blocks = (
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            text="첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.",
            page_number=1,
            section_path=("섹션1",),
            sequence=0,
        ),
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            text="네 번째 문장입니다. 다섯 번째 문장입니다. 여섯 번째 문장입니다.",
            page_number=1,
            section_path=("섹션1",),
            sequence=1,
        ),
    )
    
    doc = ParsedDocument(
        document_id=doc_id,
        source_path="doc.pdf",
        file_name="doc.pdf",
        file_type=FileType.PDF,
        file_size=500,
        md5_hash="md5hash123",
        modified_at=datetime.now(),
        blocks=blocks,
    )
    
    chunks = chunker.chunk_document(
        document=doc,
        document_id=doc_id,
        version_id=version_id,
        index_profile_id="profile_1",
    )
    
    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.document_id == doc_id
        assert c.version_id == version_id
        assert c.chunk_index == idx
        assert c.page_number == 1
        assert c.section_title == "섹션1"
        assert c.token_count > 0


def test_chunker_table_linearization():
    config = ChunkingConfig(min_chunk_tokens=1)
    chunker = Chunker(config=config)
    doc_id = uuid4()
    version_id = uuid4()
    
    table_text = "| 이름 | 나이 | 직업 |\n|---|---|---|\n| 홍길동 | 30 | 개발자 |\n| 이순신 | 40 | 장군 |"
    blocks = (
        ParsedBlock(
            block_type=BlockType.TABLE,
            text=table_text,
            page_number=1,
            sequence=0,
        ),
    )
    
    doc = ParsedDocument(
        document_id=doc_id,
        source_path="table_doc.pdf",
        file_name="table_doc.pdf",
        file_type=FileType.PDF,
        file_size=500,
        md5_hash="table_md5",
        modified_at=datetime.now(),
        blocks=blocks,
    )
    
    chunks = chunker.chunk_document(
        document=doc,
        document_id=doc_id,
        version_id=version_id,
        index_profile_id="profile_table",
    )
    
    assert len(chunks) == 1
    content = chunks[0].content
    assert "[Header]" in content
    assert "[Row 1]" in content
    assert "홍길동" in content
