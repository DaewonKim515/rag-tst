"""
Unit tests for TextNormalizer in processing/normalizer.py.
"""

from datetime import datetime
from uuid import uuid4
import pytest

from rag_app.domain.enums import BlockType, FileType, OCRStatus
from rag_app.domain.models import ParsedBlock, ParsedDocument
from rag_app.processing.normalizer import NormalizationConfig, TextNormalizer, create_normalizer


def test_normalize_control_characters():
    normalizer = TextNormalizer()
    raw_text = "Hello\x00 World!\x07 This is a test\x1F."
    normalized = normalizer.normalize_text(raw_text)
    assert normalized == "Hello World! This is a test."


def test_normalize_whitespace_and_newlines():
    normalizer = TextNormalizer()
    raw_text = "Line 1   with   spaces\r\n\r\n\r\n\r\nLine 2\n\n\nLine 3"
    normalized = normalizer.normalize_text(raw_text)
    assert normalized == "Line 1 with spaces\n\nLine 2\n\nLine 3"


def test_normalize_parsed_document():
    normalizer = TextNormalizer()
    doc_id = uuid4()
    
    blocks = (
        ParsedBlock(
            block_type=BlockType.TITLE,
            text="   Document   Title \x01  ",
            page_number=1,
            sequence=0,
        ),
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            text="Paragraph 1 line 1.\r\nParagraph 1 line 2.\n\n\nParagraph 1 line 3.",
            page_number=1,
            sequence=1,
        ),
    )
    
    doc = ParsedDocument(
        document_id=doc_id,
        source_path="test.pdf",
        file_name="test.pdf",
        file_type=FileType.PDF,
        file_size=1024,
        md5_hash="abc123md5",
        modified_at=datetime.now(),
        blocks=blocks,
    )
    
    norm_doc = normalizer.normalize(doc)
    
    assert len(norm_doc.blocks) == 2
    assert norm_doc.blocks[0].text == "Document Title"
    assert norm_doc.blocks[1].text == "Paragraph 1 line 1.\nParagraph 1 line 2.\n\nParagraph 1 line 3."


def test_normalize_empty_block_filtering():
    normalizer = TextNormalizer()
    doc_id = uuid4()
    
    blocks = (
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            text="Valid Block",
            page_number=1,
            sequence=0,
        ),
    )
    
    doc = ParsedDocument(
        document_id=doc_id,
        source_path="test.txt",
        file_name="test.txt",
        file_type=FileType.TXT,
        file_size=100,
        md5_hash="hash123",
        modified_at=datetime.now(),
        blocks=blocks,
    )
    
    norm_doc = normalizer.normalize(doc)
    assert len(norm_doc.blocks) == 1
    assert norm_doc.blocks[0].text == "Valid Block"
