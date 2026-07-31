"""
Unit tests for document parsers (PDF, DOCX, TXT, MD).
"""

from uuid import uuid4
import pytest
from pathlib import Path

from rag_app.domain.enums import BlockType, FileType
from rag_app.domain.exceptions import ParserError
from rag_app.parsing.text_parser import TextParser
from rag_app.parsing.md_parser import MarkdownParser
from rag_app.parsing.pdf_parser import PdfParser
from rag_app.parsing.docx_parser import DocxParser


def test_text_parser(tmp_path: Path):
    parser = TextParser()
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("첫 번째 문단입니다.\n\n두 번째 문단입니다.", encoding="utf-8")
    
    doc_id = uuid4()
    doc = parser.parse(sample_file, document_id=doc_id, md5_hash="md5_txt")
    
    assert doc.document_id == doc_id
    assert doc.file_type == FileType.TXT
    assert len(doc.blocks) == 2
    assert doc.blocks[0].text == "첫 번째 문단입니다."
    assert doc.blocks[1].text == "두 번째 문단입니다."


def test_md_parser(tmp_path: Path):
    parser = MarkdownParser()
    sample_file = tmp_path / "sample.md"
    content = "# 제목 1\n\n본문 내용 1입니다.\n\n## 소제목 1.1\n\n본문 내용 2입니다."
    sample_file.write_text(content, encoding="utf-8")
    
    doc_id = uuid4()
    doc = parser.parse(sample_file, document_id=doc_id, md5_hash="md5_md")
    
    assert doc.document_id == doc_id
    assert doc.file_type == FileType.MD
    assert len(doc.blocks) >= 4
    assert doc.blocks[0].block_type == BlockType.TITLE
    assert doc.blocks[0].text == "제목 1"
    assert doc.blocks[2].block_type == BlockType.HEADING
    assert doc.blocks[2].text == "소제목 1.1"


def test_pdf_parser_file_not_found():
    parser = PdfParser()
    with pytest.raises(ParserError):
        parser.parse(Path("non_existent.pdf"), uuid4(), "hash")


def test_docx_parser_file_not_found():
    parser = DocxParser()
    with pytest.raises(ParserError):
        parser.parse(Path("non_existent.docx"), uuid4(), "hash")
