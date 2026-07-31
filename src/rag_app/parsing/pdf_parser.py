"""
PDF parser using PyMuPDF (fitz) with page-level extraction and scanned PDF detection.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config.schema import AppConfig
from ..domain.enums import BlockType, FileType, OCRStatus
from ..domain.exceptions import ParserError
from ..domain.models import ParsedBlock, ParsedDocument
from .registry import DocumentParser


class PdfParser(DocumentParser):
    """Parser for PDF files using PyMuPDF."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.supported_exts = {".pdf"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_exts
    
    def parse(self, file_path: Path, document_id: UUID, md5_hash: str) -> ParsedDocument:
        """Parse PDF document and extract page text and metadata."""
        file_path = file_path.resolve()
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        blocks = []
        sequence = 0
        total_pages = 0
        pages_with_low_text = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                try:
                    page = doc.load_page(page_idx)
                    page_text = page.get_text("text").strip()
                except Exception:
                    page_text = ""

                if len(page_text) < 20:
                    pages_with_low_text.append(page_num)

                if page_text:
                    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                    if not paragraphs:
                        paragraphs = [page_text]
                    for p in paragraphs:
                        blocks.append(ParsedBlock(
                            block_type=BlockType.PARAGRAPH,
                            text=p,
                            page_number=page_num,
                            sequence=sequence,
                        ))
                        sequence += 1
            doc.close()
        except Exception as fitz_err:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    page_num = page_idx + 1
                    try:
                        page_text = (page.extract_text() or "").strip()
                    except Exception:
                        page_text = ""

                    if len(page_text) < 20:
                        pages_with_low_text.append(page_num)

                    if page_text:
                        blocks.append(ParsedBlock(
                            block_type=BlockType.PARAGRAPH,
                            text=page_text,
                            page_number=page_num,
                            sequence=sequence,
                        ))
                        sequence += 1
            except Exception as pypdf_err:
                raise ParserError(f"Failed to parse PDF {file_path.name}: fitz error ({fitz_err}), pypdf error ({pypdf_err})") from pypdf_err
        
        # Scanned PDF evaluation
        ocr_status = OCRStatus.NOT_NEEDED
        if len(pages_with_low_text) == total_pages and total_pages > 0:
            ocr_status = OCRStatus.REQUIRED
        elif len(pages_with_low_text) > 0:
            ocr_status = OCRStatus.PARTIAL
        
        if not blocks:
            blocks.append(ParsedBlock(
                block_type=BlockType.PARAGRAPH,
                text=f"(No text extracted from PDF. OCR status: {ocr_status.value})",
                page_number=1,
                sequence=0,
            ))
        
        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=FileType.PDF,
            file_size=stat.st_size,
            md5_hash=md5_hash,
            modified_at=modified_at,
            blocks=tuple(blocks),
            language="ko",
            ocr_status=ocr_status,
            ocr_pages=tuple(pages_with_low_text),
        )
    
    def supported_extensions(self) -> set[str]:
        return {".pdf"}
