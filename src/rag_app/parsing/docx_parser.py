"""
DOCX document parser using python-docx.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config.schema import AppConfig
from ..domain.enums import BlockType, FileType
from ..domain.exceptions import ParserError
from ..domain.models import ParsedBlock, ParsedDocument
from .registry import DocumentParser


class DocxParser(DocumentParser):
    """Parser for Microsoft Word documents (.docx)."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.supported_exts = {".docx"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_exts
    
    def parse(self, file_path: Path, document_id: UUID, md5_hash: str) -> ParsedDocument:
        """Parse DOCX document extracting paragraphs, headings, and tables."""
        file_path = file_path.resolve()
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        try:
            import docx
        except ImportError:
            raise ParserError("python-docx is required for DOCX parsing but is not installed.")
        
        try:
            doc = docx.Document(str(file_path))
        except Exception as e:
            raise ParserError(f"Failed to open DOCX file {file_path.name}: {e}") from e
        
        blocks = []
        sequence = 0
        current_section_path: list[str] = []
        
        # Iterate over paragraphs and tables
        for element in doc.element.body:
            tag = element.tag.split("}")[-1]
            
            if tag == "p":
                # Paragraph
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()
                if not text:
                    continue
                
                style_name = p.style.name if p.style else ""
                
                if style_name.startswith("Heading"):
                    try:
                        level = int(style_name.replace("Heading", "").strip())
                    except ValueError:
                        level = 1
                    
                    while len(current_section_path) >= level:
                        current_section_path.pop()
                    current_section_path.append(text)
                    
                    blocks.append(ParsedBlock(
                        block_type=BlockType.HEADING if level > 1 else BlockType.TITLE,
                        text=text,
                        section_path=tuple(current_section_path),
                        sequence=sequence,
                    ))
                    sequence += 1
                else:
                    blocks.append(ParsedBlock(
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        section_path=tuple(current_section_path),
                        sequence=sequence,
                    ))
                    sequence += 1
                    
            elif tag == "tbl":
                # Table
                t = docx.table.Table(element, doc)
                table_lines = []
                for row in t.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    table_lines.append(" | ".join(cells))
                
                table_text = "\n".join(table_lines).strip()
                if table_text:
                    blocks.append(ParsedBlock(
                        block_type=BlockType.TABLE,
                        text=table_text,
                        section_path=tuple(current_section_path),
                        sequence=sequence,
                    ))
                    sequence += 1
        
        if not blocks:
            blocks.append(ParsedBlock(
                block_type=BlockType.PARAGRAPH,
                text="(empty docx document)",
                sequence=0,
            ))
        
        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=FileType.DOCX,
            file_size=stat.st_size,
            md5_hash=md5_hash,
            modified_at=modified_at,
            blocks=tuple(blocks),
            language="ko",
        )
    
    def supported_extensions(self) -> set[str]:
        return {".docx"}
