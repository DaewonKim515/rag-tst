"""
Plain text document parser with automatic encoding detection.
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


class TextParser(DocumentParser):
    """Parser for plain text files (.txt, .log, .csv)."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.supported_exts = {".txt", ".log", ".csv"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_exts
    
    def parse(self, file_path: Path, document_id: UUID, md5_hash: str) -> ParsedDocument:
        """Parse text file and extract paragraph blocks."""
        file_path = file_path.resolve()
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        try:
            encoding = "utf-8"
            try:
                import chardet
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected.get("encoding") or "utf-8"
            except (ImportError, Exception):
                with open(file_path, "rb") as f:
                    raw_data = f.read()
            
            # Decode text
            try:
                content = raw_data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                content = raw_data.decode("utf-8", errors="replace")
                encoding = "utf-8"
                
        except Exception as e:
            raise ParserError(f"Failed to read/decode text file {file_path.name}: {e}") from e
        
        # Split into paragraph blocks
        content_normalized = content.replace("\r\n", "\n")
        raw_paragraphs = content_normalized.split("\n\n")
        blocks = []
        sequence = 0
        
        for p in raw_paragraphs:
            p_text = p.strip()
            if not p_text:
                continue
            
            blocks.append(ParsedBlock(
                block_type=BlockType.PARAGRAPH,
                text=p_text,
                sequence=sequence,
                metadata={"encoding": encoding},
            ))
            sequence += 1
        
        # Fallback if no non-empty blocks found
        if not blocks:
            blocks.append(ParsedBlock(
                block_type=BlockType.PARAGRAPH,
                text=content.strip() or "(empty document)",
                sequence=0,
                metadata={"encoding": encoding},
            ))
        
        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=FileType.TXT,
            file_size=stat.st_size,
            md5_hash=md5_hash,
            modified_at=modified_at,
            blocks=tuple(blocks),
            language="ko",
        )
    
    def supported_extensions(self) -> set[str]:
        return {".txt", ".log"}
