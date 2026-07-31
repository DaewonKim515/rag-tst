"""
Markdown parser preserving heading hierarchy, code blocks, and lists.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config.schema import AppConfig
from ..domain.enums import BlockType, FileType
from ..domain.exceptions import ParserError
from ..domain.models import ParsedBlock, ParsedDocument
from .registry import DocumentParser


class MarkdownParser(DocumentParser):
    """Parser for Markdown files (.md, .markdown)."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.supported_exts = {".md", ".markdown"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_exts
    
    def parse(self, file_path: Path, document_id: UUID, md5_hash: str) -> ParsedDocument:
        """Parse markdown file into structured blocks with heading hierarchy."""
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
            
            try:
                content = raw_data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                content = raw_data.decode("utf-8", errors="replace")
        except Exception as e:
            raise ParserError(f"Failed to read markdown file {file_path.name}: {e}") from e
        
        blocks = []
        sequence = 0
        current_section_stack: list[tuple[int, str]] = []  # (level, title)
        
        # Regex patterns
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        code_block_fence = re.compile(r'^```')
        list_item_pattern = re.compile(r'^(\*|-|\+|\d+\.)\s+(.+)$')
        
        lines = content.splitlines()
        idx = 0
        n = len(lines)
        
        while idx < n:
            line = lines[idx]
            stripped = line.strip()
            
            if not stripped:
                idx += 1
                continue
            
            # Code block handling
            if code_block_fence.match(stripped):
                lang = stripped.lstrip('`').strip()
                code_lines = []
                idx += 1
                while idx < n and not code_block_fence.match(lines[idx].strip()):
                    code_lines.append(lines[idx])
                    idx += 1
                if idx < n:
                    idx += 1  # closing fence
                
                code_text = "\n".join(code_lines)
                if code_text.strip():
                    section_path = tuple(t for _, t in current_section_stack)
                    blocks.append(ParsedBlock(
                        block_type=BlockType.CODE,
                        text=code_text,
                        section_path=section_path,
                        sequence=sequence,
                        metadata={"language": lang},
                    ))
                    sequence += 1
                continue
            
            # Heading handling
            h_match = heading_pattern.match(stripped)
            if h_match:
                level = len(h_match.group(1))
                h_text = h_match.group(2).strip()
                
                # Update heading stack
                while current_section_stack and current_section_stack[-1][0] >= level:
                    current_section_stack.pop()
                current_section_stack.append((level, h_text))
                
                section_path = tuple(t for _, t in current_section_stack)
                b_type = BlockType.TITLE if level == 1 else BlockType.HEADING
                blocks.append(ParsedBlock(
                    block_type=b_type,
                    text=h_text,
                    section_path=section_path,
                    sequence=sequence,
                    metadata={"heading_level": level},
                ))
                sequence += 1
                idx += 1
                continue
            
            # List item handling
            if list_item_pattern.match(stripped):
                list_lines = []
                while idx < n and list_item_pattern.match(lines[idx].strip()):
                    list_lines.append(lines[idx].strip())
                    idx += 1
                
                list_text = "\n".join(list_lines)
                section_path = tuple(t for _, t in current_section_stack)
                blocks.append(ParsedBlock(
                    block_type=BlockType.LIST,
                    text=list_text,
                    section_path=section_path,
                    sequence=sequence,
                ))
                sequence += 1
                continue
            
            # General Paragraph
            para_lines = []
            while idx < n:
                curr_strip = lines[idx].strip()
                if not curr_strip or heading_pattern.match(curr_strip) or code_block_fence.match(curr_strip) or list_item_pattern.match(curr_strip):
                    break
                para_lines.append(curr_strip)
                idx += 1
            
            para_text = " ".join(para_lines).strip()
            if para_text:
                section_path = tuple(t for _, t in current_section_stack)
                blocks.append(ParsedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=para_text,
                    section_path=section_path,
                    sequence=sequence,
                ))
                sequence += 1
        
        if not blocks:
            blocks.append(ParsedBlock(
                block_type=BlockType.PARAGRAPH,
                text=content.strip() or "(empty markdown)",
                sequence=0,
            ))
        
        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=FileType.MD,
            file_size=stat.st_size,
            md5_hash=md5_hash,
            modified_at=modified_at,
            blocks=tuple(blocks),
            language="ko",
        )
    
    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown"}
