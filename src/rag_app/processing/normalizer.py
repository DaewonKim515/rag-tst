"""
Text normalization for document processing.

Removes control characters, normalizes whitespace, handles repeated newlines,
and preserves paragraph boundaries for downstream chunking.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from ..config.schema import AppConfig
from ..domain.models import ParsedBlock, ParsedDocument
from ..domain.enums import BlockType
from ..observability.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Configuration for text normalization."""
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    normalize_unicode: bool = True
    collapse_repeated_newlines: bool = True
    max_consecutive_newlines: int = 2
    preserve_paragraph_boundaries: bool = True
    trim_lines: bool = True


DEFAULT_CONFIG = NormalizationConfig()


class TextNormalizer:
    """
    Normalizes text content from parsed documents.
    
    Handles:
    - Control character removal (except \n, \t)
    - Unicode normalization (NFC)
    - Whitespace normalization (spaces, tabs)
    - Repeated newline collapsing
    - Paragraph boundary preservation
    """
    
    def __init__(self, config: NormalizationConfig = DEFAULT_CONFIG):
        self.config = config
        
        # Compile regex patterns
        self._control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200b\u200c\u200d\ufeff]')
        self._whitespace_pattern = re.compile(r'[ \t]+')
        self._repeated_newline_pattern = re.compile(r'\n{3,}')
        self._trailing_spaces_pattern = re.compile(r'[ \t]+$', re.MULTILINE)
        self._leading_spaces_pattern = re.compile(r'^[ \t]+', re.MULTILINE)
    
    def normalize(self, document: ParsedDocument) -> ParsedDocument:
        """
        Normalize all text blocks in a document.
        
        Args:
            document: ParsedDocument to normalize.
            
        Returns:
            New ParsedDocument with normalized blocks.
        """
        normalized_blocks = []
        
        for block in document.blocks:
            normalized_block = self._normalize_block(block)
            if normalized_block is not None:
                normalized_blocks.append(normalized_block)
        
        return ParsedDocument(
            document_id=document.document_id,
            source_path=document.source_path,
            file_name=document.file_name,
            file_type=document.file_type,
            file_size=document.file_size,
            md5_hash=document.md5_hash,
            modified_at=document.modified_at,
            blocks=tuple(normalized_blocks),
            parser_version=document.parser_version,
            language=document.language,
            ocr_status=document.ocr_status,
            ocr_pages=document.ocr_pages,
        )
    
    def _normalize_block(self, block: ParsedBlock) -> Optional[ParsedBlock]:
        """Normalize a single block's text."""
        text = block.text
        
        if not text or not text.strip():
            return None
        
        # Clean invalid surrogate characters (e.g. \udfc1, \ud800-\udfff)
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

        # Normalize line endings (\r\n or \r to \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove control characters (keep \n, \t)
        if self.config.remove_control_chars:
            text = self._control_char_pattern.sub('', text)
        
        # Normalize whitespace within lines
        if self.config.normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
        
        # Trim lines if configured
        if self.config.trim_lines:
            text = self._trailing_spaces_pattern.sub('', text)
            text = self._leading_spaces_pattern.sub('', text)
        
        # Collapse repeated newlines
        if self.config.collapse_repeated_newlines:
            text = self._repeated_newline_pattern.sub(
                '\n' * self.config.max_consecutive_newlines, text
            )
        
        # Final strip
        text = text.strip()
        
        if not text:
            return None
        
        return ParsedBlock(
            block_type=block.block_type,
            text=text,
            page_number=block.page_number,
            sheet_name=block.sheet_name,
            slide_number=block.slide_number,
            section_path=block.section_path,
            sequence=block.sequence,
            metadata=block.metadata,
        )
    
    def normalize_text(self, text: str) -> str:
        """Normalize a plain text string."""
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        if self.config.remove_control_chars:
            text = self._control_char_pattern.sub('', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
        
        # Trim lines
        if self.config.trim_lines:
            text = self._trailing_spaces_pattern.sub('', text)
            text = self._leading_spaces_pattern.sub('', text)
        
        # Collapse repeated newlines
        if self.config.collapse_repeated_newlines:
            text = self._repeated_newline_pattern.sub(
                '\n' * self.config.max_consecutive_newlines, text
            )
        
        return text.strip()


def create_normalizer(config: AppConfig) -> TextNormalizer:
    """Factory function to create TextNormalizer from AppConfig."""
    return TextNormalizer(DEFAULT_CONFIG)