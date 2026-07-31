"""
Sentence splitting for Korean, English, and Japanese text.

Uses kiwipiepy for Korean morphological analysis and sentence boundary detection,
with regex-based fallbacks for English and Japanese.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from ..config.schema import AppConfig
from ..domain.enums import BlockType
from ..domain.models import ParsedBlock
from ..observability.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Sentence:
    """A single sentence with metadata."""
    text: str
    start_char: int
    end_char: int
    language: str
    confidence: float = 1.0


class SentenceSplitter(ABC):
    """Abstract base class for sentence splitters."""
    
    @abstractmethod
    def split(self, text: str) -> List[Sentence]:
        """Split text into sentences."""
        pass
    
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass


class KoreanSentenceSplitter(SentenceSplitter):
    """Korean sentence splitter using kiwipiepy."""
    
    def __init__(self):
        self._kiwi = None
        self._init_kiwi()
    
    def _init_kiwi(self):
        """Initialize kiwipiepy analyzer."""
        try:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi()
            # Load user dictionary if needed
            logger.info("korean_splitter_initialized", "Kiwi Korean analyzer initialized")
        except ImportError:
            logger.warning("kiwipiepy_not_available", "kiwipiepy not installed, using regex fallback")
            self._kiwi = None
        except Exception as e:
            logger.error("kiwi_init_failed", f"Failed to initialize Kiwi: {e}")
            self._kiwi = None
    
    def split(self, text: str) -> List[Sentence]:
        """Split Korean text into sentences using kiwipiepy."""
        if not text or not text.strip():
            return []
        
        if self._kiwi is None:
            return self._regex_fallback(text)
        
        try:
            # kiwi.split_into_sents returns list of Sentence objects or tuples
            results = self._kiwi.split_into_sents(text)
            
            sentences = []
            for res in results:
                if hasattr(res, 'text'):
                    sent_text = getattr(res, 'text')
                    start = getattr(res, 'start', 0)
                    end = getattr(res, 'end', len(sent_text))
                elif isinstance(res, (tuple, list)) and len(res) >= 3:
                    sent_text, start, end = res[0], res[1], res[2]
                else:
                    sent_text = str(res)
                    start, end = 0, len(sent_text)

                sent_text = sent_text.strip()
                if sent_text:
                    sentences.append(Sentence(
                        text=sent_text,
                        start_char=start,
                        end_char=end,
                        language="ko",
                        confidence=0.95,
                    ))
            
            return sentences
            
        except Exception as e:
            logger.warning("korean_split_failed", f"Kiwi splitting failed, using fallback: {e}")
            return self._regex_fallback(text)
    
    def _regex_fallback(self, text: str) -> List[Sentence]:
        """Regex-based fallback for Korean sentence splitting."""
        # Korean sentence endings: . ? ! ~ 。 ？ ！
        # Also handle quoted endings
        pattern = r'([^.?!~。？！]+[.?!~。？！]+(?:\s*[\'\"」』]*)?)'
        
        sentences = []
        for match in re.finditer(pattern, text):
            sent_text = match.group(1).strip()
            if sent_text:
                sentences.append(Sentence(
                    text=sent_text,
                    start_char=match.start(),
                    end_char=match.end(),
                    language="ko",
                    confidence=0.7,
                ))
        
        # If no matches, treat whole text as one sentence
        if not sentences and text.strip():
            sentences.append(Sentence(
                text=text.strip(),
                start_char=0,
                end_char=len(text),
                language="ko",
                confidence=0.5,
            ))
        
        return sentences
    
    def supported_languages(self) -> List[str]:
        return ["ko"]


class EnglishSentenceSplitter(SentenceSplitter):
    """English sentence splitter using regex."""
    
    _abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'vs', 'etc',
        'e.g', 'i.e', 'cf', 'al', 'et', 'fig', 'eq', 'no', 'vol',
        'pp', 'ch', 'sec', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
        'aug', 'sep', 'oct', 'nov', 'dec', 'am', 'pm', 'a.m', 'p.m',
    }
    
    def __init__(self):
        # Match sentence end punctuation followed by whitespace or EOF
        self._pattern = re.compile(r'([.?!]+(?:\s+|$))')
    
    def split(self, text: str) -> List[Sentence]:
        """Split English text into sentences."""
        if not text or not text.strip():
            return []
        
        sentences = []
        start = 0
        
        # Iteratively split by sentence punctuation
        for match in self._pattern.finditer(text):
            end = match.end()
            sent_text = text[start:end].strip()
            
            if not sent_text:
                start = end
                continue
            
            # Check if trailing word is an abbreviation
            words = sent_text.split()
            if words:
                last_word = words[-1].lower().rstrip('.!?')
                if last_word in self._abbreviations:
                    # Ignore split at abbreviation
                    continue
            
            sentences.append(Sentence(
                text=sent_text,
                start_char=start,
                end_char=end,
                language="en",
                confidence=0.85,
            ))
            start = end
        
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append(Sentence(
                    text=remaining,
                    start_char=start,
                    end_char=len(text),
                    language="en",
                    confidence=0.7,
                ))
        
        return sentences
    
    def supported_languages(self) -> List[str]:
        return ["en"]


class JapaneseSentenceSplitter(SentenceSplitter):
    """Japanese sentence splitter using regex."""
    
    def __init__(self):
        # Japanese sentence endings: 。 ？ ！ ．
        self._pattern = re.compile(r'([^。？！．]+[。？！．]*)')
    
    def split(self, text: str) -> List[Sentence]:
        """Split Japanese text into sentences."""
        if not text or not text.strip():
            return []
        
        sentences = []
        for match in self._pattern.finditer(text):
            sent_text = match.group(1).strip()
            if sent_text:
                sentences.append(Sentence(
                    text=sent_text,
                    start_char=match.start(),
                    end_char=match.end(),
                    language="ja",
                    confidence=0.85,
                ))
        
        if not sentences and text.strip():
            sentences.append(Sentence(
                text=text.strip(),
                start_char=0,
                end_char=len(text),
                language="ja",
                confidence=0.5,
            ))
        
        return sentences
    
    def supported_languages(self) -> List[str]:
        return ["ja"]


class CompositeSentenceSplitter:
    """
    Composite sentence splitter that routes to language-specific splitters.
    """
    
    def __init__(self):
        self._splitters = {
            "ko": KoreanSentenceSplitter(),
            "en": EnglishSentenceSplitter(),
            "ja": JapaneseSentenceSplitter(),
        }
        self._default = KoreanSentenceSplitter()  # Default to Korean
    
    def split(self, text: str, language: str = "ko") -> List[Sentence]:
        """
        Split text into sentences for the given language.
        
        Args:
            text: Text to split.
            language: Language code (ko, en, ja).
            
        Returns:
            List of Sentence objects.
        """
        splitter = self._splitters.get(language, self._default)
        return splitter.split(text)
    
    def split_blocks(self, blocks: List[ParsedBlock], language: str = "ko") -> List[ParsedBlock]:
        """
        Split blocks into sentence-level blocks.
        
        Args:
            blocks: List of ParsedBlock objects.
            language: Language code for splitting.
            
        Returns:
            List of ParsedBlock objects with sentence-level granularity.
        """
        new_blocks = []
        sequence = 0
        
        for block in blocks:
            if not block.text.strip():
                continue
            
            # Skip non-text blocks (tables, codes, etc.) or split them
            if block.block_type in (BlockType.TABLE, BlockType.CODE, BlockType.IMAGE):
                # Keep as-is
                new_blocks.append(ParsedBlock(
                    block_type=block.block_type,
                    text=block.text,
                    page_number=block.page_number,
                    sheet_name=block.sheet_name,
                    slide_number=block.slide_number,
                    section_path=block.section_path,
                    sequence=sequence,
                    metadata=block.metadata,
                ))
                sequence += 1
                continue
            
            # Split into sentences
            sentences = self.split(block.text, language)
            
            for sent in sentences:
                new_blocks.append(ParsedBlock(
                    block_type=block.block_type,
                    text=sent.text,
                    page_number=block.page_number,
                    sheet_name=block.sheet_name,
                    slide_number=block.slide_number,
                    section_path=block.section_path,
                    sequence=sequence,
                    metadata={
                        **block.metadata,
                        "sentence_start": sent.start_char,
                        "sentence_end": sent.end_char,
                        "sentence_confidence": sent.confidence,
                    },
                ))
                sequence += 1
        
        return new_blocks
    
    def get_splitter(self, language: str) -> SentenceSplitter:
        """Get splitter for specific language."""
        return self._splitters.get(language, self._default)


def create_sentence_splitter(config: AppConfig) -> CompositeSentenceSplitter:
    """Factory function to create sentence splitter."""
    return CompositeSentenceSplitter()