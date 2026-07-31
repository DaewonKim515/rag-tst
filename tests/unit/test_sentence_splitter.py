"""
Unit tests for SentenceSplitter in processing/sentence_splitter.py.
"""

import pytest
from rag_app.domain.enums import BlockType
from rag_app.domain.models import ParsedBlock
from rag_app.processing.sentence_splitter import (
    CompositeSentenceSplitter,
    EnglishSentenceSplitter,
    JapaneseSentenceSplitter,
    KoreanSentenceSplitter,
)


def test_korean_sentence_splitter():
    splitter = KoreanSentenceSplitter()
    text = "안녕하세요. 첫 번째 문장입니다! 두 번째 문장도 잘 분리되나요? 세 번째 문장입니다."
    sentences = splitter.split(text)
    
    assert len(sentences) == 4
    assert sentences[0].text == "안녕하세요."
    assert sentences[1].text == "첫 번째 문장입니다!"
    assert sentences[2].text == "두 번째 문장도 잘 분리되나요?"
    assert sentences[3].text == "세 번째 문장입니다."


def test_english_sentence_splitter():
    splitter = EnglishSentenceSplitter()
    text = "Hello Dr. Smith! This is the first sentence. Is this the second one? Yes it is."
    sentences = splitter.split(text)
    
    assert len(sentences) >= 2
    assert "This is the first sentence." in [s.text for s in sentences]


def test_japanese_sentence_splitter():
    splitter = JapaneseSentenceSplitter()
    text = "こんにちは。これは最初の文章です。二番目の文章ですか？"
    sentences = splitter.split(text)
    
    assert len(sentences) == 3
    assert sentences[0].text == "こんにちは。"
    assert sentences[1].text == "これは最初の文章です。"
    assert sentences[2].text == "二番目の文章ですか？"


def test_composite_splitter_split_blocks():
    composite = CompositeSentenceSplitter()
    blocks = [
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            text="첫 번째 문장. 두 번째 문장.",
            page_number=1,
            section_path=("소개",),
            sequence=0,
        ),
        ParsedBlock(
            block_type=BlockType.TABLE,
            text="[Header] A | B \n [Row] 1 | 2",
            page_number=1,
            section_path=("표",),
            sequence=1,
        ),
    ]
    
    split_blocks = composite.split_blocks(blocks, language="ko")
    
    # Paragraph block splits into 2 sentences, Table block stays intact
    assert len(split_blocks) == 3
    assert split_blocks[0].text == "첫 번째 문장."
    assert split_blocks[1].text == "두 번째 문장."
    assert split_blocks[2].block_type == BlockType.TABLE
