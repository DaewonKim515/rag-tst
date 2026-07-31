from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid5

from .enums import BlockType, DocumentStatus, FileType, OCRStatus


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """A structural block extracted from a document."""
    block_type: BlockType
    text: str
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    section_path: tuple[str, ...] = field(default_factory=tuple)
    sequence: int = 0
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate text is not empty for content blocks
        if self.block_type in (BlockType.PARAGRAPH, BlockType.TITLE, BlockType.HEADING, BlockType.LIST, BlockType.CODE):
            if not self.text or not self.text.strip():
                raise ValueError(f"Block type {self.block_type} requires non-empty text")
    
    @property
    def location_str(self) -> str:
        """Human-readable location string."""
        parts = []
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        if self.sheet_name:
            parts.append(f"sheet {self.sheet_name}")
        if self.slide_number is not None:
            parts.append(f"slide {self.slide_number}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return ", ".join(parts) if parts else "unknown"


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parsed document with structural blocks and metadata."""
    document_id: UUID
    source_path: str
    file_name: str
    file_type: FileType
    file_size: int
    md5_hash: str
    modified_at: datetime
    blocks: tuple[ParsedBlock, ...] = field(default_factory=tuple)
    parser_version: str = "1"
    language: str = "ko"
    ocr_status: OCRStatus = OCRStatus.NOT_NEEDED
    ocr_pages: tuple[int, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        if not self.blocks:
            raise ValueError("ParsedDocument must have at least one block")
    
    def get_blocks_by_type(self, block_type: BlockType) -> tuple[ParsedBlock, ...]:
        """Get all blocks of a specific type."""
        return tuple(b for b in self.blocks if b.block_type == block_type)
    
    def get_text_content(self) -> str:
        """Get full text content concatenated."""
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A document chunk with its embedding vector."""
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    index_profile_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    section_title: Optional[str] = None
    language: str = "ko"
    embedding: list[float] = field(default_factory=list)
    embedding_model: str = "BAAI/bge-m3"
    token_count: int = 0
    indexed_at: datetime = field(default_factory=datetime.now)
    # Source file metadata for citations
    source_path: str = ""
    file_name: str = ""
    file_type: str = ""
    
    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("EmbeddedChunk content cannot be empty")
        if not self.embedding:
            raise ValueError("EmbeddedChunk must have embedding vector")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
    
    @classmethod
    def generate_chunk_id(cls, document_id: UUID, version_id: UUID, chunk_index: int) -> UUID:
        """Generate deterministic UUIDv5 for chunk."""
        namespace = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL equivalent
        name = f"{document_id}/{version_id}/{chunk_index}"
        return uuid5(namespace, name)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A search result hit with score and metadata."""
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    index_profile_id: str
    content: str
    score: float
    chunk_index: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    section_title: Optional[str] = None
    source_path: str = ""
    file_name: str = ""
    file_type: str = ""
    language: str = "ko"
    rerank_score: Optional[float] = None
    
    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("SearchHit score must be between 0.0 and 1.0")
        if self.rerank_score is not None and not (0.0 <= self.rerank_score <= 1.0):
            raise ValueError("SearchHit rerank_score must be between 0.0 and 1.0")
    
    @property
    def location_str(self) -> str:
        """Human-readable location string for citation."""
        parts = [self.file_name]
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        elif self.sheet_name:
            parts.append(f"sheet {self.sheet_name}")
        elif self.slide_number is not None:
            parts.append(f"slide {self.slide_number}")
        if self.section_title:
            parts.append(f"section: {self.section_title}")
        return " | ".join(parts)
    
    def to_source_dict(self, citation_id: int) -> dict:
        """Convert to source dictionary for LLM context."""
        return {
            "citation_id": citation_id,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "location": self.location_str,
            "content": self.content,
            "chunk_id": str(self.chunk_id),
        }


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Search query with embedding and filters."""
    query_text: str
    query_embedding: list[float]
    candidate_count: int = 30
    final_top_k: int = 5
    minimum_score: float = 0.25
    max_chunks_per_document: int = 3
    filters: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.query_text or not self.query_text.strip():
            raise ValueError("query_text cannot be empty")
        if not self.query_embedding:
            raise ValueError("query_embedding cannot be empty")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Request for LLM generation."""
    question: str = ""
    context_sources: tuple[SearchHit, ...] = ()
    context: str = ""
    system_prompt: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: Optional[int] = None
    prompt: str = ""

    def __post_init__(self):
        # Support prompt as alias for question
        q = self.prompt or self.question
        if not q or not q.strip():
            raise ValueError("question cannot be empty")
        object.__setattr__(self, "question", q)
        if not self.prompt:
            object.__setattr__(self, "prompt", q)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result from LLM generation."""
    answer: str
    citations: Any = field(default_factory=list)  # list or dict
    model_id: str = "Qwen3.5-9B"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    def get_source_list(self) -> list[dict]:
        """Get formatted source list for output."""
        sources = []
        for cite_id in sorted(self.citations.keys()):
            hit = self.citations[cite_id]
            sources.append({
                "citation_id": cite_id,
                "file_name": hit.file_name,
                "source_path": hit.source_path,
                "location": hit.location_str,
            })
        return sources


@dataclass(frozen=True, slots=True)
class Answer:
    """Complete answer with sources and metadata."""
    question: str
    answer: str
    sources: list[dict]
    model_id: str
    retrieval_time_ms: float
    rerank_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "success"
    error: Optional[str] = None
    
    def format_cli(self) -> str:
        """Format for CLI output."""
        lines = []
        lines.append("답변:")
        lines.append(self.answer)
        lines.append("")
        lines.append("출처:")
        for src in self.sources:
            lines.append(f"[S{src['citation_id']}] {src['file_name']}, {src['location']}")
        lines.append("")
        lines.append("처리 시간:")
        lines.append(f"검색 {self.retrieval_time_ms/1000:.2f}초 / "
                     f"리랭킹 {self.rerank_time_ms/1000:.2f}초 / "
                     f"생성 {self.generation_time_ms/1000:.2f}초 / "
                     f"전체 {self.total_time_ms/1000:.2f}초")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Document metadata record for SQLite storage."""
    document_id: UUID
    source_path: str
    file_name: str
    file_type: str
    file_size: int
    md5_hash: str
    modified_at: datetime
    active_version_id: Optional[UUID] = None
    status: DocumentStatus = DocumentStatus.DISCOVERED
    error_message: Optional[str] = None
    parser_version: str = "1"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class FileVersionRecord:
    """File version record for SQLite storage (MD5 table)."""
    version_id: UUID
    document_id: UUID
    md5_hash: str
    index_profile_id: str
    parser_version: str
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.DISCOVERED
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class IndexJobRecord:
    """Index job record for SQLite storage."""
    job_id: UUID
    job_type: str  # scheduled, startup, rebuild
    status: str = "running"
    discovered_count: int = 0
    new_count: int = 0
    changed_count: int = 0
    deleted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None