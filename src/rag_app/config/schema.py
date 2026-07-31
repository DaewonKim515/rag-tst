"""
Configuration schema using Pydantic models.
Defines the structure and validation for all configuration settings.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    """Filesystem paths configuration."""
    document_root: Path = Field(default=Path("./document"), description="Root directory for documents")
    metadata_db: Path = Field(default=Path("./data/metadata.db"), description="SQLite metadata database path")
    log_dir: Path = Field(default=Path("./data/logs"), description="Log directory")

    @field_validator("document_root", "metadata_db", "log_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def validate_paths(self) -> "PathsConfig":
        # Ensure document_root exists or can be created
        self.document_root.mkdir(parents=True, exist_ok=True)
        # Ensure parent directories for files exist
        self.metadata_db.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


class IndexingConfig(BaseModel):
    """Indexing pipeline configuration."""
    enabled: bool = True
    scan_on_start: bool = True
    scan_interval_seconds: int = Field(default=300, ge=10, le=86400)
    hash_algorithm: Literal["md5"] = "md5"
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    supported_extensions: list[str] = Field(default=[".pdf", ".docx", ".txt", ".md"])
    max_file_size_mb: int = Field(default=200, ge=1, le=2048)

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, v: list[str]) -> list[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in v]


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    target_tokens: int = Field(default=700, ge=100, le=8000)
    overlap_tokens: int = Field(default=100, ge=0, le=4000)
    max_tokens: int = Field(default=900, ge=100, le=8000)
    preserve_page_boundary: bool = True

    @model_validator(mode="after")
    def validate_tokens(self) -> "ChunkingConfig":
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        return self


class LLMConfig(BaseModel):
    """LLM model configuration."""
    model_id: str = "Qwen/Qwen3.5-9B"
    max_context_tokens: int = Field(default=32768, ge=1024, le=131072)
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_id: str = "BAAI/bge-m3"
    vector_size: int = Field(default=1024, ge=256, le=4096)
    normalize: bool = True


class RerankerConfig(BaseModel):
    """Reranker model configuration."""
    enabled: bool = True
    model_id: str = "BAAI/bge-reranker-v2-m3"
    candidate_count: int = Field(default=30, ge=5, le=200)
    final_count: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_counts(self) -> "RerankerConfig":
        if self.final_count > self.candidate_count:
            raise ValueError("final_count must be <= candidate_count")
        return self


class ModelsConfig(BaseModel):
    """All model configurations."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class OCRConfig(BaseModel):
    """OCR configuration for local/on-premise Unlimited OCR service."""
    enabled: bool = False
    provider: Literal["unlimited_ocr"] = "unlimited_ocr"
    endpoint: str = "http://localhost:8000/v1/ocr"
    api_key_env: Optional[str] = "UNLIMITED_OCR_API_KEY"
    languages: list[str] = Field(default=["ko", "ja", "en"])
    timeout_seconds: int = Field(default=120, ge=10, le=600)


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    url: str = "http://localhost:6333"
    collection: str = "document_chunks_v1"
    api_key_env: Optional[str] = "QDRANT_API_KEY"
    host: str = "localhost"
    port: int = 6333
    request_timeout_seconds: int = Field(default=30, ge=5, le=300)
    prefer_grpc: bool = False
    https: bool = False
    verify_ssl: bool = True

    @property
    def api_key(self) -> Optional[str]:
        import os
        return os.getenv(self.api_key_env) if self.api_key_env else None


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    candidate_count: int = Field(default=30, ge=5, le=200)
    final_top_k: int = Field(default=5, ge=1, le=50)
    minimum_dense_score: float = Field(default=0.25, ge=0.0, le=1.0)
    max_chunks_per_document: int = Field(default=3, ge=1, le=20)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["jsonl", "text"] = "jsonl"
    retention_days: int = Field(default=30, ge=1, le=365)
    include_query_text: bool = False


class AppConfig(BaseSettings):
    """Main application configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ollama_host: str = "http://localhost:11434"

    def get_secret(self, env_name: str) -> Optional[str]:
        """Get secret value from environment variable."""
        import os
        return os.getenv(env_name)

    def get_qdrant_api_key(self) -> Optional[str]:
        return self.get_secret(self.qdrant.api_key_env) if self.qdrant.api_key_env else None

    def get_ocr_api_key(self) -> Optional[str]:
        return self.get_secret(self.ocr.api_key_env) if self.ocr.api_key_env else None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "AppConfig":
        # Ensure chunking max_tokens doesn't exceed LLM context
        llm_context = self.models.llm.max_context_tokens
        # Reserve tokens for system prompt, question, output
        reserved = 2000 + self.models.llm.max_output_tokens
        available = llm_context - reserved
        if self.chunking.max_tokens * self.retrieval.final_top_k > available:
            raise ValueError(
                f"chunking.max_tokens * retrieval.final_top_k ({self.chunking.max_tokens * self.retrieval.final_top_k}) "
                f"exceeds available context ({available})"
            )
        return self