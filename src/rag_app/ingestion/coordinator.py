"""
IndexCoordinator orchestrating the end-to-end indexing pipeline:
Scan -> ChangeDetect -> Parse -> Normalize -> Chunk -> Embed -> Qdrant Stage -> SQLite Atomic Swap.
"""

from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Optional

from ..config.schema import AppConfig
from ..domain.enums import DocumentStatus
from ..domain.models import DocumentRecord, EmbeddedChunk
from ..domain.exceptions import RAGSystemError
from ..ingestion.scanner import FileScanner
from ..ingestion.hasher import Hasher
from ..ingestion.change_detector import ChangeDetector, FileClassification
from ..parsing.registry import ParserRegistry
from ..processing.normalizer import TextNormalizer
from ..processing.chunker import Chunker
from ..models.embedding import BgeM3Embedder
from ..indexing.qdrant_store import QdrantStore
from ..indexing.version_manager import VersionManager
from ..persistence.sqlite import DatabaseManager
from ..persistence.repositories import DocumentRepository, IndexJobRepository
from ..observability.logging import get_logger, new_job_context

logger = get_logger(__name__)


class IndexCoordinator:
    """
    Orchestrates indexing pipeline per file version with atomic activation and per-file error isolation.
    """
    
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager,
        embedder: Optional[BgeM3Embedder] = None,
        qdrant_store: Optional[QdrantStore] = None,
    ):
        self.config = config
        self.db_manager = db_manager
        self.scanner = FileScanner(config)
        self.hasher = Hasher(config)
        self.change_detector = ChangeDetector(config, db_manager)
        self.parser_registry = ParserRegistry(config)
        self.normalizer = TextNormalizer()
        self.chunker = Chunker()
        self.embedder = embedder or BgeM3Embedder(config.models.embedding.model_id)
        self.qdrant_store = qdrant_store or QdrantStore(config)
        self.version_manager = VersionManager(db_manager)
        self.doc_repo = DocumentRepository(db_manager)
        self.job_repo = IndexJobRepository(db_manager)
        
        self.index_profile_id = self.change_detector.current_index_profile_id

    def run_indexing_job(self, job_type: str = "manual") -> dict:
        """
        Run a full scan and index job across document root.
        """
        job_id = uuid4()
        with new_job_context(job_id):
            logger.info("job_started", f"Starting indexing job {job_id} ({job_type})")
            
            # Discover files
            discovered = self.scanner.scan()
            
            # Build discovered dict: relative_path -> (absolute_path, md5_hash)
            disc_dict = {}
            for f in discovered:
                rel_str = str(f.relative_path)
                md5_res = self.hasher.calculate_md5(f.file_path)
                disc_dict[rel_str] = (f.file_path, md5_res.md5_hash)

            # Detect changes
            change_result = self.change_detector.detect_changes(disc_dict)
            
            summary = {
                "job_id": str(job_id),
                "discovered": len(discovered),
                "new": len(change_result.new_files),
                "changed": len(change_result.changed_files),
                "deleted": len(change_result.deleted_files),
                "unchanged": len(change_result.unchanged_files),
                "success_count": 0,
                "failed_count": 0,
            }
            
            # Process deleted files
            for item in change_result.deleted_files:
                self._handle_deleted_file(item.source_path)
            
            # Process new & changed files
            files_to_process = self.change_detector.get_files_to_process(change_result)
            disc_by_rel = {str(f.relative_path): f for f in discovered}

            for item in files_to_process:
                disc_file = disc_by_rel.get(item.source_path)
                if disc_file:
                    md5_val = disc_dict[item.source_path][1]
                    success = self._process_single_file(disc_file, md5_val)
                    if success:
                        summary["success_count"] += 1
                    else:
                        summary["failed_count"] += 1
            
            logger.info("job_finished", f"Finished indexing job {job_id}: {summary}")
            return summary

    def _process_single_file(self, discovered_file, md5_hash: str) -> bool:
        """
        Process a single file through Parse -> Normalize -> Chunk -> Embed -> Stage -> Activate.
        Implements per-file error isolation.
        """
        source_path = str(discovered_file.relative_path)
        abs_path = discovered_file.file_path
        
        logger.info("processing_file_start", f"Processing file: {source_path}")
        
        # 1. Get or create document record in SQLite
        doc_record = self.doc_repo.get_by_source_path(source_path)
        if not doc_record:
            doc_id = uuid4()
            doc_record = DocumentRecord(
                document_id=doc_id,
                source_path=source_path,
                file_name=discovered_file.file_name,
                file_type=discovered_file.file_type.value,
                file_size=discovered_file.file_size,
                md5_hash=md5_hash,
                modified_at=datetime.fromtimestamp(discovered_file.modified_at),
                status=DocumentStatus.DISCOVERED,
            )
            self.doc_repo.create(doc_record)
        else:
            doc_id = doc_record.document_id

        # 2. Create pending version record
        version_record = self.version_manager.create_pending_version(
            document_id=doc_id,
            md5_hash=md5_hash,
            index_profile_id=self.index_profile_id,
        )
        version_id = version_record.version_id

        try:
            # Step A: Parse Document
            self.version_manager.update_status(version_id, DocumentStatus.PARSING)
            parse_result = self.parser_registry.parse(abs_path, str(doc_id), md5_hash)
            parsed_doc = parse_result.document
            
            # Step B: Normalize Text
            normalized_doc = self.normalizer.normalize(parsed_doc)
            
            # Step C: Chunk Document
            self.version_manager.update_status(version_id, DocumentStatus.CHUNKING)
            chunks = self.chunker.chunk_document(
                document=normalized_doc,
                document_id=doc_id,
                version_id=version_id,
                index_profile_id=self.index_profile_id,
            )
            
            if not chunks:
                logger.warning("no_chunks_generated", f"No chunks created for file {source_path}")
                self.version_manager.mark_failed(doc_id, version_id, "No chunks generated")
                return False

            # Step D: Generate Embeddings
            self.version_manager.update_status(version_id, DocumentStatus.EMBEDDING)
            texts = [c.content for c in chunks]
            vectors = self.embedder.embed_documents(texts)
            
            # Attach embeddings to chunks
            embedded_chunks = []
            for chunk, vec in zip(chunks, vectors):
                embedded_chunks.append(EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    index_profile_id=chunk.index_profile_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    sheet_name=chunk.sheet_name,
                    slide_number=chunk.slide_number,
                    section_title=chunk.section_title,
                    language=chunk.language,
                    embedding=vec,
                    embedding_model=self.config.models.embedding.model_id,
                    token_count=chunk.token_count,
                    source_path=doc_record.source_path,
                    file_name=doc_record.file_name,
                    file_type=doc_record.file_type,
                ))

            # Step E: Qdrant Staging
            self.version_manager.update_status(version_id, DocumentStatus.STAGING)
            self.qdrant_store.stage(embedded_chunks)
            
            # Step F: Atomic Activation
            self.version_manager.update_status(
                version_id, DocumentStatus.ACTIVE, chunk_count=len(embedded_chunks)
            )
            self.version_manager.activate_version(doc_id, version_id)
            
            logger.info("file_indexed_successfully", f"Successfully indexed {source_path} ({len(embedded_chunks)} chunks)")
            return True

        except Exception as e:
            # Per-file Error Isolation: Catch error, log it, update version status to FAILED
            error_msg = f"Indexing failed for {source_path}: {e}"
            logger.error("file_indexing_failed", error_msg)
            self.version_manager.mark_failed(doc_id, version_id, str(e))
            return False

    def _handle_deleted_file(self, rel_path: str):
        """Clean up Qdrant vectors and SQLite record for deleted file."""
        doc = self.doc_repo.get_by_source_path(rel_path)
        if doc and doc.active_version_id:
            self.qdrant_store.delete_version(doc.document_id, doc.active_version_id)
            self.doc_repo.update_status(doc.document_id, DocumentStatus.DELETED)
            logger.info("file_deleted_cleanup", f"Cleaned up deleted document {rel_path}")
