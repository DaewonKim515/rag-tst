"""
Change detector for incremental indexing.

Compares current filesystem state with database records to classify
files as NEW, CHANGED, UNCHANGED, REINDEX_REQUIRED, or DELETED.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from uuid import UUID

from ..config.schema import AppConfig
from ..domain.models import DocumentRecord, FileVersionRecord
from ..domain.enums import DocumentStatus
from ..persistence.repositories import DocumentRepository, FileVersionRepository
from ..observability.logging import get_logger, get_current_job_id


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FileClassification:
    """Classification result for a file."""
    class ChangeType:
        NEW = "new"
        CHANGED = "changed"
        UNCHANGED = "unchanged"
        REINDEX_REQUIRED = "reindex_required"
        DELETED = "deleted"
        EXCLUDED = "excluded"
    
    source_path: str
    change_type: str
    document_id: Optional[UUID] = None
    version_id: Optional[UUID] = None
    current_md5: Optional[str] = None
    stored_md5: Optional[str] = None
    index_profile_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ChangeDetectionResult:
    """Complete change detection result."""
    classifications: List[FileClassification]
    new_files: List[FileClassification]
    changed_files: List[FileClassification]
    unchanged_files: List[FileClassification]
    reindex_required_files: List[FileClassification]
    deleted_files: List[FileClassification]
    excluded_files: List[FileClassification]


class ChangeDetector:
    """Detects file changes for incremental indexing."""
    
    def __init__(
        self, 
        config: AppConfig,
        db_manager: Optional[Any] = None,
        doc_repo: Optional[DocumentRepository] = None,
        version_repo: Optional[FileVersionRepository] = None,
    ):
        self.config = config
        self.doc_repo = doc_repo or DocumentRepository(db_manager)
        self.version_repo = version_repo or FileVersionRepository(db_manager)
        self.current_index_profile_id = self._compute_index_profile_id()
    
    def _compute_index_profile_id(self) -> str:
        """Compute hash of indexing configuration for change detection."""
        # Create a stable hash of chunking config + embedding model + parser version
        config_parts = [
            str(self.config.chunking.target_tokens),
            str(self.config.chunking.overlap_tokens),
            str(self.config.chunking.max_tokens),
            str(self.config.chunking.preserve_page_boundary),
            self.config.models.embedding.model_id,
            str(self.config.models.embedding.vector_size),
            str(self.config.models.embedding.normalize),
            "parser_v1",  # parser version
        ]
        config_str = "|".join(config_parts)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def detect_changes(
        self, 
        discovered_files: Dict[str, Tuple[Path, str]]  # relative_path -> (file_path, md5_hash)
    ) -> ChangeDetectionResult:
        """
        Detect changes by comparing discovered files with database.
        
        Args:
            discovered_files: Mapping of relative_path -> (absolute_path, md5_hash)
            
        Returns:
            ChangeDetectionResult with all classifications.
        """
        job_id = get_current_job_id()
        
        # Get all documents from database
        stored_docs = self.doc_repo.get_all()
        stored_by_path = {doc.source_path: doc for doc in stored_docs}
        
        # Get all active file versions
        stored_versions = {}
        for doc in stored_docs:
            active_version = self.version_repo.get_active_version(doc.document_id)
            if active_version:
                stored_versions[doc.source_path] = active_version
        
        classifications = []
        new_files = []
        changed_files = []
        unchanged_files = []
        reindex_required_files = []
        deleted_files = []
        excluded_files = []
        
        discovered_paths = set(discovered_files.keys())
        stored_paths = set(stored_by_path.keys())
        
        # Process discovered files
        for rel_path, (abs_path, md5_hash) in discovered_files.items():
            if rel_path not in stored_paths:
                # NEW file
                classification = FileClassification(
                    source_path=rel_path,
                    change_type=FileClassification.ChangeType.NEW,
                    current_md5=md5_hash,
                    reason="File not in database",
                )
                classifications.append(classification)
                new_files.append(classification)
            else:
                stored_doc = stored_by_path[rel_path]
                stored_version = stored_versions.get(rel_path)
                
                if stored_version is None:
                    # Document exists but no active version - treat as CHANGED
                    classification = FileClassification(
                        source_path=rel_path,
                        change_type=FileClassification.ChangeType.CHANGED,
                        document_id=stored_doc.document_id,
                        current_md5=md5_hash,
                        stored_md5=stored_doc.md5_hash,
                        index_profile_id=self.current_index_profile_id,
                        reason="No active version",
                    )
                    classifications.append(classification)
                    changed_files.append(classification)
                elif md5_hash != stored_version.md5_hash:
                    # MD5 changed - CHANGED
                    classification = FileClassification(
                        source_path=rel_path,
                        change_type=FileClassification.ChangeType.CHANGED,
                        document_id=stored_doc.document_id,
                        version_id=stored_version.version_id,
                        current_md5=md5_hash,
                        stored_md5=stored_version.md5_hash,
                        index_profile_id=self.current_index_profile_id,
                        reason="MD5 hash mismatch",
                    )
                    classifications.append(classification)
                    changed_files.append(classification)
                elif stored_version.index_profile_id != self.current_index_profile_id:
                    # MD5 same but index profile changed - REINDEX_REQUIRED
                    classification = FileClassification(
                        source_path=rel_path,
                        change_type=FileClassification.ChangeType.REINDEX_REQUIRED,
                        document_id=stored_doc.document_id,
                        version_id=stored_version.version_id,
                        current_md5=md5_hash,
                        stored_md5=stored_version.md5_hash,
                        index_profile_id=self.current_index_profile_id,
                        reason="Index profile changed",
                    )
                    classifications.append(classification)
                    reindex_required_files.append(classification)
                else:
                    # UNCHANGED
                    classification = FileClassification(
                        source_path=rel_path,
                        change_type=FileClassification.ChangeType.UNCHANGED,
                        document_id=stored_doc.document_id,
                        version_id=stored_version.version_id,
                        current_md5=md5_hash,
                        stored_md5=stored_version.md5_hash,
                        index_profile_id=self.current_index_profile_id,
                        reason="MD5 and profile match",
                    )
                    classifications.append(classification)
                    unchanged_files.append(classification)
        
        # Process deleted files (in DB but not on disk)
        for rel_path in stored_paths - discovered_paths:
            stored_doc = stored_by_path[rel_path]
            classification = FileClassification(
                source_path=rel_path,
                change_type=FileClassification.ChangeType.DELETED,
                document_id=stored_doc.document_id,
                stored_md5=stored_doc.md5_hash,
                reason="File deleted from filesystem",
            )
            classifications.append(classification)
            deleted_files.append(classification)
        
        logger.info("change_detection_completed", 
                   f"Changes: new={len(new_files)}, changed={len(changed_files)}, "
                   f"unchanged={len(unchanged_files)}, reindex={len(reindex_required_files)}, "
                   f"deleted={len(deleted_files)}",
                   job_id=job_id,
                   new_count=len(new_files),
                   changed_count=len(changed_files),
                   unchanged_count=len(unchanged_files),
                   reindex_count=len(reindex_required_files),
                   deleted_count=len(deleted_files))
        
        return ChangeDetectionResult(
            classifications=classifications,
            new_files=new_files,
            changed_files=changed_files,
            unchanged_files=unchanged_files,
            reindex_required_files=reindex_required_files,
            deleted_files=deleted_files,
            excluded_files=excluded_files,
        )
    
    def get_files_to_process(
        self, 
        result: ChangeDetectionResult
    ) -> List[FileClassification]:
        """Get files that need processing (new, changed, reindex_required)."""
        return (
            result.new_files + 
            result.changed_files + 
            result.reindex_required_files
        )
    
    def get_unchanged_files(self, result: ChangeDetectionResult) -> List[FileClassification]:
        """Get files that can be skipped."""
        return result.unchanged_files
    
    def get_deleted_files(self, result: ChangeDetectionResult) -> List[FileClassification]:
        """Get files that were deleted."""
        return result.deleted_files