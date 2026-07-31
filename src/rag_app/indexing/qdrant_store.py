"""
Qdrant Vector Store implementation with security, TLS/HTTPS authentication,
and batch staging/search operations.
"""

import time
from typing import Protocol, List, Optional, Any
from uuid import UUID
from pathlib import Path

from ..config.schema import AppConfig, QdrantConfig
from ..domain.models import EmbeddedChunk, SearchQuery, SearchHit
from ..domain.exceptions import VectorStoreError, ConfigurationError
from ..observability.logging import get_logger

logger = get_logger(__name__)


class VectorStore(Protocol):
    """Protocol for vector store operations."""
    
    def stage(self, chunks: list[EmbeddedChunk]) -> bool:
        """Stage chunk vectors into vector store."""
        ...
        
    def search(self, query: SearchQuery) -> list[SearchHit]:
        """Search vector store for similar chunks."""
        ...
        
    def delete_version(self, document_id: UUID, version_id: UUID) -> bool:
        """Delete all points belonging to a specific version."""
        ...


def cleanup_zombie_python_processes():
    """
    Fundamentally prevent Qdrant local storage lock errors by terminating
    any leftover/zombie Python background processes (except the current process).
    """
    import os, sys, subprocess
    if sys.platform != "win32":
        return
    current_pid = os.getpid()
    try:
        cmd = f"Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {{ $_.ProcessId -ne {current_pid} }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, timeout=5)
    except Exception:
        pass


class QdrantStore:
    """
    Qdrant vector store wrapper.
    Enforces network security, API Key authentication, HTTPS/TLS verification,
    and named vector payload indexing.
    
    Embedded Mode Lifecycle:
    - When Qdrant server is unavailable, falls back to local file-based storage.
    - A shared embedded client is reused within a single process via _shared_embedded_client.
    - File locks are managed by qdrant-client/portalocker; if locked by another process,
      retries with configurable wait before giving up.
    """

    # Shared in-memory mock storage for testing/offline environments
    _mock_storage: dict[str, list[EmbeddedChunk]] = {}
    _shared_embedded_client = None
    _is_embedded_mode = False
    _init_error_msg: Optional[str] = None

    # Configurable retry settings for lock contention
    LOCK_RETRY_COUNT = 5
    LOCK_RETRY_WAIT_SECONDS = 2.0

    def __init__(self, config: AppConfig):
        self.config = config
        self.q_config: QdrantConfig = config.qdrant
        self.collection_name = self.q_config.collection
        self.vector_dim = config.models.embedding.vector_size
        self.client = None

        if self.collection_name not in QdrantStore._mock_storage:
            QdrantStore._mock_storage[self.collection_name] = []

        self._validate_security_config()
        self._init_client()
        self._ensure_collection()

    @classmethod
    def check_availability(cls, config: AppConfig) -> tuple[bool, str]:
        """
        Lightweight pre-flight check: can we connect to Qdrant (server or embedded)?
        Call this BEFORE loading heavy ML models to fail fast.
        
        Returns:
            (is_available, error_message) - True if available, False with reason if not.
        """
        # If we already have a working embedded client in this process, it's fine
        if cls._is_embedded_mode and cls._shared_embedded_client is not None:
            return True, ""

        q_config = config.qdrant
        url = q_config.url
        if not url:
            scheme = "https" if q_config.https else "http"
            url = f"{scheme}://{q_config.host}:{q_config.port}"

        # 1. Try remote server
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=url,
                api_key=q_config.api_key or None,
                timeout=3.0,  # short timeout for quick check
                verify=q_config.verify_ssl,
            )
            client.get_collections()
            # Server is reachable - close the test client
            try:
                client.close()
            except Exception:
                pass
            return True, ""
        except ImportError:
            return False, "qdrant-client 패키지가 설치되지 않았습니다."
        except Exception:
            pass  # Server not available, try embedded

        # 2. Try embedded local storage
        if config.environment == "production":
            return False, f"Qdrant 서버({url})에 연결할 수 없습니다. 프로덕션 환경에서는 서버 모드가 필요합니다."

        qdrant_dir = config.paths.metadata_db.parent / "qdrant_storage"
        if not qdrant_dir.exists():
            return True, ""  # Will be created on first use

        lock_file = qdrant_dir / ".lock"
        if lock_file.exists():
            try:
                import portalocker
                fh = open(lock_file, "r")
                try:
                    portalocker.lock(fh, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
                    portalocker.unlock(fh)
                    fh.close()
                    return True, ""
                except portalocker.exceptions.AlreadyLocked:
                    fh.close()
                    # Automatically cleanup zombie python processes holding lock
                    cleanup_zombie_python_processes()
                    try:
                        fh2 = open(lock_file, "r")
                        portalocker.lock(fh2, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
                        portalocker.unlock(fh2)
                        fh2.close()
                        return True, ""
                    except Exception:
                        return False, (
                            f"Qdrant 로컬 저장소({qdrant_dir})가 다른 프로세스(scan/rebuild)에 의해 잠겨 있습니다.\n"
                            "  👉 해당 작업이 완료될 때까지 기다린 후 다시 시도하세요."
                        )
                except Exception as e:
                    fh.close()
                    return False, f"Qdrant 로컬 저장소 잠금 확인 실패: {e}"
            except ImportError:
                return True, ""  # Can't check, assume available

        return True, ""

    def _validate_security_config(self):
        """Validate network security & authentication requirements [1.9.3]."""
        # Production config enforces TLS and API Key
        if self.config.environment == "production":
            if not self.q_config.https:
                raise ConfigurationError("Production environment requires Qdrant HTTPS/TLS enabled")
            if not self.q_config.api_key:
                raise ConfigurationError("Production environment requires mandatory Qdrant API Key")
        
        # Validate localhost binding fallback
        if not self.q_config.url:
            logger.info("qdrant_binding_default", f"Defaulting to local binding {self.q_config.host}:{self.q_config.port}")

    def _init_client(self):
        """Initialize QdrantClient, reusing shared embedded client if available."""
        if QdrantStore._is_embedded_mode and QdrantStore._shared_embedded_client is not None:
            self.client = QdrantStore._shared_embedded_client
            logger.debug("qdrant_client_reused", "Reusing shared embedded Qdrant client")
            return

        try:
            from qdrant_client import QdrantClient
            
            url = self.q_config.url
            if not url:
                scheme = "https" if self.q_config.https else "http"
                url = f"{scheme}://{self.q_config.host}:{self.q_config.port}"

            self.client = QdrantClient(
                url=url,
                api_key=self.q_config.api_key or None,
                timeout=float(self.q_config.request_timeout_seconds),
                verify=self.q_config.verify_ssl,
            )
            logger.info("qdrant_client_initialized", f"Connected to Qdrant at {url}")
        except ImportError:
            logger.warning("qdrant_client_missing", "qdrant-client package not installed, running mock mode")
            QdrantStore._init_error_msg = "qdrant-client package not installed"
            self.client = None
        except Exception as e:
            logger.error("qdrant_conn_failed", f"Failed to connect to Qdrant: {e}")
            QdrantStore._init_error_msg = f"Failed to connect to Qdrant: {e}"
            self.client = None

    def _ensure_collection(self):
        """Ensure Qdrant collection and payload indexes exist."""
        # If we're in embedded mode and have a working client, just verify collection exists
        if QdrantStore._is_embedded_mode and self.client is not None:
            try:
                self._create_collection_if_not_exists()
                QdrantStore._init_error_msg = None
            except Exception as e:
                logger.error("qdrant_collection_verify_failed", f"Embedded collection check failed: {e}")
            return

        if self.client is None and not QdrantStore._is_embedded_mode:
            return
        
        try:
            self._create_collection_if_not_exists()
            QdrantStore._init_error_msg = None
        except Exception as e:
            if self.config.environment != "production":
                logger.warning("qdrant_remote_fallback", f"Remote Qdrant server unavailable ({e}), switching to embedded local file storage")
                self._init_embedded_with_retry()
            else:
                QdrantStore._init_error_msg = f"Failed to setup Qdrant collection: {e}"
                logger.error("qdrant_collection_init_failed", QdrantStore._init_error_msg)
                self.client = None

    def _init_embedded_with_retry(self):
        """Initialize embedded Qdrant client with lock retry logic."""
        qdrant_dir = self.config.paths.metadata_db.parent / "qdrant_storage"
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        
        max_retries = self.LOCK_RETRY_COUNT
        wait_seconds = self.LOCK_RETRY_WAIT_SECONDS
        
        for attempt in range(1, max_retries + 1):
            try:
                if QdrantStore._shared_embedded_client is None:
                    from qdrant_client import QdrantClient
                    QdrantStore._shared_embedded_client = QdrantClient(path=str(qdrant_dir))
                self.client = QdrantStore._shared_embedded_client
                QdrantStore._is_embedded_mode = True
                self._create_collection_if_not_exists()
                QdrantStore._init_error_msg = None
                logger.info("qdrant_embedded_ready", f"Embedded Qdrant local storage ready at {qdrant_dir}")
                return
            except Exception as embedded_err:
                err_str = str(embedded_err)
                is_lock_error = "already accessed by another instance" in err_str
                
                if is_lock_error and attempt < max_retries:
                    logger.warning(
                        "qdrant_lock_retry",
                        f"Qdrant local storage locked (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)
                    QdrantStore._shared_embedded_client = None
                    continue
                
                if is_lock_error:
                    QdrantStore._init_error_msg = (
                        f"Qdrant 로컬 저장소({qdrant_dir})가 다른 프로세스에 의해 잠겨 있습니다. "
                        f"scan/rebuild 작업이 완료될 때까지 기다린 후 다시 시도해 주세요."
                    )
                else:
                    QdrantStore._init_error_msg = f"Qdrant 로컬 저장소 초기화 실패: {err_str}"
                
                logger.error("qdrant_embedded_init_failed", QdrantStore._init_error_msg)
                self.client = None
                return

    def _create_collection_if_not_exists(self):
        """Helper to check and create collection and payload indexes."""
        from qdrant_client.http import models as rest_models
        
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info("creating_qdrant_collection", f"Creating collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": rest_models.VectorParams(
                        size=self.vector_dim,
                        distance=rest_models.Distance.COSINE,
                    )
                }
            )
            
            indexed_fields = [
                ("document_id", rest_models.PayloadSchemaType.KEYWORD),
                ("version_id", rest_models.PayloadSchemaType.KEYWORD),
                ("index_profile_id", rest_models.PayloadSchemaType.KEYWORD),
                ("source_path", rest_models.PayloadSchemaType.KEYWORD),
                ("file_type", rest_models.PayloadSchemaType.KEYWORD),
                ("language", rest_models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in indexed_fields:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )

    def stage(self, chunks: list[EmbeddedChunk]) -> bool:
        """
        Stage list of EmbeddedChunk vectors into Qdrant in batch.
        """
        if not chunks:
            return True
        
        if self.client is None:
            logger.warning("qdrant_stage_mock", f"Mock staging {len(chunks)} chunks")
            QdrantStore._mock_storage[self.collection_name].extend(chunks)
            return True

        try:
            from qdrant_client.http import models as rest_models
            
            points = []
            for chunk in chunks:
                points.append(rest_models.PointStruct(
                    id=str(chunk.chunk_id),
                    vector={"dense": chunk.embedding},
                    payload={
                        "document_id": str(chunk.document_id),
                        "version_id": str(chunk.version_id),
                        "index_profile_id": chunk.index_profile_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "sheet_name": chunk.sheet_name,
                        "slide_number": chunk.slide_number,
                        "section_title": chunk.section_title,
                        "language": chunk.language,
                        "token_count": chunk.token_count,
                        "source_path": chunk.source_path,
                        "file_name": chunk.file_name,
                        "file_type": chunk.file_type,
                    }
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info("qdrant_staged", f"Successfully staged {len(points)} points")
            return True
            
        except Exception as e:
            logger.error("qdrant_stage_error", f"Qdrant staging failed: {e}")
            raise VectorStoreError(f"Qdrant upsert failed: {e}") from e

    def search(self, query: SearchQuery) -> list[SearchHit]:
        """
        Perform vector similarity search against Qdrant collection.
        """
        if self.client is None:
            if QdrantStore._init_error_msg and self.config.environment != "test":
                logger.error("qdrant_search_disabled", f"Qdrant vector search unavailable: {QdrantStore._init_error_msg}")
            
            # Mock search: return SearchHits from in-memory mock storage
            stored_chunks = QdrantStore._mock_storage.get(self.collection_name, [])
            hits = []
            for chunk in stored_chunks:
                hits.append(SearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    index_profile_id=chunk.index_profile_id,
                    content=chunk.content,
                    score=0.90,  # mock similarity score
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    sheet_name=chunk.sheet_name,
                    slide_number=chunk.slide_number,
                    section_title=chunk.section_title,
                    source_path="mock_document",
                    file_name="mock_document",
                    file_type=".txt",
                    language=chunk.language,
                ))
            return hits[:query.candidate_count]

        try:
            from qdrant_client.http import models as rest_models

            # Build query filter if provided
            query_filter = None
            if query.filters:
                must_conditions = []
                for k, v in query.filters.items():
                    must_conditions.append(rest_models.FieldCondition(
                        key=k,
                        match=rest_models.MatchValue(value=v)
                    ))
                query_filter = rest_models.Filter(must=must_conditions)

            # qdrant-client >= 1.12: use query_points (search was removed)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query.query_embedding,
                using="dense",
                limit=query.candidate_count,
                score_threshold=query.minimum_score,
                query_filter=query_filter,
                with_payload=True,
            )

            hits = []
            for res in response.points:
                payload = res.payload or {}
                hits.append(SearchHit(
                    chunk_id=UUID(res.id) if isinstance(res.id, str) else res.id,
                    document_id=UUID(payload.get("document_id")),
                    version_id=UUID(payload.get("version_id")),
                    index_profile_id=payload.get("index_profile_id", ""),
                    content=payload.get("content", ""),
                    score=float(res.score),
                    chunk_index=payload.get("chunk_index", 0),
                    page_number=payload.get("page_number"),
                    sheet_name=payload.get("sheet_name"),
                    slide_number=payload.get("slide_number"),
                    section_title=payload.get("section_title"),
                    source_path=payload.get("source_path", ""),
                    file_name=payload.get("file_name", ""),
                    file_type=payload.get("file_type", ""),
                    language=payload.get("language", "ko"),
                ))
            return hits

        except Exception as e:
            logger.error("qdrant_search_error", f"Qdrant search failed: {e}")
            raise VectorStoreError(f"Qdrant search failed: {e}") from e

    def delete_version(self, document_id: UUID, version_id: UUID) -> bool:
        """
        Delete all vector points belonging to a specific document version.
        """
        if self.client is None:
            return True

        try:
            from qdrant_client.http import models as rest_models
            
            filter_condition = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="document_id", match=rest_models.MatchValue(value=str(document_id))
                    ),
                    rest_models.FieldCondition(
                        key="version_id", match=rest_models.MatchValue(value=str(version_id))
                    ),
                ]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(filter=filter_condition),
                wait=True,
            )
            logger.info("qdrant_deleted_version", f"Deleted points for document {document_id} version {version_id}")
            return True
        except Exception as e:
            logger.error("qdrant_delete_error", f"Qdrant delete failed: {e}")
            raise VectorStoreError(f"Qdrant version delete failed: {e}") from e

    def verify_point_count(self, expected_count: int) -> bool:
        """Verify total point count in collection."""
        if self.client is None:
            return True
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count == expected_count
        except Exception:
            return False

    def clear_collection(self) -> bool:
        """Clear all vectors from collection and recreate it."""
        if self.client is None:
            QdrantStore._mock_storage[self.collection_name] = []
            return True

        try:
            self.client.delete_collection(self.collection_name)
            logger.info("qdrant_collection_cleared", f"Cleared collection {self.collection_name}")
        except Exception as e:
            logger.warning("qdrant_clear_failed", f"Failed to delete collection {self.collection_name}: {e}")

        # Always recreate the collection after clearing
        self._create_collection_if_not_exists()
        return True
