"""
Dense vector retriever with active version filtering and document quota management.
"""

from typing import List, Optional, Dict
from uuid import UUID

from ..config.schema import AppConfig
from ..domain.models import SearchQuery, SearchHit
from ..domain.exceptions import RetrievalError
from ..models.embedding import BgeM3Embedder, get_embedder
from ..indexing.qdrant_store import QdrantStore
from ..persistence.sqlite import DatabaseManager, get_database_manager
from ..persistence.repositories import MetadataRepository
from ..observability.logging import get_logger, get_current_request_id

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense vector retriever filtering active document versions
    and enforcing score/document limits.
    """

    def __init__(
        self,
        config: AppConfig,
        embedder: Optional[BgeM3Embedder] = None,
        vector_store: Optional[QdrantStore] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.config = config
        self.embedder = embedder or get_embedder(config)
        self.vector_store = vector_store or QdrantStore(config)
        self.db_manager = db_manager or get_database_manager(config)
        self.metadata_repo = MetadataRepository(self.db_manager)

        self.retrieval_config = config.retrieval
        self.candidate_count = self.retrieval_config.candidate_count
        self.minimum_dense_score = self.retrieval_config.minimum_dense_score
        self.max_chunks_per_document = self.retrieval_config.max_chunks_per_document

    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[SearchHit]:
        """
        Retrieve relevant chunks for a user query.

        Args:
            query_text: User search query string.
            top_k: Optional override for candidate count.

        Returns:
            List of SearchHit objects filtered by active version & score thresholds.
        """
        req_id = get_current_request_id()
        if not query_text or not query_text.strip():
            logger.warning("empty_query", "Empty query text provided for retrieval", request_id=req_id)
            return []

        limit = top_k or self.candidate_count

        # 1. Embed Query
        try:
            query_vector = self.embedder.embed_query(query_text)
        except Exception as e:
            logger.error("query_embedding_failed", f"Failed to embed query: {e}", request_id=req_id)
            raise RetrievalError(f"Failed to embed query: {e}") from e

        # 2. Search Qdrant
        search_query = SearchQuery(
            query_text=query_text,
            query_embedding=query_vector,
            candidate_count=limit * 2,  # Search extra candidates to account for filtering
            minimum_score=self.minimum_dense_score,
        )

        raw_hits = self.vector_store.search(search_query)
        if not raw_hits:
            logger.info("retrieval_no_hits", "Qdrant returned 0 hits", request_id=req_id)
            return []

        # 3. Filter by SQLite Active Version Mapping
        active_versions = self.metadata_repo.get_active_version_mapping()  # document_id -> active_version_id
        
        filtered_hits = []
        doc_chunk_counts: Dict[UUID, int] = {}

        for hit in raw_hits:
            doc_id = hit.document_id
            ver_id = hit.version_id

            # Validate active version
            if doc_id not in active_versions or active_versions[doc_id] != ver_id:
                # Skip inactive or outdated version chunk
                continue

            # Enforce max_chunks_per_document constraint
            current_count = doc_chunk_counts.get(doc_id, 0)
            if current_count >= self.max_chunks_per_document:
                continue

            filtered_hits.append(hit)
            doc_chunk_counts[doc_id] = current_count + 1

            if len(filtered_hits) >= limit:
                break

        logger.info(
            "retrieval_completed",
            f"Retrieved {len(filtered_hits)} active hits from {len(raw_hits)} candidates",
            request_id=req_id,
            raw_count=len(raw_hits),
            filtered_count=len(filtered_hits),
        )

        return filtered_hits
