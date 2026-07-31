"""
Reranker model provider using Cross-Encoder or BAAI/bge-reranker-v2-m3.
"""

from typing import Protocol, List, Optional

# Transformers compatibility patch for XLMRobertaTokenizer in FlagReranker
# FlagEmbedding's FlagReranker uses tokenizer.prepare_for_model() which was removed in transformers >= 4.46
# We patch it to use the modern __call__ method with two sequences instead
try:
    import transformers
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    
    def _prepare_for_model_compat(self, query_ids, passage_ids, truncation='only_second', max_length=512, padding=False, **kwargs):
        """
        Compatibility wrapper for prepare_for_model.
        The FlagReranker passes pre-tokenized query_ids and passage_ids.
        We need to combine them properly for the model.
        """
        # The old prepare_for_model combined query_ids + passage_ids with proper special tokens
        # For XLMRoberta: <s> query </s></s> passage </s>
        # We need to manually construct this
        
        # Get special tokens
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 0
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 2
        
        # Truncate passage if needed (only_second truncation)
        if truncation == 'only_second' and max_length:
            # Account for special tokens: <s> query </s></s> passage </s> = query + passage + 4
            max_passage_len = max_length - len(query_ids) - 4
            if max_passage_len < 0:
                max_passage_len = 0
            passage_ids = passage_ids[:max_passage_len]
        
        # Construct: <s> query_ids </s></s> passage_ids </s>
        input_ids = [cls_token_id] + query_ids + [sep_token_id, sep_token_id] + passage_ids + [sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    # Patch both slow and fast tokenizers
    for cls_name in ["XLMRobertaTokenizer", "XLMRobertaTokenizerFast"]:
        if hasattr(transformers, cls_name):
            cls = getattr(transformers, cls_name)
            if not hasattr(cls, "prepare_for_model"):
                cls.prepare_for_model = _prepare_for_model_compat
            
except Exception:
    pass

from ..config.schema import AppConfig
from ..domain.models import SearchHit
from ..observability.logging import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    """Protocol for Reranker models."""

    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """Rerank search hits for a query."""
        ...


class BgeReranker:
    """
    Reranker implementation using BAAI/bge-reranker-v2-m3 or bge-reranker-v2-mlgn.
    Falls back to dense score ordering if model cannot be loaded.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.reranker_config = config.models.reranker
        self.model_id = self.reranker_config.model_id
        self.enabled = self.reranker_config.enabled
        self.final_count = self.reranker_config.final_count

        self.model = None
        self._is_dummy = False

        if self.enabled:
            self._init_model()

    def _init_model(self) -> None:
        """Initialize reranker model."""
        # Use GPU for reranker (RTX 3080 x2 available)
        import os
        import torch
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_device = "cuda"
            logger.info("gpu_available", f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            gpu_device = "cpu"
            logger.warning("gpu_not_available", "CUDA not available, falling back to CPU")
        
        # Try CrossEncoder first (no multiprocessing pool issues)
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_id, device=gpu_device)
            logger.info("reranker_loaded", f"Loaded CrossEncoder reranker model: {self.model_id} on {gpu_device.upper()}")
        except Exception as e:
            logger.warning("reranker_crossencoder_failed", f"CrossEncoder failed: {e}")
            # Fallback to FlagReranker if CrossEncoder not available
            try:
                from FlagEmbedding import FlagReranker
                use_fp16 = (gpu_device == "cuda")
                self.model = FlagReranker(self.model_id, use_fp16=use_fp16, device=gpu_device)
                logger.info("reranker_loaded", f"Loaded FlagReranker model: {self.model_id} on {gpu_device.upper()} (fp16={use_fp16})")
            except Exception as e2:
                logger.warning(
                    "reranker_fallback",
                    f"Reranker backend not available ({e2}), fallback to dense score ordering",
                    model_id=self.model_id,
                )
                self._is_dummy = True

    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """
        Rerank hits using Cross-Encoder model.

        Args:
            query: User search query.
            hits: Candidate SearchHit list.
            top_k: Optional final count override.

        Returns:
            Reranked list of SearchHit objects.
        """
        if not hits:
            return []

        limit = top_k or self.final_count

        if not self.enabled or self._is_dummy or self.model is None:
            # Fallback: sort by dense score descending
            sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
            return sorted_hits[:limit]

        try:
            # Form (query, text) pairs as tuples for FlagReranker
            pairs = [(query, hit.content) for hit in hits]
            
            # Predict scores
            if hasattr(self.model, "compute_score"):
                # Use GPU if model is on GPU, otherwise CPU
                import torch
                compute_device = "cuda" if torch.cuda.is_available() else "cpu"
                scores = self.model.compute_score(pairs, normalize=True, device=compute_device)
                logger.debug("rerank_compute_score", f"compute_score returned: {type(scores)} = {scores}")
            else:
                # CrossEncoder expects List[List[str]]
                pairs_list = [[q, t] for q, t in pairs]
                scores = self.model.predict(pairs_list)
                logger.debug("rerank_predict", f"predict returned: {type(scores)} = {scores}")
                if isinstance(scores, (float, int)):
                    scores = [scores]
            
            # Validate scores
            if scores is None:
                logger.warning("rerank_no_scores", "Reranker returned None scores, fallback to dense score ordering")
                sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
                return sorted_hits[:limit]
            
            # Ensure scores is a flat list of floats
            # Handle various return types: list, tuple, numpy array, nested lists
            import numpy as np
            
            # Convert numpy array to list
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Flatten nested lists (e.g., [[score1], [score2], ...] or [array([...])])
            if isinstance(scores, (list, tuple)):
                # Check if it's a nested list
                if scores and isinstance(scores[0], (list, tuple, np.ndarray)):
                    scores = [float(s[0]) if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 else float(s) for s in scores]
                else:
                    scores = [float(s) for s in scores]
            else:
                scores = [float(scores)]
            
            # Debug: check scores content
            logger.debug("rerank_scores_debug", f"Scores length: {len(scores)}, first few: {scores[:3] if len(scores) > 3 else scores}")
            
            # Attach rerank score and sort
            scored_hits = []
            for hit, score in zip(hits, scores):
                if score is None:
                    float_score = 0.0
                else:
                    float_score = float(score)
                # Create a new SearchHit with rerank_score
                new_hit = SearchHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    version_id=hit.version_id,
                    index_profile_id=hit.index_profile_id,
                    chunk_index=hit.chunk_index,
                    content=hit.content,
                    source_path=hit.source_path,
                    file_name=hit.file_name,
                    file_type=hit.file_type,
                    page_number=hit.page_number,
                    sheet_name=hit.sheet_name,
                    slide_number=hit.slide_number,
                    section_title=hit.section_title,
                    language=hit.language,
                    score=hit.score,
                    rerank_score=float_score,
                )
                scored_hits.append(new_hit)

            scored_hits.sort(key=lambda h: (h.rerank_score if h.rerank_score is not None else h.score), reverse=True)
            logger.info("rerank_completed", f"Reranked {len(hits)} candidates to top {limit}")
            return scored_hits[:limit]

        except Exception as e:
            logger.error("rerank_error", f"Rerank failed: {e}, fallback to dense score ordering")
            sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
            return sorted_hits[:limit]


def get_reranker(config: AppConfig) -> Reranker:
    """Factory function for Reranker."""
    return BgeReranker(config)
