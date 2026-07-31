"""
Embedding provider module using BAAI/bge-m3.

Provides EmbeddingProvider protocol and BgeM3Embedder implementation with batching,
vector normalization, and dimension validation.
"""

from typing import Protocol, List, Optional
import math
from ..domain.exceptions import ModelError
from ..observability.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding model providers."""
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        ...
        
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...
        
    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        ...


def normalize_l2(vector: list[float]) -> list[float]:
    """Normalize vector to unit length (L2 norm = 1.0)."""
    sq_sum = sum(v * v for v in vector)
    if sq_sum <= 0:
        return vector
    norm = math.sqrt(sq_sum)
    return [v / norm for v in vector]


class BgeM3Embedder:
    """
    Embedding provider for BAAI/bge-m3 model.
    Supports local FlagEmbedding / SentenceTransformers / Transformers with CPU/GPU fallback.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size) if batch_size else 32
        self.normalize_embeddings = bool(normalize_embeddings)
        self.device = device
        self._expected_dim = 1024
        self._model = None
        self._init_model()

    def _init_model(self):
        """Lazy or immediate model initialization."""
        # Use GPU for embedding (RTX 3080 x2 available)
        import os
        import torch
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_device = "cuda"
            logger.info("gpu_available", f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            gpu_device = "cpu"
            logger.warning("gpu_not_available", "CUDA not available, falling back to CPU")
        
        # Try SentenceTransformer first (no multiprocessing pool issues)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=gpu_device)
            logger.info("st_loaded", f"Loaded SentenceTransformer model {self.model_name} on {gpu_device.upper()}")
            return
        except Exception as e:
            logger.warning("st_load_failed", f"SentenceTransformer load failed: {e}")

        # Fallback to FlagEmbedding if SentenceTransformer not available
        try:
            from FlagEmbedding import BGEM3FlagModel
            use_fp16 = (gpu_device == "cuda")
            self._model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16, device=gpu_device)
            logger.info("bgem3_loaded", f"Loaded FlagEmbedding model {self.model_name} on {gpu_device.upper()} (fp16={use_fp16})")
            return
        except Exception as e:
            logger.warning("bgem3_load_failed", f"FlagEmbedding load failed: {e}")

        logger.warning("embedder_init_fallback", f"No embedding backend found for {self.model_name}, fallback dummy mode")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of document texts in batches.
        """
        if not texts:
            return []
        
        # Handle empty/blank texts
        processed_texts = [t.strip() if t and t.strip() else "(empty)" for t in texts]
        
        if self._model is not None:
            try:
                if hasattr(self._model, "encode"):
                    model_type = type(self._model).__name__
                    b_size = int(self.batch_size)
                    if "BGEM3" in model_type or "Flag" in model_type or "M3Embedder" in model_type:
                        # Use GPU if model is on GPU, otherwise CPU
                        import torch
                        encode_device = "cuda" if torch.cuda.is_available() else "cpu"
                        output = self._model.encode(processed_texts, batch_size=b_size, max_length=8192, device=encode_device)
                        dense_vecs = output["dense_vecs"] if isinstance(output, dict) and "dense_vecs" in output else output
                    else:
                        try:
                            output = self._model.encode(
                                processed_texts, 
                                batch_size=b_size,
                                normalize_embeddings=self.normalize_embeddings,
                            )
                        except TypeError:
                            output = self._model.encode(processed_texts, batch_size=b_size)
                        # Always extract dense_vecs from dict output (safety)
                        dense_vecs = output["dense_vecs"] if isinstance(output, dict) and "dense_vecs" in output else output
                    
                    results = []
                    for vec in dense_vecs:
                        v_list = vec.tolist() if hasattr(vec, "tolist") else [float(x) for x in vec]
                        if self.normalize_embeddings:
                            v_list = normalize_l2(v_list)
                        self._validate_dimension(v_list)
                        results.append(v_list)
                    return results
            except Exception as e:
                logger.error("embedding_error", f"Embedding generation failed: {e}")
                raise ModelError(f"Embedding failed: {e}") from e
        
        # Fallback dummy embeddings (for testing/offline environments)
        results = []
        for text in processed_texts:
            dummy_vec = [0.01 * (i % 10 + 1) for i in range(self._expected_dim)]
            if self.normalize_embeddings:
                dummy_vec = normalize_l2(dummy_vec)
            results.append(dummy_vec)
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        res = self.embed_documents([text])
        return res[0] if res else [0.0] * self._expected_dim

    @property
    def dimension(self) -> int:
        return self._expected_dim

    def _validate_dimension(self, vector: list[float]):
        """Ensure vector dimension matches expected 1024."""
        if len(vector) != self._expected_dim:
            raise ModelError(f"Vector dimension mismatch: expected {self._expected_dim}, got {len(vector)}")


def get_embedder(config) -> BgeM3Embedder:
    """Factory function for BgeM3Embedder."""
    emb_cfg = config.models.embedding
    return BgeM3Embedder(
        model_name=emb_cfg.model_id,
        normalize_embeddings=emb_cfg.normalize,
    )
