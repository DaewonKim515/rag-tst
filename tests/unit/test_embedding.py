"""
Unit tests for BgeM3Embedder in models/embedding.py.
"""

import pytest
from rag_app.models.embedding import BgeM3Embedder, normalize_l2


def test_normalize_l2():
    vec = [3.0, 4.0]
    norm_vec = normalize_l2(vec)
    assert pytest.approx(norm_vec[0]) == 0.6
    assert pytest.approx(norm_vec[1]) == 0.8


def test_bge_m3_embedder_dummy_fallback():
    embedder = BgeM3Embedder(model_name="non_existent_model_name_for_fallback")
    assert embedder.dimension == 1024
    
    texts = ["안녕하세요", "RAG 시스템 구현 테스트입니다."]
    embeddings = embedder.embed_documents(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert len(embeddings[1]) == 1024


def test_bge_m3_embed_query():
    embedder = BgeM3Embedder(model_name="non_existent_model_name_for_fallback")
    query_vec = embedder.embed_query("테스트 질의")
    
    assert len(query_vec) == 1024
