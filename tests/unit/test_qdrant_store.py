"""
Unit tests for QdrantStore security and operations.
"""

from uuid import uuid4
import pytest

from rag_app.config.schema import AppConfig
from rag_app.domain.exceptions import ConfigurationError
from rag_app.indexing.qdrant_store import QdrantStore


def test_qdrant_production_security_validation():
    prod_config = AppConfig(environment="production")
    prod_config.qdrant.https = False  # Violation
    
    with pytest.raises(ConfigurationError):
        QdrantStore(prod_config)


def test_qdrant_store_mock_staging_and_delete():
    dev_config = AppConfig(environment="development")
    store = QdrantStore(dev_config)
    
    doc_id = uuid4()
    version_id = uuid4()
    
    # Mock staging and delete should complete without raising errors
    assert store.stage([]) is True
    assert store.delete_version(doc_id, version_id) is True
