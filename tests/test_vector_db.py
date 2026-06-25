"""
week5/tests/test_vector_db.py
=============================
Unit tests for the FAISS Vector Indexer module.
Verifies L2/IP search metrics, validation rules, and disk serialization cycles.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.config_manager import VectorDbConfig
from src.rag.vector_db.vector_indexer import VectorIndexer


class TestVectorIndexer:
    """Validate FAISS search accuracy, index settings, serialization, and exceptions."""

    def test_indexer_initialization_and_clear(self):
        """Verify default metrics match configurations and clear resets values."""
        cfg = VectorDbConfig(index_type="FlatL2")
        indexer = VectorIndexer(config=cfg, dimension=128)
        assert indexer.dimension == 128
        assert indexer.index.ntotal == 0
        assert len(indexer.item_ids) == 0

        # Adding elements
        embeddings = np.random.randn(3, 128).astype(np.float32)
        indexer.add_items(["a", "b", "c"], embeddings)
        assert indexer.index.ntotal == 3

        # Reset
        indexer.clear()
        assert indexer.index.ntotal == 0
        assert len(indexer.item_ids) == 0

    def test_dimension_validation(self):
        """Verify adding vectors with mismatched dimensions raises ValueError."""
        indexer = VectorIndexer(dimension=64)
        embeddings = np.random.randn(2, 128).astype(np.float32)  # Dimension mismatch (128 vs 64)

        with pytest.raises(ValueError):
            indexer.add_items(["a", "b"], embeddings)

        with pytest.raises(ValueError):
            indexer.search(np.random.randn(128).astype(np.float32))

        # Size mismatch (3 ids vs 2 embeddings)
        with pytest.raises(ValueError):
            indexer.add_items(["a", "b", "c"], np.random.randn(2, 64).astype(np.float32))

    def test_l2_similarity_search(self):
        """Verify IndexFlatL2 L2 distances resolve nearest neighbors."""
        indexer = VectorIndexer(dimension=4)
        
        # We index 3 target vectors
        # v1 is exact match to query, v2 is close, v3 is far
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        indexer.add_items(["v1", "v2", "v3"], vectors)
        
        # Query matching v1 exactly
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = indexer.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "v1"
        assert abs(results[0][1] - 0.0) < 1e-5  # L2 distance is 0.0
        assert results[1][0] == "v2"

    def test_inner_product_similarity_search(self):
        """Verify IndexFlatIP normalized inner products evaluate cosine similarity."""
        cfg = VectorDbConfig(index_type="InnerProduct")
        indexer = VectorIndexer(config=cfg, dimension=4)
        
        # Index 3 vectors
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        indexer.add_items(["v1", "v2", "v3"], vectors)
        
        # Query along v1 direction
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = indexer.search(query, top_k=3)
        
        assert len(results) == 3
        # In InnerProduct/Cosine, higher scores mean closer similarity
        assert results[0][0] == "v1"
        assert abs(results[0][1] - 1.0) < 1e-5  # Max similarity is 1.0
        assert results[1][0] == "v2"
        assert results[2][0] == "v3"

    def test_disk_serialization(self):
        """Verify FAISS index binary and ID lists save/load cycle accurately."""
        cfg = VectorDbConfig(index_type="FlatL2", auto_save=False)
        indexer = VectorIndexer(config=cfg, dimension=4)
        
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        indexer.add_items(["id_1", "id_2"], vectors)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "index_dir"
            
            # Save
            indexer.save(storage_path)
            assert (storage_path / "index.faiss").exists()
            assert (storage_path / "mapping.json").exists()
            
            # Load in a fresh indexer
            indexer_loaded = VectorIndexer(config=cfg, dimension=4)
            indexer_loaded.load(storage_path)
            
            assert indexer_loaded.index.ntotal == 2
            assert indexer_loaded.item_ids == ["id_1", "id_2"]
            
            # Validate query match
            query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            results = indexer_loaded.search(query, top_k=1)
            assert results[0][0] == "id_2"
            assert abs(results[0][1] - 0.0) < 1e-5
