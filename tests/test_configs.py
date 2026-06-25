"""
week5/tests/test_configs.py
===========================
Unit tests verifying the Week 5 Pydantic configuration validation system.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config_manager import (
    Week5Config,
    EmbeddingConfig,
    VectorDbConfig,
    RetrievalConfig,
    RecommendationConfig,
    TrendConfig,
    get_default_config
)


class TestWeek5Configurations:
    """Validate Pydantic configurations boundaries, yaml serializations, and defaults."""

    def test_default_config_loading(self):
        """Verify defaults match architectural specs."""
        cfg = get_default_config()
        assert isinstance(cfg, Week5Config)
        assert cfg.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.embeddings.dimension == 384
        assert cfg.embeddings.device == "cpu"
        assert cfg.vector_db.index_type == "FlatL2"
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.hybrid_search is True
        assert cfg.recommendations.max_recommendations == 10
        assert cfg.trends.time_window_days == 30

    def test_embedding_device_regex(self):
        """Verify device settings only accept cpu, cuda, or mps."""
        with pytest.raises(ValidationError):
            EmbeddingConfig(device="tpu")

        cfg = EmbeddingConfig(device="cuda")
        assert cfg.device == "cuda"

    def test_vector_db_index_type_regex(self):
        """Verify index_type settings only accept FlatL2 or InnerProduct."""
        with pytest.raises(ValidationError):
            VectorDbConfig(index_type="HNSW")

        cfg = VectorDbConfig(index_type="InnerProduct")
        assert cfg.index_type == "InnerProduct"

    def test_retrieval_weights_normalization(self):
        """Verify retrieval weights sum to exactly 1.0 (auto-normalization)."""
        cfg = RetrievalConfig(keyword_weight=0.2, vector_weight=0.6)
        # 0.2 + 0.6 = 0.8 -> normalized to 0.2 / 0.8 = 0.25 and 0.6 / 0.8 = 0.75
        assert abs(cfg.keyword_weight - 0.25) < 1e-4
        assert abs(cfg.vector_weight - 0.75) < 1e-4

    def test_recommendation_boundaries(self):
        """Verify recommendation threshold and similarity ranges."""
        with pytest.raises(ValidationError):
            RecommendationConfig(similarity_threshold=-0.1)

        with pytest.raises(ValidationError):
            RecommendationConfig(similarity_threshold=1.5)

        cfg = RecommendationConfig(similarity_threshold=0.8)
        assert cfg.similarity_threshold == 0.8

    def test_config_sync_output_paths(self):
        """Verify model validators synchronize paths dynamically."""
        cfg = Week5Config(output_root="custom_outputs")
        assert cfg.embeddings.cache_folder == str((Path("custom_outputs") / "embeddings" / "cache").as_posix())
        assert cfg.vector_db.storage_path == str((Path("custom_outputs") / "vector_db" / "faiss_index").as_posix())

    def test_yaml_serialization_cycle(self):
        """Verify configurations can save to YAML and parse back accurately."""
        cfg = Week5Config()
        cfg.embeddings.dimension = 768
        cfg.retrieval.top_k = 15
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_yaml = Path(tmpdir) / "config.yaml"
            cfg.save(temp_yaml)
            
            assert temp_yaml.exists()
            
            # Load back
            loaded = Week5Config.from_yaml(temp_yaml)
            assert loaded.embeddings.dimension == 768
            assert loaded.retrieval.top_k == 15
            assert loaded.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
