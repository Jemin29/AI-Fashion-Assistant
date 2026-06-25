"""
week5/tests/test_fashion_embeddings.py
======================================
Unit tests for the Fashion Embedding Engine.
Verifies model configuration, generation of style/trend/brand embeddings, and saving/loading npz.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.config_manager import EmbeddingConfig
from src.rag.embeddings.fashion_embeddings import FashionEmbeddingEngine
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder


class TestFashionEmbeddingEngine:
    """Validate BAAI model configuration, bulk fashion object embedding, and serialization."""

    def test_engine_initialization_and_model(self):
        """Verify model name is explicitly configured as BAAI/bge-small-en-v1.5."""
        cfg = EmbeddingConfig()
        # Even if config is default, the engine must force bge-small-en-v1.5
        engine = FashionEmbeddingEngine(config=cfg, force_mock=True)
        assert engine.config.model_name == "BAAI/bge-small-en-v1.5"

    def test_generate_save_load_cycle(self):
        """Verify embeddings generate correctly, save to disk, and load back accurately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            kb = FashionKnowledgeBase(db_path=tmp_path / "kb.json")
            trend_builder = TrendDatasetBuilder(db_path=tmp_path / "trends.json")

            engine = FashionEmbeddingEngine(
                kb=kb,
                trend_builder=trend_builder,
                force_mock=True
            )

            # Generate
            embs = engine.generate_embeddings()
            assert len(embs) > 0
            
            # Assert styles, brands, trends, and forecasts (recommendations) are present
            # Styles (e.g. streetwear)
            assert "kb_fashion_styles_streetwear" in embs
            # Brands (e.g. nike)
            assert "kb_brand_profiles_nike" in embs
            # Trends (e.g. utility cargoes)
            assert "trend_streetwear_oversized_utility_cargoes" in embs
            # Recommendations/Forecasts (e.g. biophilic responsive textiles)
            assert "trend_forecast_biophilic_responsive_textiles" in embs

            # Save
            save_path = tmp_path / "fashion_embs.npz"
            engine.save_embeddings(save_path)
            assert save_path.exists()

            # Load in a fresh engine
            engine_new = FashionEmbeddingEngine(
                kb=kb,
                trend_builder=trend_builder,
                force_mock=True
            )
            loaded_embs = engine_new.load_embeddings(save_path)
            
            assert len(loaded_embs) == len(embs)
            assert "kb_brand_profiles_nike" in loaded_embs
            np.testing.assert_allclose(loaded_embs["kb_brand_profiles_nike"], embs["kb_brand_profiles_nike"])

    def test_engine_edge_cases(self):
        """Verify empty generation, saving empty cache, missing file loading, and errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Setup completely empty databases (by resolve to non-existent custom temp paths without seeding)
            kb_empty = FashionKnowledgeBase(db_path=tmp_path / "empty_kb.json")
            kb_empty.items.clear()
            
            tb_empty = TrendDatasetBuilder(db_path=tmp_path / "empty_tb.json")
            tb_empty.trends.clear()
            
            engine = FashionEmbeddingEngine(
                kb=kb_empty,
                trend_builder=tb_empty,
                force_mock=True
            )
            
            # 1. Generate empty
            res = engine.generate_embeddings()
            assert len(res) == 0
            
            # 2. Save empty does not crash and does not write
            save_path = tmp_path / "will_not_exist.npz"
            engine.save_embeddings(save_path)
            assert not save_path.exists()
            
            # 3. Load non-existent returns empty dict
            loaded = engine.load_embeddings(tmp_path / "non_existent.npz")
            assert len(loaded) == 0

            # 4. Save error (via mock patch)
            with pytest.raises(Exception):
                engine.embeddings_dict = {"a": np.array([1.0], dtype=np.float32)}
                import unittest.mock as mock
                with mock.patch("numpy.savez_compressed", side_effect=IOError("Mock write error")):
                    engine.save_embeddings(tmp_path / "test_error.npz")
            
            # 5. Load error (via mock patch)
            with pytest.raises(Exception):
                test_file = tmp_path / "corrupted.npz"
                test_file.touch()
                import unittest.mock as mock
                with mock.patch("numpy.load", side_effect=ValueError("Corrupted format")):
                    engine.load_embeddings(test_file)


