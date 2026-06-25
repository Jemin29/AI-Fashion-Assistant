"""
week5/tests/test_embeddings.py
==============================
Unit tests for the Embeddings Generator module.
Verifies mock embedding shapes, normalizations, and item representation vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.config_manager import EmbeddingConfig
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendItem


class TestEmbeddingsGenerator:
    """Validate embedding generation shapes, formats, and deterministic fallbacks."""

    def test_mock_embedding_initialization(self):
        """Verify generator initializes correctly in forced mock mode."""
        cfg = EmbeddingConfig(dimension=128)
        generator = EmbeddingsGenerator(config=cfg, force_mock=True)
        assert generator.is_mock_mode is True
        assert generator.config.dimension == 128

    def test_single_and_batch_embedding_shapes(self):
        """Verify vector shapes and array datatypes on mock outputs."""
        generator = EmbeddingsGenerator(force_mock=True)
        dim = generator.config.dimension

        # Single text
        vec = generator.embed_text("Minimalist Linen Pants")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (dim,)
        assert vec.dtype == np.float32

        # Batch texts
        texts = ["Streetwear Hoodie", "Luxury Silk Dress", "Summer Hat"]
        batch = generator.embed_batch(texts)
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (3, dim)
        assert batch.dtype == np.float32

        # Empty list check
        empty = generator.embed_batch([])
        assert empty.shape == (0, dim)

    def test_deterministic_and_normalized(self):
        """Verify that identical texts yield identical unit-length vectors."""
        generator = EmbeddingsGenerator(force_mock=True)
        
        # Deterministic check
        vec1 = generator.embed_text("Sneakers")
        vec2 = generator.embed_text("Sneakers")
        np.testing.assert_allclose(vec1, vec2, rtol=1e-6)

        # Different texts yield different vectors
        vec3 = generator.embed_text("Blazer")
        assert not np.allclose(vec1, vec3)

        # Unit length normalisation check (L2 norm should be 1.0)
        norm = np.linalg.norm(vec1)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_items(self):
        """Verify item-to-text formatting and encoding mapping."""
        generator = EmbeddingsGenerator(force_mock=True)
        dim = generator.config.dimension

        # Test KnowledgeItem
        k_item = KnowledgeItem(
            id="kb_style_streetwear",
            category="fashion_styles",
            name="Streetwear",
            content="Casual youth garments.",
            tags=["skate", "boxy"],
            metadata={"fit": "oversized"}
        )

        # Test TrendItem
        t_item = TrendItem(
            id="trend_luxury_velvet",
            category="luxury",
            name="Velvet Gowns",
            description="Opulent luxury gown trend."
        )

        embeddings = generator.embed_items([k_item, t_item])
        assert embeddings.shape == (2, dim)
        assert embeddings.dtype == np.float32
