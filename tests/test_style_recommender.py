"""
week5/tests/test_style_recommender.py
=====================================
Unit tests for the Fashion Style Recommendation System.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.recommendations.style_recommender import StyleRecommender
from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.vector_db.chromadb_manager import ChromaDbManager


@pytest.fixture
def mock_embedder():
    """EmbeddingsGenerator in mock mode."""
    return EmbeddingsGenerator(force_mock=True)


@pytest.fixture
def mock_db_manager():
    """ChromaDbManager in mock mode."""
    return ChromaDbManager(force_mock=True)


@pytest.fixture
def mock_retriever(mock_embedder, mock_db_manager):
    """FashionRetriever initialized with mock engines."""
    return FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)


class TestStyleRecommender:
    """Validate rule matches, fallbacks, semantic similarity querying, and personalized blends."""

    def test_recommender_initialization(self, mock_retriever, mock_db_manager):
        """Verify StyleRecommender initializes correctly."""
        recommender = StyleRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        assert recommender.retriever == mock_retriever
        assert recommender.db_manager == mock_db_manager

    def test_recommend_by_rules(self, mock_retriever, mock_db_manager):
        """Verify style rule matches and fallback mappings."""
        recommender = StyleRecommender(retriever=mock_retriever, db_manager=mock_db_manager)

        # 1. Exact streetwear + black match
        prefs_1 = {"favorite_style": "streetwear", "favorite_color": "black"}
        recs_1 = recommender.recommend_by_rules(prefs_1)
        assert recs_1 == ["Techwear", "Urban Minimal", "Oversized Casual"]

        # 2. Exact streetwear + white match
        prefs_2 = {"favorite_style": "streetwear", "favorite_color": "white"}
        recs_2 = recommender.recommend_by_rules(prefs_2)
        assert recs_2 == ["Minimalist Street", "Skater Casual"]

        # 3. Style fallback (streetwear + unrecognized color red)
        prefs_3 = {"favorite_style": "streetwear", "favorite_color": "red"}
        recs_3 = recommender.recommend_by_rules(prefs_3)
        assert recs_3 == ["Oversized Casual", "Urban Minimal", "Skater Casual"]

        # 4. Default fallback (unrecognized style and color)
        prefs_4 = {"favorite_style": "boho-chic", "favorite_color": "green"}
        recs_4 = recommender.recommend_by_rules(prefs_4)
        assert recs_4 == ["Classic Casual", "Smart Chic", "Minimalist Modern"]

    def test_recommend_by_similarity(self, mock_retriever, mock_db_manager):
        """Verify similarity query mappings from database records."""
        recommender = StyleRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        col_name = "fashion_styles"

        # 1. Test empty db fallback
        prefs_empty = {"favorite_style": "vintage"}
        recs_empty = recommender.recommend_by_similarity(prefs_empty, n_results=2)
        assert recs_empty == ["Classic Casual", "Smart Chic"]

        # 2. Test mock results extraction
        # Insert style documents with metadata
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["style_1", "style_2"],
            documents=["Retro Americana styling with denim jackets.", "Bohemian Indie floral dresses."],
            metadatas=[{"style": "Retro Americana"}, {"style": "Bohemian Indie"}]
        )

        recs_sim = recommender.recommend_by_similarity({"favorite_style": "vintage"}, n_results=2)
        assert len(recs_sim) == 2
        assert "Retro Americana" in recs_sim
        assert "Bohemian Indie" in recs_sim

        # Test fallback to document text parsing if metadata is missing style/name keys
        # Clear previous styles so style_3 is the only candidate in the collection
        mock_db_manager.delete_documents(collection_name="fashion_styles", ids=["style_1", "style_2"])
        mock_db_manager.insert_documents(
            collection_name="fashion_styles",
            ids=["style_3"],
            documents=["Gothic Grunge leather attire."],
            metadatas=[{}]  # Empty metadata
        )
        recs_text = recommender.recommend_by_similarity({"favorite_style": "gothic"}, n_results=1)
        assert len(recs_text) == 1
        assert recs_text[0] == "Gothic Grunge leather attire"

        # Test empty input query fallback
        recs_empty_query = recommender.recommend_by_similarity({}, n_results=1)
        assert recs_empty_query == ["Classic Casual"]

    def test_recommend_personalized(self, mock_retriever, mock_db_manager):
        """Verify personalized blending returns rules first, similarity second, with deduplication."""
        recommender = StyleRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        col_name = "fashion_styles"

        # Seed database styles
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["style_A", "style_B"],
            documents=["Techwear high functional outerwear.", "Gorpcore trail styling."],
            metadatas=[{"style": "Techwear"}, {"style": "Gorpcore"}]
        )

        prefs = {"favorite_style": "streetwear", "favorite_color": "black"}
        # Rule output: ["Techwear", "Urban Minimal", "Oversized Casual"]
        # Similarity output: ["Techwear", "Gorpcore"]
        # Blended deduped: ["Techwear", "Urban Minimal", "Oversized Casual", "Gorpcore"]
        recs = recommender.recommend_personalized(prefs, n_results=4)
        
        assert len(recs) == 4
        # First 3 should match expert rule outputs exactly
        assert recs[0] == "Techwear"
        assert recs[1] == "Urban Minimal"
        assert recs[2] == "Oversized Casual"
        # 4th should be the similarity unique match
        assert recs[3] == "Gorpcore"
