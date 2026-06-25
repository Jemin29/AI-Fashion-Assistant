"""
week5/tests/test_brand_recommender.py
=====================================
Unit tests for the Fashion Brand Recommendation Engine.
"""

from __future__ import annotations

import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.recommendations.brand_recommender import BrandRecommender
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


class TestBrandRecommender:
    """Validate brand rule matches, fallbacks, semantic similarity querying, and personalized blends."""

    def test_recommender_initialization(self, mock_retriever, mock_db_manager):
        """Verify BrandRecommender initializes correctly."""
        recommender = BrandRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        assert recommender.retriever == mock_retriever
        assert recommender.db_manager == mock_db_manager

    def test_recommend_by_rules(self, mock_retriever, mock_db_manager):
        """Verify brand style rule matches and fallback mappings."""
        recommender = BrandRecommender(retriever=mock_retriever, db_manager=mock_db_manager)

        # 1. Exact streetwear match
        profile_1 = {"favorite_style": "streetwear"}
        recs_1 = recommender.recommend_by_rules(profile_1)
        assert recs_1 == ["Nike", "Supreme", "Stussy"]

        # 2. Exact luxury match
        profile_2 = {"style": "luxury"}
        recs_2 = recommender.recommend_by_rules(profile_2)
        assert recs_2 == ["Gucci", "Prada", "Louis Vuitton"]

        # 3. Partial key match
        profile_3 = {"favorite_style": "modern minimalist"}
        recs_3 = recommender.recommend_by_rules(profile_3)
        assert recs_3 == ["Zara", "Uniqlo", "COS"]

        # 4. Default fallback (unrecognized style)
        profile_4 = {"favorite_style": "boho-chic"}
        recs_4 = recommender.recommend_by_rules(profile_4)
        assert recs_4 == ["Nike", "Zara", "Gucci", "H&M"]

    def test_recommend_by_similarity(self, mock_retriever, mock_db_manager):
        """Verify similarity query mappings from database records."""
        recommender = BrandRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        col_name = "brand_knowledge"

        # 1. Test empty db fallback
        profile_empty = {"favorite_style": "vintage"}
        recs_empty = recommender.recommend_by_similarity(profile_empty, n_results=2)
        assert recs_empty == ["Nike", "Zara"]

        # 2. Test mock results extraction using metadata keys
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["brand_1", "brand_2"],
            documents=["Nike athletic footwear and sportswear.", "Adidas activewear and sneakers."],
            metadatas=[{"brand": "Nike"}, {"name": "Adidas"}]
        )

        recs_sim = recommender.recommend_by_similarity({"favorite_style": "athletic"}, n_results=2)
        assert len(recs_sim) == 2
        assert "Nike" in recs_sim
        assert "Adidas" in recs_sim

        # 3. Test fallback to document text parsing if metadata is missing brand/name/category keys
        # Clear previous documents first
        mock_db_manager.delete_documents(collection_name=col_name, ids=["brand_1", "brand_2"])
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["brand_3"],
            documents=["Puma is a global sports brand."],
            metadatas=[{}]  # Empty metadata
        )
        recs_text = recommender.recommend_by_similarity({"favorite_style": "sports"}, n_results=1)
        assert len(recs_text) == 1
        assert recs_text[0] == "Puma is a global sports brand" or recs_text[0] == "Puma"

        # 4. Test profile parameters as list values
        recs_list = recommender.recommend_by_similarity({"colors": ["black", "grey"]}, n_results=1)
        assert len(recs_list) == 1

        # 5. Test parsing very long brand names in document
        mock_db_manager.delete_documents(collection_name=col_name, ids=["brand_3"])
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["brand_4"],
            documents=["BrandWithVeryLongNameOfMoreThanThirtyCharacters AndAdditionalText."],
            metadatas=[{}]  # Empty metadata
        )
        recs_long = recommender.recommend_by_similarity({"favorite_style": "long"}, n_results=1)
        assert len(recs_long) == 1
        assert recs_long[0] == "BrandWithVeryLongNameOfMoreThanThirtyCharacters"

        # 6. Test empty profile inputs fallback
        recs_empty_profile = recommender.recommend_by_similarity({}, n_results=2)
        assert recs_empty_profile == ["Nike", "Zara"]

    def test_recommend_personalized(self, mock_retriever, mock_db_manager):
        """Verify personalized blending returns rules first, similarity second, with deduplication."""
        recommender = BrandRecommender(retriever=mock_retriever, db_manager=mock_db_manager)
        col_name = "brand_knowledge"

        # Seed database brands
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["brand_A", "brand_B"],
            documents=["Nike athletic garments.", "Champion vintage sweatshirts."],
            metadatas=[{"brand": "Nike"}, {"brand": "Champion"}]
        )

        # Style rule output for streetwear: ["Nike", "Supreme", "Stussy"]
        # Similarity output: ["Nike", "Champion"]
        # Blended deduped output: ["Nike", "Supreme", "Stussy", "Champion"]
        profile = {"favorite_style": "streetwear"}
        recs = recommender.recommend_personalized(profile, n_results=4)

        assert len(recs) == 4
        # First 3 should match expert rules
        assert recs[0] == "Nike"
        assert recs[1] == "Supreme"
        assert recs[2] == "Stussy"
        # 4th should be similarity unique match
        assert recs[3] == "Champion"
