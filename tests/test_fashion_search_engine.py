"""
week5/tests/test_fashion_search_engine.py
=========================================
Unit tests for the Fashion Search Engine.
"""

from __future__ import annotations

import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.retrieval.fashion_search_engine import FashionSearchEngine
from src.rag.vector_db.chromadb_manager import ChromaDbManager


@pytest.fixture
def mock_db_manager():
    """ChromaDbManager in mock mode."""
    return ChromaDbManager(force_mock=True)


@pytest.fixture
def mock_retriever(mock_db_manager):
    """FashionRetriever initialized with mock engines."""
    embedder = EmbeddingsGenerator(force_mock=True)
    return FashionRetriever(embedder=embedder, db_manager=mock_db_manager)


class TestFashionSearchEngine:
    """Validate multi-type searches (keyword, semantic, style, brand, trend), metadata filtering, ranking, and sorting."""

    def test_search_engine_initialization(self, mock_retriever, mock_db_manager):
        """Verify FashionSearchEngine initializes correctly."""
        engine = FashionSearchEngine(retriever=mock_retriever, db_manager=mock_db_manager)
        assert engine.retriever == mock_retriever
        assert engine.db_manager == mock_db_manager

    def test_search_keyword(self, mock_retriever, mock_db_manager):
        """Verify keyword-based text searching routing."""
        engine = FashionSearchEngine(retriever=mock_retriever, db_manager=mock_db_manager)
        col_name = "fashion_styles"

        # Insert test styles
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["style_1", "style_2"],
            documents=["Denim jacket casual streetwear wear.", "Tailored luxury formal suit."],
            metadatas=[{"popularity_score": 0.8}, {"popularity_score": 0.5}]
        )

        # Keyword search
        results = engine.search(query="streetwear", search_type="keyword", limit=2)
        assert len(results) > 0
        assert "Denim jacket" in results[0]["document"]

    def test_search_semantic_and_categories(self, mock_retriever, mock_db_manager):
        """Verify semantic, style, brand, and trend category searches routing."""
        engine = FashionSearchEngine(retriever=mock_retriever, db_manager=mock_db_manager)

        # 1. Semantic search
        results_sem = engine.search(query="casual sneakers", search_type="semantic", limit=2)
        # Mock database contains Denim jacket, suit, etc., should return ranked candidates
        assert len(results_sem) <= 2

        # Seed brand knowledge
        mock_db_manager.insert_documents(
            collection_name="brand_knowledge",
            ids=["brand_1"],
            documents=["Nike athletic streetwear brand profile."],
            metadatas=[{"popularity_score": 0.9}]
        )

        # 2. Brand search
        results_brand = engine.search(query="Nike", search_type="brand", limit=1)
        assert len(results_brand) == 1
        assert "Nike" in results_brand[0]["document"]

        # Seed trends
        mock_db_manager.insert_documents(
            collection_name="trends",
            ids=["trend_1"],
            documents=["Hypebeast windbreaker layering trends."],
            metadatas=[{"growth_rate": 0.25}]
        )

        # 3. Trend search
        results_trend = engine.search(query="layering", search_type="trend", limit=1)
        assert len(results_trend) == 1
        assert "windbreaker" in results_trend[0]["document"]

    def test_search_filtering(self, mock_retriever, mock_db_manager):
        """Verify metadata filters mapping in search query calls."""
        engine = FashionSearchEngine(retriever=mock_retriever, db_manager=mock_db_manager)

        mock_db_manager.insert_documents(
            collection_name="fashion_styles",
            ids=["style_A", "style_B"],
            documents=["Red winter coat jacket.", "Blue summer crop top jacket."],
            metadatas=[{"color": "red", "season": "winter"}, {"color": "blue", "season": "summer"}]
        )

        # Query with filter matching color red
        filtered_results = engine.search(
            query="jacket",
            search_type="style",
            filter_metadata={"color": "red"},
            limit=2
        )
        assert len(filtered_results) == 1
        assert filtered_results[0]["id"] == "style_A"
        assert filtered_results[0]["metadata"]["color"] == "red"

    def test_search_sorting(self, mock_retriever, mock_db_manager):
        """Verify sorting search outputs by root-level and nested metadata attributes."""
        engine = FashionSearchEngine(retriever=mock_retriever, db_manager=mock_db_manager)

        # Clear and seed documents
        mock_db_manager.delete_documents(collection_name="fashion_styles", ids=["style_1", "style_2", "style_A", "style_B"])
        mock_db_manager.insert_documents(
            collection_name="fashion_styles",
            ids=["item_low", "item_high"],
            documents=["Standard wardrobe basic tee.", "Luxurious designer blazer jacket."],
            metadatas=[{"popularity_score": 0.3, "price": 20}, {"popularity_score": 0.9, "price": 500}]
        )

        # 1. Sort by root blended_score descending (ascending=False)
        weights = {"vector_similarity": 0.0, "popularity": 1.0, "growth_rate": 0.0}
        results_blend_desc = engine.search(query="wardrobe blazer", sort_by="blended_score", scoring_weights=weights, ascending=False)
        assert len(results_blend_desc) == 2
        # Blended score is computed from similarity + popularity. item_high has popularity 0.9, item_low has 0.3.
        assert results_blend_desc[0]["id"] == "item_high"

        # 2. Sort by root blended_score ascending (ascending=True)
        results_blend_asc = engine.search(query="wardrobe blazer", sort_by="blended_score", scoring_weights=weights, ascending=True)
        assert results_blend_asc[0]["id"] == "item_low"

        # 3. Sort by nested metadata attribute: 'price' descending
        results_price_desc = engine.search(query="wardrobe blazer", sort_by="price", ascending=False)
        assert results_price_desc[0]["metadata"]["price"] == 500
        assert results_price_desc[1]["metadata"]["price"] == 20

        # 4. Sort by non-existent key fallback handling (should handle gracefully)
        results_fallback = engine.search(query="wardrobe blazer", sort_by="non_existent_key")
        assert len(results_fallback) == 2

        # 5. Sort by incompatible types (triggers exception handling)
        mock_db_manager.delete_documents(collection_name="fashion_styles", ids=["item_low", "item_high"])
        mock_db_manager.insert_documents(
            collection_name="fashion_styles",
            ids=["item_str", "item_num"],
            documents=["Item with string price.", "Item with numerical price."],
            metadatas=[{"price": "free"}, {"price": 10.0}]
        )
        # Should complete successfully despite the TypeErrors during sort comparisons
        results_incompat = engine.search(query="price", sort_by="price")
        assert len(results_incompat) == 2

        # 6. Test search empty results
        mock_db_manager.delete_documents(collection_name="fashion_styles", ids=["item_str", "item_num"])
        results_empty = engine.search(query="nonexistent_query_term_xyz", limit=1)
        # Mock search handles empty matches by returning empty lists
        assert len(results_empty) == 0

