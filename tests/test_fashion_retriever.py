"""
week5/tests/test_fashion_retriever.py
=====================================
Unit tests for the Fashion Retrieval Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
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


class TestFashionRetriever:
    """Validate retrieval routing, similarity search, and weighted ranking functions."""

    def test_retriever_initialization(self, mock_embedder, mock_db_manager):
        """Verify the retriever initializes correctly."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)
        assert retriever.embedder == mock_embedder
        assert retriever.db_manager == mock_db_manager

    def test_retrieve_routing(self, mock_embedder, mock_db_manager):
        """Verify retrieve() routes search_type correctly to defaults and override collections."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)

        # Pre-populate collections
        collections = ["trends", "fashion_styles", "brand_knowledge", "custom_col"]
        for col in collections:
            mock_db_manager.insert_documents(
                collection_name=col,
                ids=[f"{col}_item_1", f"{col}_item_2"],
                documents=[f"Content from {col} item one", f"Content from {col} item two"],
                metadatas=[{"popularity_score": 0.9}, {"popularity_score": 0.3}]
            )

        # 1. Semantic Search (defaults to fashion_styles)
        res_semantic = retriever.retrieve("streetwear hoodie", search_type="semantic")
        assert len(res_semantic) == 2
        assert "fashion_styles_item" in res_semantic[0]["id"]

        # 2. Trend Search (defaults to trends)
        res_trend = retriever.retrieve("oversized pants", search_type="trend")
        assert len(res_trend) == 2
        assert "trends_item" in res_trend[0]["id"]

        # 3. Style Search (defaults to fashion_styles)
        res_style = retriever.retrieve("vintage denim", search_type="style")
        assert len(res_style) == 2
        assert "fashion_styles_item" in res_style[0]["id"]

        # 4. Brand Search (defaults to brand_knowledge)
        res_brand = retriever.retrieve("sustainable cotton", search_type="brand")
        assert len(res_brand) == 2
        assert "brand_knowledge_item" in res_brand[0]["id"]

        # 5. Collection override
        res_override = retriever.retrieve("luxury gown", collection_name="custom_col")
        assert len(res_override) == 2
        assert "custom_col_item" in res_override[0]["id"]

    def test_search_similar(self, mock_embedder, mock_db_manager):
        """Verify search_similar() retrieves details and excludes the target item itself."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)
        col_name = "similarity_collection"

        # Seed items
        ids = ["doc_A", "doc_B", "doc_C"]
        docs = ["Black leather boots", "Brown leather shoes", "Red velvet dress"]
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=ids,
            documents=docs
        )

        # Find items similar to doc_A. It should exclude doc_A.
        results = retriever.search_similar(item_id="doc_A", collection_name=col_name, n_results=2)
        assert len(results) == 2
        # Verify self-exclusion
        assert not any(item["id"] == "doc_A" for item in results)
        assert set(item["id"] for item in results) == {"doc_B", "doc_C"}

        # Search for non-existent item should fail gracefully
        res_missing = retriever.search_similar(item_id="doc_missing", collection_name=col_name)
        assert len(res_missing) == 0

    def test_rank_results(self, mock_embedder, mock_db_manager):
        """Verify weighted blending of vector similarity and metadata parameters."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)

        # Mock query outputs
        results = [
            {
                "id": "item_1",
                "document": "High quality wool sweater",
                "distance": 0.4,  # Similarity = 1.0 - 0.4 = 0.6
                "metadata": {"popularity": "high", "growth_rate": 0.8}  # popularity maps to 1.0
            },
            {
                "id": "item_2",
                "document": "Heavy denim jacket",
                "distance": 0.1,  # Similarity = 1.0 - 0.1 = 0.9
                "metadata": {"popularity": "low", "growth_rate": -0.2}  # popularity maps to 0.1
            }
        ]

        # 1. Test Default Weights: similarity=0.8, popularity=0.2, growth_rate=0.0
        # item_1 blended score: 0.8 * 0.6 + 0.2 * 1.0 = 0.48 + 0.2 = 0.68
        # item_2 blended score: 0.8 * 0.9 + 0.2 * 0.1 = 0.72 + 0.02 = 0.74
        # item_2 should rank first
        ranked_default = retriever.rank_results(results)
        assert len(ranked_default) == 2
        assert ranked_default[0]["id"] == "item_2"
        assert abs(ranked_default[0]["blended_score"] - 0.74) < 1e-5
        assert abs(ranked_default[1]["blended_score"] - 0.68) < 1e-5

        # 2. Test Popularity Dominant weights: similarity=0.2, popularity=0.8, growth_rate=0.0
        # item_1 blended score: 0.2 * 0.6 + 0.8 * 1.0 = 0.12 + 0.80 = 0.92
        # item_2 blended score: 0.2 * 0.9 + 0.8 * 0.1 = 0.18 + 0.08 = 0.26
        # item_1 should rank first
        scoring_weights = {"vector_similarity": 0.2, "popularity": 0.8, "growth_rate": 0.0}
        ranked_pop = retriever.rank_results(results, scoring_weights=scoring_weights)
        assert ranked_pop[0]["id"] == "item_1"
        assert abs(ranked_pop[0]["blended_score"] - 0.92) < 1e-5

        # 3. Test Growth Dominant weights: similarity=0.0, popularity=0.0, growth_rate=1.0
        # item_1 normalized growth: (0.8 + 1.0)/2 = 0.9 -> score = 0.9
        # item_2 normalized growth: (-0.2 + 1.0)/2 = 0.4 -> score = 0.4
        # item_1 should rank first
        weights_growth = {"vector_similarity": 0.0, "popularity": 0.0, "growth_rate": 1.0}
        ranked_growth = retriever.rank_results(results, scoring_weights=weights_growth)
        assert ranked_growth[0]["id"] == "item_1"
        assert abs(ranked_growth[0]["blended_score"] - 0.9) < 1e-5

    def test_retrieve_empty_query(self, mock_embedder, mock_db_manager):
        """Verify empty or None query strings return empty lists gracefully."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)
        assert retriever.retrieve("") == []
        assert retriever.retrieve(None) == []

    def test_rank_results_empty(self, mock_embedder, mock_db_manager):
        """Verify rank_results handles empty inputs and invalid weight totals."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)
        assert retriever.rank_results([]) == []

        # Zero weights fallback (should default to vector similarity only)
        results = [
            {"id": "item_1", "distance": 0.2},
            {"id": "item_2", "distance": 0.8}
        ]
        zero_weights = {"vector_similarity": 0.0, "popularity": 0.0, "growth_rate": 0.0}
        ranked = retriever.rank_results(results, scoring_weights=zero_weights)
        assert len(ranked) == 2
        assert ranked[0]["id"] == "item_1"

    def test_rank_results_invalid_growth(self, mock_embedder, mock_db_manager):
        """Verify rank_results handles non-numeric growth_rate values gracefully."""
        retriever = FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)
        results = [{
            "id": "item_A",
            "distance": 0.5,
            "metadata": {"growth_rate": "invalid_string_rate"}
        }]
        weights = {"vector_similarity": 0.0, "popularity": 0.0, "growth_rate": 1.0}
        ranked = retriever.rank_results(results, scoring_weights=weights)
        # Invalid growth defaults to 0.0, which normalizes to (0.0 + 1.0)/2 = 0.5
        assert ranked[0]["growth_rate_normalized"] == 0.5

    def test_retriever_native_path(self, mock_embedder):
        """Verify the retriever works correctly on native ChromaDB client."""
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_manager = ChromaDbManager(persist_directory=tmpdir, force_mock=False)
            if db_manager.is_mock_mode:
                pytest.skip("Native ChromaDB client is not available in this environment.")

            retriever = FashionRetriever(embedder=mock_embedder, db_manager=db_manager)
            col_name = "native_similarity"

            # Seed items with actual float list embeddings
            ids = ["item_X", "item_Y"]
            docs = ["Striped silk blouse", "Pleated skirt"]
            embeddings = np.random.randn(2, 384).astype(np.float32)

            db_manager.insert_documents(
                collection_name=col_name,
                ids=ids,
                documents=docs,
                embeddings=embeddings
            )

            # Test search_similar which hits native path (collection.get)
            results = retriever.search_similar(item_id="item_X", collection_name=col_name, n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "item_Y"

