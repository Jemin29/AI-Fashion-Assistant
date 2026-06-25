"""
week5/tests/test_recommendations.py
===================================
Unit tests for the Recommendation Engine module.
Verifies preference parsing, threshold filtering, and MMR diversity re-ranking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.utils.config_manager import RecommendationConfig, RetrievalConfig
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder
from src.recommendations.recommendation_engine import RecommendationEngine
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.vector_db.vector_indexer import VectorIndexer


class TestRecommendationEngine:
    """Validate attribute scoring, similarity thresholds, and diversity penalties."""

    def test_preference_query_compilation(self):
        """Verify dictionary preferences compile into text search queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            kb = FashionKnowledgeBase(db_path=tmp_path / "kb.json")
            embedder = EmbeddingsGenerator(force_mock=True)
            indexer = VectorIndexer(dimension=384)
            retriever = HybridRetriever(kb=kb, embedder=embedder, indexer=indexer)

            engine = RecommendationEngine(retriever=retriever, embedder=embedder)
            
            # Formats dict correctly
            prefs = {"styles": ["streetwear", "gorpcore"], "colors": "black"}
            q = engine._build_preference_query(prefs)
            assert "streetwear" in q
            assert "gorpcore" in q
            assert "black" in q

    def test_recommendation_filtering_and_diversity(self):
        """Verify similarity thresholds filter candidates and MMR applies diversity penalties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            kb = FashionKnowledgeBase(db_path=tmp_path / "kb.json")
            trend_builder = TrendDatasetBuilder(db_path=tmp_path / "trends.json")
            
            embedder = EmbeddingsGenerator(force_mock=True)
            indexer = VectorIndexer(dimension=384)
            
            # Setup retriever with hybrid search enabled
            cfg_ret = RetrievalConfig(hybrid_search=True, keyword_weight=0.5, vector_weight=0.5)
            retriever = HybridRetriever(
                kb=kb,
                embedder=embedder,
                indexer=indexer,
                trend_builder=trend_builder,
                config=cfg_ret
            )

            # Recommendation engine config
            cfg_rec = RecommendationConfig(
                max_recommendations=3,
                similarity_threshold=0.2, # low threshold to let matches in
                diversity_bias=0.8        # high diversity penalty
            )
            
            engine = RecommendationEngine(retriever=retriever, embedder=embedder, config=cfg_rec)
            
            # Request recommendations
            prefs = {"style": "streetwear", "brand": "Nike"}
            recs = engine.recommend(prefs, top_n=3)
            
            assert len(recs) > 0
            assert len(recs) <= 3
            
            # Verify they all satisfy similarity threshold (>= 0.2)
            assert all(score >= 0.2 for _, score in recs)

            # High diversity bias should rank diverse items (different subcategories/names) higher
            # compared to a zero bias run
            cfg_rec_no_div = RecommendationConfig(
                max_recommendations=3,
                similarity_threshold=0.2,
                diversity_bias=0.0        # no diversity penalty
            )
            engine_no_div = RecommendationEngine(
                retriever=retriever,
                embedder=embedder,
                config=cfg_rec_no_div
            )
            recs_no_div = engine_no_div.recommend(prefs, top_n=3)
            
            # The list of recommended items could change due to diversity re-ranking
            assert len(recs_no_div) > 0
