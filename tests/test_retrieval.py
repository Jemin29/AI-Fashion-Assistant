"""
week5/tests/test_retrieval.py
============================
Unit tests for the Keyword and Hybrid Retriever modules.
Verifies TF-IDF matching, dense vector routing, and score normalization/blending.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.utils.config_manager import RetrievalConfig
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder
from src.rag.retrieval.hybrid_retriever import HybridRetriever, KeywordRetriever
from src.rag.vector_db.vector_indexer import VectorIndexer


class TestHybridRetriever:
    """Validate TF-IDF keyword tokenization and hybrid dense scoring combinations."""

    def test_keyword_retriever_fit_and_search(self):
        """Verify TF-IDF term calculations and matching normalization."""
        kr = KeywordRetriever()
        
        # Build mock database records
        db = FashionKnowledgeBase(db_path=Path(tempfile.gettempdir()) / "test_kb_kr.json")
        items = db.list_items()
        
        kr.fit(items)
        assert len(kr.docs) == len(items)
        assert len(kr.doc_tfs) == len(items)

        # Search matching term
        results = kr.search("streetwear cargoes", top_k=5)
        # Streetwear item should have score 1.0 (since min-max scales the single match)
        assert len(results) > 0
        assert results[0][1] == 1.0

        # Non-matching search returns empty
        empty = kr.search("nonexistenttermqueryword", top_k=5)
        assert len(empty) == 0

    def test_hybrid_retriever_routing(self):
        """Verify keyword/vector weight distribution and combined scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            kb = FashionKnowledgeBase(db_path=tmp_path / "kb.json")
            trend_builder = TrendDatasetBuilder(db_path=tmp_path / "trends.json")
            
            embedder = EmbeddingsGenerator(force_mock=True)
            indexer = VectorIndexer(dimension=384)
            
            # 1. Pure Keyword retrieval setup (keyword_weight=1.0, vector_weight=0.0)
            cfg_kw = RetrievalConfig(hybrid_search=False, keyword_weight=1.0, vector_weight=0.0)
            retriever_kw = HybridRetriever(
                kb=kb,
                embedder=embedder,
                indexer=indexer,
                trend_builder=trend_builder,
                config=cfg_kw
            )
            
            # Search streetwear
            res_kw = retriever_kw.retrieve("Streetwear hoodies", top_k=3)
            assert len(res_kw) > 0
            assert res_kw[0][0].id.startswith("kb_") or res_kw[0][0].id.startswith("trend_")

            # 2. Pure Vector retrieval setup (keyword_weight=0.0, vector_weight=1.0)
            cfg_vec = RetrievalConfig(hybrid_search=False, keyword_weight=0.0, vector_weight=1.0)
            retriever_vec = HybridRetriever(
                kb=kb,
                embedder=embedder,
                indexer=indexer,
                trend_builder=trend_builder,
                config=cfg_vec
            )
            res_vec = retriever_vec.retrieve("Linen Pants", top_k=3)
            assert len(res_vec) > 0

            # 3. Hybrid search (0.5 keyword, 0.5 vector)
            cfg_hy = RetrievalConfig(hybrid_search=True, keyword_weight=0.5, vector_weight=0.5)
            retriever_hy = HybridRetriever(
                kb=kb,
                embedder=embedder,
                indexer=indexer,
                trend_builder=trend_builder,
                config=cfg_hy
            )
            res_hy = retriever_hy.retrieve("Nike Gowns", top_k=5)
            assert len(res_hy) > 0
            # Scores should be blended and normalized
            assert all(0.0 <= score <= 1.0 for _, score in res_hy)
