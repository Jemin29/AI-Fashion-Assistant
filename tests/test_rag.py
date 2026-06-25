"""
week5/tests/test_rag.py
=======================
Unit tests for the central Fashion RAG Pipeline Coordinator.
Verifies prompt augmentations, RAG query loops, and grounded response citations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.utils.config_manager import get_default_config
from src.data.knowledge_base.fashion_knowledge_base import KnowledgeItem
from src.rag.fashion_rag import FashionRAG


class TestFashionRAG:
    """Validate prompt context formatting and complete RAG cycle execution."""

    def test_prompt_augmentation(self):
        """Verify context items are properly formatted in the augmented prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rag = FashionRAG(
                config=get_default_config(),
                kb_path=str(tmp_path / "kb.json"),
                trend_db_path=str(tmp_path / "trends.json"),
                force_mock_embeddings=True
            )

            # Create dummy items
            k_item = KnowledgeItem(
                id="kb_style_streetwear",
                category="fashion_styles",
                name="Streetwear",
                content="Casual clothing cuts.",
                tags=["skate", "cargoes"]
            )
            
            aug_prompt = rag.augment_prompt("Design a new streetwear jacket", [k_item])
            
            assert "FASHION RETRIEVAL CONTEXT INJECTED" in aug_prompt
            assert "kb_style_streetwear" in aug_prompt
            assert "Streetwear" in aug_prompt
            assert "Design a new streetwear jacket" in aug_prompt

    def test_complete_rag_query_loop(self):
        """Verify the RAG query returns retrieved items, trends, recommendations, and responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rag = FashionRAG(
                config=get_default_config(),
                kb_path=str(tmp_path / "kb.json"),
                trend_db_path=str(tmp_path / "trends.json"),
                force_mock_embeddings=True
            )

            # Query the system
            result = rag.query("Nike streetwear designs", top_k=3)
            
            assert isinstance(result, dict)
            assert "query" in result
            assert result["query"] == "Nike streetwear designs"
            assert "retrieved_items" in result
            assert "active_trends" in result
            assert "recommendations" in result
            assert "response" in result
            assert "latency_seconds" in result
            
            # Grounding check: verify output text exists and mentions cited source IDs
            response_text = result["response"]
            assert len(response_text) > 0
            assert "### Fashion AI Assistant" in response_text
            
            # Since Nike brand profile was loaded, it should cite [kb_brand_profiles_nike]
            # (or other default seeded items)
            assert "kb_brand_profiles_nike" in response_text or "Nike" in response_text
