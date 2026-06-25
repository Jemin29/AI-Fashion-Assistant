"""
week5/tests/test_fashion_assistant.py
=====================================
Unit tests for the Context-Aware Fashion Assistant.
"""

from __future__ import annotations

import os
import tempfile
import pytest

from src.rag.fashion_assistant import FashionAssistant


@pytest.fixture
def temp_dbs():
    """Create temp database paths for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = os.path.join(tmpdir, "kb.json")
        trend_path = os.path.join(tmpdir, "trends.json")
        profile_path = os.path.join(tmpdir, "profiles.json")
        yield kb_path, trend_path, profile_path


@pytest.fixture
def assistant(temp_dbs):
    """Initialize FashionAssistant in mock mode with temp database paths."""
    kb_path, trend_path, profile_path = temp_dbs
    # Instantiate assistant forcing mock embeddings and in-memory mock ChromaDB
    return FashionAssistant(
        kb_path=kb_path,
        trend_db_path=trend_path,
        user_profile_db_path=profile_path,
        force_mock_embeddings=True
    )


class TestFashionAssistant:
    """Validate all features of the unified Context-Aware Fashion Assistant."""

    def test_assistant_initialization(self, assistant):
        """Verify all sub-components initialize and link correctly."""
        assert assistant.rag_coordinator is not None
        assert assistant.db_manager is not None
        assert assistant.style_recommender is not None
        assert assistant.brand_recommender is not None
        assert assistant.user_profile_manager is not None
        assert assistant.trend_forecaster is not None

        # Verify default seeding is executed on ChromaDB manager
        styles_col = assistant.db_manager.create_collection("fashion_styles")
        assert len(styles_col.ids) > 0

    def test_answer_question(self, assistant):
        """Verify RAG question-answering returns grounded answers and logs history."""
        user_id = "test_user_qa"
        query = "What is Streetwear?"
        res = assistant.answer_question(query=query, user_id=user_id)

        assert "response" in res
        assert "retrieved_items" in res
        assert len(res["retrieved_items"]) > 0

        # Verify user profile recorded the search query
        profile = assistant.user_profile_manager.get_profile(user_id)
        assert profile is not None
        assert query in profile.search_history

    def test_recommend_styles(self, assistant):
        """Verify style recommendations return recommendations and log preferences."""
        user_id = "test_user_styles"
        preferences = {"favorite_style": "streetwear", "favorite_color": "black"}

        recs = assistant.recommend_styles(preferences=preferences, user_id=user_id, n_results=2)
        assert len(recs) == 2
        # Default streetwear+black rule yields Techwear, Urban Minimal, Oversized Casual
        assert recs[0] in ["Techwear", "Urban Minimal", "Oversized Casual"]

        # Verify history logging
        profile = assistant.user_profile_manager.get_profile(user_id)
        assert len(profile.recommendation_history) > 0
        assert any("style_recommendation" in s for s in profile.search_history)

    def test_recommend_brands(self, assistant):
        """Verify brand suggestions return brands and log preferences."""
        user_id = "test_user_brands"
        profile_query = {"favorite_style": "luxury"}

        brands = assistant.recommend_brands(profile=profile_query, user_id=user_id, n_results=2)
        assert len(brands) == 2
        assert brands[0] in ["Gucci", "Prada", "Louis Vuitton"]

        # Verify history logging
        profile = assistant.user_profile_manager.get_profile(user_id)
        assert len(profile.recommendation_history) > 0
        assert any("brand_recommendation" in s for s in profile.search_history)

    def test_explain_trend(self, assistant):
        """Verify trend explanation retrieves description and forecast drivers."""
        res = assistant.explain_trend("Cyber Utility Wear")
        assert res["trend"] == "Cyber Utility Wear"
        assert len(res["explanation"]) > 0
        assert res["confidence"] > 0.0
        assert "cargo" in res["reasoning"] or "utility" in res["reasoning"]

        # Test non-existent trend fallback description
        res_fallback = assistant.explain_trend("Futuristic Neon Goth")
        assert "Futuristic Neon Goth" in res_fallback["explanation"]

    def test_get_trend_forecast(self, assistant):
        """Verify seasonal trend forecasting returns predictions."""
        forecasts = assistant.get_trend_forecast(season="winter", n_predictions=2)
        assert len(forecasts) == 2
        assert "trend" in forecasts[0]
        assert "confidence" in forecasts[0]
        assert "reasoning" in forecasts[0]

    def test_explain_fabric(self, assistant):
        """Verify fabric descriptions pull material details from the KB."""
        res_linen = assistant.explain_fabric("Linen")
        assert res_linen["fabric"] == "Linen"
        assert "flax" in res_linen["explanation"].lower() or "breathability" in res_linen["explanation"].lower()
        assert res_linen["breathability"] == "extreme"

        # Check unknown fabric fallback
        res_unknown = assistant.explain_fabric("Carbon Fiber Alloy")
        assert res_unknown["durability"] == "unknown"
        assert "Carbon Fiber Alloy" in res_unknown["explanation"]

    def test_explain_style(self, assistant):
        """Verify style aesthetic lookups pull data from the KB."""
        res_street = assistant.explain_style("Streetwear")
        assert res_street["style"] == "Streetwear"
        assert "oversized" in res_street["explanation"].lower() or "hip-hop" in res_street["explanation"].lower()
        assert res_street["fit"] == "oversized"

        # Check unknown style fallback
        res_unknown = assistant.explain_style("Cyberpunk Goth")
        assert res_unknown["fit"] == "unknown"
        assert "Cyberpunk Goth" in res_unknown["explanation"]

    def test_chat_intents(self, assistant):
        """Verify chat router classifies messages and executes correct workflows."""
        user_id = "chat_user_1"

        # 1. Chat Fabric Explanation
        res_fabric = assistant.chat("Tell me about linen material.", user_id=user_id)
        assert res_fabric["intent"] == "fabric_explanation"
        assert res_fabric["fabric"] == "Linen"
        assert "Linen" in res_fabric["response"]

        # 2. Chat Style Explanation
        res_style = assistant.chat("Explain the streetwear style.", user_id=user_id)
        assert res_style["intent"] == "style_explanation"
        assert res_style["style"] == "Streetwear"
        assert "streetwear" in res_style["response"].lower()

        # 3. Chat Brand Suggestion
        res_brand = assistant.chat("Suggest brands for a luxury look.", user_id=user_id)
        assert res_brand["intent"] == "brand_recommendation"
        assert len(res_brand["brands"]) > 0

        # 4. Chat Style Recommendation
        res_rec = assistant.chat("Recommend a style in black", user_id=user_id)
        assert res_rec["intent"] == "style_recommendation"
        assert len(res_rec["recommendations"]) > 0

        # 5. Chat Trend Forecast
        res_forecast = assistant.chat("Give me a trend forecast for summer", user_id=user_id)
        assert res_forecast["intent"] == "trend_forecast"
        assert res_forecast["season"] == "summer"

        # 6. Chat Trend Explanation
        res_trend = assistant.chat("Tell me about active loungewear tech trend details", user_id=user_id)
        assert res_trend["intent"] == "trend_explanation"
        assert "active loungewear tech" in res_trend["trend"].lower()

        # 7. Chat General QA Fallback
        res_fallback = assistant.chat("Who invented the modern sewing machine?", user_id=user_id)
        assert res_fallback["intent"] == "general_qa"
        assert len(res_fallback["response"]) > 0
