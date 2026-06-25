"""
week5/tests/test_trend_forecaster.py
====================================
Unit tests for the Fashion Trend Forecasting Engine.
"""

from __future__ import annotations

import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.trends.trend_analyzer import TrendAnalyzer
from src.trends.trend_forecaster import TrendForecaster
from src.rag.vector_db.chromadb_manager import ChromaDbManager


@pytest.fixture
def mock_analyzer():
    """TrendAnalyzer instance for testing."""
    return TrendAnalyzer()


@pytest.fixture
def mock_retriever():
    """FashionRetriever initialized with mock engines."""
    embedder = EmbeddingsGenerator(force_mock=True)
    db_manager = ChromaDbManager(force_mock=True)
    return FashionRetriever(embedder=embedder, db_manager=db_manager)


class TestTrendForecaster:
    """Validate trend forecasting engine, confidence calculations, and seasonal correlations."""

    def test_forecaster_initialization(self, mock_analyzer, mock_retriever):
        """Verify TrendForecaster initializes correctly."""
        forecaster = TrendForecaster(analyzer=mock_analyzer, retriever=mock_retriever)
        assert forecaster.analyzer == mock_analyzer
        assert forecaster.retriever == mock_retriever

    def test_forecast_trends_default(self, mock_analyzer, mock_retriever):
        """Verify default template forecasting without seasonal or keyword inputs."""
        forecaster = TrendForecaster(analyzer=mock_analyzer, retriever=mock_retriever)

        predictions = forecaster.forecast_trends(n_predictions=2)
        assert len(predictions) == 2
        # Verify output formats
        assert "trend" in predictions[0]
        assert "confidence" in predictions[0]
        assert "category" in predictions[0]
        assert "reasoning" in predictions[0]

        # Verify predictions are sorted by confidence descending
        assert predictions[0]["confidence"] >= predictions[1]["confidence"]

    def test_forecast_trends_seasonal_boost(self, mock_analyzer, mock_retriever):
        """Verify that season parameter boosts and penalizes matching templates."""
        forecaster = TrendForecaster(analyzer=mock_analyzer, retriever=mock_retriever)

        # Baseline forecast (Cyber Utility Wear has base confidence 0.82)
        base_recs = {p["trend"]: p["confidence"] for p in forecaster.forecast_trends(n_predictions=4)}

        # Summer forecast (Organic Linen base 0.74 should get boost, Cyber Utility base 0.82 should get penalty)
        summer_recs = {p["trend"]: p["confidence"] for p in forecaster.forecast_trends(current_season="summer", n_predictions=4)}

        # Cyber Utility Wear: out of season penalty (0.82 - 0.05 = 0.77)
        assert summer_recs["Cyber Utility Wear"] < base_recs["Cyber Utility Wear"]
        assert summer_recs["Cyber Utility Wear"] == 0.77

        # Organic Linen Loungewear: seasonal boost (0.74 + 0.07 = 0.81)
        assert summer_recs["Organic Linen Loungewear"] > base_recs["Organic Linen Loungewear"]
        assert summer_recs["Organic Linen Loungewear"] == 0.81

    def test_forecast_trends_keyword_correlation(self, mock_analyzer, mock_retriever):
        """Verify that analyzer growth mentions boost corresponding template confidence levels."""
        forecaster = TrendForecaster(analyzer=mock_analyzer, retriever=mock_retriever)

        # 1. Base run
        base_recs = {p["trend"]: p["confidence"] for p in forecaster.forecast_trends(n_predictions=4)}

        # 2. Add mentions for Cyber Utility Wear keywords
        # We need to satisfy min_mention_count (defaults to 3 or 5 depending on config)
        # Let's seed mentions of 'utility' with growth velocity (recent half mentions > older half mentions)
        mock_analyzer.add_mentions(["utility", "utility", "utility", "cargo", "cargo"])
        
        # Add another set to older half to establish baseline and show growth
        import time
        older_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 86400 * 5))
        mock_analyzer.add_mention("utility", timestamp=older_time)

        # Active trends should now list 'utility' as active/rising
        active_list = mock_analyzer.get_active_trends()
        assert any(t["element"] == "utility" for t in active_list)

        # Run forecast
        boosted_recs = {p["trend"]: p["confidence"] for p in forecaster.forecast_trends(n_predictions=4)}

        # 3. Add mentions for stable trend 'grey' (Athletic Loungewear Tech keyword)
        # We need min_mention_count=3. Let's add 3 in older half, 3 in recent half.
        stable_older_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 86400 * 20))
        for _ in range(3):
            mock_analyzer.add_mention("grey")
            mock_analyzer.add_mention("grey", timestamp=stable_older_time)

        # Verify 'grey' is in stable trends (count >= 3, growth = 0.0)
        forecast_data = mock_analyzer.get_trend_forecast()
        stable_names = {t["element"] for t in forecast_data.get("stable_trends", [])}
        assert "grey" in stable_names

        # Run forecast again
        boosted_recs_2 = {p["trend"]: p["confidence"] for p in forecaster.forecast_trends(n_predictions=4)}

        # Active Loungewear Tech should get a stable trend boost (0.72 + 0.02 = 0.74)
        assert boosted_recs_2["Active Loungewear Tech"] > base_recs["Active Loungewear Tech"]
        assert boosted_recs_2["Active Loungewear Tech"] == 0.74
