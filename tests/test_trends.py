"""
week5/tests/test_trends.py
==========================
Unit tests for the Trend Analyzer module.
Verifies keyword tracking, lookback window filtering, and growth rate forecasting.
"""

from __future__ import annotations

import time

import pytest

from src.utils.config_manager import TrendConfig
from src.trends.trend_analyzer import TrendAnalyzer


class TestTrendAnalyzer:
    """Validate sliding window calculations, keyword tracking, and forecasts."""

    def test_trend_analyzer_ingestion_and_lookback(self):
        """Verify mentions ingest correctly and older logs fall outside lookback window."""
        cfg = TrendConfig(time_window_days=10, min_mention_count=2, growth_threshold=0.1)
        analyzer = TrendAnalyzer(config=cfg)

        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        old_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 15 * 86400)) # 15 days ago

        # Add 3 current mentions for "linen" and 1 old mention
        analyzer.add_mentions(["linen", "linen", "linen"])
        analyzer.add_mention("linen", timestamp=old_str)

        # Cutoff is 10 days, so old_str mention (15 days ago) is filtered out
        active = analyzer.get_active_trends()
        assert len(active) == 1
        assert active[0]["element"] == "linen"
        assert active[0]["mentions_count"] == 3

    def test_growth_velocity_calculation(self):
        """Verify growth rates accurately classify active vs stable trends."""
        cfg = TrendConfig(time_window_days=30, min_mention_count=3, growth_threshold=0.2)
        analyzer = TrendAnalyzer(config=cfg)

        now = time.time()
        # Older half (e.g. 20 days ago)
        old_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now - 20 * 86400))
        # Recent half (e.g. 5 days ago)
        new_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now - 5 * 86400))

        # "velvet" is rising: 1 old mention, 4 new mentions.
        # Growth = (4 - 1) / max(1, 1) = 3.0 (which is >= 0.2)
        analyzer.add_mention("velvet", timestamp=old_time)
        analyzer.add_mentions(["velvet", "velvet", "velvet", "velvet"], timestamp=new_time)

        # "denim" is declining/stable: 4 old mentions, 1 new mention.
        # Growth = (1 - 4) / max(1, 4) = -0.75 (not >= 0.2)
        analyzer.add_mentions(["denim", "denim", "denim", "denim"], timestamp=old_time)
        analyzer.add_mention("denim", timestamp=new_time)

        active = analyzer.get_active_trends()
        # Only velvet meets the growth_threshold (0.2) and min_mention_count (3)
        assert len(active) == 1
        assert active[0]["element"] == "velvet"
        assert active[0]["growth_rate"] == 3.0

        # Check forecasts
        forecast = analyzer.get_trend_forecast()
        assert any(t["element"] == "velvet" for t in forecast["rising_trends"])
        assert any(t["element"] == "denim" for t in forecast["stable_trends"])
