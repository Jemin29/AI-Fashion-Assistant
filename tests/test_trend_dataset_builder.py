"""
week5/tests/test_trend_dataset_builder.py
=========================================
Unit tests verifying the Trend Dataset Builder and trend validation constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.knowledge_base.trend_dataset_builder import (
    TrendDatasetBuilder,
    TrendItem
)


class TestTrendDatasetBuilder:
    """Validate trend schema checks, CRUD operations, and seeding features."""

    def test_trend_item_validation(self):
        """Verify TrendItem validation constraints and default values."""
        # Valid item
        item = TrendItem(
            id="trend_streetwear_boxy_tees",
            category="streetwear",
            name="Boxy Tees",
            description="Boxy drop shoulder cotton tees."
        )
        assert item.category == "streetwear"
        assert item.popularity_score == 0.5
        assert item.growth_rate == 0.0
        assert isinstance(item.metadata, dict)

        # Invalid category rejection
        with pytest.raises(ValidationError):
            TrendItem(
                id="trend_invalid",
                category="footwear",  # Not in streetwear, luxury, seasonal, color, fabric, forecast
                name="Sneakers",
                description="Cool kicks"
            )

        # Score boundaries
        with pytest.raises(ValidationError):
            TrendItem(
                id="trend_invalid_score",
                category="streetwear",
                name="Tees",
                description="Desc",
                popularity_score=1.5  # Must be <= 1.0
            )

    def test_builder_seeding_and_crud(self):
        """Verify initialization seeds defaults and supports complete CRUD cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_trends.json"
            
            # Initial load should trigger default seeding
            builder = TrendDatasetBuilder(db_path=db_path)
            assert db_path.exists()
            assert len(builder.trends) > 0
            
            # Verify we have items in all 6 categories
            categories = {item.category for item in builder.trends.values()}
            assert categories == {"streetwear", "luxury", "seasonal", "color", "fabric", "forecast"}

            # Test Create (add_trend)
            new_item = builder.add_trend(
                category="color",
                name="Cyberpunk Purple",
                description="Neon violet shades for digital and physical clothing overlays.",
                popularity_score=0.75,
                growth_rate=0.25,
                metadata={"hex": "#8A2BE2"}
            )
            assert new_item.id == "trend_color_cyberpunk_purple"
            assert builder.get_trend("trend_color_cyberpunk_purple") is not None
            
            # Duplicate prevention
            with pytest.raises(ValueError):
                builder.add_trend(category="color", name="Cyberpunk Purple", description="Duplicate")

            # Test Read (list_trends and get_trend)
            color_trends = builder.list_trends(category="color")
            assert any(t.name == "Cyberpunk Purple" for t in color_trends)

            # Test Update (update_trend)
            updated = builder.update_trend(
                trend_id="trend_color_cyberpunk_purple",
                popularity_score=0.95,
                metadata={"hex": "#8A2BE2", "glow": True}
            )
            assert updated.popularity_score == 0.95
            assert updated.metadata["glow"] is True
            
            # Verify update fails on missing keys
            with pytest.raises(KeyError):
                builder.update_trend(trend_id="non_existent", popularity_score=0.5)

            # Test Delete (delete_trend)
            success = builder.delete_trend("trend_color_cyberpunk_purple")
            assert success is True
            assert builder.get_trend("trend_color_cyberpunk_purple") is None
            
            # Deleting non-existent returns False
            success_fail = builder.delete_trend("trend_color_cyberpunk_purple")
            assert success_fail is False

    def test_database_persistence(self):
        """Verify that saved dataset matches loaded dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persist_trends.json"
            
            builder1 = TrendDatasetBuilder(db_path=db_path)
            builder1.add_trend(
                category="luxury",
                name="Satin Silk Cape",
                description="Flowing capes."
            )
            
            # Load in a second instance
            builder2 = TrendDatasetBuilder(db_path=db_path)
            assert "trend_luxury_satin_silk_cape" in builder2.trends
            assert builder2.get_trend("trend_luxury_satin_silk_cape").description == "Flowing capes."
