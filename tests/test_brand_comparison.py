"""
week4/tests/test_brand_comparison.py
====================================
Unit tests for the BrandComparisonFramework.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.week4_brand_comparison import BrandComparisonFramework


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBrandComparisonFramework:
    """Verify initialization, L1 pixel differences, brand pair comparison, and report serialization."""

    def test_initialization(self, temp_workspace):
        """Verify framework output directory is set up correctly."""
        framework = BrandComparisonFramework(output_dir=temp_workspace / "reports")
        assert framework.output_dir == (temp_workspace / "reports").resolve()
        assert framework.output_dir.exists()

    def test_compute_visual_differences(self):
        """Verify pixel differences return expected bounds for identical vs different images."""
        framework = BrandComparisonFramework()
        
        img1 = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img2 = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img3 = Image.new("RGB", (100, 100), color=(0, 0, 0))

        # Identical images
        assert framework.compute_visual_differences(img1, img2) == 0.0

        # Completely opposite black/white images
        assert framework.compute_visual_differences(img1, img3) == 1.0

    def test_compare_pair(self):
        """Verify metric calculation on a specific brand pair."""
        framework = BrandComparisonFramework()
        
        img_nike = Image.new("RGB", (100, 100), color=(0, 0, 0))
        img_gucci = Image.new("RGB", (100, 100), color=(139, 69, 19))
        
        res = framework.compare_pair(
            image1=img_nike,
            image2=img_gucci,
            prompt="Sporty chic jacket",
            brand1="nike",
            brand2="gucci"
        )
        
        assert res["comparison"] == "NIKE vs GUCCI"
        assert res["brand1"] == "nike"
        assert res["brand2"] == "gucci"
        assert "style_similarity" in res
        assert "visual_differences" in res
        assert "nike_clip_score" in res["metrics"]
        assert "gucci_clip_score" in res["metrics"]

    def test_generate_comparison_report_success(self, temp_workspace):
        """Verify report compilation, JSON serialization, and disk persistence."""
        framework = BrandComparisonFramework(output_dir=temp_workspace / "reports")
        
        brand_images = {
            "nike": Image.new("RGB", (100, 100), color=(0, 0, 0)),
            "gucci": Image.new("RGB", (100, 100), color=(139, 69, 19)),
            "zara": Image.new("RGB", (100, 100), color=(245, 245, 220)),
            "h&m": Image.new("RGB", (100, 100), color=(128, 128, 128))
        }
        
        prompt = "Casual street hoodie"
        report = framework.generate_comparison_report(
            brand_images=brand_images,
            prompt=prompt,
            report_name="test_report.json"
        )
        
        assert report["prompt"] == prompt
        assert len(report["comparisons"]) == 4
        assert "summary" in report
        assert "mean_style_similarity" in report["summary"]
        assert "mean_visual_differences" in report["summary"]

        # Check file exists on disk
        report_file = temp_workspace / "reports" / "test_report.json"
        assert report_file.exists()
        
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data["prompt"] == prompt
        assert len(data["comparisons"]) == 4

    def test_generate_comparison_report_missing_brand(self):
        """Verify ValueError raised when one of the required brands is missing."""
        framework = BrandComparisonFramework()
        
        brand_images = {
            "nike": Image.new("RGB", (100, 100), color=(0, 0, 0)),
            "gucci": Image.new("RGB", (100, 100), color=(139, 69, 19)),
            "zara": Image.new("RGB", (100, 100), color=(245, 245, 220))
            # H&M is missing
        }
        
        with pytest.raises(ValueError, match="Missing images for required brands"):
            framework.generate_comparison_report(
                brand_images=brand_images,
                prompt="Missing brand test"
            )
