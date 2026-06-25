"""
week4/tests/test_style_evaluator.py
===================================
Unit tests for the FashionStyleEvaluator system.
"""

from __future__ import annotations

import pytest
from PIL import Image

from src.evaluation.week4_style_evaluator import FashionStyleEvaluator


class TestFashionStyleEvaluator:
    """Verify metrics calculation, brand similarity comparisons, quality estimation, and schema checks."""

    def test_image_quality_scoring(self):
        """Verify image quality maps to reasonable scores for plain vs contrast images."""
        evaluator = FashionStyleEvaluator()
        
        # Solid white image (very low sharpness/contrast)
        img_solid = Image.new("RGB", (100, 100), color=(255, 255, 255))
        score_solid = evaluator.measure_image_quality(img_solid)
        assert 0.0 <= score_solid <= 0.1

        # Checked contrast image (higher sharpness/contrast)
        img_checked = Image.new("RGB", (100, 100), color=(255, 255, 255))
        pixels = img_checked.load()
        if pixels is not None:
            for i in range(100):
                for j in range(100):
                    if (i + j) % 2 == 0:
                        pixels[i, j] = (0, 0, 0)
        
        score_checked = evaluator.measure_image_quality(img_checked)
        assert score_checked > score_solid

    def test_style_evaluator_nike(self):
        """Verify evaluation outputs on Nike style settings."""
        evaluator = FashionStyleEvaluator()
        img = Image.new("RGB", (100, 100), color=(0, 0, 0)) # black matches Nike palette
        
        metrics = evaluator.evaluate(
            image=img,
            prompt="A sleek Nike performance hoodie",
            brand="nike"
        )
        
        assert "style_similarity" in metrics
        assert "prompt_alignment" in metrics
        assert isinstance(metrics["style_similarity"], float)
        assert isinstance(metrics["prompt_alignment"], float)
        assert 0.0 <= metrics["style_similarity"] <= 1.0
        assert 0.0 <= metrics["prompt_alignment"] <= 1.0

    def test_style_evaluator_zara(self):
        """Verify evaluation outputs on Zara style settings."""
        evaluator = FashionStyleEvaluator()
        img = Image.new("RGB", (100, 100), color=(245, 245, 220)) # beige matches Zara palette
        
        metrics = evaluator.evaluate(
            image=img,
            prompt="A contemporary Zara beige linen blazer",
            brand="zara"
        )
        
        assert "style_similarity" in metrics
        assert "prompt_alignment" in metrics
        assert metrics["style_similarity"] > 0.5
        assert metrics["prompt_alignment"] > 0.5

    def test_style_evaluator_with_reference(self):
        """Verify structural consistency (SSIM) is integrated when a reference image is passed."""
        evaluator = FashionStyleEvaluator()
        img1 = Image.new("RGB", (100, 100), color=(245, 245, 220))
        img2 = Image.new("RGB", (100, 100), color=(0, 0, 0))
        
        # Identical reference image
        metrics_self = evaluator.evaluate(
            image=img1,
            prompt="Neutral tones blazer",
            brand="zara",
            reference_image=img1
        )
        
        # Mismatched reference image
        metrics_other = evaluator.evaluate(
            image=img1,
            prompt="Neutral tones blazer",
            brand="zara",
            reference_image=img2
        )
        
        # Style similarity should be higher when evaluated against itself versus a black image
        assert metrics_self["style_similarity"] > metrics_other["style_similarity"]
