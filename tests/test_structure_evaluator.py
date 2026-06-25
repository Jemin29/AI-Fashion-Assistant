"""
week3/tests/test_structure_evaluator.py
========================================
Unit tests for the StructureEvaluator engine.
Verifies identical matching (1.0), diverse image metrics, and ControlNet alignment dominance.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path
from src.evaluation.week3_structure_evaluator import StructureEvaluator


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def evaluator():
    return StructureEvaluator()


@pytest.fixture
def shapes():
    """Create structured shapes to evaluate metrics with."""
    black_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    white_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    
    # Square shape
    square_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    draw_sq = ImageDraw.Draw(square_img)
    draw_sq.rectangle([20, 20, 100, 100], fill=(255, 255, 255))
    
    # Circle shape
    circle_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    draw_ci = ImageDraw.Draw(circle_img)
    draw_ci.ellipse([20, 20, 100, 100], fill=(255, 255, 255))
    
    return {
        "black": black_img,
        "white": white_img,
        "square": square_img,
        "circle": circle_img
    }


# =============================================================================
# ── Test Suite
# =============================================================================

class TestStructureEvaluator:

    def test_evaluate_identical_images(self, evaluator, shapes):
        # Comparing an image with itself must yield 1.0 for all structural metrics
        res = evaluator.evaluate_structure(shapes["square"], shapes["square"])
        
        assert abs(res["ssim"] - 1.0) < 1e-4
        assert abs(res["edge_preservation"] - 1.0) < 1e-4
        assert abs(res["layout_consistency"] - 1.0) < 1e-4
        assert abs(res["shape_preservation"] - 1.0) < 1e-4

    def test_evaluate_completely_different_images(self, evaluator, shapes):
        # Solid white vs solid black should yield low metrics
        res = evaluator.evaluate_structure(shapes["black"], shapes["white"])
        
        assert res["ssim"] < 0.2
        assert res["layout_consistency"] == 0.0  # Zero correlation

    def test_compare_images_controlnet_advantage(self, evaluator, shapes):
        # reference: circle
        # standard: square (differs structurally)
        # controlnet: circle (matching reference layout)
        res = evaluator.compare_images(
            standard_img=shapes["square"],
            controlnet_img=shapes["circle"],
            reference_img=shapes["circle"]
        )
        
        # Verify structure
        assert "metrics" in res
        assert "improvements" in res
        assert "controlnet_advantage" in res
        
        metrics = res["metrics"]
        assert "standard" in metrics
        assert "controlnet" in metrics
        
        # ControlNet must match perfectly (1.0 metrics)
        cnet = metrics["controlnet"]
        assert cnet["ssim"] == 1.0
        assert cnet["edge_preservation"] == 1.0
        assert cnet["layout_consistency"] == 1.0
        assert cnet["shape_preservation"] == 1.0
        
        # Standard unconditioned square must match poorly (< 1.0 metrics)
        std = metrics["standard"]
        assert std["ssim"] < 1.0
        assert std["edge_preservation"] < 1.0
        assert std["layout_consistency"] < 1.0
        assert std["shape_preservation"] < 1.0

        # Check improvements values are positive
        improvements = res["improvements"]
        for key in ["ssim", "edge_preservation", "layout_consistency", "shape_preservation"]:
            assert improvements[key] > 0.0

        # Confirm ControlNet dominance
        assert res["controlnet_advantage"] is True
