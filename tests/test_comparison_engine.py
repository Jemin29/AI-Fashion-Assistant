"""
week3/tests/test_comparison_engine.py
======================================
Unit tests for the evaluation ComparisonEngine.
Validates math calculations, SSIM correctness, batch evaluation, and JSON reports.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.week3_comparison_engine import ComparisonEngine


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def test_images():
    """Create a set of PIL images representing standard, ControlNet, and condition images."""
    # Solid black
    black_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    # Solid white
    white_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    
    # Standard image (has a square shape)
    standard_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    draw_std = ImageDraw.Draw(standard_img)
    draw_std.rectangle([20, 20, 100, 100], fill=(255, 255, 255))
    
    # Condition image (has a circular shape, differing from standard square)
    condition_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    draw_cond = ImageDraw.Draw(condition_img)
    draw_cond.ellipse([40, 40, 80, 80], fill=(255, 255, 255))
    
    # ControlNet image (perfect match of condition image)
    controlnet_img = condition_img.copy()
    
    return {
        "black": black_img,
        "white": white_img,
        "standard": standard_img,
        "controlnet": controlnet_img,
        "condition": condition_img
    }


@pytest.fixture
def engine():
    """Return a ComparisonEngine instance in mock mode."""
    return ComparisonEngine(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestComparisonEngine:

    def test_init_propagates_mock(self):
        comp = ComparisonEngine(mock=True)
        assert comp.mock is True

    def test_compute_ssim_identical_images(self, engine, test_images):
        # Identical images must have SSIM close to 1.0
        val = engine.compute_ssim(test_images["black"], test_images["black"])
        assert abs(val - 1.0) < 1e-4

        val_design = engine.compute_ssim(test_images["standard"], test_images["standard"])
        assert abs(val_design - 1.0) < 1e-4

    def test_compute_ssim_different_images(self, engine, test_images):
        # A black and a white image should have very low or negative SSIM
        val = engine.compute_ssim(test_images["black"], test_images["white"])
        assert val < 0.2

    def test_compute_visual_consistency(self, engine, test_images):
        # Consistency of single image or identical images should be 1.0 (or close to 1.0)
        images = [test_images["standard"], test_images["standard"], test_images["standard"]]
        val = engine.compute_visual_consistency(images)
        assert abs(val - 1.0) < 1e-3

        # Visual consistency of diverse images should be lower
        mixed = [test_images["black"], test_images["white"]]
        val_mixed = engine.compute_visual_consistency(mixed)
        assert val_mixed < 0.5

    def test_evaluate_pair_mock(self, engine, test_images):
        res = engine.evaluate_pair(
            standard_img=test_images["standard"],
            controlnet_img=test_images["controlnet"],
            condition_img=test_images["condition"],
            prompt="A vibrant red designer dress"
        )
        
        assert "prompt" in res
        assert "metrics" in res
        
        metrics = res["metrics"]
        assert "standard" in metrics
        assert "controlnet" in metrics
        
        # Check standard metrics keys
        std = metrics["standard"]
        assert "clip_score" in std
        assert "ssim" in std
        assert "prompt_alignment" in std
        
        # Check ControlNet metrics keys
        cnet = metrics["controlnet"]
        assert "clip_score" in cnet
        assert "ssim" in cnet
        assert "prompt_alignment" in cnet

        # ControlNet image is identical to condition, so its SSIM should be 1.0
        assert cnet["ssim"] == 1.0
        # Standard image has square, condition has circle, so their SSIM should be lower
        assert std["ssim"] < 0.5

    def test_evaluate_batch_mock(self, engine, test_images):
        std_list = [test_images["standard"], test_images["standard"]]
        cnet_list = [test_images["controlnet"], test_images["controlnet"]]
        cond_list = [test_images["condition"], test_images["condition"]]
        prompts = ["Red design pattern", "Dark fashion garment"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "comparison_report.json"
            
            report = engine.evaluate_batch(
                standard_imgs=std_list,
                controlnet_imgs=cnet_list,
                condition_imgs=cond_list,
                prompts=prompts,
                output_json=report_path
            )
            
            # Assert report structure
            assert "summary" in report
            assert "pairs" in report
            assert report["summary"]["num_samples"] == 2
            
            aggregate = report["summary"]["aggregate_metrics"]
            assert "standard" in aggregate
            assert "controlnet" in aggregate
            
            assert "visual_consistency" in aggregate["standard"]
            assert "visual_consistency" in aggregate["controlnet"]
            
            # File exists
            assert report_path.exists()
            
            # Reload file to verify JSON format
            with open(report_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["summary"]["num_samples"] == 2
            assert len(saved["pairs"]) == 2
            assert saved["pairs"][0]["prompt"] == "Red design pattern"

    def test_evaluate_batch_length_mismatch(self, engine, test_images):
        std_list = [test_images["standard"]]
        cnet_list = [test_images["controlnet"], test_images["controlnet"]] # length 2 vs length 1
        cond_list = [test_images["condition"]]
        prompts = ["Red design pattern"]
        
        with pytest.raises(ValueError, match="All input lists must have matching lengths"):
            engine.evaluate_batch(
                standard_imgs=std_list,
                controlnet_imgs=cnet_list,
                condition_imgs=cond_list,
                prompts=prompts
            )

    def test_compute_visual_consistency_empty_or_single(self, engine, test_images):
        # Empty list
        assert engine.compute_visual_consistency([]) == 1.0
        # Single image
        assert engine.compute_visual_consistency([test_images["standard"]]) == 1.0

    def test_evaluate_batch_save_exception(self, engine, test_images):
        std_list = [test_images["standard"]]
        cnet_list = [test_images["controlnet"]]
        cond_list = [test_images["condition"]]
        prompts = ["Red design pattern"]
        
        # Pass a directory path instead of a file path to force an open() exception
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir)
            
            # It should catch the exception, log it, and return the report dict anyway!
            report = engine.evaluate_batch(
                standard_imgs=std_list,
                controlnet_imgs=cnet_list,
                condition_imgs=cond_list,
                prompts=prompts,
                output_json=invalid_path
            )
            assert report["summary"]["num_samples"] == 1

    def test_clip_computation_exception(self, engine, test_images):
        engine.mock = False
        from unittest.mock import patch
        with patch("transformers.CLIPModel.from_pretrained", side_effect=Exception("No internet")):
            score = engine._compute_clip_score(test_images["standard"], "A velvet coat")
            assert isinstance(score, float)
            assert 0.20 <= score <= 0.32


