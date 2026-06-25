"""
week2/tests/test_benchmark.py
=================================
Unit tests for the fashion generation benchmarking framework.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.week2_benchmark import (
    FashionBenchmarker,
    BenchmarkRunResult,
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Isolated output path for benchmarks."""
    return tmp_path / "benchmarks"


def test_benchmark_run_result_to_dict():
    """Verify serialization properties of BenchmarkRunResult."""
    res = BenchmarkRunResult(
        run_id="run_001_test",
        prompt="A luxury handbag",
        seed=10,
        width=512,
        height=512,
        cfg_scale=8.0,
        generation_time=2.3,
        image_path="images/run_001_test.png",
        clip_score=0.82,
        alignment="high",
        quality_metrics={"overall": 85, "quality": "excellent"},
    )
    
    d = res.to_dict()
    assert d["run_id"] == "run_001_test"
    assert d["prompt"] == "A luxury handbag"
    assert d["width"] == 512
    assert d["cfg_scale"] == 8.0
    assert d["generation_time"] == 2.3
    assert d["clip_score"] == 0.82
    assert d["quality_metrics"]["overall"] == 85


def test_benchmarker_init(temp_output_dir):
    """Verify that benchmarker sets up directories on init."""
    bench = FashionBenchmarker(output_dir=temp_output_dir)
    assert temp_output_dir.exists()
    assert (temp_output_dir / "images").exists()


def test_benchmarker_run_stub_mode(temp_output_dir):
    """Test full benchmarking pipeline using default stub generator and mock evaluators."""
    bench = FashionBenchmarker(output_dir=temp_output_dir)
    
    prompts = ["a streetwear hoodie", "a luxury gown"]
    seeds = [42]
    resolutions = ["square_512", (512, 1024)]
    cfg_scales = [7.5, 9.0]
    
    # 2 prompts * 1 seed * 2 resolutions * 2 cfgs = 8 runs
    report = bench.run_benchmark(
        prompts=prompts,
        seeds=seeds,
        resolutions=resolutions,
        cfg_scales=cfg_scales,
        run_name="stub_test",
    )
    
    # Verify report summaries
    assert report["metadata"]["total_runs"] == 8
    assert len(report["runs"]) == 8
    
    assert "mean_clip_score" in report["summary"]
    assert "mean_quality_score" in report["summary"]
    assert "mean_generation_time_s" in report["summary"]
    
    # Verify group stats
    stats = report["grouped_statistics"]
    assert "by_prompt" in stats
    assert "by_seed" in stats
    assert "by_resolution" in stats
    assert "by_cfg_scale" in stats
    
    # Verify that file paths were resolved and files were written
    json_path = temp_output_dir / "benchmark_report.json"
    html_path = temp_output_dir / "benchmark_report.html"
    grid_path = temp_output_dir / "benchmark_grid.png"
    
    assert json_path.exists()
    assert html_path.exists()
    assert grid_path.exists()
    
    # Verify JSON content matches
    with json_path.open("r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["metadata"]["total_runs"] == 8
    assert len(saved_data["runs"]) == 8
    
    # Verify individual run dimensions
    # First run: square_512 -> 512x512
    run1 = saved_data["runs"][0]
    assert run1["width"] == 512
    assert run1["height"] == 512
    assert run1["cfg_scale"] == 7.5
    
    # Second run: square_512 with cfg 9.0
    run2 = saved_data["runs"][1]
    assert run2["cfg_scale"] == 9.0


def test_benchmarker_with_mock_generator(temp_output_dir):
    """Verify that benchmarker correctly invokes custom generators and evaluators."""
    mock_gen = MagicMock()
    mock_out = MagicMock()
    
    # Create a small dummy PIL image
    from PIL import Image
    dummy_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    mock_out.images = [dummy_img]
    mock_gen.generate_image.return_value = mock_out

    mock_clip = MagicMock()
    mock_clip.evaluate.return_value = {"clip_score": 0.89, "alignment": "very high"}

    mock_quality = MagicMock()
    class MockQualityReport:
        def to_dict(self):
            return {"overall": 92, "quality": "excellent"}
    mock_quality.assess.return_value = MockQualityReport()

    bench = FashionBenchmarker(
        generator=mock_gen,
        output_dir=temp_output_dir,
        clip_evaluator=mock_clip,
        quality_assessor=mock_quality,
    )

    prompts = ["stylish boots"]
    seeds = [100]
    resolutions = [(512, 512)]
    cfg_scales = [8.5]

    report = bench.run_benchmark(
        prompts=prompts,
        seeds=seeds,
        resolutions=resolutions,
        cfg_scales=cfg_scales,
        run_name="mock_test",
    )

    # Verify calls
    mock_gen.generate_image.assert_called_once_with(
        prompt="stylish boots",
        width=512,
        height=512,
        seed=100,
        guidance_scale=8.5,
        num_inference_steps=30,
        scheduler="euler",
    )
    
    mock_clip.evaluate.assert_called_once()
    mock_quality.assess.assert_called_once()

    # Verify scores inside JSON
    assert report["runs"][0]["clip_score"] == 0.89
    assert report["runs"][0]["quality_metrics"]["overall"] == 92
