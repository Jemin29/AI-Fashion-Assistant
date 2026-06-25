"""
week3/tests/test_controlnet_tracker.py
======================================
Unit tests for the ControlNetExperimentTracker.
Validates logging runs, loading run details, and querying best metrics/statistics.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.week3_controlnet_tracker import ControlNetExperimentTracker


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def temp_tracker():
    """Sets up a tracker writing to a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ControlNetExperimentTracker(experiment_dir=tmpdir)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestControlNetExperimentTracker:

    def test_init_creates_files(self, temp_tracker):
        assert temp_tracker.experiment_dir.exists()
        assert temp_tracker.registry_path.exists()
        
        # Registry must start empty
        runs = temp_tracker.get_all_experiments()
        assert len(runs) == 0

    def test_log_experiment_persists_files(self, temp_tracker):
        exp_id = temp_tracker.log_experiment(
            prompt="A sleek leather jacket",
            control_type="sketch",
            control_weight=0.8,
            seed=42,
            clip_score=0.28,
            ssim_score=0.65,
            generation_time=1.5,
            extra_metadata={"resolution": "1024x1024", "version": "1.0.0"}
        )
        
        # Verify unique ID was generated
        assert exp_id.startswith("exp_")
        
        # Standalone run file should exist
        run_file = temp_tracker.experiment_dir / f"run_{exp_id}.json"
        assert run_file.exists()
        
        # Check run file contents
        with open(run_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiment_id"] == exp_id
        assert data["hyperparameters"]["prompt"] == "A sleek leather jacket"
        assert data["metrics"]["ssim_score"] == 0.65
        assert data["metadata"]["resolution"] == "1024x1024"
        
        # Central registry should now contain 1 entry
        runs = temp_tracker.get_all_experiments()
        assert len(runs) == 1
        assert runs[0]["experiment_id"] == exp_id
        assert runs[0]["hyperparameters"]["control_type"] == "sketch"

    def test_get_experiment(self, temp_tracker):
        exp_id = temp_tracker.log_experiment(
            prompt="Simple white t-shirt",
            control_type="depth",
            control_weight=0.6,
            seed=101,
            clip_score=0.31,
            ssim_score=0.74,
            generation_time=1.2
        )
        
        loaded = temp_tracker.get_experiment(exp_id)
        assert loaded is not None
        assert loaded["experiment_id"] == exp_id
        assert loaded["hyperparameters"]["seed"] == 101
        
        # Get missing experiment should return None
        assert temp_tracker.get_experiment("exp_invalid_id") is None

    def test_get_best_run(self, temp_tracker):
        # Log 3 experiments with different scores
        id1 = temp_tracker.log_experiment(
            prompt="Outfit 1", control_type="sketch", control_weight=0.5, seed=1,
            clip_score=0.22, ssim_score=0.60, generation_time=2.0
        )
        id2 = temp_tracker.log_experiment(
            prompt="Outfit 2", control_type="sketch", control_weight=0.8, seed=2,
            clip_score=0.35, ssim_score=0.85, generation_time=1.8
        )
        id3 = temp_tracker.log_experiment(
            prompt="Outfit 3", control_type="pose", control_weight=0.9, seed=3,
            clip_score=0.29, ssim_score=0.72, generation_time=1.1
        )
        
        # 1. Best SSIM overall should be Outfit 2
        best_ssim = temp_tracker.get_best_run(metric="ssim")
        assert best_ssim["experiment_id"] == id2
        assert best_ssim["metrics"]["ssim_score"] == 0.85

        # 2. Best CLIP overall should be Outfit 2
        best_clip = temp_tracker.get_best_run(metric="clip")
        assert best_clip["experiment_id"] == id2

        # 3. Best Latency (minimum time) should be Outfit 3
        best_time = temp_tracker.get_best_run(metric="time")
        assert best_time["experiment_id"] == id3
        assert best_time["metrics"]["generation_time"] == 1.1

        # 4. Best SSIM filtered by control_type="pose" should be Outfit 3
        best_pose = temp_tracker.get_best_run(metric="ssim", control_type="pose")
        assert best_pose["experiment_id"] == id3
        assert best_pose["hyperparameters"]["control_type"] == "pose"

    def test_get_statistics(self, temp_tracker):
        # Log runs
        temp_tracker.log_experiment(
            prompt="Outfit 1", control_type="sketch", control_weight=0.5, seed=1,
            clip_score=0.20, ssim_score=0.60, generation_time=2.0
        )
        temp_tracker.log_experiment(
            prompt="Outfit 2", control_type="sketch", control_weight=0.8, seed=2,
            clip_score=0.30, ssim_score=0.80, generation_time=1.0
        )
        temp_tracker.log_experiment(
            prompt="Outfit 3", control_type="pose", control_weight=0.9, seed=3,
            clip_score=0.25, ssim_score=0.70, generation_time=1.5
        )
        
        stats = temp_tracker.get_statistics()
        assert stats["total_runs"] == 3
        
        # Check global means
        assert stats["global"]["clip_mean"] == pytest.approx(0.25)
        assert stats["global"]["ssim_mean"] == pytest.approx(0.70)
        assert stats["global"]["time_mean"] == pytest.approx(1.5)
        
        # Check control type breakdown
        by_type = stats["by_control_type"]
        assert "sketch" in by_type
        assert "pose" in by_type
        
        assert by_type["sketch"]["count"] == 2
        assert by_type["sketch"]["clip_mean"] == pytest.approx(0.25)
        assert by_type["sketch"]["ssim_mean"] == pytest.approx(0.70)
        assert by_type["sketch"]["time_mean"] == pytest.approx(1.5)
        
        assert by_type["pose"]["count"] == 1
        assert by_type["pose"]["clip_mean"] == pytest.approx(0.25)
        assert by_type["pose"]["ssim_mean"] == pytest.approx(0.70)
        assert by_type["pose"]["time_mean"] == pytest.approx(1.5)
