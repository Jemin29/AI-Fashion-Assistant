"""
week4/tests/test_lora_tracker.py
================================
Unit tests for the LoraExperimentTracker.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.week4_lora_tracker import LoraExperimentTracker


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestLoraExperimentTracker:
    """Verify serialization, query search parameters, metric optimization targets, and aggregation logic."""

    def test_initialization(self, temp_workspace):
        """Verify registry file creation and initialization."""
        tracker = LoraExperimentTracker(output_dir=temp_workspace / "experiments")
        assert tracker.output_dir == (temp_workspace / "experiments").resolve()
        assert tracker.registry_path.exists()
        
        with open(tracker.registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert registry == {}

    def test_log_experiment(self, temp_workspace):
        """Verify logging creates detailed run files and indexes them in registry."""
        tracker = LoraExperimentTracker(output_dir=temp_workspace / "experiments")
        
        params = {"rank": 8, "alpha": 16, "lr": 1e-4}
        run = tracker.log_experiment(
            brand="nike",
            lora_version="nike_v1",
            training_loss=0.04562,
            validation_score=0.88,
            clip_score=0.31,
            style_similarity=0.89,
            parameters=params
        )
        
        exp_id = run["experiment_id"]
        assert exp_id.startswith("lora_nike_")
        assert run["lora_version"] == "nike_v1"
        assert run["parameters"] == params
        
        # Verify run file exists on disk
        run_file = temp_workspace / "experiments" / f"run_{exp_id}.json"
        assert run_file.exists()
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)
        assert run_data["experiment_id"] == exp_id
        
        # Verify indexed in registry index
        with open(tracker.registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert exp_id in registry
        assert registry[exp_id]["lora_version"] == "nike_v1"

    def test_get_experiment(self, temp_workspace):
        """Verify retrieving detailed run details."""
        tracker = LoraExperimentTracker(output_dir=temp_workspace / "experiments")
        run = tracker.log_experiment(
            brand="nike",
            lora_version="v1",
            training_loss=0.05,
            validation_score=0.85,
            clip_score=0.30,
            style_similarity=0.80
        )
        
        exp_id = run["experiment_id"]
        retrieved = tracker.get_experiment(exp_id)
        assert retrieved is not None
        assert retrieved["experiment_id"] == exp_id
        assert retrieved["training_loss"] == 0.05
        
        # Non-existent run
        assert tracker.get_experiment("missing_id") is None

    def test_get_best_run(self, temp_workspace):
        """Verify best run optimization logic for minimizing loss vs maximizing scores."""
        tracker = LoraExperimentTracker(output_dir=temp_workspace / "experiments")
        
        # Run 1
        tracker.log_experiment(
            brand="nike",
            lora_version="nike_v1",
            training_loss=0.05,
            validation_score=0.80,
            clip_score=0.28,
            style_similarity=0.85
        )
        
        # Run 2 (better loss, worse scores)
        run2 = tracker.log_experiment(
            brand="nike",
            lora_version="nike_v2",
            training_loss=0.02,
            validation_score=0.75,
            clip_score=0.25,
            style_similarity=0.78
        )
        
        # Run 3 (better scores, worse loss, gucci brand)
        run3 = tracker.log_experiment(
            brand="gucci",
            lora_version="gucci_v1",
            training_loss=0.08,
            validation_score=0.92,
            clip_score=0.34,
            style_similarity=0.91
        )
        
        # Optimize loss (should yield run2)
        best_loss = tracker.get_best_run(metric="training_loss")
        assert best_loss["experiment_id"] == run2["experiment_id"]

        # Optimize loss filtered by Gucci (should yield run3)
        best_loss_gucci = tracker.get_best_run(metric="training_loss", brand="gucci")
        assert best_loss_gucci["experiment_id"] == run3["experiment_id"]
        
        # Optimize clip similarity globally (should yield run3)
        best_clip = tracker.get_best_run(metric="clip_score")
        assert best_clip["experiment_id"] == run3["experiment_id"]
        
        # Optimize style similarity filtered by Nike (should yield run 1, which had style_similarity 0.85)
        best_style_nike = tracker.get_best_run(metric="style_similarity", brand="nike")
        assert best_style_nike["lora_version"] == "nike_v1"

        # Unsupported metric
        with pytest.raises(ValueError):
            tracker.get_best_run(metric="unknown_metric")

    def test_get_statistics(self, temp_workspace):
        """Verify aggregated metric statistics globally and grouped by brand."""
        tracker = LoraExperimentTracker(output_dir=temp_workspace / "experiments")
        
        # Empty registry
        assert tracker.get_statistics()["total_runs"] == 0
        
        # Log runs
        tracker.log_experiment(
            brand="nike",
            lora_version="nike_v1",
            training_loss=0.06,
            validation_score=0.80,
            clip_score=0.28,
            style_similarity=0.84
        )
        tracker.log_experiment(
            brand="nike",
            lora_version="nike_v2",
            training_loss=0.04,
            validation_score=0.90,
            clip_score=0.32,
            style_similarity=0.88
        )
        tracker.log_experiment(
            brand="gucci",
            lora_version="gucci_v1",
            training_loss=0.02,
            validation_score=0.95,
            clip_score=0.35,
            style_similarity=0.92
        )
        
        stats = tracker.get_statistics()
        assert stats["total_runs"] == 3
        
        # Global means
        assert stats["global"]["count"] == 3
        assert stats["global"]["mean_training_loss"] == round((0.06 + 0.04 + 0.02) / 3, 6)
        assert stats["global"]["mean_validation_score"] == round((0.80 + 0.90 + 0.95) / 3, 4)

        # Brand specific means
        assert stats["by_brand"]["nike"]["count"] == 2
        assert stats["by_brand"]["nike"]["mean_training_loss"] == round((0.06 + 0.04) / 2, 6)
        
        assert stats["by_brand"]["gucci"]["count"] == 1
        assert stats["by_brand"]["gucci"]["mean_training_loss"] == 0.02
