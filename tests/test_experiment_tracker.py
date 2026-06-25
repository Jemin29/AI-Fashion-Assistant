"""
week5/tests/test_experiment_tracker.py
======================================
Unit tests for the Experiment Tracker.
Verifies logging correctness, file persistence, and summary statistics calculations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.experiment_tracker import ExperimentRun, ExperimentTracker


class TestExperimentTracker:
    """Validate experiment run schema validation, CRUD operations, stats calculations, and persistence."""

    def test_experiment_run_validation(self):
        """Verify ExperimentRun validation constraints and default values."""
        # Valid run
        run = ExperimentRun(
            run_id="run_123",
            query="test query",
            retrieved_documents=["doc1", "doc2"],
            recommendation_quality=0.85,
            confidence_score=0.9,
            latency_seconds=0.015,
            metadata={"environment": "testing"}
        )
        assert run.run_id == "run_123"
        assert run.query == "test query"
        assert run.recommendation_quality == 0.85
        assert run.confidence_score == 0.9
        assert run.latency_seconds == 0.015
        assert run.metadata["environment"] == "testing"
        assert run.timestamp is not None

        # Invalid score types rejection
        with pytest.raises(ValidationError):
            ExperimentRun(
                run_id="run_invalid",
                query="Q",
                retrieved_documents=[],
                recommendation_quality="high",  # Must be float
                confidence_score=0.9,
                latency_seconds=0.015
            )

    def test_tracker_logging_and_crud(self):
        """Verify adding logs, retrieving runs, statistics computation, and clearing logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_runs.json"

            tracker = ExperimentTracker(log_path=log_path)
            assert len(tracker.list_runs()) == 0

            # Test empty statistics
            stats_empty = tracker.get_stats()
            assert stats_empty["total_runs"] == 0
            assert stats_empty["average_latency_seconds"] == 0.0

            # Log first run
            run1 = tracker.log_run(
                query="First query",
                retrieved_documents=["doc1"],
                recommendation_quality=0.8,
                confidence_score=0.7,
                latency_seconds=0.010,
                metadata={"test": "run1"}
            )
            assert run1.run_id is not None
            assert tracker.get_run(run1.run_id) is not None
            assert len(tracker.list_runs()) == 1

            # Log second run
            run2 = tracker.log_run(
                query="Second query",
                retrieved_documents=["doc2", "doc3"],
                recommendation_quality=0.9,
                confidence_score=0.9,
                latency_seconds=0.020
            )

            # Test statistics calculations
            stats = tracker.get_stats()
            assert stats["total_runs"] == 2
            assert stats["average_latency_seconds"] == pytest.approx(0.015)
            assert stats["average_confidence_score"] == pytest.approx(0.8)
            assert stats["average_recommendation_quality"] == pytest.approx(0.85)
            assert stats["min_latency_seconds"] == 0.010
            assert stats["max_latency_seconds"] == 0.020

            # Test clearing logs
            tracker.clear_logs()
            assert len(tracker.list_runs()) == 0
            assert tracker.get_stats()["total_runs"] == 0

    def test_database_persistence(self):
        """Verify that logged experiments persist across tracker instances on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "persist_runs.json"

            tracker1 = ExperimentTracker(log_path=log_path)
            logged_run = tracker1.log_run(
                query="Persistent query",
                retrieved_documents=["doc_persist"],
                recommendation_quality=0.95,
                confidence_score=0.98,
                latency_seconds=0.005
            )

            # Re-load in a second instance
            tracker2 = ExperimentTracker(log_path=log_path)
            recovered_run = tracker2.get_run(logged_run.run_id)
            assert recovered_run is not None
            assert recovered_run.query == "Persistent query"
            assert recovered_run.latency_seconds == 0.005
            assert recovered_run.retrieved_documents == ["doc_persist"]
