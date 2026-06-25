"""
week2/tests/test_experiment_tracker.py
========================================
Comprehensive test suite for ExperimentTracker.

Strategy
--------
Every test uses a fresh ExperimentTracker backed by a tmp_path fixture
(isolated SQLite DB per test) — no side effects between tests.

Coverage targets
----------------
  - TRACKED_METRICS, CSV_COLUMNS constants
  - ExperimentRecord dataclass: all fields, to_dict, to_csv_row, summary,
    composite_score, passed, repr
  - ExperimentTracker.__init__, __repr__, __len__
  - log_experiment() — all 6 required + optional fields
  - log_batch()
  - get_experiment() — found / not found
  - list_experiments() — all filters (style, run_name, status, tags,
    min_clip_score, max_fid_score, model_version, limit, order_by)
  - best_experiment() — clip_score, fid_score (minimize), generation_time,
    composite_score, style filter
  - compare_experiments() — winner logic, delta, missing ID
  - statistics() — all aggregate fields
  - update_experiment() — allowed / disallowed fields
  - delete_experiment() — found / not found
  - clear_all() — with/without confirm flag
  - count()
  - export_csv() — file written, valid CSV, correct headers
  - export_json() — file written, valid JSON, statistics block
  - to_csv_string() — in-memory, no file
  - to_json_string() — in-memory, no file
  - generate_report() — all sections present
  - track_run() context manager — success, failure, timing
  - Thread-safety: concurrent inserts
"""

from __future__ import annotations

import csv
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.experiment_tracker import (
    ExperimentTracker,
    ExperimentRecord,
    TRACKED_METRICS,
    CSV_COLUMNS,
)


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def tracker(tmp_path):
    """Fresh ExperimentTracker with isolated SQLite DB."""
    return ExperimentTracker(experiments_dir=tmp_path)

@pytest.fixture
def populated_tracker(tmp_path):
    """Tracker pre-loaded with 8 diverse experiment records."""
    t = ExperimentTracker(experiments_dir=tmp_path)
    records = [
        {"prompt": "a black streetwear hoodie", "seed": 1, "style": "streetwear",
         "run_name": "baseline", "model_version": "sdxl-1.0",
         "clip_score": 0.91, "fid_score": 28.4, "generation_time": 4.32,
         "quality_score": 80.0, "quality_rating": "very good",
         "sharpness": 85.0, "brightness": 72.0, "contrast": 78.0,
         "color_distribution": 66.0, "noise_level": 90.0,
         "tags": ["batch_01", "baseline"]},
        {"prompt": "an emerald luxury gown", "seed": 2, "style": "luxury",
         "run_name": "baseline", "model_version": "sdxl-1.0",
         "clip_score": 0.88, "fid_score": 22.1, "generation_time": 3.90,
         "quality_score": 85.0, "tags": ["batch_01"]},
        {"prompt": "a casual white t-shirt", "seed": 3, "style": "casual",
         "run_name": "run_02", "model_version": "sdxl-refiner",
         "clip_score": 0.75, "fid_score": 40.0, "generation_time": 5.10,
         "quality_score": 65.0, "tags": ["batch_02"]},
        {"prompt": "a formal navy suit", "seed": 4, "style": "formal",
         "run_name": "run_02", "model_version": "sdxl-refiner",
         "clip_score": 0.82, "fid_score": 30.0, "generation_time": 4.50,
         "quality_score": 75.0, "tags": ["batch_02", "formal"]},
        {"prompt": "techwear cargo pants", "seed": 5, "style": "techwear",
         "run_name": "run_03", "model_version": "sdxl-lora",
         "clip_score": 0.79, "fid_score": 35.0, "generation_time": 3.20,
         "quality_score": 70.0, "status": "completed"},
        {"prompt": "vintage 90s denim jacket", "seed": 6, "style": "vintage",
         "run_name": "run_03", "model_version": "sdxl-lora",
         "clip_score": 0.86, "fid_score": 25.0, "generation_time": 3.80,
         "quality_score": 78.0},
        {"prompt": "athleisure yoga set", "seed": 7, "style": "athleisure",
         "run_name": "run_04", "model_version": "sdxl-1.0",
         "clip_score": 0.77, "fid_score": 38.0, "generation_time": 4.00,
         "quality_score": 68.0, "status": "completed"},
        {"prompt": "failed generation", "seed": 8, "style": "casual",
         "run_name": "run_04", "model_version": "sdxl-1.0",
         "clip_score": None, "fid_score": None, "generation_time": 1.20,
         "status": "failed", "notes": "OOM error"},
    ]
    t.log_batch(records)
    return t


# =============================================================================
# ── Module Constants
# =============================================================================

class TestModuleConstants:

    def test_tracked_metrics_tuple(self):
        assert isinstance(TRACKED_METRICS, tuple)

    def test_tracked_metrics_has_required(self):
        required = {"clip_score", "fid_score", "generation_time"}
        assert required.issubset(set(TRACKED_METRICS))

    def test_csv_columns_is_list(self):
        assert isinstance(CSV_COLUMNS, list)

    def test_csv_columns_has_required_fields(self):
        required = {"experiment_id", "prompt", "seed", "model_version",
                    "clip_score", "fid_score", "generation_time"}
        assert required.issubset(set(CSV_COLUMNS))


# =============================================================================
# ── ExperimentRecord Dataclass
# =============================================================================

class TestExperimentRecord:

    def test_minimum_instantiation(self):
        r = ExperimentRecord(prompt="a hoodie")
        assert r.prompt == "a hoodie"
        assert r.experiment_id != ""

    def test_auto_experiment_id(self):
        r1 = ExperimentRecord(prompt="a")
        r2 = ExperimentRecord(prompt="b")
        assert r1.experiment_id != r2.experiment_id

    def test_auto_timestamp(self):
        r = ExperimentRecord(prompt="a")
        assert "T" in r.timestamp   # ISO-8601

    def test_default_status_completed(self):
        r = ExperimentRecord(prompt="a")
        assert r.status == "completed"

    def test_passed_completed_with_good_clip(self):
        r = ExperimentRecord(prompt="a", clip_score=0.85, status="completed")
        assert r.passed is True

    def test_passed_false_when_failed_status(self):
        r = ExperimentRecord(prompt="a", clip_score=0.90, status="failed")
        assert r.passed is False

    def test_passed_false_when_low_clip(self):
        r = ExperimentRecord(prompt="a", clip_score=0.10, status="completed")
        assert r.passed is False

    def test_passed_true_when_no_clip(self):
        r = ExperimentRecord(prompt="a", clip_score=None, status="completed")
        assert r.passed is True   # No clip = assume pass

    def test_composite_score_none_when_no_metrics(self):
        r = ExperimentRecord(prompt="a")
        assert r.composite_score is None

    def test_composite_score_with_clip(self):
        r = ExperimentRecord(prompt="a", clip_score=0.90)
        assert r.composite_score is not None
        assert 0.0 <= r.composite_score <= 1.0

    def test_composite_score_with_all_metrics(self):
        r = ExperimentRecord(
            prompt="a", clip_score=0.90, fid_score=20.0,
            quality_score=85.0, sharpness=90.0, noise_level=88.0,
        )
        assert r.composite_score is not None
        assert 0.0 <= r.composite_score <= 1.0

    def test_composite_higher_clip_gives_higher_score(self):
        r_hi = ExperimentRecord(prompt="a", clip_score=0.95)
        r_lo = ExperimentRecord(prompt="a", clip_score=0.50)
        assert r_hi.composite_score > r_lo.composite_score

    def test_to_dict_required_keys(self):
        r = ExperimentRecord(
            prompt="a hoodie", seed=42,
            model_version="sdxl", clip_score=0.91,
            fid_score=28.4, generation_time=4.32,
        )
        d = r.to_dict()
        for key in ["experiment_id", "prompt", "seed", "model_version",
                    "clip_score", "fid_score", "generation_time",
                    "style", "tags", "status", "passed"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_json_serialisable(self):
        r = ExperimentRecord(
            prompt="a hoodie", seed=42, clip_score=0.91,
            tags=["tag1", "tag2"], extra={"k": "v"},
        )
        json.dumps(r.to_dict())

    def test_to_dict_include_composite(self):
        r = ExperimentRecord(prompt="a", clip_score=0.88)
        d = r.to_dict(include_composite=True)
        assert "composite_score" in d

    def test_to_csv_row_all_columns(self):
        r = ExperimentRecord(
            prompt="a hoodie", seed=42, clip_score=0.91,
            fid_score=28.4, generation_time=4.32,
            model_version="sdxl", tags=["t1"],
        )
        row = r.to_csv_row()
        for col in CSV_COLUMNS:
            assert col in row, f"CSV missing column: {col}"

    def test_to_csv_row_numeric_fields_as_strings(self):
        r = ExperimentRecord(prompt="a", clip_score=0.91, seed=42)
        row = r.to_csv_row()
        assert row["seed"] == "42"
        assert row["clip_score"] == "0.91"

    def test_to_csv_row_none_as_empty_string(self):
        r = ExperimentRecord(prompt="a", fid_score=None)
        row = r.to_csv_row()
        assert row["fid_score"] == ""

    def test_to_csv_row_tags_json(self):
        r = ExperimentRecord(prompt="a", tags=["t1", "t2"])
        row = r.to_csv_row()
        parsed = json.loads(row["tags"])
        assert parsed == ["t1", "t2"]

    def test_summary_returns_string(self):
        r = ExperimentRecord(prompt="a hoodie", seed=42, clip_score=0.91)
        s = r.summary()
        assert isinstance(s, str)
        assert "0.910" in s

    def test_summary_contains_clip(self):
        r = ExperimentRecord(prompt="a", clip_score=0.88)
        assert "CLIP=0.880" in r.summary()

    def test_summary_na_when_no_scores(self):
        r = ExperimentRecord(prompt="a")
        s = r.summary()
        assert "N/A" in s

    def test_repr_eq_summary(self):
        r = ExperimentRecord(prompt="a", clip_score=0.9)
        assert repr(r) == r.summary()


# =============================================================================
# ── ExperimentTracker — Init
# =============================================================================

class TestTrackerInit:

    def test_creates_db_file(self, tmp_path):
        t = ExperimentTracker(experiments_dir=tmp_path)
        assert (tmp_path / "experiments.db").exists()

    def test_creates_experiments_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        t = ExperimentTracker(experiments_dir=nested)
        assert nested.exists()

    def test_custom_db_name(self, tmp_path):
        t = ExperimentTracker(experiments_dir=tmp_path, db_name="custom.db")
        assert (tmp_path / "custom.db").exists()

    def test_repr(self, tracker):
        r = repr(tracker)
        assert "ExperimentTracker" in r
        assert "experiments.db" in r

    def test_len_initially_zero(self, tracker):
        assert len(tracker) == 0

    def test_count_initially_zero(self, tracker):
        assert tracker.count() == 0


# =============================================================================
# ── log_experiment()
# =============================================================================

class TestLogExperiment:

    def test_returns_string_id(self, tracker):
        exp_id = tracker.log_experiment(prompt="a hoodie")
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_six_required_fields_tracked(self, tracker):
        exp_id = tracker.log_experiment(
            prompt          = "a black hoodie",
            seed            = 42,
            model_version   = "sdxl-1.0",
            clip_score      = 0.91,
            fid_score       = 28.4,
            generation_time = 4.32,
        )
        rec = tracker.get_experiment(exp_id)
        assert rec.prompt          == "a black hoodie"
        assert rec.seed            == 42
        assert rec.model_version   == "sdxl-1.0"
        assert rec.clip_score      == pytest.approx(0.91)
        assert rec.fid_score       == pytest.approx(28.4)
        assert rec.generation_time == pytest.approx(4.32)

    def test_increments_count(self, tracker):
        tracker.log_experiment(prompt="a")
        tracker.log_experiment(prompt="b")
        assert tracker.count() == 2

    def test_optional_metadata_stored(self, tracker):
        exp_id = tracker.log_experiment(
            prompt="a hoodie", style="streetwear",
            quality_score=80.0, quality_rating="very good",
            sharpness=85.0, brightness=72.0, contrast=78.0,
            color_distribution=66.0, noise_level=90.0,
            image_width=1024, image_height=1024,
            num_inference_steps=50, guidance_scale=7.5,
            tags=["tag1", "tag2"],
            notes="test note",
            run_name="my_run",
        )
        rec = tracker.get_experiment(exp_id)
        assert rec.style              == "streetwear"
        assert rec.quality_score      == pytest.approx(80.0)
        assert rec.sharpness          == pytest.approx(85.0)
        assert rec.image_width        == 1024
        assert rec.num_inference_steps== 50
        assert rec.guidance_scale     == pytest.approx(7.5)
        assert "tag1" in rec.tags
        assert rec.notes              == "test note"
        assert rec.run_name           == "my_run"

    def test_pipeline_config_stored(self, tracker):
        config = {"scheduler": "DPMSolver", "vae": "sdxl-vae"}
        exp_id = tracker.log_experiment(prompt="a", pipeline_config=config)
        rec    = tracker.get_experiment(exp_id)
        assert rec.pipeline_config == config

    def test_extra_data_stored(self, tracker):
        extra  = {"custom_key": "custom_value", "num": 42}
        exp_id = tracker.log_experiment(prompt="a", extra=extra)
        rec    = tracker.get_experiment(exp_id)
        assert rec.extra == extra

    def test_failed_status_stored(self, tracker):
        exp_id = tracker.log_experiment(prompt="a", status="failed")
        rec    = tracker.get_experiment(exp_id)
        assert rec.status == "failed"

    def test_custom_experiment_id(self, tracker):
        exp_id = tracker.log_experiment(prompt="a", experiment_id="custom_abc123")
        rec    = tracker.get_experiment("custom_abc123")
        assert rec is not None

    def test_overwrite_with_same_id(self, tracker):
        tracker.log_experiment(prompt="original", experiment_id="dup_id")
        tracker.log_experiment(prompt="overwritten", experiment_id="dup_id")
        rec = tracker.get_experiment("dup_id")
        assert rec.prompt == "overwritten"
        assert tracker.count() == 1   # Still just one record


# =============================================================================
# ── log_batch()
# =============================================================================

class TestLogBatch:

    def test_returns_list_of_ids(self, tracker):
        ids = tracker.log_batch([
            {"prompt": "a", "seed": 1},
            {"prompt": "b", "seed": 2},
        ])
        assert isinstance(ids, list)
        assert len(ids) == 2

    def test_all_records_persisted(self, tracker):
        tracker.log_batch([{"prompt": f"img_{i}", "seed": i} for i in range(5)])
        assert tracker.count() == 5

    def test_empty_batch_returns_empty(self, tracker):
        ids = tracker.log_batch([])
        assert ids == []
        assert tracker.count() == 0

    def test_batch_with_full_metadata(self, tracker):
        ids = tracker.log_batch([{
            "prompt": "luxury gown", "seed": 42,
            "clip_score": 0.91, "fid_score": 22.0,
            "generation_time": 3.5, "style": "luxury",
            "tags": ["t1", "t2"],
        }])
        rec = tracker.get_experiment(ids[0])
        assert rec.clip_score == pytest.approx(0.91)
        assert "t1" in rec.tags

    def test_batch_ids_are_unique(self, tracker):
        ids = tracker.log_batch([{"prompt": f"p{i}"} for i in range(5)])
        assert len(set(ids)) == 5


# =============================================================================
# ── get_experiment()
# =============================================================================

class TestGetExperiment:

    def test_returns_record(self, tracker):
        exp_id = tracker.log_experiment(prompt="a hoodie", clip_score=0.88)
        rec    = tracker.get_experiment(exp_id)
        assert isinstance(rec, ExperimentRecord)

    def test_returns_none_when_missing(self, tracker):
        rec = tracker.get_experiment("nonexistent_id_xyz")
        assert rec is None

    def test_correct_fields_returned(self, tracker):
        exp_id = tracker.log_experiment(
            prompt="a luxury gown", seed=99,
            clip_score=0.92, fid_score=20.0,
            generation_time=3.8, style="luxury",
        )
        rec = tracker.get_experiment(exp_id)
        assert rec.seed      == 99
        assert rec.clip_score== pytest.approx(0.92)
        assert rec.style     == "luxury"


# =============================================================================
# ── list_experiments()
# =============================================================================

class TestListExperiments:

    def test_returns_all(self, populated_tracker):
        recs = populated_tracker.list_experiments()
        assert len(recs) == 8

    def test_filter_by_style(self, populated_tracker):
        recs = populated_tracker.list_experiments(style="streetwear")
        assert len(recs) == 1
        assert all(r.style == "streetwear" for r in recs)

    def test_filter_by_run_name(self, populated_tracker):
        recs = populated_tracker.list_experiments(run_name="baseline")
        assert len(recs) == 2

    def test_filter_by_status(self, populated_tracker):
        failed = populated_tracker.list_experiments(status="failed")
        assert len(failed) == 1
        assert failed[0].status == "failed"

    def test_filter_by_min_clip_score(self, populated_tracker):
        recs = populated_tracker.list_experiments(min_clip_score=0.88)
        assert all(r.clip_score >= 0.88 for r in recs if r.clip_score is not None)

    def test_filter_by_max_fid_score(self, populated_tracker):
        recs = populated_tracker.list_experiments(max_fid_score=25.0)
        assert all(r.fid_score <= 25.0 for r in recs if r.fid_score is not None)

    def test_filter_by_model_version(self, populated_tracker):
        recs = populated_tracker.list_experiments(model_version="sdxl-lora")
        assert len(recs) == 2
        assert all(r.model_version == "sdxl-lora" for r in recs)

    def test_filter_by_tags(self, populated_tracker):
        recs = populated_tracker.list_experiments(tags=["batch_01"])
        assert len(recs) == 2
        assert all("batch_01" in r.tags for r in recs)

    def test_filter_by_multiple_tags(self, populated_tracker):
        recs = populated_tracker.list_experiments(tags=["batch_01", "baseline"])
        assert all("batch_01" in r.tags and "baseline" in r.tags for r in recs)

    def test_limit(self, populated_tracker):
        recs = populated_tracker.list_experiments(limit=3)
        assert len(recs) == 3

    def test_order_by_clip_score_desc(self, populated_tracker):
        recs = populated_tracker.list_experiments(order_by="clip_score", descending=True)
        clips = [r.clip_score for r in recs if r.clip_score is not None]
        assert clips == sorted(clips, reverse=True)

    def test_order_by_generation_time_asc(self, populated_tracker):
        recs = populated_tracker.list_experiments(order_by="generation_time", descending=False)
        times = [r.generation_time for r in recs]
        assert times == sorted(times)

    def test_empty_tracker_returns_empty(self, tracker):
        assert tracker.list_experiments() == []


# =============================================================================
# ── best_experiment()
# =============================================================================

class TestBestExperiment:

    def test_best_clip_score(self, populated_tracker):
        best = populated_tracker.best_experiment("clip_score")
        assert best is not None
        all_clips = [r.clip_score for r in populated_tracker.list_experiments()
                     if r.clip_score is not None]
        assert best.clip_score == max(all_clips)

    def test_best_fid_score_minimize(self, populated_tracker):
        best = populated_tracker.best_experiment("fid_score", minimize=True)
        assert best is not None
        all_fids = [r.fid_score for r in populated_tracker.list_experiments()
                    if r.fid_score is not None]
        assert best.fid_score == min(all_fids)

    def test_fastest_generation_time(self, populated_tracker):
        fastest = populated_tracker.best_experiment("generation_time", minimize=True)
        assert fastest is not None
        all_times = [r.generation_time for r in populated_tracker.list_experiments()]
        assert fastest.generation_time == min(all_times)

    def test_best_quality_score(self, populated_tracker):
        best = populated_tracker.best_experiment("quality_score")
        assert best is not None
        all_q = [r.quality_score for r in populated_tracker.list_experiments()
                 if r.quality_score is not None]
        assert best.quality_score == max(all_q)

    def test_best_composite_score(self, populated_tracker):
        best = populated_tracker.best_experiment("composite_score")
        assert best is not None
        assert best.composite_score is not None

    def test_best_with_style_filter(self, populated_tracker):
        best = populated_tracker.best_experiment("clip_score", style="luxury")
        assert best is not None
        assert best.style == "luxury"

    def test_returns_none_when_empty(self, tracker):
        result = tracker.best_experiment("clip_score")
        assert result is None

    def test_best_run_name_filter(self, populated_tracker):
        best = populated_tracker.best_experiment("clip_score", run_name="run_02")
        assert best is not None
        assert best.run_name == "run_02"


# =============================================================================
# ── compare_experiments()
# =============================================================================

class TestCompareExperiments:

    def test_returns_dict(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.91, fid_score=28.4)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.80, fid_score=35.0)
        diff = tracker.compare_experiments(id_a, id_b)
        assert isinstance(diff, dict)

    def test_has_required_keys(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.91)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.80)
        diff = tracker.compare_experiments(id_a, id_b)
        for key in ["a", "b", "delta", "winner", "overall_winner"]:
            assert key in diff

    def test_winner_a_for_better_clip(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.91)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.75)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["winner"]["clip_score"] == "a"

    def test_winner_b_for_better_clip(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.75)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.91)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["winner"]["clip_score"] == "b"

    def test_winner_tie_when_equal(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.88)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.88)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["winner"]["clip_score"] == "tie"

    def test_fid_minimized_winner_b(self, tracker):
        id_a = tracker.log_experiment(prompt="a", fid_score=40.0)
        id_b = tracker.log_experiment(prompt="b", fid_score=20.0)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["winner"]["fid_score"] == "b"   # lower FID wins

    def test_delta_correct(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.91)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.81)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["delta"]["clip_score"] == pytest.approx(0.10, abs=0.001)

    def test_a_b_dicts_populated(self, tracker):
        id_a = tracker.log_experiment(prompt="a test")
        id_b = tracker.log_experiment(prompt="b test")
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["a"]["prompt"] == "a test"
        assert diff["b"]["prompt"] == "b test"

    def test_missing_id_returns_error(self, tracker):
        exp_id = tracker.log_experiment(prompt="a")
        diff   = tracker.compare_experiments(exp_id, "nonexistent_xyz")
        assert "error" in diff

    def test_both_missing_returns_error(self, tracker):
        diff = tracker.compare_experiments("missing_a", "missing_b")
        assert "error" in diff

    def test_overall_winner_a_or_b_or_tie(self, tracker):
        id_a = tracker.log_experiment(prompt="a", clip_score=0.91)
        id_b = tracker.log_experiment(prompt="b", clip_score=0.75)
        diff = tracker.compare_experiments(id_a, id_b)
        assert diff["overall_winner"] in ("a", "b", "tie")

    def test_a_wins_count_populated(self, tracker):
        id_a = tracker.log_experiment(prompt="a")
        id_b = tracker.log_experiment(prompt="b")
        diff = tracker.compare_experiments(id_a, id_b)
        assert "a_wins" in diff
        assert "b_wins" in diff


# =============================================================================
# ── statistics()
# =============================================================================

class TestStatistics:

    def test_returns_dict(self, populated_tracker):
        stats = populated_tracker.statistics()
        assert isinstance(stats, dict)

    def test_total_count(self, populated_tracker):
        stats = populated_tracker.statistics()
        assert stats["total"] == 8

    def test_completed_count(self, populated_tracker):
        stats = populated_tracker.statistics()
        assert stats["completed"] == 7   # 7 completed, 1 failed

    def test_failed_count(self, populated_tracker):
        stats = populated_tracker.statistics()
        assert stats["failed"] == 1

    def test_pass_rate_range(self, populated_tracker):
        stats = populated_tracker.statistics()
        assert 0.0 <= stats["pass_rate"] <= 1.0

    def test_clip_score_stats(self, populated_tracker):
        stats = populated_tracker.statistics()
        cs    = stats["clip_score"]
        assert "mean" in cs
        assert "min"  in cs
        assert "max"  in cs
        assert "std"  in cs
        assert cs["min"] <= cs["mean"] <= cs["max"]

    def test_fid_score_stats(self, populated_tracker):
        stats = populated_tracker.statistics()
        fs    = stats["fid_score"]
        assert fs["min"] <= fs["mean"] <= fs["max"]

    def test_generation_time_stats(self, populated_tracker):
        stats = populated_tracker.statistics()
        gt    = stats["generation_time"]
        assert gt["min"] > 0.0

    def test_style_breakdown_keys(self, populated_tracker):
        stats = populated_tracker.statistics()
        breakdown = stats["style_breakdown"]
        assert "streetwear" in breakdown
        assert "luxury"     in breakdown

    def test_style_breakdown_counts(self, populated_tracker):
        stats = populated_tracker.statistics()
        breakdown = stats["style_breakdown"]
        assert breakdown["streetwear"] == 1
        assert breakdown["luxury"]     == 1
        assert breakdown["casual"]     == 2   # casual + failed

    def test_top_models_populated(self, populated_tracker):
        stats = populated_tracker.statistics()
        top   = stats["top_models"]
        assert "sdxl-1.0" in top

    def test_empty_tracker_stats(self, tracker):
        stats = tracker.statistics()
        assert stats.get("total", 0) == 0

    def test_style_filter(self, populated_tracker):
        stats = populated_tracker.statistics(style="luxury")
        assert stats["total"] == 1

    def test_json_serialisable(self, populated_tracker):
        stats = populated_tracker.statistics()
        json.dumps(stats)


# =============================================================================
# ── update_experiment()
# =============================================================================

class TestUpdateExperiment:

    def test_returns_true_on_success(self, tracker):
        exp_id = tracker.log_experiment(prompt="a", clip_score=0.80)
        result = tracker.update_experiment(exp_id, clip_score=0.93)
        assert result is True

    def test_clip_score_updated(self, tracker):
        exp_id = tracker.log_experiment(prompt="a", clip_score=0.80)
        tracker.update_experiment(exp_id, clip_score=0.93)
        rec = tracker.get_experiment(exp_id)
        assert rec.clip_score == pytest.approx(0.93)

    def test_notes_updated(self, tracker):
        exp_id = tracker.log_experiment(prompt="a")
        tracker.update_experiment(exp_id, notes="updated notes")
        rec = tracker.get_experiment(exp_id)
        assert rec.notes == "updated notes"

    def test_status_updated(self, tracker):
        exp_id = tracker.log_experiment(prompt="a", status="pending")
        tracker.update_experiment(exp_id, status="completed")
        rec = tracker.get_experiment(exp_id)
        assert rec.status == "completed"

    def test_disallowed_field_ignored(self, tracker):
        exp_id  = tracker.log_experiment(prompt="original")
        tracker.update_experiment(exp_id, prompt="modified")   # prompt not in allowed
        rec = tracker.get_experiment(exp_id)
        assert rec.prompt == "original"   # Unchanged

    def test_returns_false_for_nonexistent(self, tracker):
        result = tracker.update_experiment("nonexistent_id", clip_score=0.99)
        assert result is False

    def test_no_updates_returns_false(self, tracker):
        exp_id = tracker.log_experiment(prompt="a")
        result = tracker.update_experiment(exp_id)   # No kwargs
        assert result is False


# =============================================================================
# ── delete_experiment() / clear_all()
# =============================================================================

class TestDeleteAndClear:

    def test_delete_returns_true(self, tracker):
        exp_id = tracker.log_experiment(prompt="a")
        result = tracker.delete_experiment(exp_id)
        assert result is True

    def test_record_gone_after_delete(self, tracker):
        exp_id = tracker.log_experiment(prompt="a")
        tracker.delete_experiment(exp_id)
        assert tracker.get_experiment(exp_id) is None

    def test_count_decrements_after_delete(self, tracker):
        id1 = tracker.log_experiment(prompt="a")
        id2 = tracker.log_experiment(prompt="b")
        assert tracker.count() == 2
        tracker.delete_experiment(id1)
        assert tracker.count() == 1

    def test_delete_nonexistent_returns_false(self, tracker):
        result = tracker.delete_experiment("nope_xyz")
        assert result is False

    def test_clear_all_requires_confirm(self, tracker):
        tracker.log_experiment(prompt="a")
        with pytest.raises(ValueError, match="confirm"):
            tracker.clear_all()

    def test_clear_all_deletes_all(self, tracker):
        tracker.log_batch([{"prompt": f"p{i}"} for i in range(5)])
        n = tracker.clear_all(confirm=True)
        assert n == 5
        assert tracker.count() == 0

    def test_clear_all_returns_count(self, tracker):
        tracker.log_batch([{"prompt": f"p{i}"} for i in range(3)])
        n = tracker.clear_all(confirm=True)
        assert n == 3


# =============================================================================
# ── export_csv()
# =============================================================================

class TestExportCSV:

    def test_returns_path(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a", clip_score=0.88)
        p = tracker.export_csv(tmp_path / "test.csv")
        assert isinstance(p, Path)
        assert p.exists()

    def test_csv_has_header(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a")
        p = tracker.export_csv(tmp_path / "test.csv")
        with p.open() as fh:
            reader = csv.DictReader(fh)
            assert set(CSV_COLUMNS).issubset(set(reader.fieldnames))

    def test_csv_row_count(self, populated_tracker, tmp_path):
        p = populated_tracker.export_csv(tmp_path / "test.csv")
        with p.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 8

    def test_csv_clip_score_value(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a", clip_score=0.912345)
        p = tracker.export_csv(tmp_path / "test.csv")
        with p.open() as fh:
            row = next(csv.DictReader(fh))
        assert "0.912345" in row["clip_score"]

    def test_csv_tags_json(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a", tags=["tag_x", "tag_y"])
        p = tracker.export_csv(tmp_path / "test.csv")
        with p.open() as fh:
            row = next(csv.DictReader(fh))
        tags = json.loads(row["tags"])
        assert "tag_x" in tags

    def test_csv_auto_path_when_none(self, tracker):
        tracker.log_experiment(prompt="a")
        p = tracker.export_csv()
        assert p.exists()
        assert p.suffix == ".csv"

    def test_csv_filter_by_style(self, populated_tracker, tmp_path):
        p = populated_tracker.export_csv(tmp_path / "filtered.csv", style="luxury")
        with p.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["style"] == "luxury"


# =============================================================================
# ── export_json()
# =============================================================================

class TestExportJSON:

    def test_returns_path(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a")
        p = tracker.export_json(tmp_path / "test.json")
        assert isinstance(p, Path)
        assert p.exists()

    def test_json_valid(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a", clip_score=0.90)
        p    = tracker.export_json(tmp_path / "test.json")
        data = json.loads(p.read_text())
        assert isinstance(data, dict)

    def test_json_has_experiments_key(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a")
        p    = tracker.export_json(tmp_path / "test.json")
        data = json.loads(p.read_text())
        assert "experiments" in data

    def test_json_experiment_count(self, populated_tracker, tmp_path):
        p    = populated_tracker.export_json(tmp_path / "test.json")
        data = json.loads(p.read_text())
        assert data["total"] == 8
        assert len(data["experiments"]) == 8

    def test_json_statistics_included(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a")
        p    = tracker.export_json(tmp_path / "test.json", include_stats=True)
        data = json.loads(p.read_text())
        assert "statistics" in data

    def test_json_statistics_excluded(self, tracker, tmp_path):
        tracker.log_experiment(prompt="a")
        p    = tracker.export_json(tmp_path / "test.json", include_stats=False)
        data = json.loads(p.read_text())
        assert "statistics" not in data

    def test_json_auto_path(self, tracker):
        tracker.log_experiment(prompt="a")
        p = tracker.export_json()
        assert p.exists()
        assert p.suffix == ".json"

    def test_json_filter_by_run(self, populated_tracker, tmp_path):
        p    = populated_tracker.export_json(tmp_path / "filtered.json", run_name="baseline")
        data = json.loads(p.read_text())
        assert data["total"] == 2


# =============================================================================
# ── In-Memory String Exports
# =============================================================================

class TestStringExports:

    def test_to_csv_string_returns_str(self, tracker):
        tracker.log_experiment(prompt="a", clip_score=0.88)
        s = tracker.to_csv_string()
        assert isinstance(s, str)

    def test_to_csv_string_has_header(self, tracker):
        tracker.log_experiment(prompt="a")
        s = tracker.to_csv_string()
        reader = csv.DictReader(io.StringIO(s))
        assert "experiment_id" in reader.fieldnames

    def test_to_csv_string_row_count(self, populated_tracker):
        s    = populated_tracker.to_csv_string()
        rows = list(csv.DictReader(io.StringIO(s)))
        assert len(rows) == 8

    def test_to_json_string_returns_str(self, tracker):
        tracker.log_experiment(prompt="a")
        s = tracker.to_json_string()
        assert isinstance(s, str)

    def test_to_json_string_valid_json(self, tracker):
        tracker.log_experiment(prompt="a", clip_score=0.91)
        s    = tracker.to_json_string()
        data = json.loads(s)
        assert "experiments" in data

    def test_to_json_string_count(self, populated_tracker):
        s    = populated_tracker.to_json_string()
        data = json.loads(s)
        assert data["total"] == 8

    def test_to_csv_string_with_records(self, tracker):
        recs = [
            ExperimentRecord(prompt="a", clip_score=0.88),
            ExperimentRecord(prompt="b", clip_score=0.75),
        ]
        s    = tracker.to_csv_string(records=recs)
        rows = list(csv.DictReader(io.StringIO(s)))
        assert len(rows) == 2


# =============================================================================
# ── generate_report()
# =============================================================================

class TestGenerateReport:

    def test_returns_string(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert isinstance(r, str)

    def test_contains_summary_section(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "SUMMARY" in r

    def test_contains_total_experiments(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "Total experiments" in r
        assert "8" in r

    def test_contains_clip_score_stats(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "CLIP Score" in r

    def test_contains_style_breakdown(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "STYLE BREAKDOWN" in r
        assert "streetwear" in r

    def test_contains_top_models(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "TOP MODELS" in r

    def test_contains_top_experiments(self, populated_tracker):
        r = populated_tracker.generate_report()
        assert "TOP" in r and "EXPERIMENTS" in r

    def test_style_filter_applied(self, populated_tracker):
        r = populated_tracker.generate_report(style="luxury")
        assert "Total experiments : 1" in r

    def test_empty_tracker_no_crash(self, tracker):
        r = tracker.generate_report()
        assert isinstance(r, str)
        assert "0" in r

    def test_top_n_respected(self, populated_tracker):
        r = populated_tracker.generate_report(top_n=3)
        assert "TOP 3" in r


# =============================================================================
# ── track_run() Context Manager
# =============================================================================

class TestTrackRun:

    def test_logs_experiment_on_exit(self, tracker):
        with tracker.track_run("a hoodie", seed=42) as run:
            run["clip_score"] = 0.88
        assert tracker.count() == 1

    def test_generation_time_auto_set(self, tracker):
        with tracker.track_run("a hoodie") as run:
            time.sleep(0.01)
            run["clip_score"] = 0.88
        recs = tracker.list_experiments()
        assert recs[0].generation_time >= 0.01

    def test_yields_dict(self, tracker):
        with tracker.track_run("a hoodie") as run:
            assert isinstance(run, dict)
            assert run["prompt"] == "a hoodie"

    def test_extra_fields_in_run_dict(self, tracker):
        with tracker.track_run("a hoodie", seed=99, style="luxury") as run:
            run["fid_score"] = 25.0
        recs = tracker.list_experiments()
        assert recs[0].seed  == 99
        assert recs[0].style == "luxury"
        assert recs[0].fid_score == pytest.approx(25.0)

    def test_exception_sets_failed_status(self, tracker):
        # track_run catches exceptions internally — it does NOT re-raise them
        with tracker.track_run("a hoodie") as run:
            raise RuntimeError("test failure")
        # After the context exits (exception suppressed), status should be "failed"
        recs = tracker.list_experiments()
        assert recs[0].status == "failed"

    def test_exception_still_logs_experiment(self, tracker):
        # track_run swallows the exception and logs with status="failed"
        with tracker.track_run("a hoodie") as run:
            raise RuntimeError("test failure")
        assert tracker.count() == 1

    def test_context_manager_prompt_set(self, tracker):
        with tracker.track_run("unique_prompt_xzy", seed=1) as run:
            pass
        recs = tracker.list_experiments()
        assert recs[0].prompt == "unique_prompt_xzy"


# =============================================================================
# ── Thread-Safety
# =============================================================================

class TestThreadSafety:

    def test_concurrent_inserts(self, tracker):
        """50 threads insert simultaneously — no data loss, no crash."""
        errors: List[Exception] = []
        ids: List[str] = []
        lock = threading.Lock()

        def insert(i: int) -> None:
            try:
                eid = tracker.log_experiment(
                    prompt          = f"thread_{i}",
                    seed            = i,
                    clip_score      = 0.75 + (i % 10) * 0.02,
                    generation_time = 1.0,
                )
                with lock:
                    ids.append(eid)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=insert, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert tracker.count() == 50
        assert len(set(ids)) == 50   # All IDs unique


# =============================================================================
# ── Persistence (DB survives re-init)
# =============================================================================

class TestPersistence:

    def test_records_persist_across_reinit(self, tmp_path):
        t1 = ExperimentTracker(experiments_dir=tmp_path)
        t1.log_experiment(prompt="persisted prompt", seed=7, clip_score=0.85)
        t1_id = t1.list_experiments()[0].experiment_id

        # Re-create from same directory
        t2  = ExperimentTracker(experiments_dir=tmp_path)
        rec = t2.get_experiment(t1_id)
        assert rec is not None
        assert rec.prompt == "persisted prompt"
        assert rec.clip_score == pytest.approx(0.85)

    def test_exports_survive_reinit(self, tmp_path):
        t1 = ExperimentTracker(experiments_dir=tmp_path)
        t1.log_experiment(prompt="a", clip_score=0.90)
        csv_path = t1.export_csv()

        t2 = ExperimentTracker(experiments_dir=tmp_path)
        assert csv_path.exists()
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
