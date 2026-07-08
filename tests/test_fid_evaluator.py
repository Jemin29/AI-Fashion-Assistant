"""
week2/tests/test_fid_evaluator.py
=====================================
Comprehensive test suite for FIDEvaluator.

Strategy
--------
All tests run in stub/mock mode:
  - No real GPU / pytorch-fid / torchmetrics required
  - The core FID computation is mocked to inject controlled values
  - All dataclasses, quality thresholds, backend logic, and
    public API functions are tested

Coverage targets
----------------
  - FIDScore, DatasetComparison, BenchmarkReport dataclasses
  - FIDEvaluator.__init__, backend detection, properties
  - calculate_fid() — success, stub, error paths
  - compare_with_dataset() — with/without dataset_dir
  - benchmark_results() — multi-run, aggregation, best/worst
  - Quality rating and quality score thresholds
  - _frechet_distance static method
  - _rate_quality static method
  - _fid_to_quality_score static method
  - _resolve_device static method
  - _load_images_from_dir (with tmp files)
  - _save_images_to_dir (with tmp files)
  - _create_synthetic_reference
  - _batch_iter
  - load_inception_stats / _save_inception_stats
  - Module-level: calculate_fid, compare_with_dataset, benchmark_results
  - FID_QUALITY_THRESHOLDS constant
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.week2_fid_evaluator import (
    FIDEvaluator,
    FIDScore,
    DatasetComparison,
    BenchmarkReport,
    FID_QUALITY_THRESHOLDS,
    MIN_IMAGES_FOR_FID,
    INCEPTION_FEATURE_DIM,
    calculate_fid,
    compare_with_dataset,
    benchmark_results,
)


# =============================================================================
# ── Helpers / Fixtures
# =============================================================================

def _make_image(color: tuple = (128, 64, 200), size: int = 64) -> PILImage.Image:
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    return PILImage.fromarray(arr)

def _make_images(n: int = 5) -> List[PILImage.Image]:
    return [_make_image((i * 40 % 255, i * 70 % 255, i * 120 % 255)) for i in range(n)]

@pytest.fixture
def imgs():
    return _make_images(5)

@pytest.fixture
def real_imgs():
    return _make_images(5)

@pytest.fixture
def gen_imgs():
    return _make_images(4)

@pytest.fixture
def evaluator():
    return FIDEvaluator(device="cpu", batch_size=2)

@pytest.fixture
def mock_evaluator(evaluator):
    """FIDEvaluator with _compute_fid mocked to return FID=28.4."""
    with patch.object(evaluator, "_compute_fid", return_value=(28.4, [0.1]*8, [0.2]*8)):
        yield evaluator


# =============================================================================
# ── Module Constants
# =============================================================================

class TestModuleConstants:

    def test_fid_quality_thresholds_keys(self):
        expected = {"excellent", "very good", "good", "fair", "poor", "very poor"}
        assert set(FID_QUALITY_THRESHOLDS.keys()) == expected

    def test_fid_quality_thresholds_ordered(self):
        labels = ["excellent", "very good", "good", "fair", "poor", "very poor"]
        lows   = [FID_QUALITY_THRESHOLDS[l][0] for l in labels]
        assert lows == sorted(lows), "Quality thresholds should be ascending"

    def test_min_images_for_fid(self):
        assert MIN_IMAGES_FOR_FID >= 2

    def test_inception_feature_dim(self):
        assert INCEPTION_FEATURE_DIM == 2048


# =============================================================================
# ── FIDScore Dataclass
# =============================================================================

class TestFIDScore:

    def test_default_instantiation(self):
        sc = FIDScore()
        assert sc.fid_score == float("inf")
        assert sc.quality_rating == "unknown"
        assert sc.passed is False

    def test_valid_score_passed(self):
        sc = FIDScore(fid_score=28.4, quality_rating="good")
        assert sc.passed is True

    def test_inf_score_not_passed(self):
        sc = FIDScore(fid_score=float("inf"))
        assert sc.passed is False

    def test_poor_quality_not_passed(self):
        sc = FIDScore(fid_score=250.0, quality_rating="very poor")
        assert sc.passed is False

    def test_to_dict_returns_dict(self):
        sc = FIDScore(fid_score=25.0, quality_rating="very good", n_real=100, n_generated=50)
        d  = sc.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_required_keys(self):
        sc = FIDScore(fid_score=25.0, quality_rating="good")
        d  = sc.to_dict()
        for key in ["fid_score", "quality_rating", "quality_score", "passed",
                    "n_real", "n_generated", "feature_dim", "backend",
                    "device", "elapsed_s", "run_id", "error"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_fid_score_rounded(self):
        sc = FIDScore(fid_score=28.123456789)
        d  = sc.to_dict()
        assert d["fid_score"] == round(28.123456789, 4)

    def test_to_dict_inf_fid_is_none(self):
        sc = FIDScore(fid_score=float("inf"))
        d  = sc.to_dict()
        assert d["fid_score"] is None

    def test_to_dict_json_serialisable(self):
        sc = FIDScore(fid_score=28.4, quality_rating="good", n_real=50, n_generated=50)
        json.dumps(sc.to_dict())   # Must not raise

    def test_summary_returns_string(self):
        sc = FIDScore(fid_score=28.4, quality_rating="good", backend="pytorch_fid")
        s  = sc.summary()
        assert isinstance(s, str)
        assert "28.4" in s

    def test_summary_contains_quality(self):
        sc = FIDScore(fid_score=28.4, quality_rating="good")
        assert "good" in sc.summary()

    def test_repr_eq_summary(self):
        sc = FIDScore(fid_score=28.4, quality_rating="good")
        assert repr(sc) == sc.summary()

    def test_auto_run_id(self):
        sc1 = FIDScore()
        sc2 = FIDScore()
        assert sc1.run_id != sc2.run_id

    def test_quality_score_for_good_fid(self):
        sc = FIDScore(fid_score=28.4, quality_score=0.858)
        assert 0.0 <= sc.quality_score <= 1.0


# =============================================================================
# ── DatasetComparison Dataclass
# =============================================================================

class TestDatasetComparison:

    def _make(self, fid=28.4, quality="good"):
        return DatasetComparison(
            dataset_name = "test_dataset",
            fid_result   = FIDScore(fid_score=fid, quality_rating=quality),
            style        = "luxury",
        )

    def test_instantiation(self):
        dc = self._make()
        assert dc.dataset_name == "test_dataset"
        assert dc.style        == "luxury"

    def test_to_dict_keys(self):
        dc = self._make()
        d  = dc.to_dict()
        for key in ["dataset_name", "style", "fid_result", "improvement",
                    "percentile_rank", "notes"]:
            assert key in d

    def test_to_dict_json_serialisable(self):
        dc = self._make()
        json.dumps(dc.to_dict())

    def test_summary_returns_string(self):
        dc = self._make()
        s  = dc.summary()
        assert isinstance(s, str)
        assert "test_dataset" in s

    def test_improvement_positive(self):
        dc = DatasetComparison(
            dataset_name  = "ds",
            fid_result    = FIDScore(fid_score=30.0, quality_rating="good"),
            improvement   = 10.0,
        )
        assert dc.improvement == 10.0

    def test_improvement_negative(self):
        dc = DatasetComparison(
            dataset_name  = "ds",
            fid_result    = FIDScore(fid_score=50.0, quality_rating="fair"),
            improvement   = -5.0,
        )
        assert dc.improvement == -5.0

    def test_percentile_rank_in_dict(self):
        dc = DatasetComparison(
            dataset_name   = "ds",
            fid_result     = FIDScore(),
            percentile_rank= 0.75,
        )
        d = dc.to_dict()
        assert d["percentile_rank"] == pytest.approx(0.75)


# =============================================================================
# ── BenchmarkReport Dataclass
# =============================================================================

class TestBenchmarkReport:

    def _make_report(self):
        report = BenchmarkReport(reference_dir="data/real/")
        report.runs = {
            "baseline": FIDScore(fid_score=45.0, quality_rating="fair"),
            "improved": FIDScore(fid_score=28.4, quality_rating="good"),
            "refined":  FIDScore(fid_score=18.2, quality_rating="very good"),
        }
        report.best_run  = "refined"
        report.worst_run = "baseline"
        report.mean_fid  = 30.5
        report.std_fid   = 11.2
        return report

    def test_instantiation(self):
        report = BenchmarkReport()
        assert isinstance(report.runs, dict)

    def test_to_dict_keys(self):
        report = self._make_report()
        d = report.to_dict()
        for key in ["generated_at", "reference_dir", "best_run", "worst_run",
                    "mean_fid", "std_fid", "elapsed_s", "runs"]:
            assert key in d

    def test_to_dict_runs_populated(self):
        report = self._make_report()
        d = report.to_dict()
        assert "baseline" in d["runs"]
        assert "improved" in d["runs"]
        assert "refined"  in d["runs"]

    def test_to_dict_json_serialisable(self):
        report = self._make_report()
        json.dumps(report.to_dict())

    def test_summary_returns_string(self):
        report = self._make_report()
        s = report.summary()
        assert isinstance(s, str)

    def test_summary_contains_best_run(self):
        report = self._make_report()
        s = report.summary()
        assert "refined" in s

    def test_summary_contains_mean_fid(self):
        report = self._make_report()
        s = report.summary()
        assert "30.5" in s

    def test_summary_best_marker(self):
        report = self._make_report()
        s = report.summary()
        assert "BEST" in s


# =============================================================================
# ── FIDEvaluator — Initialisation
# =============================================================================

class TestFIDEvaluatorInit:

    def test_default_instantiation(self):
        ev = FIDEvaluator()
        assert ev is not None

    def test_device_cpu_explicit(self):
        ev = FIDEvaluator(device="cpu")
        assert ev.device == "cpu"

    def test_device_auto_resolves(self):
        ev = FIDEvaluator(device="auto")
        assert ev.device in ("cpu", "cuda")

    def test_batch_size_set(self):
        ev = FIDEvaluator(batch_size=16)
        assert ev.batch_size == 16

    def test_feature_dim_set(self):
        ev = FIDEvaluator(feature_dim=2048)
        assert ev.feature_dim == 2048

    def test_backend_property(self):
        ev = FIDEvaluator()
        assert isinstance(ev.backend, str)
        assert ev.backend in ("pytorch_fid", "torchmetrics", "manual", "stub")

    def test_is_available_property(self):
        ev = FIDEvaluator()
        assert isinstance(ev.is_available, bool)

    def test_repr(self):
        ev = FIDEvaluator()
        r = repr(ev)
        assert "FIDEvaluator" in r
        assert "backend" in r


# =============================================================================
# ── calculate_fid() — via mock
# =============================================================================

class TestCalculateFID:

    def test_returns_fid_score(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert isinstance(result, FIDScore)

    def test_fid_score_value(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.fid_score == pytest.approx(28.4, abs=0.01)

    def test_quality_rating_populated(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.quality_rating in FID_QUALITY_THRESHOLDS

    def test_quality_score_in_range(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert 0.0 <= result.quality_score <= 1.0

    def test_n_real_populated(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.n_real == len(real_imgs)

    def test_n_generated_populated(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.n_generated == len(gen_imgs)

    def test_device_in_result(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.device == "cpu"

    def test_elapsed_nonnegative(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.elapsed_s >= 0.0

    def test_no_error_on_success(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        assert result.error == ""

    def test_custom_run_id(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs, run_id="my_run_001")
        assert result.run_id == "my_run_001"

    def test_too_few_real_images_returns_error(self, evaluator, gen_imgs):
        """1 real image should return an error (needs >= 2)."""
        result = evaluator.calculate_fid([_make_image()], gen_imgs)
        assert result.error != ""
        assert result.fid_score == float("inf")

    def test_too_few_gen_images_returns_error(self, evaluator, real_imgs):
        result = evaluator.calculate_fid(real_imgs, [_make_image()])
        assert result.error != ""

    def test_empty_real_returns_error(self, evaluator, gen_imgs):
        result = evaluator.calculate_fid([], gen_imgs)
        assert result.error != ""

    def test_to_dict_json_serialisable(self, mock_evaluator, real_imgs, gen_imgs):
        result = mock_evaluator.calculate_fid(real_imgs, gen_imgs)
        json.dumps(result.to_dict())

    def test_directory_source_real(self, mock_evaluator, gen_imgs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Save 3 images to directory
            for i in range(3):
                img = _make_image()
                img.save(str(tmp_path / f"img_{i}.png"))
            result = mock_evaluator.calculate_fid(tmp_path, gen_imgs)
            assert isinstance(result, FIDScore)

    def test_stub_mode_returns_error(self, real_imgs, gen_imgs):
        ev = FIDEvaluator()
        with patch.object(ev, "_backend", "stub"):
            result = ev.calculate_fid(real_imgs, gen_imgs)
        # stub backend returns inf — error message added
        assert result.fid_score == float("inf") or result.error != ""

    def test_passed_property_good_fid(self):
        sc = FIDScore(fid_score=25.0, quality_rating="good")
        assert sc.passed is True

    def test_passed_property_excellent_fid(self):
        sc = FIDScore(fid_score=5.0, quality_rating="excellent")
        assert sc.passed is True

    def test_passed_property_poor_fid(self):
        sc = FIDScore(fid_score=150.0, quality_rating="poor")
        assert sc.passed is False


# =============================================================================
# ── compare_with_dataset()
# =============================================================================

class TestCompareWithDataset:

    def test_returns_dataset_comparison(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "deepfashion")
        assert isinstance(result, DatasetComparison)

    def test_dataset_name_set(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "fashion_mnist")
        assert result.dataset_name == "fashion_mnist"

    def test_fid_result_is_fid_score(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "test_ds")
        assert isinstance(result.fid_result, FIDScore)

    def test_style_set(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "ds", style="luxury")
        assert result.style == "luxury"

    def test_improvement_calculated_when_baseline_given(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(
            gen_imgs, "ds", baseline_fid=45.0
        )
        # improvement = baseline - current = 45.0 - 28.4 = 16.6
        assert result.improvement is not None
        assert result.improvement == pytest.approx(16.6, abs=0.1)

    def test_improvement_negative_when_worse_than_baseline(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(
            gen_imgs, "ds", baseline_fid=20.0
        )
        # current FID=28.4 > baseline=20.0 → regression
        assert result.improvement is not None
        assert result.improvement < 0

    def test_no_improvement_when_no_baseline(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "ds")
        assert result.improvement is None

    def test_percentile_rank_with_known_benchmarks(self, mock_evaluator, gen_imgs):
        benchmarks = {"model_a": 50.0, "model_b": 30.0, "model_c": 20.0}
        result = mock_evaluator.compare_with_dataset(
            gen_imgs, "ds", known_benchmarks=benchmarks
        )
        assert result.percentile_rank is not None
        assert 0.0 <= result.percentile_rank <= 1.0

    def test_notes_populated(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "ds")
        assert isinstance(result.notes, list)

    def test_no_dataset_dir_uses_synthetic_reference(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "no_dir_ds")
        assert isinstance(result, DatasetComparison)
        synthetic_note = any("synthetic" in n.lower() for n in result.notes)
        assert synthetic_note

    def test_nonexistent_dataset_dir_falls_back(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(
            gen_imgs, "ds", dataset_dir="/nonexistent/path/xyzabc"
        )
        assert isinstance(result, DatasetComparison)

    def test_valid_dataset_dir(self, mock_evaluator, gen_imgs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(3):
                _make_image().save(str(tmp_path / f"ref_{i}.png"))
            result = mock_evaluator.compare_with_dataset(
                gen_imgs, "local_ds", dataset_dir=tmp_path
            )
            assert isinstance(result, DatasetComparison)

    def test_to_dict_json_serialisable(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "ds")
        json.dumps(result.to_dict())

    def test_summary_string(self, mock_evaluator, gen_imgs):
        result = mock_evaluator.compare_with_dataset(gen_imgs, "deepfashion")
        assert "deepfashion" in result.summary()


# =============================================================================
# ── benchmark_results()
# =============================================================================

class TestBenchmarkResults:

    def test_returns_benchmark_report(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"run_a": gen_imgs, "run_b": gen_imgs}
        )
        assert isinstance(report, BenchmarkReport)

    def test_all_runs_present(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"v1": gen_imgs, "v2": gen_imgs, "v3": gen_imgs}
        )
        assert set(report.runs.keys()) == {"v1", "v2", "v3"}

    def test_best_run_set(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"run_a": gen_imgs, "run_b": gen_imgs}
        )
        assert report.best_run in report.runs

    def test_worst_run_set(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"run_a": gen_imgs, "run_b": gen_imgs}
        )
        assert report.worst_run in report.runs

    def test_mean_fid_populated(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"run_a": gen_imgs, "run_b": gen_imgs}
        )
        assert report.mean_fid != float("inf")
        assert report.mean_fid == pytest.approx(28.4, abs=0.01)

    def test_elapsed_nonnegative(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(runs={"run_a": gen_imgs})
        assert report.elapsed_s >= 0.0

    def test_empty_runs_returns_empty_report(self, evaluator):
        report = evaluator.benchmark_results(runs={})
        assert len(report.runs) == 0

    def test_single_run(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(runs={"only": gen_imgs})
        assert "only" in report.runs
        assert report.best_run == "only"

    def test_with_reference_dir(self, mock_evaluator, gen_imgs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(3):
                _make_image().save(str(tmp_path / f"ref_{i}.png"))
            report = mock_evaluator.benchmark_results(
                runs={"run_a": gen_imgs},
                reference_dir=tmp_path,
            )
            assert "run_a" in report.runs

    def test_save_report_writes_file(self, mock_evaluator, gen_imgs, tmp_path):
        report_file = tmp_path / "test_report.json"
        mock_evaluator.benchmark_results(
            runs        = {"run_a": gen_imgs},
            save_report = True,
            report_path = report_file,
        )
        assert report_file.exists()

    def test_saved_report_valid_json(self, mock_evaluator, gen_imgs, tmp_path):
        report_file = tmp_path / "test_report.json"
        mock_evaluator.benchmark_results(
            runs        = {"run_a": gen_imgs},
            save_report = True,
            report_path = report_file,
        )
        data = json.loads(report_file.read_text())
        assert "runs" in data
        assert "best_run" in data

    def test_to_dict_json_serialisable(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(runs={"run_a": gen_imgs})
        json.dumps(report.to_dict())

    def test_summary_string(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"run_a": gen_imgs, "run_b": gen_imgs}
        )
        s = report.summary()
        assert isinstance(s, str)
        assert "BENCHMARK" in s

    def test_no_reference_uses_first_run_as_baseline(self, mock_evaluator, gen_imgs):
        report = mock_evaluator.benchmark_results(
            runs={"base": gen_imgs, "new": gen_imgs},
        )
        assert "auto" in report.reference_dir.lower() or report.reference_dir != ""


# =============================================================================
# ── Static Methods
# =============================================================================

class TestStaticMethods:

    @pytest.mark.parametrize("fid,expected_label", [
        (0.0,   "excellent"),
        (5.0,   "excellent"),
        (9.99,  "excellent"),
        (10.0,  "very good"),
        (24.99, "very good"),
        (25.0,  "good"),
        (49.99, "good"),
        (50.0,  "fair"),
        (99.99, "fair"),
        (100.0, "poor"),
        (199.99,"poor"),
        (200.0, "very poor"),
        (500.0, "very poor"),
    ])
    def test_rate_quality_thresholds(self, fid, expected_label):
        assert FIDEvaluator._rate_quality(fid) == expected_label

    def test_rate_quality_inf(self):
        assert FIDEvaluator._rate_quality(float("inf")) == "unknown"

    def test_rate_quality_nan(self):
        assert FIDEvaluator._rate_quality(float("nan")) == "unknown"

    @pytest.mark.parametrize("fid,expected_approx", [
        (0.0,   1.0),
        (100.0, 0.5),
        (200.0, 0.0),
        (400.0, 0.0),   # clamped
    ])
    def test_fid_to_quality_score(self, fid, expected_approx):
        result = FIDEvaluator._fid_to_quality_score(fid)
        assert result == pytest.approx(expected_approx, abs=0.01)

    def test_fid_to_quality_score_inf(self):
        assert FIDEvaluator._fid_to_quality_score(float("inf")) == 0.0

    @pytest.mark.parametrize("device,expected", [
        ("cpu",  "cpu"),
        ("cuda", "cuda"),
    ])
    def test_resolve_device_explicit(self, device, expected):
        assert FIDEvaluator._resolve_device(device) == expected

    def test_resolve_device_auto(self):
        result = FIDEvaluator._resolve_device("auto")
        assert result in ("cpu", "cuda")

    def test_frechet_distance_identical(self):
        """FID between identical distributions should be near zero."""
        mu    = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3) * 0.5
        fid   = FIDEvaluator._frechet_distance(mu, sigma, mu, sigma)
        assert fid == pytest.approx(0.0, abs=1e-3)

    def test_frechet_distance_different(self):
        mu1 = np.zeros(4)
        mu2 = np.ones(4) * 5.0
        s   = np.eye(4)
        fid = FIDEvaluator._frechet_distance(mu1, s, mu2, s)
        assert fid > 0.0

    def test_frechet_distance_symmetric(self):
        mu1 = np.array([0.0, 1.0])
        mu2 = np.array([1.0, 0.0])
        s   = np.eye(2) * 0.3
        fid_ab = FIDEvaluator._frechet_distance(mu1, s, mu2, s)
        fid_ba = FIDEvaluator._frechet_distance(mu2, s, mu1, s)
        assert fid_ab == pytest.approx(fid_ba, abs=1e-3)


# =============================================================================
# ── Image Utilities
# =============================================================================

class TestImageUtils:

    def test_load_images_from_dir(self, tmp_path):
        for i in range(3):
            _make_image().save(str(tmp_path / f"img_{i}.png"))
        imgs = FIDEvaluator._load_images_from_dir(tmp_path)
        assert len(imgs) == 3
        for img in imgs:
            assert isinstance(img, PILImage.Image)

    def test_load_images_from_dir_filters_non_images(self, tmp_path):
        _make_image().save(str(tmp_path / "good.png"))
        (tmp_path / "readme.txt").write_text("not an image")
        imgs = FIDEvaluator._load_images_from_dir(tmp_path)
        assert len(imgs) == 1

    def test_load_images_from_dir_empty(self, tmp_path):
        imgs = FIDEvaluator._load_images_from_dir(tmp_path)
        assert imgs == []

    def test_save_images_to_dir(self, tmp_path):
        images = _make_images(3)
        FIDEvaluator._save_images_to_dir(images, tmp_path, prefix="out")
        saved = list(tmp_path.glob("*.png"))
        assert len(saved) == 3

    def test_batch_iter_correct_batches(self):
        items  = list(range(10))
        batches= list(FIDEvaluator._batch_iter(items, batch_size=3))
        assert batches == [[0,1,2], [3,4,5], [6,7,8], [9]]

    def test_batch_iter_single_batch(self):
        items  = list(range(4))
        batches= list(FIDEvaluator._batch_iter(items, 10))
        assert batches == [list(range(4))]

    def test_batch_iter_empty(self):
        assert list(FIDEvaluator._batch_iter([], 4)) == []

    def test_resolve_images_list(self, evaluator, imgs):
        result = evaluator._resolve_images(imgs, "test")
        assert result == imgs

    def test_resolve_images_drops_none(self, evaluator, imgs):
        with_none = imgs + [None]
        result = evaluator._resolve_images(with_none, "test")
        assert None not in result
        assert len(result) == len(imgs)

    def test_resolve_images_directory(self, evaluator, tmp_path):
        for i in range(2):
            _make_image().save(str(tmp_path / f"img_{i}.png"))
        result = evaluator._resolve_images(tmp_path, "test")
        assert len(result) == 2

    def test_resolve_images_nonexistent_dir_raises(self, evaluator):
        with pytest.raises(FileNotFoundError):
            evaluator._resolve_images(Path("/totally/nonexistent/xyzabc"), "test")

    def test_resolve_images_invalid_type_raises(self, evaluator):
        with pytest.raises(TypeError):
            evaluator._resolve_images(12345, "test")

    def test_create_synthetic_reference(self, evaluator, gen_imgs):
        synthetic = evaluator._create_synthetic_reference(gen_imgs)
        # May return augmented (same count) or halved fallback — must be >= MIN
        assert len(synthetic) >= MIN_IMAGES_FOR_FID
        assert len(synthetic) <= len(gen_imgs)
        for img in synthetic:
            assert isinstance(img, PILImage.Image)


# =============================================================================
# ── Inception Statistics Caching
# =============================================================================

class TestInceptionStatsCache:

    def test_save_and_load_stats(self, tmp_path):
        ev = FIDEvaluator(stats_cache_dir=tmp_path, save_stats=True)
        mu    = np.random.randn(2048)
        sigma = np.eye(2048)
        ev._save_inception_stats("test_key", mu, sigma)
        loaded = ev.load_inception_stats("test_key")
        assert loaded is not None
        loaded_mu, loaded_sigma = loaded
        np.testing.assert_allclose(loaded_mu, mu, atol=1e-5)

    def test_load_nonexistent_returns_none(self, tmp_path):
        ev = FIDEvaluator(stats_cache_dir=tmp_path)
        result = ev.load_inception_stats("nonexistent_key")
        assert result is None

    def test_no_cache_dir_returns_none_on_load(self):
        ev = FIDEvaluator(stats_cache_dir=None)
        assert ev.load_inception_stats("any_key") is None


# =============================================================================
# ── Module-Level Convenience Functions
# =============================================================================

class TestModuleFunctions:

    def _patch_compute(self, evaluator):
        """Helper: patch _compute_fid on any evaluator instance."""
        return patch.object(evaluator, "_compute_fid", return_value=(28.4, None, None))

    def test_calculate_fid_returns_fid_score(self, real_imgs, gen_imgs):
        with patch("src.evaluation.week2_fid_evaluator.FIDEvaluator.calculate_fid") as mock_fn:
            mock_fn.return_value = FIDScore(fid_score=28.4, quality_rating="good")
            result = calculate_fid(real_imgs, gen_imgs)
        assert isinstance(result, FIDScore)

    def test_compare_with_dataset_returns_comparison(self, gen_imgs):
        with patch("src.evaluation.week2_fid_evaluator.FIDEvaluator.compare_with_dataset") as mock_fn:
            mock_fn.return_value = DatasetComparison(
                dataset_name="ds",
                fid_result=FIDScore(fid_score=28.4, quality_rating="good"),
            )
            result = compare_with_dataset(gen_imgs, "ds")
        assert isinstance(result, DatasetComparison)

    def test_benchmark_results_returns_report(self, gen_imgs):
        with patch("src.evaluation.week2_fid_evaluator.FIDEvaluator.benchmark_results") as mock_fn:
            mock_fn.return_value = BenchmarkReport()
            result = benchmark_results(runs={"r1": gen_imgs})
        assert isinstance(result, BenchmarkReport)


# =============================================================================
# ── Integration: Full pipeline (stub mode, no model)
# =============================================================================

class TestIntegration:

    def test_full_calculate_fid_pipeline(self, gen_imgs):
        ev = FIDEvaluator(device="cpu")
        with patch.object(ev, "_compute_fid", return_value=(35.2, [0.1]*8, [0.2]*8)):
            result = ev.calculate_fid(gen_imgs, gen_imgs)
        assert result.fid_score == pytest.approx(35.2, abs=0.01)
        assert result.quality_rating == "good"
        assert result.quality_score  > 0.0
        assert result.n_real         == len(gen_imgs)
        assert result.n_generated    == len(gen_imgs)
        assert result.error          == ""

    def test_full_benchmark_pipeline_three_runs(self, gen_imgs):
        ev     = FIDEvaluator(device="cpu")
        fid_vals = [45.0, 28.4, 15.0]
        calls  = [0]
        def mock_compute(real, gen, cache=None):
            v = fid_vals[calls[0] % len(fid_vals)]
            calls[0] += 1
            return v, None, None
        with patch.object(ev, "_compute_fid", side_effect=mock_compute):
            report = ev.benchmark_results(
                runs = {
                    "baseline": gen_imgs,
                    "improved": gen_imgs,
                    "refined":  gen_imgs,
                }
            )
        assert report.best_run  == "refined"
        assert report.worst_run == "baseline"
        assert report.mean_fid  == pytest.approx((45.0 + 28.4 + 15.0) / 3, abs=0.1)

    def test_compare_with_improvement_tracking(self, gen_imgs):
        ev = FIDEvaluator(device="cpu")
        with patch.object(ev, "_compute_fid", return_value=(28.4, None, None)):
            result = ev.compare_with_dataset(
                gen_imgs, "fashion_reference",
                style        = "luxury",
                baseline_fid = 45.0,
            )
        assert result.improvement == pytest.approx(16.6, abs=0.1)
        assert result.style == "luxury"
        assert result.dataset_name == "fashion_reference"
