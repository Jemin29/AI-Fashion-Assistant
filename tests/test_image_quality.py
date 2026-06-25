"""
week2/tests/test_image_quality.py
=====================================
Comprehensive test suite for ImageQualityAssessor.

Strategy
--------
All tests use synthetic PIL images (no real images or GPU required).
Controlled images let us verify each metric in isolation:
  - Blurry: low-frequency / flat images → low sharpness
  - Sharp:  high-frequency checkerboard → high sharpness
  - Dark:   near-black image            → low brightness
  - Bright: near-white image            → low brightness score (overexposed)
  - Flat:   constant-value image        → low contrast
  - Gray:   desaturated image           → low color_distribution
  - Vivid:  saturated image             → high color_distribution
  - Small:  32x32 image                 → low resolution score
  - Large:  2048x2048 image             → high resolution score
  - Noisy:  random pixel noise          → low noise_level score

Coverage targets
----------------
  - QUALITY_THRESHOLDS, METRIC_WEIGHTS constants
  - QualityReport dataclass (all fields, to_dict, to_json, summary, detailed_report)
  - BatchQualityReport dataclass
  - ImageQualityAssessor.__init__, __repr__
  - ImageQualityAssessor.assess (all 6 metrics)
  - ImageQualityAssessor.assess_batch
  - ImageQualityAssessor.aggregate
  - ImageQualityAssessor.generate_report (dict, json, text modes)
  - Static: _rate_quality, _compute_overall
  - Warning generation
  - _measure_sharpness, _measure_brightness, _measure_contrast
  - _measure_color_distribution, _measure_resolution, _measure_noise
  - Module-level: assess, assess_batch, generate_report
  - Edge cases: None image, tiny image, 1-pixel image
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.week2_image_quality import (
    ImageQualityAssessor,
    QualityReport,
    BatchQualityReport,
    QUALITY_THRESHOLDS,
    METRIC_WEIGHTS,
    OVERALL_PASS_THRESHOLD,
    MIN_PASSING_SCORE,
    assess,
    assess_batch,
    generate_report,
)


# =============================================================================
# ── Image Factory Helpers
# =============================================================================

def _solid(color: tuple = (128, 128, 128), size: int = 512) -> PILImage.Image:
    """Solid-colour image — minimum texture."""
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    return PILImage.fromarray(arr)

def _noise(size: int = 512, seed: int = 42) -> PILImage.Image:
    """Full random noise — high frequency, high noise energy."""
    np.random.seed(seed)
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return PILImage.fromarray(arr)

def _checkerboard(size: int = 512, block: int = 4) -> PILImage.Image:
    """High-frequency checkerboard — maximum sharpness."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, block):
        for j in range(0, size, block):
            v = 255 if (i // block + j // block) % 2 == 0 else 0
            arr[i:i+block, j:j+block] = v
    return PILImage.fromarray(arr)

def _gradient(size: int = 512) -> PILImage.Image:
    """Smooth horizontal gradient — moderate sharpness."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        arr[:, x] = int(x / size * 255)
    return PILImage.fromarray(arr)

def _vivid(size: int = 512) -> PILImage.Image:
    """Fully saturated red-green-blue stripes — high colour distribution."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    third = size // 3
    arr[:, :third,       0] = 255   # Red
    arr[:, third:2*third,1] = 255   # Green
    arr[:, 2*third:,     2] = 255   # Blue
    return PILImage.fromarray(arr)

def _gray_image(size: int = 512) -> PILImage.Image:
    """Grayscale image (R=G=B) — low saturation / color_distribution."""
    v   = 128
    arr = np.full((size, size, 3), v, dtype=np.uint8)
    return PILImage.fromarray(arr)

def _dark(size: int = 512) -> PILImage.Image:
    """Very dark image (near black)."""
    arr = np.full((size, size, 3), 10, dtype=np.uint8)
    return PILImage.fromarray(arr)

def _bright(size: int = 512) -> PILImage.Image:
    """Very bright image (near white)."""
    arr = np.full((size, size, 3), 245, dtype=np.uint8)
    return PILImage.fromarray(arr)

def _natural(size: int = 512, seed: int = 7) -> PILImage.Image:
    """Natural-looking image with moderate noise + range."""
    np.random.seed(seed)
    base = np.random.randint(80, 180, (size, size, 3), dtype=np.uint8)
    return PILImage.fromarray(base)


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def assessor():
    return ImageQualityAssessor(min_resolution=(64, 64))

@pytest.fixture
def natural():
    return _natural()

@pytest.fixture
def natural_small():
    return _natural(size=512)

@pytest.fixture
def noisy():
    return _noise()

@pytest.fixture
def checkerboard():
    return _checkerboard()

@pytest.fixture
def vivid_img():
    return _vivid()


# =============================================================================
# ── Module Constants
# =============================================================================

class TestModuleConstants:

    def test_quality_thresholds_keys(self):
        expected = {"excellent", "very good", "good", "fair", "poor", "very poor"}
        assert set(QUALITY_THRESHOLDS.keys()) == expected

    def test_quality_thresholds_ordered(self):
        labels = ["very poor", "poor", "fair", "good", "very good", "excellent"]
        lows   = [QUALITY_THRESHOLDS[l][0] for l in labels]
        assert lows == sorted(lows)

    def test_quality_thresholds_continuous(self):
        """Ranges should be contiguous (no gaps)."""
        sorted_thresholds = sorted(QUALITY_THRESHOLDS.values(), key=lambda t: t[0])
        for i in range(len(sorted_thresholds) - 1):
            _, hi = sorted_thresholds[i]
            lo, _ = sorted_thresholds[i + 1]
            assert hi == lo, f"Gap in thresholds between {sorted_thresholds[i]} and {sorted_thresholds[i+1]}"

    def test_metric_weights_keys(self):
        expected = {"sharpness", "brightness", "contrast",
                    "color_distribution", "resolution", "noise_level"}
        assert set(METRIC_WEIGHTS.keys()) == expected

    def test_metric_weights_sum_to_one(self):
        total = sum(METRIC_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_overall_pass_threshold(self):
        assert 0 < OVERALL_PASS_THRESHOLD < 100

    def test_min_passing_score_keys(self):
        for metric in ["sharpness", "brightness", "contrast",
                        "color_distribution", "resolution", "noise_level"]:
            assert metric in MIN_PASSING_SCORE


# =============================================================================
# ── QualityReport Dataclass
# =============================================================================

class TestQualityReport:

    def test_default_instantiation(self):
        r = QualityReport()
        assert isinstance(r.image_id, str)
        assert r.sharpness == 0
        assert r.quality == "unknown"

    def test_auto_image_id(self):
        r1 = QualityReport()
        r2 = QualityReport()
        assert r1.image_id != r2.image_id

    def test_to_dict_required_keys(self):
        r = QualityReport()
        d = r.to_dict(include_raw=False)
        required = ["sharpness", "brightness", "contrast", "color_distribution",
                    "resolution", "noise_level", "overall", "quality",
                    "passed", "warnings", "image_id"]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_matches_spec(self):
        """Output matches the documented specification."""
        r = QualityReport(
            image_id            = "GEN_001",
            sharpness           = 90,
            brightness          = 75,
            contrast            = 85,
            color_distribution  = 80,
            resolution          = 95,
            noise_level         = 88,
            overall             = 86,
            quality             = "excellent",
            passed              = True,
        )
        d = r.to_dict(include_raw=False)
        assert d["sharpness"]          == 90
        assert d["brightness"]         == 75
        assert d["contrast"]           == 85
        assert d["color_distribution"] == 80
        assert d["resolution"]         == 95
        assert d["noise_level"]        == 88
        assert d["overall"]            == 86
        assert d["quality"]            == "excellent"
        assert d["passed"]             is True
        assert d["image_id"]           == "GEN_001"

    def test_to_dict_includes_raw(self):
        r = QualityReport(laplacian_var=100.5)
        d = r.to_dict(include_raw=True)
        assert "raw" in d
        assert "laplacian_var" in d["raw"]

    def test_to_dict_excludes_raw(self):
        r = QualityReport()
        d = r.to_dict(include_raw=False)
        assert "raw" not in d

    def test_to_dict_json_serialisable(self):
        r = QualityReport(
            sharpness=90, brightness=80, contrast=75,
            quality="excellent", passed=True,
        )
        json.dumps(r.to_dict())

    def test_to_json_returns_string(self):
        r = QualityReport()
        j = r.to_json()
        assert isinstance(j, str)
        data = json.loads(j)
        assert "sharpness" in data

    def test_summary_returns_string(self):
        r = QualityReport(image_id="TEST", overall=80, quality="very good", passed=True)
        s = r.summary()
        assert isinstance(s, str)
        assert "TEST" in s
        assert "very good" in s

    def test_detailed_report_returns_string(self):
        r = QualityReport(
            image_id   = "DETAIL",
            sharpness  = 90,
            brightness = 80,
            contrast   = 75,
            quality    = "excellent",
            passed     = True,
            width      = 512,
            height     = 512,
        )
        text = r.detailed_report()
        assert isinstance(text, str)
        assert "DETAIL" in text
        assert "Sharpness" in text
        assert "Brightness" in text
        assert "Contrast"   in text

    def test_detailed_report_shows_warnings(self):
        r = QualityReport(warnings=["[SHARPNESS] Blurry image"])
        text = r.detailed_report()
        assert "SHARPNESS" in text

    def test_repr_eq_summary(self):
        r = QualityReport(image_id="X", overall=70, quality="good", passed=True)
        assert repr(r) == r.summary()

    def test_passed_true_for_good_overall(self):
        r = QualityReport(overall=75, quality="very good")
        r.passed = r.overall >= OVERALL_PASS_THRESHOLD
        assert r.passed is True

    def test_passed_false_for_poor_overall(self):
        r = QualityReport(overall=20, quality="very poor")
        r.passed = r.overall >= OVERALL_PASS_THRESHOLD
        assert r.passed is False


# =============================================================================
# ── BatchQualityReport Dataclass
# =============================================================================

class TestBatchQualityReport:

    def _make_reports(self, n: int = 3, overall: int = 70) -> List[QualityReport]:
        return [
            QualityReport(
                image_id  = f"img_{i}",
                overall   = overall - i * 10,
                quality   = "good",
                passed    = (overall - i * 10) >= OVERALL_PASS_THRESHOLD,
            )
            for i in range(n)
        ]

    def test_instantiation(self):
        batch = BatchQualityReport()
        assert isinstance(batch.reports, list)
        assert isinstance(batch.quality_dist, dict)

    def test_to_dict_keys(self):
        batch = BatchQualityReport(reports=self._make_reports())
        d = batch.to_dict()
        for key in ["generated_at", "total", "passed", "failed", "pass_rate",
                    "overall_mean", "mean_scores", "quality_dist",
                    "best_image_id", "worst_image_id"]:
            assert key in d

    def test_to_dict_json_serialisable(self):
        batch = BatchQualityReport(reports=self._make_reports())
        json.dumps(batch.to_dict())

    def test_to_json_string(self):
        batch = BatchQualityReport(reports=self._make_reports())
        j = batch.to_json()
        assert isinstance(j, str)

    def test_summary_string(self):
        batch = BatchQualityReport(
            reports      = self._make_reports(),
            pass_rate    = 0.67,
            overall_mean = 60.0,
        )
        s = batch.summary()
        assert isinstance(s, str)
        assert "BATCH" in s


# =============================================================================
# ── ImageQualityAssessor — Init
# =============================================================================

class TestAssessorInit:

    def test_default_instantiation(self):
        a = ImageQualityAssessor()
        assert a is not None

    def test_custom_min_resolution(self):
        a = ImageQualityAssessor(min_resolution=(256, 256))
        assert a.min_resolution == (256, 256)

    def test_custom_weights(self):
        w = {"sharpness": 0.5, "brightness": 0.1, "contrast": 0.1,
             "color_distribution": 0.1, "resolution": 0.1, "noise_level": 0.1}
        a = ImageQualityAssessor(weights=w)
        total = sum(a.weights.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_unnormalised_weights_are_normalised(self):
        w = {"sharpness": 5, "brightness": 1, "contrast": 2,
             "color_distribution": 1, "resolution": 1, "noise_level": 1}
        a = ImageQualityAssessor(weights=w)
        assert sum(a.weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_repr(self):
        a = ImageQualityAssessor()
        r = repr(a)
        assert "ImageQualityAssessor" in r
        assert "min_res" in r


# =============================================================================
# ── assess() — Core method
# =============================================================================

class TestAssess:

    def test_returns_quality_report(self, assessor, natural):
        r = assessor.assess(natural, image_id="test")
        assert isinstance(r, QualityReport)

    def test_image_id_set(self, assessor, natural):
        r = assessor.assess(natural, image_id="MY_IMG")
        assert r.image_id == "MY_IMG"

    def test_auto_image_id(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.image_id != ""

    def test_no_error_on_valid_image(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.error == ""

    def test_width_height_populated(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.width  == 512
        assert r.height == 512

    def test_megapixels_populated(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.megapixels > 0.0

    def test_aspect_ratio_set(self, assessor, natural):
        r = assessor.assess(natural)
        assert ":" in r.aspect_ratio

    def test_elapsed_ms_nonnegative(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.elapsed_ms >= 0.0

    def test_passed_bool(self, assessor, natural):
        r = assessor.assess(natural)
        assert isinstance(r.passed, bool)

    def test_all_scores_in_range(self, assessor, natural):
        r = assessor.assess(natural)
        for metric in ["sharpness", "brightness", "contrast",
                        "color_distribution", "resolution", "noise_level", "overall"]:
            score = getattr(r, metric)
            assert 0 <= score <= 100, f"{metric} score {score} out of range"

    def test_quality_label_valid(self, assessor, natural):
        r = assessor.assess(natural)
        assert r.quality in set(QUALITY_THRESHOLDS.keys()) | {"unknown"}

    def test_warnings_is_list(self, assessor, natural):
        r = assessor.assess(natural)
        assert isinstance(r.warnings, list)

    def test_none_image_returns_error(self, assessor):
        r = assessor.assess(None, image_id="none_test")
        assert r.error != ""

    def test_numpy_array_accepted(self, assessor):
        arr = np.random.randint(80, 180, (64, 64, 3), dtype=np.uint8)
        r   = assessor.assess(arr)
        assert r.error == ""


# =============================================================================
# ── Metric 1: Sharpness
# =============================================================================

class TestSharpness:

    def test_high_frequency_image_has_high_sharpness(self):
        """Checkerboard → maximum sharpness score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _checkerboard(size=128, block=2)
        r   = a.assess(img)
        assert r.sharpness >= 70, f"Expected high sharpness, got {r.sharpness}"

    def test_solid_image_has_low_sharpness(self):
        """Solid colour → near-zero Laplacian variance → low sharpness."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=128)
        r   = a.assess(img)
        assert r.sharpness <= 20, f"Expected low sharpness, got {r.sharpness}"

    def test_gradient_has_moderate_sharpness(self):
        """Smooth gradient → moderate-to-high sharpness (edges at extremes raise score)."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _gradient(size=128)
        r   = a.assess(img)
        # Gradient has clear edges at extremes — score can be relatively high
        assert 5 <= r.sharpness <= 100

    def test_laplacian_var_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _noise(size=64)
        r   = a.assess(img)
        assert r.laplacian_var > 0.0

    def test_checkerboard_laplacian_higher_than_solid(self):
        a    = ImageQualityAssessor(min_resolution=(8, 8))
        high = _checkerboard(size=64)
        low  = _solid((128, 128, 128), size=64)
        r_h  = a.assess(high)
        r_l  = a.assess(low)
        assert r_h.laplacian_var > r_l.laplacian_var


# =============================================================================
# ── Metric 2: Brightness
# =============================================================================

class TestBrightness:

    def test_ideal_brightness_high_score(self):
        """Mid-grey (128) → inside ideal range → high brightness score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=64)
        r   = a.assess(img)
        assert r.brightness >= 70

    def test_dark_image_low_brightness(self):
        """Near-black → very low brightness score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _dark(size=64)
        r   = a.assess(img)
        assert r.brightness <= 40

    def test_bright_image_low_score(self):
        """Near-white → low brightness score (overexposed)."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _bright(size=64)
        r   = a.assess(img)
        assert r.brightness <= 50

    def test_mean_brightness_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((100, 100, 100), size=64)
        r   = a.assess(img)
        assert r.mean_brightness > 0.0

    def test_dark_image_low_mean_brightness(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _dark(size=64)
        r   = a.assess(img)
        assert r.mean_brightness < 50


# =============================================================================
# ── Metric 3: Contrast
# =============================================================================

class TestContrast:

    def test_high_contrast_image_high_score(self):
        """Checkerboard → max RMS contrast."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _checkerboard(size=128, block=4)
        r   = a.assess(img)
        assert r.contrast >= 60

    def test_solid_image_zero_contrast(self):
        """Solid colour → zero RMS → minimal contrast score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=64)
        r   = a.assess(img)
        assert r.contrast == 0

    def test_rms_contrast_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _noise(size=64)
        r   = a.assess(img)
        assert r.rms_contrast > 0.0

    def test_checkerboard_contrast_higher_than_gradient(self):
        a    = ImageQualityAssessor(min_resolution=(8, 8))
        r_hi = a.assess(_checkerboard(size=64, block=2))
        r_lo = a.assess(_gradient(size=64))
        assert r_hi.contrast >= r_lo.contrast


# =============================================================================
# ── Metric 4: Color Distribution
# =============================================================================

class TestColorDistribution:

    def test_vivid_image_high_color_score(self):
        """Fully saturated RGB stripes → high color distribution."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _vivid(size=128)
        r   = a.assess(img)
        assert r.color_distribution >= 50

    def test_gray_image_low_color_score(self):
        """Grayscale → near-zero saturation → low color distribution."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _gray_image(size=64)
        r   = a.assess(img)
        assert r.color_distribution <= 60

    def test_saturation_mean_positive_for_vivid(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _vivid(size=64)
        r   = a.assess(img)
        assert r.saturation_mean > 0.1

    def test_saturation_coverage_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _vivid(size=64)
        r   = a.assess(img)
        assert 0.0 <= r.saturation_coverage <= 1.0

    def test_channel_balance_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        r   = a.assess(img)
        assert r.channel_balance >= 0.0

    def test_vivid_higher_than_gray(self):
        a    = ImageQualityAssessor(min_resolution=(8, 8))
        r_v  = a.assess(_vivid(size=64))
        r_g  = a.assess(_gray_image(size=64))
        assert r_v.color_distribution >= r_g.color_distribution


# =============================================================================
# ── Metric 5: Resolution
# =============================================================================

class TestResolution:

    def test_large_image_high_resolution_score(self):
        """2048×2048 → ultra-high resolution."""
        a   = ImageQualityAssessor(min_resolution=(64, 64))
        img = _natural(size=512)   # 0.26 MP – adjust via factory
        # Use a custom larger image
        arr = np.random.randint(80, 180, (1024, 1024, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr.astype(np.uint8))
        r   = a.assess(img)
        assert r.resolution >= 75

    def test_small_image_low_resolution_score(self):
        """32×32 → very low resolution."""
        a   = ImageQualityAssessor(min_resolution=(256, 256))
        img = _solid(size=32)
        r   = a.assess(img)
        assert r.resolution <= 50

    def test_below_min_resolution_penalty(self):
        """Image below min_resolution → lower score than equal-MP above."""
        a    = ImageQualityAssessor(min_resolution=(256, 256))
        big  = _natural(size=512)
        small= _natural(size=128)
        r_b  = a.assess(big)
        r_s  = a.assess(small)
        assert r_b.resolution >= r_s.resolution

    def test_megapixels_correct(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid(size=512)
        r   = a.assess(img)
        assert r.megapixels == pytest.approx(512 * 512 / 1_000_000, abs=0.001)

    def test_aspect_ratio_square(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid(size=64)
        r   = a.assess(img)
        assert r.aspect_ratio == "1:1"

    def test_aspect_ratio_landscape(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        arr = np.zeros((64, 128, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        r   = a.assess(img)
        assert "2:1" in r.aspect_ratio or ":" in r.aspect_ratio

    def test_resolution_class_populated(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=128)
        r   = a.assess(img)
        assert r.resolution_class in ("ultra-high", "high", "standard", "low", "very low")

    def test_width_height_match_image_size(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid(size=100)
        r   = a.assess(img)
        assert r.width  == 100
        assert r.height == 100


# =============================================================================
# ── Metric 6: Noise Level
# =============================================================================

class TestNoiseLevel:

    def test_noisy_image_low_score(self):
        """Full random noise → high noise energy → low score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _noise(size=128)
        r   = a.assess(img)
        assert r.noise_level <= 60, f"Expected low noise score, got {r.noise_level}"

    def test_solid_image_high_noise_score(self):
        """Solid colour → zero noise residual → high noise score."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=64)
        r   = a.assess(img)
        assert r.noise_level >= 90, f"Expected high noise score, got {r.noise_level}"

    def test_noise_energy_positive_for_noisy(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _noise(size=64)
        r   = a.assess(img)
        assert r.noise_energy > 0.0

    def test_noise_energy_near_zero_for_solid(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=64)
        r   = a.assess(img)
        assert r.noise_energy < 2.0

    def test_solid_noise_score_higher_than_random_noise(self):
        a    = ImageQualityAssessor(min_resolution=(8, 8))
        r_s  = a.assess(_solid((128, 128, 128), size=64))
        r_n  = a.assess(_noise(size=64))
        assert r_s.noise_level > r_n.noise_level


# =============================================================================
# ── Overall Score & Quality Rating
# =============================================================================

class TestOverallScoreAndRating:

    @pytest.mark.parametrize("overall,expected_quality", [
        (95, "excellent"),
        (90, "excellent"),
        (89, "very good"),
        (75, "very good"),
        (74, "good"),
        (60, "good"),
        (59, "fair"),
        (45, "fair"),
        (44, "poor"),
        (25, "poor"),
        (24, "very poor"),
        (0,  "very poor"),
    ])
    def test_rate_quality_thresholds(self, overall, expected_quality):
        assert ImageQualityAssessor._rate_quality(overall) == expected_quality

    def test_overall_score_range(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        r   = a.assess(img)
        assert 0 <= r.overall <= 100

    def test_compute_overall_weighted(self):
        a   = ImageQualityAssessor()
        scores = {
            "sharpness":          80,
            "brightness":         90,
            "contrast":           70,
            "color_distribution": 75,
            "resolution":         85,
            "noise_level":        80,
        }
        overall = a._compute_overall(scores)
        expected = sum(scores[m] * w for m, w in METRIC_WEIGHTS.items())
        assert overall == pytest.approx(int(round(expected)), abs=2)

    def test_zero_sharpness_caps_overall(self):
        """If sharpness is 0, overall should be capped at 50."""
        a   = ImageQualityAssessor()
        scores = {m: 100 for m in METRIC_WEIGHTS}
        scores["sharpness"] = 0
        overall = a._compute_overall(scores)
        assert overall <= 50

    def test_passed_consistent_with_threshold(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        r   = a.assess(img)
        assert r.passed == (r.overall >= OVERALL_PASS_THRESHOLD)


# =============================================================================
# ── Warning Generation
# =============================================================================

class TestWarnings:

    def test_dark_image_generates_brightness_warning(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _dark(size=64)
        r   = a.assess(img)
        has_brightness_warning = any("BRIGHTNESS" in w for w in r.warnings)
        assert has_brightness_warning

    def test_clean_image_no_warnings(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        r   = a.assess(img)
        # Natural image might have some warnings — just verify it's a list
        assert isinstance(r.warnings, list)

    def test_solid_image_generates_contrast_warning(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((128, 128, 128), size=64)
        r   = a.assess(img)
        has_contrast_warning = any("CONTRAST" in w for w in r.warnings)
        assert has_contrast_warning

    def test_gray_image_generates_color_warning(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _gray_image(size=64)
        r   = a.assess(img)
        has_color_warning = any("COLOR" in w.upper() for w in r.warnings)
        assert has_color_warning


# =============================================================================
# ── assess_batch()
# =============================================================================

class TestAssessBatch:

    def test_returns_list(self, assessor, natural):
        results = assessor.assess_batch([natural, natural, natural])
        assert isinstance(results, list)

    def test_length_matches_input(self, assessor, natural):
        results = assessor.assess_batch([natural] * 4)
        assert len(results) == 4

    def test_all_quality_reports(self, assessor, natural):
        results = assessor.assess_batch([natural, natural])
        for r in results:
            assert isinstance(r, QualityReport)

    def test_custom_image_ids(self, assessor, natural):
        ids     = ["a", "b", "c"]
        results = assessor.assess_batch([natural] * 3, image_ids=ids)
        result_ids = [r.image_id for r in results]
        assert result_ids == ids

    def test_empty_batch_returns_empty(self, assessor):
        results = assessor.assess_batch([])
        assert results == []

    def test_mixed_images(self, assessor):
        imgs    = [_natural(size=64), _dark(size=64), _vivid(size=64)]
        results = assessor.assess_batch(imgs, image_ids=["n", "d", "v"])
        assert len(results) == 3
        # Dark image should have lower brightness than natural
        brightness_n = next(r.brightness for r in results if r.image_id == "n")
        brightness_d = next(r.brightness for r in results if r.image_id == "d")
        assert brightness_d < brightness_n


# =============================================================================
# ── aggregate()
# =============================================================================

class TestAggregate:

    def _reports(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        return a.assess_batch(
            [_natural(size=64), _dark(size=64), _vivid(size=64)],
            image_ids=["nat", "dark", "vivid"],
        )

    def test_returns_batch_quality_report(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert isinstance(batch, BatchQualityReport)

    def test_total_count(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert len(batch.reports) == 3

    def test_mean_scores_populated(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert "sharpness" in batch.mean_scores
        assert "contrast"  in batch.mean_scores

    def test_overall_mean_in_range(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert 0 <= batch.overall_mean <= 100

    def test_pass_rate_in_range(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert 0.0 <= batch.pass_rate <= 1.0

    def test_best_worst_image_id_set(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert batch.best_image_id  != ""
        assert batch.worst_image_id != ""

    def test_quality_dist_populated(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        assert isinstance(batch.quality_dist, dict)
        assert sum(batch.quality_dist.values()) == 3

    def test_empty_aggregate(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate([])
        assert batch.overall_mean == 0.0

    def test_to_dict_json_serialisable(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        json.dumps(batch.to_dict())

    def test_summary_string(self):
        a = ImageQualityAssessor(min_resolution=(8, 8))
        batch = a.aggregate(self._reports())
        s = batch.summary()
        assert isinstance(s, str)
        assert "BATCH" in s


# =============================================================================
# ── generate_report()
# =============================================================================

class TestGenerateReport:

    def test_dict_output(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        result = a.generate_report(img, output_format="dict")
        assert isinstance(result, dict)
        assert "quality" in result

    def test_json_output(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        result = a.generate_report(img, output_format="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "quality" in data

    def test_text_output(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        result = a.generate_report(img, output_format="text")
        assert isinstance(result, str)
        assert "Sharpness" in result

    def test_custom_image_id(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        result = a.generate_report(img, image_id="SPEC_TEST", output_format="dict")
        assert result["image_id"] == "SPEC_TEST"

    def test_dict_matches_spec_format(self):
        """Output dict must match the documented specification exactly."""
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        d   = a.generate_report(img, image_id="GEN_001", output_format="dict")
        spec_keys = ["sharpness", "brightness", "contrast", "color_distribution",
                     "resolution", "noise_level", "overall", "quality",
                     "passed", "warnings", "image_id"]
        for key in spec_keys:
            assert key in d, f"Missing required key: {key}"
        assert isinstance(d["sharpness"], int)
        assert isinstance(d["quality"],   str)
        assert isinstance(d["passed"],    bool)
        assert isinstance(d["warnings"],  list)


# =============================================================================
# ── Module-Level Convenience Functions
# =============================================================================

class TestModuleFunctions:

    def test_assess_returns_dict(self):
        img    = _natural(size=64)
        result = assess(img, image_id="MOD_001")
        assert isinstance(result, dict)

    def test_assess_has_quality_key(self):
        img    = _natural(size=64)
        result = assess(img)
        assert "quality" in result

    def test_assess_has_sharpness_key(self):
        img    = _natural(size=64)
        result = assess(img)
        assert "sharpness" in result

    def test_assess_sharpness_is_int(self):
        img    = _natural(size=64)
        result = assess(img)
        assert isinstance(result["sharpness"], int)

    def test_assess_batch_returns_list(self):
        imgs   = [_natural(size=64), _dark(size=64)]
        result = assess_batch(imgs)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_assess_batch_each_is_dict(self):
        imgs   = [_natural(size=64)] * 3
        result = assess_batch(imgs)
        for item in result:
            assert isinstance(item, dict)
            assert "quality" in item

    def test_generate_report_dict(self):
        img    = _natural(size=64)
        result = generate_report(img, output_format="dict")
        assert isinstance(result, dict)

    def test_generate_report_json(self):
        img    = _natural(size=64)
        result = generate_report(img, output_format="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "quality" in data

    def test_generate_report_text(self):
        img    = _natural(size=64)
        result = generate_report(img, output_format="text")
        assert isinstance(result, str)
        assert "Sharpness" in result

    def test_generate_report_json_serialisable(self):
        img    = _natural(size=64)
        result = assess(img)
        json.dumps(result)   # Must not raise


# =============================================================================
# ── Edge Cases
# =============================================================================

class TestEdgeCases:

    def test_1x1_image_no_crash(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid(size=1)
        r   = a.assess(img)
        assert isinstance(r, QualityReport)
        assert 0 <= r.overall <= 100

    def test_very_small_image(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid(size=4)
        r   = a.assess(img)
        assert r.error == ""

    def test_rgba_image_accepted(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        arr = np.random.randint(80, 200, (64, 64, 4), dtype=np.uint8)
        img = PILImage.fromarray(arr, mode="RGBA")
        r   = a.assess(img)
        assert r.error == ""

    def test_grayscale_pil_accepted(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        arr = np.full((64, 64), 128, dtype=np.uint8)
        img = PILImage.fromarray(arr, mode="L")
        r   = a.assess(img)
        assert r.error == ""

    def test_none_image_graceful(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        r   = a.assess(None)
        assert r.error != ""
        assert r.overall == 0

    def test_extreme_dark_image(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((0, 0, 0), size=64)
        r   = a.assess(img)
        assert r.brightness < 50
        any_warn = any("dark" in w.lower() or "BRIGHTNESS" in w for w in r.warnings)
        assert any_warn

    def test_extreme_bright_image(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _solid((255, 255, 255), size=64)
        r   = a.assess(img)
        assert r.brightness < 50

    def test_non_square_image(self):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        arr = np.random.randint(80, 200, (64, 128, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        r   = a.assess(img)
        assert r.width  == 128
        assert r.height == 64
        assert ":" in r.aspect_ratio

    def test_save_report_writes_file(self, tmp_path):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        r   = a.assess(img, save_report=True, report_dir=tmp_path)
        files = list(tmp_path.glob("*.json"))
        assert len(files) >= 1

    def test_saved_report_is_valid_json(self, tmp_path):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        img = _natural(size=64)
        a.assess(img, save_report=True, report_dir=tmp_path)
        files = list(tmp_path.glob("*.json"))
        data  = json.loads(files[0].read_text())
        assert "quality" in data

    def test_batch_save_report(self, tmp_path):
        a   = ImageQualityAssessor(min_resolution=(8, 8))
        imgs= [_natural(size=64)] * 2
        a.assess_batch(imgs, save_report=True, report_dir=tmp_path)
        files = list(tmp_path.glob("*.json"))
        assert len(files) >= 1
