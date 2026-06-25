"""
week2/evaluation/image_quality.py
=====================================
Production-Grade Image Quality Assessment System
AI-Powered Fashion Design Assistant — Week 2

Measures 6 perceptual quality dimensions for fashion-generated images:

  1. Sharpness          — Laplacian variance (edge crispness)
  2. Brightness         — Perceptual luminance from YCbCr luma channel
  3. Contrast           — RMS contrast across the image
  4. Color Distribution — Saturation coverage, histogram spread, channel balance
  5. Resolution         — Megapixels, aspect ratio, resolution class
  6. Noise Level        — High-frequency noise estimate via residual analysis

Output schema (matches specification)::

    {
        "sharpness":          90,
        "brightness":         75,
        "contrast":           85,
        "color_distribution": 80,
        "resolution":         95,
        "noise_level":        88,
        "overall":            86,
        "quality":            "excellent",
        "passed":             true,
        "warnings":           [],
        "image_id":           "GEN_001"
    }

All metrics are normalised to [0, 100] integer scores.
All computation is pure NumPy + Pillow — no GPU or ML model required.
Gracefully degrades on any unavailable dependency.

Public API
----------
    from src.evaluation.week2_image_quality import ImageQualityAssessor

    assessor = ImageQualityAssessor()
    report   = assessor.assess(pil_image, image_id="GEN_001")

    print(report.sharpness)          # 90
    print(report.quality)            # "excellent"
    print(report.to_dict())          # Full JSON-serialisable dict

    # Batch
    reports  = assessor.assess_batch(images, image_ids=[...])
    aggregate= assessor.aggregate(reports)

Module-level shortcuts::

    from src.evaluation.week2_image_quality import assess, assess_batch, generate_report
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging as _log
    logger = _log.getLogger("image_quality")   # type: ignore[assignment]

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None       # type: ignore[assignment]
    _NUMPY = False

try:
    from PIL import Image as PILImage, ImageFilter, ImageStat
    _PIL = True
except ImportError:
    PILImage = None     # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageStat = None    # type: ignore[assignment]
    _PIL = False

try:
    import scipy.ndimage as _ndi
    _SCIPY = True
except ImportError:
    _ndi = None     # type: ignore[assignment]
    _SCIPY = False


# =============================================================================
# ── Constants
# =============================================================================

# Quality rating thresholds (overall score 0-100)
QUALITY_THRESHOLDS: Dict[str, Tuple[int, int]] = {
    "excellent": (90, 101),
    "very good": (75,  90),
    "good":      (60,  75),
    "fair":      (45,  60),
    "poor":      (25,  45),
    "very poor": ( 0,  25),
}

# Per-metric weights for overall score computation
METRIC_WEIGHTS: Dict[str, float] = {
    "sharpness":          0.25,
    "brightness":         0.10,
    "contrast":           0.20,
    "color_distribution": 0.15,
    "resolution":         0.15,
    "noise_level":        0.15,
}

# Sharpness: Laplacian variance — calibrated for 512×512 fashion images
SHARPNESS_MAX_VAR  = 3000.0   # Variance → score=100
SHARPNESS_MIN_VAR  = 10.0     # Variance → score=0

# Brightness: ideal luminance range for fashion photography
BRIGHTNESS_IDEAL_MIN = 80     # Y value (0–255)
BRIGHTNESS_IDEAL_MAX = 200

# Contrast: RMS contrast → score calibration
CONTRAST_MAX_RMS = 80.0

# Noise: high-frequency residual energy thresholds
NOISE_MIN_ENERGY = 0.5        # Below → very clean (score≈100)
NOISE_MAX_ENERGY = 40.0       # Above → very noisy (score≈0)

# Resolution quality classes (megapixels)
RESOLUTION_CLASSES: List[Tuple[float, str, int]] = [
    (4.0,  "ultra-high", 100),
    (2.0,  "high",        90),
    (1.0,  "standard",    75),
    (0.25, "low",         50),
    (0.0,  "very low",    25),
]

# Minimum passing score per metric
MIN_PASSING_SCORE: Dict[str, int] = {
    "sharpness":          30,
    "brightness":         20,
    "contrast":           25,
    "color_distribution": 20,
    "resolution":         40,
    "noise_level":        25,
}

# Minimum overall score to pass
OVERALL_PASS_THRESHOLD = 45


# =============================================================================
# ── Quality Report Dataclass
# =============================================================================

@dataclass
class QualityReport:
    """
    Complete image quality assessment report.

    All per-metric scores are integers in [0, 100].
    Higher = better for all metrics EXCEPT noise_level
    (higher noise_level score means less noise detected).

    Attributes
    ----------
    image_id           : str
    sharpness          : int    Laplacian variance → edge crispness.
    brightness         : int    Perceptual luminance comfort score.
    contrast           : int    RMS contrast score.
    color_distribution : int    Saturation + histogram spread score.
    resolution         : int    Megapixel + aspect ratio score.
    noise_level        : int    Inverse noise energy score (100 = no noise).
    overall            : int    Weighted composite score.
    quality            : str    "excellent" | "very good" | "good" | "fair" |
                                "poor" | "very poor"
    passed             : bool   overall >= OVERALL_PASS_THRESHOLD.
    warnings           : list   Per-metric warning messages.
    error              : str    Non-empty if assessment failed.
    elapsed_ms         : float
    width              : int    Image width (px).
    height             : int    Image height (px).
    megapixels         : float
    aspect_ratio       : str    e.g. "1:1", "4:3", "16:9"
    mean_brightness    : float  Raw mean luminance (0–255).
    rms_contrast       : float  Raw RMS contrast value.
    laplacian_var      : float  Raw Laplacian variance.
    noise_energy       : float  Raw high-frequency energy estimate.
    saturation_mean    : float  Mean HSV saturation (0–1).
    saturation_coverage: float  Fraction of pixels with saturation > 0.1.
    channel_balance    : float  RGB channel standard deviation balance score.
    resolution_class   : str    "ultra-high" | "high" | "standard" | "low" | "very low"
    """
    image_id:            str            = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # ── Per-metric scores [0, 100] ──────────────────────────────────────────
    sharpness:           int            = 0
    brightness:          int            = 0
    contrast:            int            = 0
    color_distribution:  int            = 0
    resolution:          int            = 0
    noise_level:         int            = 0
    # ── Overall ────────────────────────────────────────────────────────────
    overall:             int            = 0
    quality:             str            = "unknown"
    passed:              bool           = False
    warnings:            List[str]      = field(default_factory=list)
    error:               str            = ""
    elapsed_ms:          float          = 0.0
    # ── Raw measurements ───────────────────────────────────────────────────
    width:               int            = 0
    height:              int            = 0
    megapixels:          float          = 0.0
    aspect_ratio:        str            = "unknown"
    mean_brightness:     float          = 0.0
    rms_contrast:        float          = 0.0
    laplacian_var:       float          = 0.0
    noise_energy:        float          = 0.0
    saturation_mean:     float          = 0.0
    saturation_coverage: float          = 0.0
    channel_balance:     float          = 0.0
    resolution_class:    str            = "unknown"

    # =========================================================================
    # ── Serialisation
    # =========================================================================

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        """
        Serialise to a JSON-compatible dict.

        Matches the output specification::

            {
                "sharpness":          90,
                "brightness":         75,
                "contrast":           85,
                "color_distribution": 80,
                "resolution":         95,
                "noise_level":        88,
                "overall":            86,
                "quality":            "excellent",
                "passed":             true,
                "warnings":           [],
                "image_id":           "GEN_001"
            }

        Parameters
        ----------
        include_raw : bool
            Include raw measurement values (laplacian_var, rms_contrast etc.)
            in addition to the normalised scores.
        """
        d: Dict[str, Any] = {
            # ── Core output (matches spec) ─────────────────────────────────
            "sharpness":           self.sharpness,
            "brightness":          self.brightness,
            "contrast":            self.contrast,
            "color_distribution":  self.color_distribution,
            "resolution":          self.resolution,
            "noise_level":         self.noise_level,
            "overall":             self.overall,
            "quality":             self.quality,
            "passed":              self.passed,
            "warnings":            list(self.warnings),
            "image_id":            self.image_id,
            "error":               self.error,
            "elapsed_ms":          round(self.elapsed_ms, 2),
            # ── Image metadata ─────────────────────────────────────────────
            "width":               self.width,
            "height":              self.height,
            "megapixels":          round(self.megapixels, 3),
            "aspect_ratio":        self.aspect_ratio,
            "resolution_class":    self.resolution_class,
        }
        if include_raw:
            d.update({
                "raw": {
                    "laplacian_var":       round(self.laplacian_var, 2),
                    "mean_brightness":     round(self.mean_brightness, 2),
                    "rms_contrast":        round(self.rms_contrast, 2),
                    "noise_energy":        round(self.noise_energy, 4),
                    "saturation_mean":     round(self.saturation_mean, 4),
                    "saturation_coverage": round(self.saturation_coverage, 4),
                    "channel_balance":     round(self.channel_balance, 4),
                }
            })
        return d

    def to_json(self, indent: int = 2, include_raw: bool = True) -> str:
        """Return a formatted JSON string."""
        return json.dumps(self.to_dict(include_raw=include_raw), indent=indent)

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"QualityReport [{status}] | {self.image_id} | "
            f"overall={self.overall} | quality={self.quality!r} | "
            f"sharpness={self.sharpness} | contrast={self.contrast} | "
            f"{self.width}x{self.height}"
        )

    def detailed_report(self) -> str:
        """Multi-line formatted quality report."""
        sep   = "=" * 62
        lines = [
            sep,
            f"  IMAGE QUALITY REPORT  [{self.image_id}]",
            sep,
            f"  Resolution   : {self.width}x{self.height} ({self.megapixels:.2f} MP) — {self.resolution_class}",
            f"  Aspect Ratio : {self.aspect_ratio}",
            "",
            "  QUALITY SCORES  (0-100, higher = better)",
            f"  {'Metric':<22} {'Score':>6}  {'Bar':<30}",
            "  " + "-" * 58,
        ]
        metrics = [
            ("Sharpness",          self.sharpness),
            ("Brightness",         self.brightness),
            ("Contrast",           self.contrast),
            ("Color Distribution", self.color_distribution),
            ("Resolution",         self.resolution),
            ("Noise Level",        self.noise_level),
        ]
        for name, score in metrics:
            bar_len = score // 5   # max 20 chars
            bar     = "#" * bar_len + "-" * (20 - bar_len)
            lines.append(f"  {name:<22} {score:>5}   [{bar}]")

        lines += [
            "  " + "-" * 58,
            f"  {'OVERALL':<22} {self.overall:>5}   quality={self.quality!r}",
            f"  {'RESULT':<22} {'PASSED' if self.passed else 'FAILED':>5}",
            "",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    [!] {w}")
            lines.append("")
        if self.error:
            lines.append(f"  Error: {self.error}")
        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# ── Aggregate Report
# =============================================================================

@dataclass
class BatchQualityReport:
    """
    Aggregate quality statistics across a batch of images.

    Attributes
    ----------
    reports       : list of QualityReport
    mean_scores   : dict  Per-metric mean scores.
    overall_mean  : float
    pass_rate     : float
    quality_dist  : dict  Distribution of quality labels.
    best_image_id : str
    worst_image_id: str
    generated_at  : str
    """
    reports:        List[QualityReport]     = field(default_factory=list)
    mean_scores:    Dict[str, float]        = field(default_factory=dict)
    overall_mean:   float                   = 0.0
    pass_rate:      float                   = 0.0
    quality_dist:   Dict[str, int]          = field(default_factory=dict)
    best_image_id:  str                     = ""
    worst_image_id: str                     = ""
    generated_at:   str                     = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elapsed_ms:     float                   = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at":  self.generated_at,
            "total":         len(self.reports),
            "passed":        sum(1 for r in self.reports if r.passed),
            "failed":        sum(1 for r in self.reports if not r.passed),
            "pass_rate":     round(self.pass_rate, 4),
            "overall_mean":  round(self.overall_mean, 2),
            "mean_scores":   {k: round(v, 2) for k, v in self.mean_scores.items()},
            "quality_dist":  self.quality_dist,
            "best_image_id": self.best_image_id,
            "worst_image_id":self.worst_image_id,
            "elapsed_ms":    round(self.elapsed_ms, 2),
            "per_image":     [r.to_dict(include_raw=False) for r in self.reports],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        sep   = "=" * 62
        lines = [
            sep,
            "  BATCH IMAGE QUALITY REPORT",
            sep,
            f"  Total images  : {len(self.reports)}",
            f"  Passed        : {sum(1 for r in self.reports if r.passed)}",
            f"  Failed        : {sum(1 for r in self.reports if not r.passed)}",
            f"  Pass rate     : {self.pass_rate:.1%}",
            f"  Overall mean  : {self.overall_mean:.1f}",
            "",
            "  Mean Scores:",
        ]
        for metric, score in self.mean_scores.items():
            lines.append(f"    {metric:<22} {score:>6.1f}")
        lines += [
            "",
            "  Quality Distribution:",
        ]
        for label, count in self.quality_dist.items():
            lines.append(f"    {label:<12} {count:>4}")
        lines += [
            f"  Best  image   : {self.best_image_id}",
            f"  Worst image   : {self.worst_image_id}",
            sep,
        ]
        return "\n".join(lines)


# =============================================================================
# ── ImageQualityAssessor
# =============================================================================

class ImageQualityAssessor:
    """
    Production-grade image quality assessment for fashion-generated images.

    Measures 6 perceptual dimensions using pure NumPy + Pillow (no GPU needed).
    All scores are normalised to [0, 100] integers.

    Parameters
    ----------
    min_resolution : tuple of int
        Minimum acceptable (width, height) in pixels.
    sharpness_max_var : float
        Laplacian variance mapped to sharpness score 100.
    brightness_ideal : tuple of int
        (min, max) ideal luminance range for fashion photography.
    contrast_max_rms : float
        RMS contrast mapped to score 100.
    noise_max_energy : float
        Noise energy mapped to score 0 (noisiest).
    weights : dict, optional
        Custom per-metric weights for overall score.
        Must sum to 1.0.

    Example
    -------
        from src.evaluation.week2_image_quality import ImageQualityAssessor

        assessor = ImageQualityAssessor()
        report   = assessor.assess(pil_image, image_id="SDXL_001")

        print(report.sharpness)      # 90
        print(report.quality)        # "excellent"
        print(report.to_dict())      # Full output dict
        print(report.detailed_report())
    """

    def __init__(
        self,
        min_resolution:    Tuple[int, int]         = (512, 512),
        sharpness_max_var: float                    = SHARPNESS_MAX_VAR,
        brightness_ideal:  Tuple[int, int]          = (BRIGHTNESS_IDEAL_MIN, BRIGHTNESS_IDEAL_MAX),
        contrast_max_rms:  float                    = CONTRAST_MAX_RMS,
        noise_max_energy:  float                    = NOISE_MAX_ENERGY,
        weights:           Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_resolution    = min_resolution
        self.sharpness_max_var = sharpness_max_var
        self.brightness_ideal  = brightness_ideal
        self.contrast_max_rms  = contrast_max_rms
        self.noise_max_energy  = noise_max_energy
        self.weights           = weights or METRIC_WEIGHTS.copy()

        # Normalise weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(
                "Metric weights sum to {:.3f}, not 1.0 — normalising.", total
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(
            "ImageQualityAssessor initialised | min_res={}x{} | weights={}",
            min_resolution[0], min_resolution[1], self.weights,
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def assess(
        self,
        image:     Any,
        image_id:  Optional[str] = None,
        *,
        save_report: bool         = False,
        report_dir:  Optional[Path] = None,
    ) -> QualityReport:
        """
        Assess a single PIL image across all 6 quality dimensions.

        Parameters
        ----------
        image    : PIL.Image.Image
        image_id : str, optional   Identifier (auto-generated if None).
        save_report : bool         Write JSON report to disk.
        report_dir  : Path, optional   Output directory.

        Returns
        -------
        QualityReport

        Output dict matches the specification::

            {
                "sharpness":          90,
                "brightness":         75,
                "contrast":           85,
                "color_distribution": 80,
                "resolution":         95,
                "noise_level":        88,
                "overall":            86,
                "quality":            "excellent",
                "passed":             true,
                "warnings":           [],
                "image_id":           "GEN_001"
            }

        Example
        -------
            report = assessor.assess(pil_image)
            print(report.quality)        # "excellent"
            print(report.sharpness)      # 90
        """
        t0     = time.perf_counter()
        img_id = image_id or str(uuid.uuid4())[:8]
        report = QualityReport(image_id=img_id)

        if not _PIL or not _NUMPY:
            report.error = "Pillow and NumPy are required for quality assessment."
            logger.error("assess: missing dependencies — {}", report.error)
            return report

        if image is None:
            report.error = "image is None"
            return report

        try:
            # Ensure PIL image
            if not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(image)

            rgb = image.convert("RGB")
            arr = np.array(rgb, dtype=np.float32)   # [H, W, 3] float32

            # ── 1. Resolution ──────────────────────────────────────────────
            res_score, res_meta = self._measure_resolution(rgb)
            report.resolution       = res_score
            report.width            = res_meta["width"]
            report.height           = res_meta["height"]
            report.megapixels       = res_meta["megapixels"]
            report.aspect_ratio     = res_meta["aspect_ratio"]
            report.resolution_class = res_meta["resolution_class"]

            # ── 2. Sharpness ───────────────────────────────────────────────
            sharp_score, lap_var = self._measure_sharpness(arr)
            report.sharpness    = sharp_score
            report.laplacian_var= lap_var

            # ── 3. Brightness ──────────────────────────────────────────────
            bright_score, mean_lum = self._measure_brightness(arr)
            report.brightness      = bright_score
            report.mean_brightness = mean_lum

            # ── 4. Contrast ────────────────────────────────────────────────
            contrast_score, rms = self._measure_contrast(arr)
            report.contrast     = contrast_score
            report.rms_contrast = rms

            # ── 5. Color Distribution ──────────────────────────────────────
            color_score, color_meta = self._measure_color_distribution(arr)
            report.color_distribution  = color_score
            report.saturation_mean     = color_meta["saturation_mean"]
            report.saturation_coverage = color_meta["saturation_coverage"]
            report.channel_balance     = color_meta["channel_balance"]

            # ── 6. Noise Level ─────────────────────────────────────────────
            noise_score, noise_energy = self._measure_noise(arr)
            report.noise_level  = noise_score
            report.noise_energy = noise_energy

            # ── Overall composite score ────────────────────────────────────
            report.overall = self._compute_overall({
                "sharpness":          report.sharpness,
                "brightness":         report.brightness,
                "contrast":           report.contrast,
                "color_distribution": report.color_distribution,
                "resolution":         report.resolution,
                "noise_level":        report.noise_level,
            })
            report.quality = self._rate_quality(report.overall)
            report.passed  = report.overall >= OVERALL_PASS_THRESHOLD

            # ── Per-metric warnings ────────────────────────────────────────
            report.warnings = self._generate_warnings(report)

        except Exception as exc:
            logger.error("assess failed for {}: {}", img_id, exc)
            report.error = str(exc)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "QualityReport | {} | overall={} | quality={!r} | {:.1f}ms",
            img_id, report.overall, report.quality, report.elapsed_ms,
        )

        if save_report:
            self._save_report(report, report_dir)

        return report

    def assess_batch(
        self,
        images:     Sequence[Any],
        image_ids:  Optional[List[str]] = None,
        *,
        save_report: bool               = False,
        report_dir:  Optional[Path]     = None,
    ) -> List[QualityReport]:
        """
        Assess a batch of images.

        Parameters
        ----------
        images    : sequence of PIL.Image.Image
        image_ids : list of str, optional
        save_report : bool
        report_dir  : Path, optional

        Returns
        -------
        list of QualityReport

        Example
        -------
            reports = assessor.assess_batch(pil_images)
            for r in reports:
                print(r.summary())
        """
        n   = len(images)
        ids = image_ids or [str(uuid.uuid4())[:8] for _ in range(n)]

        logger.info("assess_batch | n={}", n)
        results = []
        for img, img_id in zip(images, ids):
            results.append(self.assess(img, image_id=img_id))

        passed = sum(1 for r in results if r.passed)
        logger.info(
            "assess_batch complete | n={} | passed={} | mean_overall={:.1f}",
            n, passed,
            sum(r.overall for r in results) / n if n else 0.0,
        )

        if save_report:
            batch = self.aggregate(results)
            self._save_batch_report(batch, report_dir)

        return results

    def aggregate(self, reports: List[QualityReport]) -> BatchQualityReport:
        """
        Aggregate a list of QualityReports into batch statistics.

        Parameters
        ----------
        reports : list of QualityReport

        Returns
        -------
        BatchQualityReport

        Example
        -------
            batch = assessor.aggregate(reports)
            print(batch.summary())
            print(batch.pass_rate)    # 0.85
        """
        if not reports:
            return BatchQualityReport()

        n = len(reports)
        metrics = ["sharpness", "brightness", "contrast",
                   "color_distribution", "resolution", "noise_level"]

        mean_scores = {
            m: sum(getattr(r, m) for r in reports) / n
            for m in metrics
        }
        overall_mean = sum(r.overall for r in reports) / n
        pass_rate    = sum(1 for r in reports if r.passed) / n

        quality_dist: Dict[str, int] = {}
        for r in reports:
            quality_dist[r.quality] = quality_dist.get(r.quality, 0) + 1

        sorted_by_overall = sorted(reports, key=lambda r: r.overall, reverse=True)
        best_id  = sorted_by_overall[0].image_id  if reports else ""
        worst_id = sorted_by_overall[-1].image_id if reports else ""

        return BatchQualityReport(
            reports        = reports,
            mean_scores    = mean_scores,
            overall_mean   = overall_mean,
            pass_rate      = pass_rate,
            quality_dist   = quality_dist,
            best_image_id  = best_id,
            worst_image_id = worst_id,
        )

    def generate_report(
        self,
        image:        Any,
        image_id:     Optional[str] = None,
        *,
        output_format: str          = "dict",   # "dict" | "json" | "text"
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a quality report in the requested output format.

        Parameters
        ----------
        image         : PIL.Image.Image
        image_id      : str, optional
        output_format : str
            ``"dict"``  — returns ``dict`` (JSON-serialisable).
            ``"json"``  — returns formatted JSON ``str``.
            ``"text"``  — returns a human-readable multi-line report ``str``.

        Returns
        -------
        dict or str

        Example
        -------
            report = assessor.generate_report(pil_image, output_format="dict")
            # {"sharpness": 90, "contrast": 85, "quality": "excellent", ...}

            text = assessor.generate_report(pil_image, output_format="text")
            print(text)
        """
        qr = self.assess(image, image_id=image_id)
        if output_format == "json":
            return qr.to_json()
        if output_format == "text":
            return qr.detailed_report()
        return qr.to_dict()

    # =========================================================================
    # ── Metric 1: Sharpness
    # =========================================================================

    def _measure_sharpness(self, arr: Any) -> Tuple[int, float]:
        """
        Measure image sharpness via Laplacian variance.

        The Laplacian operator highlights rapid intensity changes (edges).
        Its variance indicates the presence of fine detail and focus quality.

        Method
        ------
        1. Convert to grayscale (luminance).
        2. Apply the discrete Laplacian kernel:
               [[0,  1, 0],
                [1, -4, 1],
                [0,  1, 0]]
        3. Compute variance of the response.
        4. Map variance → score via logarithmic scaling.

        Score Interpretation
        --------------------
        score 0–30   : Blurry, out-of-focus, or over-smoothed
        score 30–60  : Moderate sharpness
        score 60–85  : Good focus and detail
        score 85–100 : Excellent sharpness (fine textures preserved)

        Returns
        -------
        (score: int, laplacian_variance: float)
        """
        # Luminance (BT.601 coefficients)
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2])

        # Laplacian kernel via convolution
        if _SCIPY:
            kernel = np.array([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]], dtype=np.float32)
            lap = _ndi.convolve(gray, kernel, mode="reflect")
        else:
            # Manual: use NumPy diff approximation
            lap = (
                np.roll(gray,  1, axis=0) + np.roll(gray, -1, axis=0) +
                np.roll(gray,  1, axis=1) + np.roll(gray, -1, axis=1) -
                4 * gray
            )

        variance = float(np.var(lap))

        # Log-scale mapping: sharp images have very high variance
        if variance <= SHARPNESS_MIN_VAR:
            score = 0
        elif variance >= self.sharpness_max_var:
            score = 100
        else:
            # Log interpolation for smoother curve
            log_var  = math.log(variance - SHARPNESS_MIN_VAR + 1.0)
            log_max  = math.log(self.sharpness_max_var - SHARPNESS_MIN_VAR + 1.0)
            score    = int(min(100, round((log_var / log_max) * 100)))

        logger.debug("Sharpness | var={:.2f} | score={}", variance, score)
        return score, round(variance, 2)

    # =========================================================================
    # ── Metric 2: Brightness
    # =========================================================================

    def _measure_brightness(self, arr: Any) -> Tuple[int, float]:
        """
        Measure perceptual brightness via the Y (luma) channel.

        Method
        ------
        1. Compute Y (luma) = 0.299R + 0.587G + 0.114B (BT.601).
        2. Compare mean Y against the ideal fashion-photography range.
        3. Apply quadratic penalty for deviation from ideal midpoint.

        Score Interpretation
        --------------------
        score 0–30   : Very dark or very overexposed
        score 30–60  : Sub-optimal lighting
        score 60–85  : Good exposure
        score 85–100 : Perfect fashion-photography luminance

        Ideal range for fashion photography: Y ∈ [80, 200]
        Midpoint: Y ≈ 140

        Returns
        -------
        (score: int, mean_luma: float)
        """
        luma     = (0.299 * arr[:, :, 0] +
                    0.587 * arr[:, :, 1] +
                    0.114 * arr[:, :, 2])
        mean_lum = float(luma.mean())

        lo, hi   = self.brightness_ideal
        mid      = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        if lo <= mean_lum <= hi:
            # Inside ideal range: full → quadratic drop toward edges
            dist_from_mid = abs(mean_lum - mid)
            score = int(round(100 - (dist_from_mid / half_range) ** 2 * 15))
        else:
            # Outside ideal range: linear penalty
            if mean_lum < lo:
                dist = lo - mean_lum
                max_dist = lo  # max distance = 0 (pure black)
            else:
                dist = mean_lum - hi
                max_dist = 255 - hi   # max distance = 255 (pure white)
            penalty = min(1.0, dist / max(max_dist, 1))
            score   = int(round(max(0, 85 - penalty * 85)))

        score = max(0, min(100, score))
        logger.debug("Brightness | luma={:.2f} | score={}", mean_lum, score)
        return score, round(mean_lum, 2)

    # =========================================================================
    # ── Metric 3: Contrast
    # =========================================================================

    def _measure_contrast(self, arr: Any) -> Tuple[int, float]:
        """
        Measure global image contrast via RMS contrast.

        Method
        ------
        RMS contrast = std(I) / mean(I) — Michelson contrast variant.

        We use: RMS_contrast = std(luma) on [0, 255] scale.

        Score Interpretation
        --------------------
        score 0–20   : Nearly flat / no contrast
        score 20–50  : Low contrast (washed out or flat lighting)
        score 50–75  : Acceptable contrast
        score 75–90  : Good contrast
        score 90–100 : Excellent contrast (punchy, high-fashion look)

        Calibration: RMS=0 → score=0 | RMS=80 → score=100

        Returns
        -------
        (score: int, rms_contrast: float)
        """
        luma = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2])
        rms  = float(np.std(luma))

        # Square-root scaling for perceptual linearity
        score = int(round(min(100, (math.sqrt(rms) / math.sqrt(self.contrast_max_rms)) * 100)))
        score = max(0, min(100, score))

        logger.debug("Contrast | rms={:.2f} | score={}", rms, score)
        return score, round(rms, 2)

    # =========================================================================
    # ── Metric 4: Color Distribution
    # =========================================================================

    def _measure_color_distribution(
        self,
        arr: Any,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Measure color richness via saturation, histogram spread, and channel balance.

        Three sub-metrics combined:

        A. Saturation Mean (40%)
           - Convert RGB → HSV, extract saturation channel.
           - Mean saturation across all pixels.
           - Penalises grayscale or near-monochrome images.

        B. Saturation Coverage (30%)
           - Fraction of pixels with saturation > 0.10.
           - Penalises images with large flat/neutral areas.

        C. Channel Balance (30%)
           - Standard deviation of per-channel means (R, G, B).
           - Balanced channels → rich colour palette.
           - High std → single-channel dominance (e.g., extreme colour cast).
           - Moderate std preferred for natural fashion colours.

        Score Interpretation
        --------------------
        score 0–30   : Grayscale or heavily desaturated
        score 30–60  : Limited colour palette
        score 60–80  : Good colour variety
        score 80–100 : Vibrant, diverse, well-balanced colours

        Returns
        -------
        (score: int, meta: dict with saturation_mean, saturation_coverage, channel_balance)
        """
        # ── A. Saturation (HSV) ───────────────────────────────────────────
        r, g, b = arr[:, :, 0] / 255.0, arr[:, :, 1] / 255.0, arr[:, :, 2] / 255.0
        cmax    = np.maximum(np.maximum(r, g), b)
        cmin    = np.minimum(np.minimum(r, g), b)
        delta   = cmax - cmin

        # Saturation = delta / cmax (avoid div by zero)
        saturation = np.where(cmax > 0, delta / (cmax + 1e-9), 0.0)
        sat_mean   = float(saturation.mean())
        sat_cov    = float(np.mean(saturation > 0.10))

        sat_score  = min(100, int(round(sat_mean * 200)))    # 0.5 sat → 100
        cov_score  = min(100, int(round(sat_cov * 120)))     # 0.83 coverage → 100

        # ── B. Channel Balance ────────────────────────────────────────────
        ch_means   = np.array([r.mean(), g.mean(), b.mean()])
        ch_std     = float(np.std(ch_means))
        # Ideal: moderate imbalance (fashion colours); extreme imbalance = cast
        # Penalty curve: 0.0 std (gray) → 50; 0.1 std → 100; 0.3+ std → 40
        if ch_std < 0.05:
            bal_score = int(round(ch_std / 0.05 * 50 + 50))  # 50–100
        elif ch_std < 0.20:
            bal_score = 100
        else:
            excess    = ch_std - 0.20
            bal_score = max(20, int(round(100 - excess * 200)))

        # ── Weighted combination ──────────────────────────────────────────
        score = int(round(0.40 * sat_score + 0.30 * cov_score + 0.30 * bal_score))
        score = max(0, min(100, score))

        meta = {
            "saturation_mean":     round(sat_mean, 4),
            "saturation_coverage": round(sat_cov, 4),
            "channel_balance":     round(ch_std, 4),
        }

        logger.debug(
            "ColorDist | sat_mean={:.3f} | cov={:.3f} | bal={:.3f} | score={}",
            sat_mean, sat_cov, ch_std, score,
        )
        return score, meta

    # =========================================================================
    # ── Metric 5: Resolution
    # =========================================================================

    def _measure_resolution(
        self,
        image: Any,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Measure resolution quality based on megapixels, aspect ratio, and min size.

        Method
        ------
        1. Megapixel score: 0.25 MP → 50, 2.0 MP → 90, 4.0+ MP → 100.
        2. Minimum dimension check: both dims ≥ min_resolution → full score.
        3. Aspect ratio penalty: extreme ratios (> 2:1 or < 1:2) penalised.

        Score Interpretation
        --------------------
        score 0–30   : Very low resolution (< 256×256)
        score 30–60  : Below standard for AI fashion generation
        score 60–80  : Standard 512×512 range
        score 80–95  : High quality 1024×1024
        score 95–100 : Ultra-high resolution 2048×2048+

        Returns
        -------
        (score: int, meta: dict)
        """
        w, h       = image.size
        mp         = (w * h) / 1_000_000

        # ── Megapixel score (base) ────────────────────────────────────────
        mp_score = 0
        res_class = "very low"
        for threshold, label, pts in RESOLUTION_CLASSES:
            if mp >= threshold:
                mp_score  = pts
                res_class = label
                break

        # Intermediate interpolation
        for i, (thresh, label, pts) in enumerate(RESOLUTION_CLASSES):
            if mp >= thresh:
                if i > 0:
                    prev_thresh, _, prev_pts = RESOLUTION_CLASSES[i - 1]
                    t   = (mp - thresh) / (prev_thresh - thresh + 1e-9)
                    mp_score = int(round(pts + t * (prev_pts - pts)))
                else:
                    mp_score = pts
                res_class = label
                break

        mp_score = max(0, min(100, mp_score))

        # ── Minimum dimension penalty ─────────────────────────────────────
        min_w, min_h = self.min_resolution
        if w < min_w or h < min_h:
            penalty = min(40, int((min_w - w) / min_w * 40 + (min_h - h) / min_h * 40))
            mp_score = max(0, mp_score - penalty)

        # ── Aspect ratio check ────────────────────────────────────────────
        ratio = w / max(h, 1)
        if ratio > 2.5 or ratio < 0.4:
            mp_score = max(0, mp_score - 10)

        # ── Aspect ratio string ───────────────────────────────────────────
        gcd = math.gcd(w, h)
        ar_str = f"{w // gcd}:{h // gcd}" if gcd > 0 else f"{w}:{h}"

        meta = {
            "width":           w,
            "height":          h,
            "megapixels":      round(mp, 4),
            "aspect_ratio":    ar_str,
            "resolution_class":res_class,
        }

        logger.debug(
            "Resolution | {}x{} | {:.3f}MP | {} | score={}",
            w, h, mp, res_class, mp_score,
        )
        return mp_score, meta

    # =========================================================================
    # ── Metric 6: Noise Level
    # =========================================================================

    def _measure_noise(self, arr: Any) -> Tuple[int, float]:
        """
        Estimate noise level via high-frequency residual analysis.

        Method
        ------
        1. Compute a smoothed version of the image (box blur / uniform filter).
        2. Subtract the smoothed image from the original → noise residual.
        3. Compute the mean absolute energy of the residual.
        4. Map energy → inverse score (more noise = lower score).

        Score Interpretation
        --------------------
        score 90–100 : Extremely clean (professional studio output)
        score 70–90  : Clean with minor grain
        score 50–70  : Moderate noise (acceptable for web)
        score 30–50  : Noisy (visible grain/artifacts)
        score 0–30   : Heavy noise / strong compression artifacts

        Calibration: energy=0.5 → score≈100 | energy=40 → score≈0

        Returns
        -------
        (score: int, noise_energy: float)
        """
        # Grayscale luminance for noise estimation
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2])

        # Smooth with a box filter (approximates Gaussian for speed)
        if _SCIPY:
            smoothed = _ndi.uniform_filter(gray, size=3)
        else:
            # Manual 3×3 box blur via numpy roll
            smoothed = (
                gray
                + np.roll(gray,  1, axis=0) + np.roll(gray, -1, axis=0)
                + np.roll(gray,  1, axis=1) + np.roll(gray, -1, axis=1)
            ) / 5.0

        residual = gray - smoothed
        energy   = float(np.mean(np.abs(residual)))

        # Inverse mapping: energy → score
        if energy <= NOISE_MIN_ENERGY:
            score = 100
        elif energy >= self.noise_max_energy:
            score = 0
        else:
            t     = (energy - NOISE_MIN_ENERGY) / (self.noise_max_energy - NOISE_MIN_ENERGY)
            score = int(round((1.0 - t) ** 1.5 * 100))   # convex decay curve

        score = max(0, min(100, score))
        logger.debug("Noise | energy={:.4f} | score={}", energy, score)
        return score, round(energy, 4)

    # =========================================================================
    # ── Overall Score + Quality Rating
    # =========================================================================

    def _compute_overall(self, scores: Dict[str, int]) -> int:
        """
        Compute weighted overall quality score [0, 100].

        Uses the weight map defined in ``self.weights``.
        Applies a soft floor: if any critical metric is 0,
        the overall is capped at 50.
        """
        weighted = sum(
            scores.get(metric, 0) * weight
            for metric, weight in self.weights.items()
        )
        overall = int(round(min(100, max(0, weighted))))

        # Soft floor: catastrophically bad single metric
        if any(scores.get(m, 0) == 0 for m in ["sharpness", "resolution"]):
            overall = min(overall, 50)

        return overall

    @staticmethod
    def _rate_quality(overall: int) -> str:
        """
        Map an overall score to a quality label.

        +-----------+----------+
        | Score     | Label    |
        +===========+==========+
        | 90 – 100  | excellent|
        | 75 – 89   | very good|
        | 60 – 74   | good     |
        | 45 – 59   | fair     |
        | 25 – 44   | poor     |
        |  0 – 24   | very poor|
        +-----------+----------+
        """
        for label, (lo, hi) in QUALITY_THRESHOLDS.items():
            if lo <= overall < hi:
                return label
        return "very poor"

    # =========================================================================
    # ── Warning Generation
    # =========================================================================

    def _generate_warnings(self, report: QualityReport) -> List[str]:
        """
        Generate human-readable warnings for sub-threshold metrics.
        """
        warnings: List[str] = []

        checks = [
            ("sharpness",          report.sharpness,
             "Image appears blurry or out of focus"),
            ("brightness",         report.brightness,
             f"Brightness sub-optimal (mean luma={report.mean_brightness:.1f})"),
            ("contrast",           report.contrast,
             f"Low contrast detected (RMS={report.rms_contrast:.1f})"),
            ("color_distribution", report.color_distribution,
             "Limited colour distribution or saturation"),
            ("resolution",         report.resolution,
             f"Resolution {report.width}x{report.height} below recommended minimum"),
            ("noise_level",        report.noise_level,
             f"High noise detected (energy={report.noise_energy:.3f})"),
        ]

        for metric, score, msg in checks:
            if score < MIN_PASSING_SCORE.get(metric, 30):
                warnings.append(f"[{metric.upper()}] {msg} (score={score})")

        # Extra contextual warnings
        if report.mean_brightness < 30:
            warnings.append("[BRIGHTNESS] Image is very dark — may be underexposed")
        if report.mean_brightness > 230:
            warnings.append("[BRIGHTNESS] Image is very bright — may be overexposed")
        if report.rms_contrast < 10:
            warnings.append("[CONTRAST] Nearly flat image detected")
        if report.saturation_mean < 0.05:
            warnings.append("[COLOR] Image appears nearly grayscale")

        return warnings

    # =========================================================================
    # ── Report Saving
    # =========================================================================

    def _save_report(
        self,
        report:     QualityReport,
        report_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Save a single QualityReport JSON to disk."""
        out_dir = Path(report_dir) if report_dir else Path("week2/outputs/quality_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"quality_{report.image_id}_{ts}.json"
        try:
            path.write_text(report.to_json(), encoding="utf-8")
            logger.debug("Quality report saved: {}", path)
            return path
        except Exception as exc:
            logger.error("Failed to save quality report: {}", exc)
            return None

    def _save_batch_report(
        self,
        batch:      BatchQualityReport,
        report_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Save a BatchQualityReport JSON to disk."""
        out_dir = Path(report_dir) if report_dir else Path("week2/outputs/quality_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"batch_quality_{ts}.json"
        try:
            path.write_text(batch.to_json(), encoding="utf-8")
            logger.debug("Batch quality report saved: {}", path)
            return path
        except Exception as exc:
            logger.error("Failed to save batch quality report: {}", exc)
            return None

    # =========================================================================
    # ── Utilities / Properties
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"ImageQualityAssessor("
            f"min_res={self.min_resolution} | "
            f"weights={self.weights})"
        )


# =============================================================================
# ── Module-Level Convenience API
# =============================================================================

_DEFAULT_ASSESSOR: Optional[ImageQualityAssessor] = None


def _get_assessor(**kwargs) -> ImageQualityAssessor:
    global _DEFAULT_ASSESSOR
    if _DEFAULT_ASSESSOR is None:
        _DEFAULT_ASSESSOR = ImageQualityAssessor(**kwargs)
    return _DEFAULT_ASSESSOR


def assess(
    image:    Any,
    image_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Module-level shortcut — assess a single image and return a dict.

    Parameters
    ----------
    image    : PIL.Image.Image
    image_id : str, optional

    Returns
    -------
    dict — ``{"sharpness": 90, "contrast": 85, "quality": "excellent", ...}``

    Example
    -------
        from src.evaluation.week2_image_quality import assess
        result = assess(pil_image, image_id="GEN_001")
        print(result["quality"])     # "excellent"
        print(result["sharpness"])   # 90
    """
    assessor = _get_assessor()
    return assessor.assess(image, image_id=image_id).to_dict()


def assess_batch(
    images:    Sequence[Any],
    image_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Module-level shortcut — assess a batch of images.

    Parameters
    ----------
    images    : sequence of PIL.Image.Image
    image_ids : list of str, optional

    Returns
    -------
    list of dict

    Example
    -------
        from src.evaluation.week2_image_quality import assess_batch
        results = assess_batch(pil_images)
        for r in results:
            print(r["quality"])
    """
    assessor = _get_assessor()
    return [r.to_dict() for r in assessor.assess_batch(images, image_ids)]


def generate_report(
    image:         Any,
    image_id:      Optional[str] = None,
    output_format: str           = "dict",
) -> Union[Dict[str, Any], str]:
    """
    Module-level shortcut — generate a quality report.

    Parameters
    ----------
    image         : PIL.Image.Image
    image_id      : str, optional
    output_format : str   ``"dict"`` | ``"json"`` | ``"text"``

    Returns
    -------
    dict or str

    Example
    -------
        from src.evaluation.week2_image_quality import generate_report
        report = generate_report(pil_image, output_format="json")
        print(report)
    """
    assessor = _get_assessor()
    return assessor.generate_report(image, image_id=image_id, output_format=output_format)
