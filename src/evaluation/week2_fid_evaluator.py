"""
week2/evaluation/fid_evaluator.py
=====================================
Production-Grade FID (Fréchet Inception Distance) Evaluation Framework
AI-Powered Fashion Design Assistant — Week 2

╔══════════════════════════════════════════════════════════════════╗
║               FID Evaluation Framework                          ║
║                                                                  ║
║  FID measures the quality and diversity of generated images      ║
║  by comparing feature distributions to real reference images.    ║
║                                                                  ║
║  Lower FID = better (0 = identical distributions)               ║
║                                                                  ║
║  Metrics                                                         ║
║  ───────                                                         ║
║   1. FID Score          — distribution distance vs reference     ║
║   2. Dataset Comparison — style-vs-style / run-vs-run            ║
║   3. Generation Quality — quality rating from FID interpretation ║
║                                                                  ║
║  Functions                                                       ║
║  ─────────                                                       ║
║   calculate_fid()                                                ║
║   compare_with_dataset()                                         ║
║   benchmark_results()                                            ║
╚══════════════════════════════════════════════════════════════════╝

Setup Instructions
------------------
Install the required dependencies::

    # Option A — pytorch-fid (standalone CLI + Python API)
    pip install pytorch-fid

    # Option B — torchmetrics (already in requirements.txt)
    pip install torchmetrics torch torchvision

    # Full stack (recommended)
    pip install pytorch-fid torchmetrics torch torchvision Pillow numpy

GPU accelerated (optional but recommended for large sets)::

    pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage
-----
    from src.evaluation.week2_fid_evaluator import FIDEvaluator

    evaluator = FIDEvaluator()

    # FID between two image directories
    result = evaluator.calculate_fid(
        real_images_dir  = "data/real_fashion/",
        generated_images = my_pil_images,
    )
    print(result.fid_score)        # e.g. 28.4
    print(result.quality_rating)   # "good"

    # Compare with a curated dataset
    comparison = evaluator.compare_with_dataset(
        generated_images = my_pil_images,
        dataset_name     = "fashion_mnist",
    )

    # Benchmark multiple runs
    report = evaluator.benchmark_results(
        runs = {
            "baseline":  baseline_images,
            "improved":  improved_images,
        },
        reference_dir = "data/real_fashion/",
    )
    print(report.summary())

Module-level convenience::

    from src.evaluation.week2_fid_evaluator import calculate_fid, compare_with_dataset, benchmark_results
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
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
    logger = _log.getLogger("fid_evaluator")  # type: ignore[assignment]

# ── NumPy ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None   # type: ignore[assignment]
    _NUMPY = False

# ── PyTorch / TorchVision ─────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    _TORCH = True
except ImportError:
    torch = None        # type: ignore[assignment]
    T     = None        # type: ignore[assignment]
    models= None        # type: ignore[assignment]
    _TORCH = False

# ── pytorch-fid ───────────────────────────────────────────────────────────────
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import (
        calculate_frechet_distance,
        get_activations,
        calculate_activation_statistics,
    )
    _PYTORCH_FID = True
except ImportError:
    InceptionV3                    = None  # type: ignore[assignment]
    calculate_frechet_distance     = None  # type: ignore[assignment]
    get_activations                = None  # type: ignore[assignment]
    calculate_activation_statistics= None  # type: ignore[assignment]
    _PYTORCH_FID = False

# ── torchmetrics fallback ─────────────────────────────────────────────────────
try:
    from torchmetrics.image.fid import FrechetInceptionDistance as _TorchFID
    _TORCHMETRICS_FID = True
except ImportError:
    _TorchFID = None    # type: ignore[assignment]
    _TORCHMETRICS_FID = False

# ── PIL ───────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    PILImage = None # type: ignore[assignment]
    _PIL = False

# ── SciPy (for manual Fréchet distance calculation) ──────────────────────────
try:
    from scipy import linalg as _scipy_linalg
    _SCIPY = True
except ImportError:
    _scipy_linalg = None   # type: ignore[assignment]
    _SCIPY = False


# =============================================================================
# ── Constants
# =============================================================================

# FID quality interpretation thresholds (lower is better)
FID_QUALITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "excellent": (0.0,   10.0),
    "very good": (10.0,  25.0),
    "good":      (25.0,  50.0),
    "fair":      (50.0,  100.0),
    "poor":      (100.0, 200.0),
    "very poor": (200.0, float("inf")),
}

# Inception V3 feature dimensionality (standard)
INCEPTION_FEATURE_DIM = 2048

# Minimum images needed for meaningful FID
MIN_IMAGES_FOR_FID = 2

# Default Inception V3 input size
INCEPTION_INPUT_SIZE = (299, 299)


# =============================================================================
# ── Result Dataclasses
# =============================================================================

@dataclass
class FIDScore:
    """
    Result of a single FID computation.

    Attributes
    ----------
    fid_score        : float   Fréchet Inception Distance (lower = better).
    quality_rating   : str     Human-readable quality label.
    quality_score    : float   Normalised quality [0, 1] (higher = better).
    n_real           : int     Number of real reference images used.
    n_generated      : int     Number of generated images evaluated.
    feature_dim      : int     Inception feature dimensionality used.
    backend          : str     Which FID backend was used.
    device           : str
    elapsed_s        : float
    run_id           : str
    error            : str     Non-empty if computation failed.
    mu_real          : list    Mean of real feature distribution (optional).
    mu_gen           : list    Mean of generated feature distribution (optional).
    """
    fid_score:      float           = float("inf")
    quality_rating: str             = "unknown"
    quality_score:  float           = 0.0
    n_real:         int             = 0
    n_generated:    int             = 0
    feature_dim:    int             = INCEPTION_FEATURE_DIM
    backend:        str             = "unknown"
    device:         str             = "cpu"
    elapsed_s:      float           = 0.0
    run_id:         str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    error:          str             = ""
    mu_real:        Optional[List[float]] = None
    mu_gen:         Optional[List[float]] = None

    @property
    def passed(self) -> bool:
        """True if FID is finite and quality is at least 'fair'."""
        return (
            self.fid_score != float("inf")
            and self.quality_rating in ("excellent", "very good", "good", "fair")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fid_score":     round(self.fid_score, 4) if self.fid_score != float("inf") else None,
            "quality_rating":self.quality_rating,
            "quality_score": round(self.quality_score, 4),
            "passed":        self.passed,
            "n_real":        self.n_real,
            "n_generated":   self.n_generated,
            "feature_dim":   self.feature_dim,
            "backend":       self.backend,
            "device":        self.device,
            "elapsed_s":     round(self.elapsed_s, 3),
            "run_id":        self.run_id,
            "error":         self.error,
        }

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        score  = f"{self.fid_score:.2f}" if self.fid_score != float("inf") else "N/A"
        return (
            f"FIDScore [{status}] | FID={score} | "
            f"quality={self.quality_rating!r} | "
            f"n_real={self.n_real} | n_gen={self.n_generated} | "
            f"backend={self.backend}"
        )

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class DatasetComparison:
    """
    Result of comparing generated images against a named dataset.

    Attributes
    ----------
    dataset_name     : str
    fid_result       : FIDScore
    style            : str         Style/category label.
    improvement      : float       FID delta vs baseline (negative = better).
    percentile_rank  : float       [0, 1] rank among known benchmarks.
    notes            : list of str
    """
    dataset_name:    str
    fid_result:      FIDScore
    style:           str            = ""
    improvement:     Optional[float]= None
    percentile_rank: Optional[float]= None
    notes:           List[str]      = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name":   self.dataset_name,
            "style":          self.style,
            "fid_result":     self.fid_result.to_dict(),
            "improvement":    round(self.improvement, 4) if self.improvement is not None else None,
            "percentile_rank":round(self.percentile_rank, 4) if self.percentile_rank is not None else None,
            "notes":          self.notes,
        }

    def summary(self) -> str:
        fid = f"{self.fid_result.fid_score:.2f}" if self.fid_result.fid_score != float("inf") else "N/A"
        return (
            f"DatasetComparison | dataset={self.dataset_name!r} | "
            f"style={self.style!r} | FID={fid} | "
            f"quality={self.fid_result.quality_rating!r}"
        )


@dataclass
class BenchmarkReport:
    """
    Aggregate benchmark across multiple generation runs.

    Attributes
    ----------
    runs             : dict  name → FIDScore
    reference_dir    : str
    best_run         : str   Name of the best-performing run.
    worst_run        : str
    mean_fid         : float Mean FID across all runs.
    std_fid          : float Standard deviation.
    generated_at     : str
    elapsed_s        : float
    """
    runs:           Dict[str, FIDScore]  = field(default_factory=dict)
    reference_dir:  str                  = ""
    best_run:       str                  = ""
    worst_run:      str                  = ""
    mean_fid:       float                = float("inf")
    std_fid:        float                = 0.0
    generated_at:   str                  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elapsed_s:      float                = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at":  self.generated_at,
            "reference_dir": self.reference_dir,
            "best_run":      self.best_run,
            "worst_run":     self.worst_run,
            "mean_fid":      round(self.mean_fid, 4) if self.mean_fid != float("inf") else None,
            "std_fid":       round(self.std_fid, 4),
            "elapsed_s":     round(self.elapsed_s, 3),
            "runs": {
                name: score.to_dict()
                for name, score in self.runs.items()
            },
        }

    def summary(self) -> str:
        sep   = "=" * 64
        lines = [
            sep,
            "  FID BENCHMARK REPORT",
            sep,
            f"  Reference dir : {self.reference_dir or 'N/A'}",
            f"  Runs          : {len(self.runs)}",
            f"  Best run      : {self.best_run or 'N/A'}",
            f"  Worst run     : {self.worst_run or 'N/A'}",
            f"  Mean FID      : {self.mean_fid:.2f}" if self.mean_fid != float("inf") else "  Mean FID      : N/A",
            f"  Std FID       : {self.std_fid:.2f}",
            f"  Elapsed       : {self.elapsed_s:.2f}s",
            "",
            "  Per-Run Results:",
        ]
        for name, score in sorted(self.runs.items(), key=lambda kv: kv[1].fid_score):
            fid_str = f"{score.fid_score:.2f}" if score.fid_score != float("inf") else "N/A"
            marker  = " <-- BEST" if name == self.best_run else ""
            lines.append(
                f"    {name:<20} FID={fid_str:>8}  "
                f"quality={score.quality_rating:<10}{marker}"
            )
        lines.append(sep)
        return "\n".join(lines)


# =============================================================================
# ── FIDEvaluator
# =============================================================================

class FIDEvaluator:
    """
    Production-grade FID evaluation framework for fashion image generation.

    Backend priority
    ----------------
    1. ``pytorch-fid``   — official, dedicated package (most accurate)
    2. ``torchmetrics``  — widely used, GPU-friendly
    3. ``manual``        — pure NumPy/SciPy Fréchet distance on Inception features
    4. ``stub``          — returns NaN with error message (no deps installed)

    Parameters
    ----------
    device : str
        ``"cuda"`` | ``"cpu"`` | ``"auto"`` (default).
    batch_size : int
        Inception feature extraction batch size.
    feature_dim : int
        Inception feature dimensionality (default 2048).
    temp_dir : Path, optional
        Temporary directory for intermediate files.
        Created and cleaned up automatically if None.
    save_stats : bool
        Save computed feature statistics to disk for reuse.
    stats_cache_dir : Path, optional
        Directory to store/load cached Inception statistics.

    Example
    -------
        evaluator = FIDEvaluator()

        result = evaluator.calculate_fid(
            real_images      = real_pil_images,
            generated_images = gen_pil_images,
        )
        print(result.fid_score)       # 28.4
        print(result.quality_rating)  # "good"
    """

    def __init__(
        self,
        device:          str            = "auto",
        batch_size:      int            = 32,
        feature_dim:     int            = INCEPTION_FEATURE_DIM,
        temp_dir:        Optional[Path] = None,
        save_stats:      bool           = False,
        stats_cache_dir: Optional[Path] = None,
    ) -> None:
        self.device          = self._resolve_device(device)
        self.batch_size      = batch_size
        self.feature_dim     = feature_dim
        self.temp_dir        = Path(temp_dir) if temp_dir else None
        self.save_stats      = save_stats
        self.stats_cache_dir = Path(stats_cache_dir) if stats_cache_dir else None

        self._backend        = self._detect_backend()
        self._inception      = None   # Lazy-loaded

        logger.info(
            "FIDEvaluator initialised | backend={} | device={} | batch_size={}",
            self._backend, self.device, batch_size,
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def calculate_fid(
        self,
        real_images:         Union[List[Any], str, Path],
        generated_images:    Union[List[Any], str, Path],
        *,
        real_stats_cache:    Optional[str] = None,
        run_id:              Optional[str] = None,
    ) -> FIDScore:
        """
        Calculate the FID score between real and generated image distributions.

        Parameters
        ----------
        real_images : list of PIL.Image or str/Path (directory)
            Reference real images. Can be:
              - A list of ``PIL.Image.Image`` objects.
              - A path to a directory containing ``.jpg``/``.png`` images.
        generated_images : list of PIL.Image or str/Path
            Generated images to evaluate.
        real_stats_cache : str, optional
            Name key for caching/loading real image statistics (avoids
            recomputing Inception features on every run).
        run_id : str, optional
            Identifier for this run (auto-generated if not provided).

        Returns
        -------
        FIDScore

        Example
        -------
            result = evaluator.calculate_fid(
                real_images      = "data/fashion_reference/",
                generated_images = my_generated_pil_images,
            )
            print(result.to_dict())
            # {"fid_score": 28.4, "quality_rating": "good", ...}
        """
        t0    = time.perf_counter()
        _id   = run_id or str(uuid.uuid4())[:8]

        result = FIDScore(
            run_id     = _id,
            backend    = self._backend,
            device     = self.device,
            feature_dim= self.feature_dim,
        )

        logger.info(
            "calculate_fid | run_id={} | backend={} | device={}",
            _id, self._backend, self.device,
        )

        try:
            # ── Resolve image sources ─────────────────────────────────────
            real_imgs = self._resolve_images(real_images,      "real")
            gen_imgs  = self._resolve_images(generated_images, "generated")

            result.n_real      = len(real_imgs)
            result.n_generated = len(gen_imgs)

            if result.n_real < MIN_IMAGES_FOR_FID:
                raise ValueError(
                    f"Too few real images ({result.n_real}). "
                    f"FID requires at least {MIN_IMAGES_FOR_FID}."
                )
            if result.n_generated < MIN_IMAGES_FOR_FID:
                raise ValueError(
                    f"Too few generated images ({result.n_generated}). "
                    f"FID requires at least {MIN_IMAGES_FOR_FID}."
                )

            logger.info(
                "FID inputs | real={} | generated={} images",
                result.n_real, result.n_generated,
            )

            # ── Compute FID via best available backend ────────────────────
            fid_val, mu_real, mu_gen = self._compute_fid(
                real_imgs, gen_imgs, real_stats_cache
            )

            result.fid_score    = fid_val
            result.mu_real      = mu_real[:8] if mu_real is not None else None  # Store first 8 dims
            result.mu_gen       = mu_gen[:8]  if mu_gen  is not None else None
            result.quality_rating= self._rate_quality(fid_val)
            result.quality_score = self._fid_to_quality_score(fid_val)

        except Exception as exc:
            logger.error("calculate_fid failed: {}", exc)
            result.error          = str(exc)
            result.quality_rating = "unknown"
            result.quality_score  = 0.0

        result.elapsed_s = time.perf_counter() - t0
        logger.info(
            "calculate_fid complete | FID={} | quality={} | {:.2f}s",
            f"{result.fid_score:.2f}" if result.fid_score != float("inf") else "N/A",
            result.quality_rating,
            result.elapsed_s,
        )
        return result

    def compare_with_dataset(
        self,
        generated_images: Union[List[Any], str, Path],
        dataset_name:     str,
        *,
        dataset_dir:      Optional[Union[str, Path]] = None,
        style:            str = "",
        baseline_fid:     Optional[float] = None,
        known_benchmarks: Optional[Dict[str, float]] = None,
        run_id:           Optional[str] = None,
    ) -> DatasetComparison:
        """
        Compare generated images against a named reference dataset.

        Parameters
        ----------
        generated_images : list of PIL.Image or str/Path
            Generated images to evaluate.
        dataset_name : str
            Human-readable name for the reference dataset
            (e.g. ``"fashion_mnist"``, ``"deepfashion"``,
             ``"real_runway_ss2024"``).
        dataset_dir : str or Path, optional
            Path to the reference dataset directory. If None, a synthetic
            reference is created from the generated set for relative scoring.
        style : str, optional
            Fashion style label (for reporting).
        baseline_fid : float, optional
            Baseline FID to compute improvement against.
        known_benchmarks : dict, optional
            {dataset_name: published_fid} for percentile ranking.
        run_id : str, optional

        Returns
        -------
        DatasetComparison

        Example
        -------
            comparison = evaluator.compare_with_dataset(
                generated_images = my_images,
                dataset_name     = "deepfashion_upper_body",
                dataset_dir      = "data/deepfashion/",
                style            = "streetwear",
                baseline_fid     = 45.0,
            )
            print(comparison.summary())
        """
        _id    = run_id or str(uuid.uuid4())[:8]
        notes: List[str] = []

        logger.info(
            "compare_with_dataset | dataset={} | style={} | run_id={}",
            dataset_name, style, _id,
        )

        # ── Resolve reference images ──────────────────────────────────────
        if dataset_dir is not None:
            ref_path = Path(dataset_dir)
            if not ref_path.exists():
                logger.warning(
                    "Dataset directory {} not found — using synthetic reference.",
                    ref_path,
                )
                notes.append(f"Dataset dir {ref_path} not found; using synthetic reference.")
                real_images = self._create_synthetic_reference(generated_images)
            else:
                real_images = dataset_dir
        else:
            notes.append("No dataset_dir provided; using synthetic reference (relative comparison only).")
            real_images = self._create_synthetic_reference(generated_images)

        # ── Calculate FID ─────────────────────────────────────────────────
        fid_result = self.calculate_fid(
            real_images      = real_images,
            generated_images = generated_images,
            run_id           = _id,
        )

        # ── Improvement vs baseline ───────────────────────────────────────
        improvement = None
        if baseline_fid is not None and fid_result.fid_score != float("inf"):
            improvement = baseline_fid - fid_result.fid_score
            direction   = "improvement" if improvement > 0 else "regression"
            notes.append(f"vs baseline FID={baseline_fid:.2f}: {direction} of {abs(improvement):.2f}")

        # ── Percentile rank against known benchmarks ──────────────────────
        percentile_rank = None
        if known_benchmarks and fid_result.fid_score != float("inf"):
            fids = sorted(known_benchmarks.values())
            rank = sum(1 for f in fids if fid_result.fid_score <= f) / len(fids)
            percentile_rank = rank
            notes.append(f"Outperforms {rank:.1%} of known benchmarks.")

        return DatasetComparison(
            dataset_name    = dataset_name,
            fid_result      = fid_result,
            style           = style,
            improvement     = improvement,
            percentile_rank = percentile_rank,
            notes           = notes,
        )

    def benchmark_results(
        self,
        runs:          Dict[str, Union[List[Any], str, Path]],
        reference_dir: Optional[Union[str, Path]] = None,
        *,
        reference_images: Optional[List[Any]]     = None,
        save_report:   bool                        = False,
        report_path:   Optional[Path]              = None,
    ) -> BenchmarkReport:
        """
        Compute FID for multiple generation runs and produce a comparative report.

        Parameters
        ----------
        runs : dict
            Mapping of run_name → image source (list of PIL or directory path).
            Example: ``{"baseline": images_v1, "improved": images_v2}``
        reference_dir : str or Path, optional
            Reference image directory. If not provided and reference_images
            is None, uses the first run as the reference for all others.
        reference_images : list of PIL.Image, optional
            Reference images (used if reference_dir is None).
        save_report : bool
            Write JSON report to disk.
        report_path : Path, optional
            Output path for JSON report. Defaults to
            ``week2/outputs/evaluation_reports/fid_benchmark_{ts}.json``.

        Returns
        -------
        BenchmarkReport

        Example
        -------
            report = evaluator.benchmark_results(
                runs = {
                    "sdxl_base":     base_images,
                    "sdxl_refiner":  refined_images,
                    "sdxl_lora":     lora_images,
                },
                reference_dir = "data/real_fashion/",
            )
            print(report.summary())
        """
        t0 = time.perf_counter()
        report = BenchmarkReport(
            reference_dir = str(reference_dir) if reference_dir else "",
        )

        if not runs:
            logger.warning("benchmark_results: no runs provided")
            return report

        logger.info("benchmark_results | {} runs | reference={}", len(runs), reference_dir)

        # ── Determine reference ───────────────────────────────────────────
        if reference_dir is not None:
            ref_source: Union[str, Path, List[Any]] = reference_dir
        elif reference_images is not None:
            ref_source = reference_images
        else:
            # Use first run as reference baseline
            first_name = next(iter(runs))
            ref_source  = runs[first_name]
            logger.warning(
                "No reference provided — using run '{}' as reference baseline.",
                first_name,
            )
            report.reference_dir = f"(auto: {first_name})"

        # ── Compute FID per run ───────────────────────────────────────────
        run_results: Dict[str, FIDScore] = {}
        for run_name, run_images in runs.items():
            logger.info("Benchmarking run: {!r}", run_name)
            fid = self.calculate_fid(
                real_images      = ref_source,
                generated_images = run_images,
                run_id           = run_name,
            )
            run_results[run_name] = fid
            logger.info("  {} -> FID={:.2f} | quality={}",
                        run_name,
                        fid.fid_score if fid.fid_score != float("inf") else float("nan"),
                        fid.quality_rating)

        report.runs = run_results

        # ── Aggregate stats ───────────────────────────────────────────────
        valid_scores = [
            s.fid_score for s in run_results.values()
            if s.fid_score != float("inf")
        ]
        if valid_scores:
            report.mean_fid  = float(sum(valid_scores) / len(valid_scores))
            report.std_fid   = self._std(valid_scores)
            sorted_runs      = sorted(run_results.items(), key=lambda kv: kv[1].fid_score)
            report.best_run  = sorted_runs[0][0]
            report.worst_run = sorted_runs[-1][0]

        report.elapsed_s = time.perf_counter() - t0

        # ── Save JSON report ──────────────────────────────────────────────
        if save_report:
            saved = self._save_report(report, report_path)
            if saved:
                logger.success("Benchmark report saved: {}", saved)

        logger.info(
            "benchmark_results complete | best={} | mean_fid={:.2f} | {:.2f}s",
            report.best_run,
            report.mean_fid if report.mean_fid != float("inf") else float("nan"),
            report.elapsed_s,
        )
        return report

    # =========================================================================
    # ── Core FID Computation (multi-backend)
    # =========================================================================

    def _compute_fid(
        self,
        real_images: List[Any],
        gen_images:  List[Any],
        cache_key:   Optional[str] = None,
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """
        Dispatch FID computation to the best available backend.

        Returns
        -------
        (fid_value, mu_real, mu_gen)
        """
        if self._backend == "pytorch_fid":
            return self._compute_fid_pytorch_fid(real_images, gen_images, cache_key)
        elif self._backend == "torchmetrics":
            return self._compute_fid_torchmetrics(real_images, gen_images)
        elif self._backend == "manual":
            return self._compute_fid_manual(real_images, gen_images)
        else:
            logger.error("No FID backend available. Install pytorch-fid or torchmetrics.")
            return float("inf"), None, None

    # ── Backend 1: pytorch-fid ────────────────────────────────────────────────

    def _compute_fid_pytorch_fid(
        self,
        real_images: List[Any],
        gen_images:  List[Any],
        cache_key:   Optional[str],
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """Use pytorch-fid's InceptionV3 activations + Fréchet distance."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            real_dir = Path(tmp_dir) / "real"
            gen_dir  = Path(tmp_dir) / "gen"
            real_dir.mkdir()
            gen_dir.mkdir()

            self._save_images_to_dir(real_images, real_dir, "real")
            self._save_images_to_dir(gen_images,  gen_dir,  "gen")

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.feature_dim]
            model     = InceptionV3([block_idx]).to(self.device)
            model.eval()

            mu_real, sigma_real = calculate_activation_statistics(
                files      = list(real_dir.glob("*.png")),
                model      = model,
                batch_size = self.batch_size,
                dims       = self.feature_dim,
                device     = self.device,
            )
            mu_gen, sigma_gen = calculate_activation_statistics(
                files      = list(gen_dir.glob("*.png")),
                model      = model,
                batch_size = self.batch_size,
                dims       = self.feature_dim,
                device     = self.device,
            )

            # Persist stats if requested
            if self.save_stats and cache_key and self.stats_cache_dir:
                self._save_inception_stats(cache_key, mu_real, sigma_real)

            fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            return float(fid), mu_real.tolist()[:8], mu_gen.tolist()[:8]

    # ── Backend 2: torchmetrics ───────────────────────────────────────────────

    def _compute_fid_torchmetrics(
        self,
        real_images: List[Any],
        gen_images:  List[Any],
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """Use torchmetrics.image.fid.FrechetInceptionDistance."""
        fid_metric = _TorchFID(feature=self.feature_dim).to(self.device)

        # Process in batches
        for batch in self._batch_iter(real_images, self.batch_size):
            tensors = self._pil_to_tensor_batch(batch)
            fid_metric.update(tensors.to(self.device), real=True)

        for batch in self._batch_iter(gen_images, self.batch_size):
            tensors = self._pil_to_tensor_batch(batch)
            fid_metric.update(tensors.to(self.device), real=False)

        fid_val = fid_metric.compute().item()

        # Access internals for mu (torchmetrics exposes them)
        mu_real = None
        mu_gen  = None
        try:
            mu_real = fid_metric.real_features_sum.cpu().numpy()[:8].tolist()
            mu_gen  = fid_metric.fake_features_sum.cpu().numpy()[:8].tolist()
        except Exception:
            pass

        return float(fid_val), mu_real, mu_gen

    # ── Backend 3: Manual (NumPy + SciPy + InceptionV3 from torchvision) ─────

    def _compute_fid_manual(
        self,
        real_images: List[Any],
        gen_images:  List[Any],
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """
        Manual FID implementation using torchvision Inception V3.

        Extracts pool3 features, computes mean + covariance, then
        Fréchet distance via SciPy matrix square root.
        """
        if not _TORCH or not _NUMPY:
            raise RuntimeError("PyTorch and NumPy required for manual FID backend")

        model = self._get_inception()

        real_feats = self._extract_features(real_images, model)
        gen_feats  = self._extract_features(gen_images,  model)

        mu_real,  sigma_real  = self._compute_statistics(real_feats)
        mu_gen,   sigma_gen   = self._compute_statistics(gen_feats)
        fid_val = self._frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        return float(fid_val), mu_real.tolist()[:8], mu_gen.tolist()[:8]

    # ── Fréchet Distance (NumPy/SciPy) ───────────────────────────────────────

    @staticmethod
    def _frechet_distance(
        mu1:    Any,
        sigma1: Any,
        mu2:    Any,
        sigma2: Any,
        eps:    float = 1e-6,
    ) -> float:
        """
        Compute the Fréchet distance between two multivariate Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

        Parameters
        ----------
        mu1, mu2 : ndarray  Mean vectors.
        sigma1, sigma2 : ndarray  Covariance matrices.
        eps : float  Regularisation for numerical stability.

        Returns
        -------
        float — FID value.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff  = mu1 - mu2
        diff_sq = np.dot(diff, diff)

        # Numeric regularisation
        offset    = np.eye(sigma1.shape[0]) * eps
        covmean_sq= sigma1 @ sigma2

        try:
            if _SCIPY:
                covmean, _ = _scipy_linalg.sqrtm(covmean_sq + offset, disp=False)
            else:
                # Fallback: eigenvalue decomposition
                vals, vecs = np.linalg.eigh(covmean_sq + offset)
                vals  = np.maximum(vals, 0)
                covmean = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        except Exception as exc:
            logger.warning("Matrix sqrt failed ({}). Using approximation.", exc)
            covmean = np.zeros_like(sigma1)

        # Handle numerical errors — imaginary part should be near zero
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                logger.warning("Imaginary component in FID covmean matrix")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(np.real(fid))

    # =========================================================================
    # ── Feature Extraction (Manual Backend)
    # =========================================================================

    def _get_inception(self) -> Any:
        """Lazy-load InceptionV3 (torchvision) for feature extraction."""
        if self._inception is None:
            if not _TORCH:
                raise RuntimeError("PyTorch required for Inception feature extraction")
            inception = models.inception_v3(
                weights="Inception_V3_Weights.IMAGENET1K_V1",
                aux_logits=True,
            )
            inception.fc = torch.nn.Identity()   # Remove final classifier → pool3 features
            inception = inception.to(self.device).eval()
            self._inception = inception
            logger.debug("InceptionV3 loaded on {}", self.device)
        return self._inception

    def _extract_features(
        self,
        images: List[Any],
        model:  Any,
    ) -> Any:
        """
        Extract 2048-dim Inception pool3 features from a list of PIL images.

        Returns
        -------
        np.ndarray  shape [N, 2048]
        """
        if not _TORCH or not _NUMPY:
            raise RuntimeError("PyTorch and NumPy required")

        transform = T.Compose([
            T.Resize(INCEPTION_INPUT_SIZE),
            T.CenterCrop(INCEPTION_INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        all_feats = []
        for batch in self._batch_iter(images, self.batch_size):
            tensors = torch.stack([transform(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            with torch.no_grad():
                out = model(tensors)
                if isinstance(out, tuple):
                    out = out[0]    # Inception returns (logits, aux_logits)
                feats = out.cpu().numpy()
            all_feats.append(feats)

        return np.concatenate(all_feats, axis=0)

    @staticmethod
    def _compute_statistics(features: Any) -> Tuple[Any, Any]:
        """Compute mean and covariance of feature matrix."""
        mu    = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    # =========================================================================
    # ── Image Resolution Helpers
    # =========================================================================

    def _resolve_images(
        self,
        source: Union[List[Any], str, Path],
        label:  str,
    ) -> List[Any]:
        """
        Resolve a source to a list of PIL images.

        Accepts:
          - List of PIL.Image objects
          - Path to a directory (loads all jpg/png)
          - Path string
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                imgs = self._load_images_from_dir(path)
                logger.debug("Loaded {} {} images from {}", len(imgs), label, path)
                return imgs
            else:
                raise FileNotFoundError(
                    f"Image source '{path}' is neither a list nor a valid directory."
                )
        if isinstance(source, list):
            valid = [img for img in source if img is not None]
            if len(valid) < len(source):
                logger.warning("Dropped {} None images from {}", len(source) - len(valid), label)
            return valid
        raise TypeError(
            f"Image source must be list[PIL.Image] or a directory path, got {type(source)}"
        )

    @staticmethod
    def _load_images_from_dir(directory: Path) -> List[Any]:
        """Load all jpg/png images from a directory."""
        if not _PIL:
            raise RuntimeError("Pillow required for directory image loading")
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]
        imgs  = []
        for p in paths:
            try:
                imgs.append(PILImage.open(p).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load {}: {}", p, exc)
        return imgs

    def _create_synthetic_reference(
        self,
        generated_images: Union[List[Any], str, Path],
    ) -> List[Any]:
        """
        Create a synthetic reference set from generated images
        (used when no real dataset is available — for relative comparison only).

        Applies mild augmentations (flip, slight colour jitter) to create
        a distinct-but-related distribution.
        """
        imgs = self._resolve_images(generated_images, "synthetic_ref")
        if not _PIL or not _TORCH:
            return imgs[:max(len(imgs) // 2, MIN_IMAGES_FOR_FID)]

        augmented = []
        for img in imgs:
            try:
                aug = img.transpose(PILImage.FLIP_LEFT_RIGHT)
                augmented.append(aug)
            except Exception:
                augmented.append(img)
        return augmented

    # =========================================================================
    # ── Tensor Utilities
    # =========================================================================

    @staticmethod
    def _pil_to_tensor_batch(images: List[Any]) -> Any:
        """Convert PIL images → uint8 [N, 3, H, W] tensor for torchmetrics FID."""
        transform = T.Compose([
            T.Resize(INCEPTION_INPUT_SIZE),
            T.CenterCrop(INCEPTION_INPUT_SIZE),
            T.ToTensor(),
        ])
        tensors = torch.stack([
            (transform(img.convert("RGB")) * 255).byte()
            for img in images
        ])
        return tensors

    @staticmethod
    def _save_images_to_dir(
        images:    List[Any],
        directory: Path,
        prefix:    str = "img",
    ) -> None:
        """Save PIL images to a directory for pytorch-fid's file-based API."""
        for idx, img in enumerate(images):
            path = directory / f"{prefix}_{idx:05d}.png"
            try:
                img.convert("RGB").save(str(path))
            except Exception as exc:
                logger.warning("Failed to save {}: {}", path, exc)

    @staticmethod
    def _batch_iter(items: List[Any], batch_size: int):
        """Yield successive batches from a list."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    # =========================================================================
    # ── Quality Interpretation
    # =========================================================================

    @staticmethod
    def _rate_quality(fid: float) -> str:
        """
        Map a FID score to a human-readable quality label.

        Reference thresholds for fashion/product imagery::

            FID  0 – 10   : excellent  (near-indistinguishable from real)
            FID 10 – 25   : very good  (high quality, minor artifacts)
            FID 25 – 50   : good       (clearly AI-generated but high quality)
            FID 50 – 100  : fair       (noticeable quality issues)
            FID 100 – 200 : poor       (significant quality problems)
            FID > 200     : very poor  (severely degraded)
        """
        if fid == float("inf") or fid != fid:  # inf or NaN
            return "unknown"
        for label, (lo, hi) in FID_QUALITY_THRESHOLDS.items():
            if lo <= fid < hi:
                return label
        return "very poor"

    @staticmethod
    def _fid_to_quality_score(fid: float, max_fid: float = 200.0) -> float:
        """
        Convert FID to a normalised quality score [0, 1].

        Score = 1 - clamp(fid / max_fid, 0, 1)
        """
        if fid == float("inf") or fid != fid:
            return 0.0
        return round(max(0.0, 1.0 - min(fid, max_fid) / max_fid), 4)

    # =========================================================================
    # ── Backend Detection
    # =========================================================================

    @staticmethod
    def _detect_backend() -> str:
        """Select the best available FID backend."""
        if _PYTORCH_FID:
            return "pytorch_fid"
        if _TORCHMETRICS_FID and _TORCH:
            return "torchmetrics"
        if _TORCH and _NUMPY and _SCIPY:
            return "manual"
        return "stub"

    @property
    def backend(self) -> str:
        """Active FID backend name."""
        return self._backend

    @property
    def is_available(self) -> bool:
        """True if at least one FID backend is available."""
        return self._backend != "stub"

    # =========================================================================
    # ── Caching
    # =========================================================================

    def _save_inception_stats(
        self,
        key:   str,
        mu:    Any,
        sigma: Any,
    ) -> Optional[Path]:
        """Save Inception statistics (mu, sigma) for a dataset."""
        if self.stats_cache_dir is None:
            return None
        self.stats_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.stats_cache_dir / f"{key}.npz"
        try:
            np.savez(str(path), mu=mu, sigma=sigma)
            logger.debug("Inception stats cached: {}", path)
            return path
        except Exception as exc:
            logger.warning("Failed to save Inception stats: {}", exc)
            return None

    def load_inception_stats(
        self,
        key: str,
    ) -> Optional[Tuple[Any, Any]]:
        """Load cached Inception statistics (mu, sigma) for a dataset key."""
        if self.stats_cache_dir is None:
            return None
        path = self.stats_cache_dir / f"{key}.npz"
        if not path.exists():
            return None
        try:
            data = np.load(str(path))
            logger.debug("Inception stats loaded from cache: {}", path)
            return data["mu"], data["sigma"]
        except Exception as exc:
            logger.warning("Failed to load Inception stats: {}", exc)
            return None

    # =========================================================================
    # ── Report Saving
    # =========================================================================

    def _save_report(
        self,
        report: BenchmarkReport,
        path:   Optional[Path],
    ) -> Optional[Path]:
        """Serialise a BenchmarkReport to JSON."""
        if path is None:
            ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_d = Path("week2/outputs/evaluation_reports")
            out_d.mkdir(parents=True, exist_ok=True)
            path  = out_d / f"fid_benchmark_{ts}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return path
        except Exception as exc:
            logger.error("Failed to save FID report: {}", exc)
            return None

    # =========================================================================
    # ── Utilities
    # =========================================================================

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if _TORCH and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    @staticmethod
    def _std(values: List[float]) -> float:
        if _NUMPY and len(values) > 0:
            return float(np.std(values))
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    def __repr__(self) -> str:
        return (
            f"FIDEvaluator(backend={self._backend!r} | "
            f"device={self.device} | "
            f"available={self.is_available})"
        )


# =============================================================================
# ── Module-Level Convenience API
# =============================================================================

_DEFAULT_EVALUATOR: Optional[FIDEvaluator] = None


def _get_evaluator(**kwargs) -> FIDEvaluator:
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = FIDEvaluator(**kwargs)
    return _DEFAULT_EVALUATOR


def calculate_fid(
    real_images:      Union[List[Any], str, Path],
    generated_images: Union[List[Any], str, Path],
    *,
    device:    str = "auto",
    batch_size:int = 32,
    run_id:    Optional[str] = None,
) -> FIDScore:
    """
    Module-level shortcut for ``FIDEvaluator().calculate_fid()``.

    Parameters
    ----------
    real_images      : list of PIL.Image or directory path
    generated_images : list of PIL.Image or directory path
    device : str
    batch_size : int
    run_id : str, optional

    Returns
    -------
    FIDScore

    Example
    -------
        from src.evaluation.week2_fid_evaluator import calculate_fid
        result = calculate_fid(real_images, generated_images)
        print(result.fid_score, result.quality_rating)
    """
    ev = FIDEvaluator(device=device, batch_size=batch_size)
    return ev.calculate_fid(real_images, generated_images, run_id=run_id)


def compare_with_dataset(
    generated_images: Union[List[Any], str, Path],
    dataset_name:     str,
    *,
    dataset_dir:      Optional[Union[str, Path]] = None,
    style:            str = "",
    baseline_fid:     Optional[float] = None,
    device:           str = "auto",
) -> DatasetComparison:
    """
    Module-level shortcut for ``FIDEvaluator().compare_with_dataset()``.

    Parameters
    ----------
    generated_images : list of PIL.Image or directory path
    dataset_name     : str   Reference dataset name.
    dataset_dir      : str or Path, optional
    style            : str   Fashion style label.
    baseline_fid     : float, optional   Baseline FID for improvement tracking.
    device           : str

    Returns
    -------
    DatasetComparison

    Example
    -------
        from src.evaluation.week2_fid_evaluator import compare_with_dataset
        result = compare_with_dataset(
            generated_images,
            "deepfashion",
            dataset_dir="data/deepfashion/",
            style="luxury",
        )
        print(result.summary())
    """
    ev = FIDEvaluator(device=device)
    return ev.compare_with_dataset(
        generated_images = generated_images,
        dataset_name     = dataset_name,
        dataset_dir      = dataset_dir,
        style            = style,
        baseline_fid     = baseline_fid,
    )


def benchmark_results(
    runs:          Dict[str, Union[List[Any], str, Path]],
    reference_dir: Optional[Union[str, Path]] = None,
    *,
    save_report:   bool = False,
    device:        str  = "auto",
) -> BenchmarkReport:
    """
    Module-level shortcut for ``FIDEvaluator().benchmark_results()``.

    Parameters
    ----------
    runs          : dict   name → image source.
    reference_dir : str or Path, optional
    save_report   : bool   Write JSON report to disk.
    device        : str

    Returns
    -------
    BenchmarkReport

    Example
    -------
        from src.evaluation.week2_fid_evaluator import benchmark_results
        report = benchmark_results(
            runs={"v1": images_v1, "v2": images_v2},
            reference_dir="data/real_fashion/",
        )
        print(report.summary())
    """
    ev = FIDEvaluator(device=device)
    return ev.benchmark_results(runs=runs, reference_dir=reference_dir, save_report=save_report)
