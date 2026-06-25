"""
week2/generator/batch_generator.py
=====================================
Production-Grade Batch Generation Framework
AI-Powered Fashion Design Assistant — Week 2

╔══════════════════════════════════════════════════════════════════╗
║                 BATCH GENERATION FRAMEWORK                       ║
║                                                                  ║
║  Reads a CSV of prompts, drives FashionSDXLGenerator in          ║
║  parallel threads, retries transient failures, tracks every      ║
║  result in ExperimentTracker, and writes a full summary          ║
║  report.                                                         ║
║                                                                  ║
║  Features                                                        ║
║  ─────────────────────────────────────────────────────────────   ║
║  1. Parallel processing  — ThreadPoolExecutor (configurable)     ║
║  2. Retry logic          — exponential back-off per item         ║
║  3. Structured logging   — loguru; per-batch rotating log file   ║
║  4. Metadata saving      — JSON sidecar per image + SQLite DB    ║
║  5. Progress tracking    — live counter + ETA + rich summary     ║
║  6. CSV input            — flexible column mapping               ║
║  7. CSV + JSON output    — full batch manifest                   ║
║                                                                  ║
║  CSV Input Schema                                                ║
║  ────────────────                                                ║
║  Required columns: prompt                                        ║
║  Optional columns: negative_prompt, seed, style, width, height, ║
║                    steps, guidance_scale, run_name, tags, notes  ║
║                                                                  ║
║  Public API                                                      ║
║  ──────────                                                      ║
║  BatchGenerator                                                  ║
║    .run_from_csv()   — load CSV → generate → save → report       ║
║    .run_from_list()  — list of dicts → same pipeline             ║
║    .generate_one()   — single item with retry                    ║
║    .export_manifest()— CSV + JSON manifest of completed batch    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Quick Start
-----------
    from src.generation.generator.batch_generator import BatchGenerator

    bg = BatchGenerator(max_workers=2, max_retries=3)
    report = bg.run_from_csv("prompts/my_batch.csv", run_name="v1_baseline")

    print(report.summary())
    print(f"Success rate: {report.success_rate:.1%}")

CSV Format
----------
    prompt,style,seed,width,height
    "a black oversized hoodie",streetwear,42,1024,1024
    "an emerald silk evening gown",luxury,7,1024,1024
    "tailored navy suit on a runway",formal,99,832,1216
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

try:
    from loguru import logger
    _LOGURU = True
except ImportError:
    import logging as _log
    logger = _log.getLogger("batch_generator")   # type: ignore[assignment]
    _LOGURU = False

# ── Optional rich progress (graceful fallback) ────────────────────────────────
try:
    from rich.console import Console as _RichConsole
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    _RICH = True
except ImportError:
    _RICH = False

# ── Internal imports (lazy so the module is importable without heavy deps) ────
try:
    from src.generation.generator.sdxl_generator import (
        FashionSDXLGenerator,
        GenerationOutput,
        SIZE_PRESETS,
    )
    _SDXL_AVAILABLE = True
except ImportError:
    FashionSDXLGenerator = None  # type: ignore[assignment,misc]
    GenerationOutput = None      # type: ignore[assignment]
    SIZE_PRESETS = {}            # type: ignore[assignment]
    _SDXL_AVAILABLE = False

try:
    from src.utils.experiment_tracker import ExperimentTracker
    _TRACKER_AVAILABLE = True
except ImportError:
    ExperimentTracker = None     # type: ignore[assignment]
    _TRACKER_AVAILABLE = False


# =============================================================================
# ── Constants
# =============================================================================

DEFAULT_OUTPUT_DIR      = Path("week2/outputs/generated")
DEFAULT_BATCH_DIR       = Path("week2/outputs/batch_runs")
DEFAULT_EXPERIMENTS_DIR = Path("week2/outputs/experiments")
DEFAULT_MAX_WORKERS     = 2
DEFAULT_MAX_RETRIES     = 3
DEFAULT_RETRY_DELAY     = 2.0   # seconds (exponential base)
DEFAULT_STEPS           = 30
DEFAULT_GUIDANCE        = 7.5
DEFAULT_WIDTH           = 1024
DEFAULT_HEIGHT          = 1024

# CSV column aliases — maps canonical names → accepted synonyms
CSV_COLUMN_ALIASES: Dict[str, List[str]] = {
    "prompt":          ["prompt", "text", "description", "caption"],
    "negative_prompt": ["negative_prompt", "negative", "neg_prompt"],
    "seed":            ["seed", "rng_seed", "random_seed"],
    "style":           ["style", "fashion_style", "category"],
    "width":           ["width", "w", "image_width"],
    "height":          ["height", "h", "image_height"],
    "steps":           ["steps", "num_steps", "inference_steps", "num_inference_steps"],
    "guidance_scale":  ["guidance_scale", "guidance", "cfg", "cfg_scale"],
    "run_name":        ["run_name", "run", "batch_name"],
    "tags":            ["tags", "tag", "labels"],
    "notes":           ["notes", "note", "comment"],
}


# =============================================================================
# ── BatchItem — single unit of work
# =============================================================================

@dataclass
class BatchItem:
    """
    A single generation task read from the CSV or passed programmatically.

    Attributes
    ----------
    item_id        : str    Auto-generated UUID (short).
    prompt         : str    Full generation prompt (required).
    negative_prompt: str    Negative prompt override.
    seed           : int    RNG seed (-1 = random per-run).
    style          : str    Fashion style tag.
    width          : int    Image width in pixels.
    height         : int    Image height in pixels.
    steps          : int    SDXL denoising steps.
    guidance_scale : float  Classifier-free guidance scale.
    run_name       : str    Human-readable run label.
    tags           : list   String tags for grouping.
    notes          : str    Free-text notes.
    row_index      : int    Source CSV row number (1-based).
    extra          : dict   Arbitrary extra CSV columns.
    """
    prompt:          str
    negative_prompt: str                     = ""
    seed:            int                     = -1
    style:           str                     = ""
    width:           int                     = DEFAULT_WIDTH
    height:          int                     = DEFAULT_HEIGHT
    steps:           int                     = DEFAULT_STEPS
    guidance_scale:  float                   = DEFAULT_GUIDANCE
    run_name:        str                     = ""
    tags:            List[str]               = field(default_factory=list)
    notes:           str                     = ""
    row_index:       int                     = 0
    extra:           Dict[str, Any]          = field(default_factory=dict)
    item_id:         str                     = field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "")[:10]
    )

    def resolved_seed(self) -> int:
        """Return seed; if -1, generate a random one."""
        return self.seed if self.seed != -1 else random.randint(0, 2 ** 32 - 1)


# =============================================================================
# ── ItemResult — outcome of a single generation attempt
# =============================================================================

@dataclass
class ItemResult:
    """
    Full outcome of processing one BatchItem.

    Attributes
    ----------
    item            : BatchItem
    success         : bool
    attempts        : int    Number of generation attempts made.
    generation_time : float  Wall-clock seconds for successful generation.
    total_time      : float  Including retries and saving.
    image_paths     : list   Saved file paths.
    experiment_id   : str    ID logged to ExperimentTracker (if available).
    clip_score      : float  CLIP score (if evaluator was attached).
    error           : str    Last exception message on failure.
    timestamp       : str    ISO-8601 UTC timestamp.
    metadata        : dict   Full GenerationOutput metadata dict.
    """
    item:            BatchItem
    success:         bool                     = False
    attempts:        int                      = 0
    generation_time: float                    = 0.0
    total_time:      float                    = 0.0
    image_paths:     List[Path]               = field(default_factory=list)
    experiment_id:   str                      = ""
    clip_score:      Optional[float]          = None
    error:           str                      = ""
    timestamp:       str                      = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata:        Dict[str, Any]           = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id":         self.item.item_id,
            "row_index":       self.item.row_index,
            "prompt":          self.item.prompt,
            "style":           self.item.style,
            "seed":            self.item.seed,
            "width":           self.item.width,
            "height":          self.item.height,
            "steps":           self.item.steps,
            "guidance_scale":  self.item.guidance_scale,
            "run_name":        self.item.run_name,
            "tags":            self.item.tags,
            "notes":           self.item.notes,
            "success":         self.success,
            "attempts":        self.attempts,
            "generation_time": round(self.generation_time, 3),
            "total_time":      round(self.total_time, 3),
            "image_paths":     [str(p) for p in self.image_paths],
            "experiment_id":   self.experiment_id,
            "clip_score":      self.clip_score,
            "error":           self.error,
            "timestamp":       self.timestamp,
            "metadata":        self.metadata,
        }

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "item_id":         self.item.item_id,
            "row_index":       str(self.item.row_index),
            "prompt":          self.item.prompt,
            "style":           self.item.style,
            "seed":            str(self.item.seed),
            "width":           str(self.item.width),
            "height":          str(self.item.height),
            "steps":           str(self.item.steps),
            "guidance_scale":  str(self.item.guidance_scale),
            "run_name":        self.item.run_name,
            "tags":            json.dumps(self.item.tags),
            "notes":           self.item.notes,
            "success":         str(self.success),
            "attempts":        str(self.attempts),
            "generation_time": str(round(self.generation_time, 3)),
            "total_time":      str(round(self.total_time, 3)),
            "image_paths":     json.dumps([str(p) for p in self.image_paths]),
            "experiment_id":   self.experiment_id,
            "clip_score":      str(self.clip_score) if self.clip_score is not None else "",
            "error":           self.error,
            "timestamp":       self.timestamp,
        }

    def summary(self) -> str:
        status = "OK" if self.success else f"FAIL({self.attempts} attempts)"
        return (
            f"[{self.item.item_id}] row={self.item.row_index:>3} | {status} | "
            f"time={self.generation_time:.1f}s | {self.item.prompt[:60]!r}"
        )

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# ── BatchReport — aggregate outcome of a full batch run
# =============================================================================

@dataclass
class BatchReport:
    """
    Aggregate report for a completed batch generation run.

    Attributes
    ----------
    run_name       : str
    run_id         : str    Auto-generated UUID.
    started_at     : str    ISO-8601 UTC.
    finished_at    : str
    total          : int    Total items submitted.
    succeeded      : int    Items that produced at least one image.
    failed         : int
    total_time     : float  Wall-clock seconds for the whole batch.
    mean_gen_time  : float  Mean generation time per successful item.
    results        : list   All ItemResult objects.
    manifest_csv   : Path   Written CSV manifest (if exported).
    manifest_json  : Path   Written JSON manifest (if exported).
    """
    run_name:      str                    = ""
    run_id:        str                    = field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "")[:10]
    )
    started_at:    str                    = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at:   str                    = ""
    total:         int                    = 0
    succeeded:     int                    = 0
    failed:        int                    = 0
    total_time:    float                  = 0.0
    mean_gen_time: float                  = 0.0
    results:       List[ItemResult]       = field(default_factory=list)
    manifest_csv:  Optional[Path]         = None
    manifest_json: Optional[Path]         = None

    @property
    def success_rate(self) -> float:
        return self.succeeded / max(self.total, 1)

    @property
    def failed_items(self) -> List[ItemResult]:
        return [r for r in self.results if not r.success]

    @property
    def successful_items(self) -> List[ItemResult]:
        return [r for r in self.results if r.success]

    def summary(self) -> str:
        sep  = "=" * 66
        thin = "-" * 66
        lines = [
            sep,
            "  BATCH GENERATION REPORT",
            sep,
            f"  Run           : {self.run_name or 'unnamed'} [{self.run_id}]",
            f"  Started       : {self.started_at}",
            f"  Finished      : {self.finished_at}",
            f"  Total time    : {self.total_time:.1f}s",
            "",
            f"  Total items   : {self.total}",
            f"  Succeeded [OK]: {self.succeeded}",
            f"  Failed [FAIL] : {self.failed}",
            f"  Success rate  : {self.success_rate:.1%}",
            f"  Mean gen time : {self.mean_gen_time:.2f}s / image",
            "",
        ]
        if self.manifest_csv:
            lines.append(f"  Manifest CSV  : {self.manifest_csv}")
        if self.manifest_json:
            lines.append(f"  Manifest JSON : {self.manifest_json}")
        if self.failed_items:
            lines += ["", "  FAILED ITEMS", thin]
            for r in self.failed_items:
                lines.append(f"  row {r.item.row_index:>3} | {r.error[:80]}")
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name":      self.run_name,
            "run_id":        self.run_id,
            "started_at":    self.started_at,
            "finished_at":   self.finished_at,
            "total":         self.total,
            "succeeded":     self.succeeded,
            "failed":        self.failed,
            "success_rate":  round(self.success_rate, 4),
            "total_time":    round(self.total_time, 3),
            "mean_gen_time": round(self.mean_gen_time, 3),
            "manifest_csv":  str(self.manifest_csv) if self.manifest_csv else None,
            "manifest_json": str(self.manifest_json) if self.manifest_json else None,
            "results":       [r.to_dict() for r in self.results],
        }

    def __repr__(self) -> str:
        return (
            f"BatchReport(run={self.run_name!r} | "
            f"{self.succeeded}/{self.total} OK | "
            f"rate={self.success_rate:.0%} | "
            f"time={self.total_time:.1f}s)"
        )


# =============================================================================
# ── BatchGenerationConfig
# =============================================================================

@dataclass
class BatchGenerationConfig:
    """
    Configuration for a BatchGenerator instance.

    Parameters
    ----------
    max_workers     : int    Parallel generation threads.
    max_retries     : int    Per-item retry attempts on transient failure.
    retry_delay     : float  Base delay (seconds) for exponential back-off.
    output_dir      : Path   Root directory for saved images.
    batch_dir       : Path   Directory for batch manifests and log files.
    experiments_dir : Path   SQLite database directory.
    default_steps   : int    SDXL denoising steps default.
    default_guidance: float  Classifier-free guidance scale default.
    default_width   : int    Default image width.
    default_height  : int    Default image height.
    save_images     : bool   Write PNG files to disk.
    save_metadata   : bool   Write JSON sidecar per image.
    track_experiments:bool   Log each result to ExperimentTracker.
    log_to_file     : bool   Write per-batch rotating log file.
    on_item_complete: callable  Optional callback(ItemResult) after each item.
    """
    max_workers:      int                         = DEFAULT_MAX_WORKERS
    max_retries:      int                         = DEFAULT_MAX_RETRIES
    retry_delay:      float                       = DEFAULT_RETRY_DELAY
    output_dir:       Path                        = DEFAULT_OUTPUT_DIR
    batch_dir:        Path                        = DEFAULT_BATCH_DIR
    experiments_dir:  Path                        = DEFAULT_EXPERIMENTS_DIR
    default_steps:    int                         = DEFAULT_STEPS
    default_guidance: float                       = DEFAULT_GUIDANCE
    default_width:    int                         = DEFAULT_WIDTH
    default_height:   int                         = DEFAULT_HEIGHT
    save_images:      bool                        = True
    save_metadata:    bool                        = True
    track_experiments:bool                        = True
    log_to_file:      bool                        = True
    on_item_complete: Optional[Callable]          = None


# =============================================================================
# ── Progress Tracker (thread-safe)
# =============================================================================

class _ProgressTracker:
    """
    Thread-safe progress counter with ETA estimation.
    Falls back to plain log lines when rich is unavailable.
    """

    def __init__(self, total: int, run_name: str = "") -> None:
        self.total      = total
        self.run_name   = run_name
        self._done      = 0
        self._succeeded = 0
        self._failed    = 0
        self._success   = False  # last item status (for display)
        self._lock      = threading.Lock()
        self._started   = time.perf_counter()
        self._progress  = None
        self._task_id   = None

        if _RICH:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=32),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
            self._task_id = self._progress.add_task(
                f"{run_name or 'batch'}", total=total
            )

    def __enter__(self):
        if self._progress:
            self._progress.start()
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()

    def update(self, success: bool, prompt: str = "") -> None:
        with self._lock:
            self._done      += 1
            if success:
                self._succeeded += 1
            else:
                self._failed    += 1
            done = self._done

        elapsed = time.perf_counter() - self._started
        eta_s   = (elapsed / done) * (self.total - done) if done > 0 else 0.0

        status = "✓" if success else "✗"
        short  = (prompt[:50] + "...") if len(prompt) > 50 else prompt

        if self._progress and self._task_id is not None:
            label = "OK" if success else "FAIL"
            self._progress.update(
                self._task_id,
                advance=1,
                description=(
                    f"{label} {done}/{self.total} ETA {eta_s:.0f}s"
                ),
            )
        else:
            label = "OK" if success else "FAIL"
            logger.info(
                "Progress {}/{} ({:.0%}) {} | ETA {:.0f}s | {}",
                done, self.total, done / self.total, label, eta_s, short,
            )

    @property
    def done(self) -> int:
        return self._done

    @property
    def succeeded(self) -> int:
        return self._succeeded

    @property
    def failed(self) -> int:
        return self._failed


# =============================================================================
# ── BatchGenerator — core class
# =============================================================================

class BatchGenerator:
    """
    Production-grade batch fashion image generation framework.

    Features
    --------
    1. **Parallel processing** — ThreadPoolExecutor with configurable workers.
    2. **Retry logic**         — Exponential back-off per item (configurable).
    3. **Structured logging**  — loguru; optional per-batch rotating log file.
    4. **Metadata saving**     — JSON sidecar per image + ExperimentTracker SQLite.
    5. **Progress tracking**   — Thread-safe counter with ETA (rich bar or log).
    6. **CSV input**           — Flexible column mapping; required: ``prompt``.
    7. **CSV + JSON output**   — Full batch manifest written to ``batch_dir``.

    Parameters
    ----------
    generator       : FashionSDXLGenerator, optional
        Pre-loaded SDXL generator. If None, a stub is used in test mode.
    config          : BatchGenerationConfig, optional
    max_workers     : int    Thread pool size (overrides config).
    max_retries     : int    Per-item retries (overrides config).
    output_dir      : str or Path
    experiments_dir : str or Path

    Example
    -------
        bg = BatchGenerator(max_workers=2, max_retries=3)
        report = bg.run_from_csv("prompts/batch.csv", run_name="v1")
        print(report.summary())
        bg.export_manifest(report)
    """

    def __init__(
        self,
        generator:       Optional[Any]                    = None,
        config:          Optional[BatchGenerationConfig]  = None,
        *,
        max_workers:     int                              = DEFAULT_MAX_WORKERS,
        max_retries:     int                              = DEFAULT_MAX_RETRIES,
        retry_delay:     float                            = DEFAULT_RETRY_DELAY,
        output_dir:      Optional[Union[str, Path]]       = None,
        batch_dir:       Optional[Union[str, Path]]       = None,
        experiments_dir: Optional[Union[str, Path]]       = None,
        save_images:     bool                             = True,
        save_metadata:   bool                             = True,
        track_experiments: bool                           = True,
        log_to_file:     bool                             = False,
        on_item_complete: Optional[Callable]              = None,
    ) -> None:
        self.config = config or BatchGenerationConfig(
            max_workers      = max_workers,
            max_retries      = max_retries,
            retry_delay      = retry_delay,
            output_dir       = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR,
            batch_dir        = Path(batch_dir)  if batch_dir  else DEFAULT_BATCH_DIR,
            experiments_dir  = Path(experiments_dir) if experiments_dir else DEFAULT_EXPERIMENTS_DIR,
            save_images      = save_images,
            save_metadata    = save_metadata,
            track_experiments= track_experiments,
            log_to_file      = log_to_file,
            on_item_complete = on_item_complete,
        )

        # Directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.batch_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Generator (may be None; lazy-loaded or injected)
        self._generator   = generator
        self._gen_lock    = threading.Lock()

        # Experiment tracker (optional)
        self._tracker: Optional[Any] = None
        if self.config.track_experiments and _TRACKER_AVAILABLE:
            try:
                self._tracker = ExperimentTracker(
                    experiments_dir=self.config.experiments_dir
                )
            except Exception as exc:
                logger.warning("ExperimentTracker init failed: {}", exc)

        logger.info(
            "BatchGenerator initialised | workers={} | retries={} | "
            "output={} | tracker={}",
            self.config.max_workers, self.config.max_retries,
            self.config.output_dir,
            "enabled" if self._tracker is not None else "disabled",
        )

    # =========================================================================
    # ── Primary Public API
    # =========================================================================

    def run_from_csv(
        self,
        csv_path:        Union[str, Path],
        *,
        run_name:        str                        = "",
        default_style:   str                        = "",
        default_seed:    int                        = -1,
        column_map:      Optional[Dict[str, str]]   = None,
        skip_header:     bool                       = True,
        export:          bool                       = True,
    ) -> BatchReport:
        """
        Load a CSV file of prompts and generate fashion images for each row.

        CSV required column: ``prompt``
        Optional columns:    ``negative_prompt``, ``seed``, ``style``,
                             ``width``, ``height``, ``steps``,
                             ``guidance_scale``, ``run_name``, ``tags``, ``notes``

        Parameters
        ----------
        csv_path     : str or Path   Path to input CSV file.
        run_name     : str           Human-readable batch label.
        default_style: str           Style override for rows without a style column.
        default_seed : int           Seed override (-1 = random per row).
        column_map   : dict, optional  ``{canonical_name: actual_csv_column}``
                       mapping for non-standard CSV headers.
        skip_header  : bool          True if first row is a header (default True).
        export       : bool          Write CSV + JSON manifests on completion.

        Returns
        -------
        BatchReport

        Example
        -------
            report = bg.run_from_csv(
                "prompts/luxury_batch.csv",
                run_name     = "luxury_v2",
                default_style= "luxury",
            )
            print(report.summary())
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        items = self._load_csv(
            csv_path,
            run_name     = run_name,
            default_style= default_style,
            default_seed = default_seed,
            column_map   = column_map or {},
        )
        logger.info("CSV loaded | {} rows | {}", len(items), csv_path)

        report = self.run_from_list(items, run_name=run_name, export=export)
        return report

    def run_from_list(
        self,
        items:    List[Union[BatchItem, Dict[str, Any]]],
        *,
        run_name: str  = "",
        export:   bool = True,
    ) -> BatchReport:
        """
        Generate images for a list of BatchItem objects or dicts.

        Parameters
        ----------
        items    : list of BatchItem or dict
            Each dict is coerced to a BatchItem; must contain ``"prompt"``.
        run_name : str   Human-readable batch label.
        export   : bool  Write manifests.

        Returns
        -------
        BatchReport

        Example
        -------
            report = bg.run_from_list([
                {"prompt": "a black hoodie", "seed": 42, "style": "streetwear"},
                {"prompt": "a red gown",     "seed": 7,  "style": "luxury"},
            ], run_name="test_batch")
        """
        # Coerce dicts → BatchItem
        batch_items: List[BatchItem] = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                batch_items.append(self._dict_to_item(item, row_index=idx, run_name=run_name))
            else:
                batch_items.append(item)

        n = len(batch_items)
        logger.info("Starting batch | run={!r} | n={} | workers={}", run_name, n, self.config.max_workers)

        report = BatchReport(run_name=run_name, total=n)
        t0 = time.perf_counter()

        # Optional log file for this batch
        log_path: Optional[Path] = None
        if self.config.log_to_file and _LOGURU:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = self.config.batch_dir / f"batch_{run_name}_{ts}.log"
            log_id   = logger.add(str(log_path), level="DEBUG", rotation="50 MB")

        try:
            results = self._run_parallel(batch_items, run_name=run_name)
        finally:
            if log_path and _LOGURU:
                logger.remove(log_id)

        # ── Populate report ───────────────────────────────────────────────
        report.results   = results
        report.succeeded = sum(1 for r in results if r.success)
        report.failed    = sum(1 for r in results if not r.success)
        report.total_time= round(time.perf_counter() - t0, 3)
        report.finished_at = datetime.now(timezone.utc).isoformat()

        gen_times = [r.generation_time for r in results if r.success]
        report.mean_gen_time = round(sum(gen_times) / len(gen_times), 3) if gen_times else 0.0

        logger.info(
            "Batch complete | run={!r} | {}/{} OK | {:.1f}s total | mean={:.2f}s/img",
            run_name, report.succeeded, n, report.total_time, report.mean_gen_time,
        )

        if export:
            self.export_manifest(report)

        return report

    def generate_one(
        self,
        item: BatchItem,
        *,
        attempt_override: Optional[int] = None,
    ) -> ItemResult:
        """
        Generate images for a single BatchItem, with retry logic.

        Retries up to ``config.max_retries`` times with exponential back-off
        on any exception. Each retry uses a fresh random seed offset.

        Parameters
        ----------
        item             : BatchItem
        attempt_override : int, optional  Override max retries for this call.

        Returns
        -------
        ItemResult

        Example
        -------
            result = bg.generate_one(
                BatchItem(prompt="a hoodie", seed=42, style="streetwear")
            )
            print(result.success, result.generation_time)
        """
        max_attempts = attempt_override or self.config.max_retries
        t0           = time.perf_counter()
        last_error   = ""
        result       = ItemResult(item=item)

        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt
            try:
                seed = item.resolved_seed()
                # Offset seed on retries so we don't repeat the same failure
                if attempt > 1:
                    seed = (seed + attempt * 1000) % (2 ** 32)
                    logger.info(
                        "[{}] Retry {}/{} | seed={} | {}",
                        item.item_id, attempt, max_attempts, seed, item.prompt[:60],
                    )

                gen_out = self._call_generator(item, seed)

                # ── Success path ─────────────────────────────────────────
                result.success         = True
                result.generation_time = gen_out.generation_time_s if hasattr(gen_out, "generation_time_s") else (time.perf_counter() - t0)
                result.metadata        = gen_out.metadata if hasattr(gen_out, "metadata") else {}

                # Save images
                if self.config.save_images or self.config.save_metadata:
                    result.image_paths = self._save_generation(gen_out, item)

                # Track in ExperimentTracker
                if self._tracker is not None:
                    result.experiment_id = self._log_to_tracker(item, result, gen_out)

                break   # ← success, exit retry loop

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[{}] Attempt {}/{} failed: {}",
                    item.item_id, attempt, max_attempts, last_error,
                )
                if attempt < max_attempts:
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    logger.debug("[{}] Back-off {:.1f}s", item.item_id, delay)
                    time.sleep(delay)

        if not result.success:
            result.error      = last_error
            result.total_time = round(time.perf_counter() - t0, 3)
            logger.error(
                "[{}] All {} attempts failed | {} | {}",
                item.item_id, max_attempts, item.prompt[:60], last_error,
            )

            # Log failed run to tracker too
            if self._tracker is not None:
                try:
                    self._tracker.log_experiment(
                        prompt         = item.prompt,
                        seed           = item.seed,
                        model_version  = self._model_version(),
                        generation_time= result.total_time,
                        style          = item.style,
                        run_name       = item.run_name,
                        tags           = item.tags,
                        notes          = f"FAILED: {last_error}",
                        status         = "failed",
                    )
                except Exception:
                    pass
        else:
            result.total_time = round(time.perf_counter() - t0, 3)

        return result

    # =========================================================================
    # ── Export API
    # =========================================================================

    def export_manifest(
        self,
        report:   BatchReport,
        *,
        out_dir:  Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Write a CSV and JSON manifest for a completed batch run.

        Parameters
        ----------
        report  : BatchReport
        out_dir : Path, optional  Defaults to ``config.batch_dir``.

        Returns
        -------
        (csv_path, json_path) — absolute Paths of written files.

        Example
        -------
            csv_path, json_path = bg.export_manifest(report)
        """
        out_dir = Path(out_dir) if out_dir else self.config.batch_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug     = (report.run_name or "batch").replace(" ", "_")[:30]
        base     = f"manifest_{slug}_{report.run_id}_{ts}"

        csv_path  = out_dir / f"{base}.csv"
        json_path = out_dir / f"{base}.json"

        # ── CSV ──────────────────────────────────────────────────────────
        csv_fields = list(report.results[0].to_csv_row().keys()) if report.results else []
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields or [], extrasaction="ignore")
            writer.writeheader()
            for r in report.results:
                writer.writerow(r.to_csv_row())

        # ── JSON ─────────────────────────────────────────────────────────
        json_payload = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "report":        report.to_dict(),
        }
        json_path.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        report.manifest_csv  = csv_path
        report.manifest_json = json_path

        logger.info("Manifest written | CSV={} | JSON={}", csv_path, json_path)
        return csv_path, json_path

    def to_csv_string(self, report: BatchReport) -> str:
        """Return the batch manifest as an in-memory CSV string."""
        if not report.results:
            return ""
        fields = list(report.results[0].to_csv_row().keys())
        buf    = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in report.results:
            writer.writerow(r.to_csv_row())
        return buf.getvalue()

    def to_json_string(self, report: BatchReport, indent: int = 2) -> str:
        """Return the batch manifest as an in-memory JSON string."""
        return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)

    # =========================================================================
    # ── CSV Parsing
    # =========================================================================

    def _load_csv(
        self,
        path:          Path,
        *,
        run_name:      str,
        default_style: str,
        default_seed:  int,
        column_map:    Dict[str, str],
    ) -> List[BatchItem]:
        """Parse a CSV file into a list of BatchItem objects."""
        items: List[BatchItem] = []

        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty or has no headers: {path}")

            # Build reverse lookup: actual CSV header → canonical name
            header_lower = {h.lower().strip(): h for h in reader.fieldnames}
            resolved: Dict[str, str] = {}

            # Apply user-provided column_map first
            for canon, actual in column_map.items():
                if actual in reader.fieldnames:
                    resolved[canon] = actual

            # Auto-detect remaining columns from aliases
            for canon, aliases in CSV_COLUMN_ALIASES.items():
                if canon in resolved:
                    continue
                for alias in aliases:
                    if alias in header_lower:
                        resolved[canon] = header_lower[alias]
                        break

            if "prompt" not in resolved:
                raise ValueError(
                    f"CSV must have a 'prompt' column. "
                    f"Found headers: {list(reader.fieldnames)}"
                )

            def _get(row: Dict, canon: str, default: Any = "") -> Any:
                col = resolved.get(canon)
                return row.get(col, default).strip() if col else default

            for row_idx, row in enumerate(reader, start=2):
                prompt = _get(row, "prompt").strip()
                if not prompt:
                    logger.debug("Row {} skipped — empty prompt", row_idx)
                    continue

                # Parse seed
                seed_raw = _get(row, "seed", str(default_seed))
                try:
                    seed = int(seed_raw) if seed_raw else default_seed
                except ValueError:
                    seed = default_seed

                # Parse size
                try:
                    w = int(_get(row, "width", str(self.config.default_width)))
                except ValueError:
                    w = self.config.default_width
                try:
                    h = int(_get(row, "height", str(self.config.default_height)))
                except ValueError:
                    h = self.config.default_height

                # Parse steps
                try:
                    steps = int(_get(row, "steps", str(self.config.default_steps)))
                except ValueError:
                    steps = self.config.default_steps

                # Parse guidance
                try:
                    guidance = float(_get(row, "guidance_scale", str(self.config.default_guidance)))
                except ValueError:
                    guidance = self.config.default_guidance

                # Tags: accept "tag1,tag2" or JSON array
                raw_tags = _get(row, "tags", "")
                if raw_tags.startswith("["):
                    try:
                        tags = json.loads(raw_tags)
                    except json.JSONDecodeError:
                        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                else:
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

                # Remaining unresolved columns → extra
                resolved_vals = set(resolved.values())
                extra = {
                    k: v for k, v in row.items()
                    if k not in resolved_vals and v
                }

                items.append(BatchItem(
                    prompt          = prompt,
                    negative_prompt = _get(row, "negative_prompt"),
                    seed            = seed,
                    style           = _get(row, "style") or default_style,
                    width           = w,
                    height          = h,
                    steps           = steps,
                    guidance_scale  = guidance,
                    run_name        = _get(row, "run_name") or run_name,
                    tags            = tags,
                    notes           = _get(row, "notes"),
                    row_index       = row_idx,
                    extra           = extra,
                ))

        return items

    # =========================================================================
    # ── Parallel Execution Engine
    # =========================================================================

    def _run_parallel(
        self,
        items:    List[BatchItem],
        run_name: str,
    ) -> List[ItemResult]:
        """
        Submit all items to the thread pool and collect results in order.

        Thread-safety notes
        -------------------
        • Each worker calls ``generate_one()`` independently.
        • The underlying FashionSDXLGenerator pipeline is NOT thread-safe
          (PyTorch CUDA context is single-threaded by default).
          We acquire ``_gen_lock`` around each generation call so that only
          one thread runs inference at a time, while I/O (save, metadata)
          happens concurrently.
        • If max_workers=1, this degrades to simple sequential processing.
        """
        n       = len(items)
        results: List[Optional[ItemResult]] = [None] * n

        with _ProgressTracker(n, run_name=run_name) as progress:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                future_to_idx: Dict[Future, int] = {
                    pool.submit(self.generate_one, item): idx
                    for idx, item in enumerate(items)
                }

                for future in as_completed(future_to_idx):
                    idx    = future_to_idx[future]
                    item   = items[idx]
                    try:
                        result = future.result()
                    except Exception as exc:
                        # Should not happen (generate_one catches all), but safety net
                        result = ItemResult(
                            item    = item,
                            success = False,
                            error   = f"Unhandled: {exc}",
                            attempts= self.config.max_retries,
                        )

                    results[idx] = result
                    progress.update(result.success, item.prompt)

                    # Fire optional callback
                    if self.config.on_item_complete:
                        try:
                            self.config.on_item_complete(result)
                        except Exception as cb_exc:
                            logger.warning("on_item_complete callback raised: {}", cb_exc)

        # Fill any None gaps (shouldn't happen)
        return [r if r is not None else ItemResult(item=items[i], error="lost") for i, r in enumerate(results)]

    # =========================================================================
    # ── Generator Wrapper
    # =========================================================================

    def _call_generator(self, item: BatchItem, seed: int) -> Any:
        """
        Call FashionSDXLGenerator.generate_image() under the generation lock.
        Falls back to a stub GenerationOutput when the generator is unavailable.
        """
        with self._gen_lock:
            if self._generator is None or not _SDXL_AVAILABLE:
                return self._stub_generation(item, seed)

            return self._generator.generate_image(
                prompt          = item.prompt,
                negative_prompt = item.negative_prompt or None,
                width           = item.width,
                height          = item.height,
                seed            = seed,
                num_inference_steps = item.steps,
                guidance_scale  = item.guidance_scale,
            )

    def _stub_generation(self, item: BatchItem, seed: int) -> Any:
        """
        Stub GenerationOutput for testing without a real SDXL pipeline.
        Returns a simple object with the expected interface.
        """
        class _StubOutput:
            def __init__(self):
                self.images            = []
                self.image_paths       = []
                self.metadata          = {
                    "prompt": item.prompt,
                    "seed":   seed,
                    "width":  item.width,
                    "height": item.height,
                    "steps":  item.steps,
                    "stub":   True,
                }
                self.generation_time_s = 0.01
                self.success           = True
                self.error             = None
                self.model_id          = "stub"
        return _StubOutput()

    # =========================================================================
    # ── Save Helpers
    # =========================================================================

    def _save_generation(self, gen_out: Any, item: BatchItem) -> List[Path]:
        """Save images + metadata sidecars for a GenerationOutput."""
        saved: List[Path] = []

        images = getattr(gen_out, "images", [])
        if not images:
            return saved

        ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        style = (item.style or "general").replace(" ", "_")[:20]

        for i, img in enumerate(images):
            img_id   = f"{item.item_id}_{i}"
            filename = f"{style}_{img_id}_{ts}.png"
            img_path = self.config.output_dir / filename

            # Save PNG
            if self.config.save_images and hasattr(img, "save"):
                try:
                    img.save(str(img_path))
                    saved.append(img_path)
                    logger.debug("Saved image: {}", img_path)
                except Exception as exc:
                    logger.error("Failed to save image: {}", exc)

            # Save JSON sidecar
            if self.config.save_metadata:
                meta_path = img_path.with_suffix(".json")
                meta = {
                    "image_id":       img_id,
                    "filename":       filename,
                    "prompt":         item.prompt,
                    "negative_prompt":item.negative_prompt,
                    "seed":           item.seed,
                    "style":          item.style,
                    "width":          item.width,
                    "height":         item.height,
                    "steps":          item.steps,
                    "guidance_scale": item.guidance_scale,
                    "run_name":       item.run_name,
                    "tags":           item.tags,
                    "notes":          item.notes,
                    "model":          getattr(gen_out, "model_id", "unknown"),
                    "generated_at":   datetime.now(timezone.utc).isoformat(),
                    **({} if not hasattr(gen_out, "metadata") else gen_out.metadata),
                }
                try:
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    logger.debug("Saved metadata: {}", meta_path)
                except Exception as exc:
                    logger.error("Failed to save metadata sidecar: {}", exc)

        return saved

    # =========================================================================
    # ── ExperimentTracker Integration
    # =========================================================================

    def _log_to_tracker(
        self,
        item:    BatchItem,
        result:  ItemResult,
        gen_out: Any,
    ) -> str:
        """Log a successful generation to ExperimentTracker. Returns experiment_id."""
        if self._tracker is None:
            return ""
        try:
            meta = getattr(gen_out, "metadata", {})
            return self._tracker.log_experiment(
                prompt          = item.prompt,
                seed            = item.seed,
                model_version   = self._model_version(),
                generation_time = result.generation_time,
                style           = item.style,
                run_name        = item.run_name,
                tags            = list(item.tags),
                notes           = item.notes,
                status          = "completed",
                image_width     = item.width,
                image_height    = item.height,
                num_inference_steps = item.steps,
                guidance_scale  = item.guidance_scale,
                clip_score      = result.clip_score,
                extra           = {
                    "item_id":    item.item_id,
                    "row_index":  item.row_index,
                    "image_paths":[str(p) for p in result.image_paths],
                    **({k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}),
                },
            )
        except Exception as exc:
            logger.warning("ExperimentTracker log failed: {}", exc)
            return ""

    def _model_version(self) -> str:
        if self._generator and hasattr(self._generator, "model_id"):
            return self._generator.model_id
        return "stub"

    # =========================================================================
    # ── Utilities
    # =========================================================================

    @staticmethod
    def _dict_to_item(d: Dict[str, Any], row_index: int, run_name: str) -> BatchItem:
        """Coerce a plain dict to a BatchItem."""
        tags_raw = d.get("tags", [])
        if isinstance(tags_raw, str):
            tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]

        return BatchItem(
            prompt          = str(d.get("prompt", "")),
            negative_prompt = str(d.get("negative_prompt", "")),
            seed            = int(d.get("seed", -1)),
            style           = str(d.get("style", "")),
            width           = int(d.get("width",  DEFAULT_WIDTH)),
            height          = int(d.get("height", DEFAULT_HEIGHT)),
            steps           = int(d.get("steps",  DEFAULT_STEPS)),
            guidance_scale  = float(d.get("guidance_scale", DEFAULT_GUIDANCE)),
            run_name        = str(d.get("run_name", run_name)),
            tags            = list(tags_raw),
            notes           = str(d.get("notes", "")),
            row_index       = row_index,
            extra           = {k: v for k, v in d.items()
                               if k not in {"prompt","negative_prompt","seed","style",
                                             "width","height","steps","guidance_scale",
                                             "run_name","tags","notes"}},
        )

    @property
    def tracker(self) -> Optional[Any]:
        """Expose the ExperimentTracker for external use."""
        return self._tracker

    def __repr__(self) -> str:
        return (
            f"BatchGenerator("
            f"workers={self.config.max_workers} | "
            f"retries={self.config.max_retries} | "
            f"tracker={'on' if self._tracker is not None else 'off'})"
        )

    def __len__(self) -> int:
        """Number of experiments tracked so far."""
        return len(self._tracker) if self._tracker is not None else 0


# =============================================================================
# ── Module-Level Convenience
# =============================================================================

def run_batch_from_csv(
    csv_path:    Union[str, Path],
    *,
    run_name:    str                        = "",
    max_workers: int                        = DEFAULT_MAX_WORKERS,
    max_retries: int                        = DEFAULT_MAX_RETRIES,
    output_dir:  Optional[Union[str, Path]] = None,
    generator:   Optional[Any]             = None,
    export:      bool                       = True,
) -> BatchReport:
    """
    One-liner convenience wrapper — load CSV and run a full batch.

    Parameters
    ----------
    csv_path    : str or Path
    run_name    : str
    max_workers : int
    max_retries : int
    output_dir  : str or Path, optional
    generator   : FashionSDXLGenerator, optional
    export      : bool

    Returns
    -------
    BatchReport

    Example
    -------
        from src.generation.generator.batch_generator import run_batch_from_csv

        report = run_batch_from_csv("prompts/batch.csv", run_name="v1")
        print(report.summary())
    """
    bg = BatchGenerator(
        generator   = generator,
        max_workers = max_workers,
        max_retries = max_retries,
        output_dir  = output_dir,
    )
    return bg.run_from_csv(csv_path, run_name=run_name, export=export)


def run_batch_from_list(
    items:       List[Dict[str, Any]],
    *,
    run_name:    str                        = "",
    max_workers: int                        = DEFAULT_MAX_WORKERS,
    max_retries: int                        = DEFAULT_MAX_RETRIES,
    output_dir:  Optional[Union[str, Path]] = None,
    generator:   Optional[Any]             = None,
    export:      bool                       = True,
) -> BatchReport:
    """
    One-liner convenience wrapper — list of dicts → batch run.

    Example
    -------
        report = run_batch_from_list([
            {"prompt": "a hoodie", "style": "streetwear", "seed": 42},
            {"prompt": "a gown",   "style": "luxury",     "seed":  7},
        ], run_name="my_batch")
    """
    bg = BatchGenerator(
        generator   = generator,
        max_workers = max_workers,
        max_retries = max_retries,
        output_dir  = output_dir,
    )
    return bg.run_from_list(items, run_name=run_name, export=export)
