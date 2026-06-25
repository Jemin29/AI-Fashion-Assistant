"""
week2/evaluation/experiment_tracker.py
========================================
Production-Grade Experiment Tracking System
AI-Powered Fashion Design Assistant — Week 2

Tracks every fashion generation run with full metadata, evaluation scores,
and provenance — stored as a persistent SQLite database and exported to
CSV / JSON on demand.

╔══════════════════════════════════════════════════════════════════╗
║              EXPERIMENT TRACKING SYSTEM                         ║
║                                                                  ║
║  Tracks per-experiment:                                          ║
║   1. Prompt           — full text + style metadata              ║
║   2. Seed             — RNG seed for full reproducibility        ║
║   3. Model Version    — SDXL model ID + pipeline config          ║
║   4. CLIP Score       — prompt-image semantic similarity         ║
║   5. FID Score        — distributional quality vs reference      ║
║   6. Generation Time  — wall-clock time in seconds              ║
║                                                                  ║
║  Storage                                                         ║
║   • SQLite DB  → outputs/experiments/experiments.db             ║
║   • JSON       → outputs/experiments/export_<ts>.json           ║
║   • CSV        → outputs/experiments/export_<ts>.csv            ║
║                                                                  ║
║  Public API                                                      ║
║   ExperimentTracker                                              ║
║     .log_experiment()    — record a single run                   ║
║     .log_batch()         — record multiple runs at once          ║
║     .get_experiment()    — retrieve by ID                        ║
║     .list_experiments()  — query/filter experiments              ║
║     .best_experiment()   — find best by metric                   ║
║     .compare_experiments()— side-by-side diff                    ║
║     .export_csv()        — write CSV file                        ║
║     .export_json()       — write JSON file                       ║
║     .generate_report()   — rich summary report                   ║
║     .delete_experiment() — remove a record                       ║
║     .clear_all()         — wipe the database                     ║
╚══════════════════════════════════════════════════════════════════╝

Quick Start
-----------
    from src.utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()

    exp_id = tracker.log_experiment(
        prompt          = "a black oversized streetwear hoodie",
        seed            = 42,
        model_version   = "stabilityai/stable-diffusion-xl-base-1.0",
        clip_score      = 0.91,
        fid_score       = 28.4,
        generation_time = 4.32,
        style           = "streetwear",
        tags            = ["batch_01", "baseline"],
    )

    # Export
    tracker.export_csv("my_experiments.csv")
    tracker.export_json("my_experiments.json")

    # Query best
    best = tracker.best_experiment(metric="clip_score")
    print(best.clip_score)

    # Full report
    print(tracker.generate_report())
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging as _log
    logger = _log.getLogger("experiment_tracker")  # type: ignore[assignment]

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None   # type: ignore[assignment]
    _NUMPY = False


# =============================================================================
# ── Constants
# =============================================================================

DEFAULT_EXPERIMENTS_DIR = Path("week2/outputs/experiments")
DEFAULT_DB_NAME         = "experiments.db"

# Metric keys tracked in every experiment
TRACKED_METRICS = ("clip_score", "fid_score", "generation_time",
                   "quality_score", "sharpness", "brightness",
                   "contrast", "color_distribution", "noise_level")

# CSV column order
CSV_COLUMNS = [
    "experiment_id", "run_name", "timestamp",
    "prompt", "style", "seed", "model_version", "pipeline_config",
    "clip_score", "fid_score", "generation_time",
    "quality_score", "quality_rating",
    "sharpness", "brightness", "contrast", "color_distribution", "noise_level",
    "image_width", "image_height", "num_inference_steps", "guidance_scale",
    "tags", "notes", "status",
]

# SQL schema
_SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    run_name            TEXT,
    timestamp           TEXT NOT NULL,
    prompt              TEXT NOT NULL,
    style               TEXT,
    seed                INTEGER,
    model_version       TEXT,
    pipeline_config     TEXT,       -- JSON blob
    clip_score          REAL,
    fid_score           REAL,
    generation_time     REAL,
    quality_score       REAL,
    quality_rating      TEXT,
    sharpness           REAL,
    brightness          REAL,
    contrast            REAL,
    color_distribution  REAL,
    noise_level         REAL,
    image_width         INTEGER,
    image_height        INTEGER,
    num_inference_steps INTEGER,
    guidance_scale      REAL,
    tags                TEXT,       -- JSON array
    notes               TEXT,
    status              TEXT DEFAULT 'completed',
    extra               TEXT        -- JSON blob for extensibility
);
"""

_SQL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_timestamp     ON experiments(timestamp);
CREATE INDEX IF NOT EXISTS idx_run_name      ON experiments(run_name);
CREATE INDEX IF NOT EXISTS idx_clip_score    ON experiments(clip_score);
CREATE INDEX IF NOT EXISTS idx_fid_score     ON experiments(fid_score);
CREATE INDEX IF NOT EXISTS idx_style         ON experiments(style);
"""


# =============================================================================
# ── ExperimentRecord Dataclass
# =============================================================================

@dataclass
class ExperimentRecord:
    """
    A complete record of a single generation experiment.

    Core tracked fields (required)
    ------------------------------
    experiment_id    : str    Auto-generated UUID (short).
    timestamp        : str    ISO-8601 UTC timestamp.
    prompt           : str    Full generation prompt.
    seed             : int    RNG seed used for generation.
    model_version    : str    Model ID (e.g. "stabilityai/stable-diffusion-xl-base-1.0").
    clip_score       : float  CLIP cosine similarity [0, 1].
    fid_score        : float  FID score (lower = better; None if not computed).
    generation_time  : float  Wall-clock seconds for generation.

    Optional metadata fields
    ------------------------
    run_name              : str    Human-readable run label.
    style                 : str    Fashion style (streetwear / luxury / ...).
    pipeline_config       : dict   Full pipeline config snapshot.
    quality_score         : float  Composite quality score [0, 100].
    quality_rating        : str    "excellent" / "very good" / "good" / ...
    sharpness             : float  Image sharpness score [0, 100].
    brightness            : float  Brightness score [0, 100].
    contrast              : float  Contrast score [0, 100].
    color_distribution    : float  Colour distribution score [0, 100].
    noise_level           : float  Noise level score [0, 100].
    image_width / height  : int    Output image dimensions.
    num_inference_steps   : int    SDXL denoising steps.
    guidance_scale        : float  Classifier-free guidance scale.
    tags                  : list   Arbitrary string tags.
    notes                 : str    Free-text notes.
    status                : str    "completed" | "failed" | "pending".
    extra                 : dict   Arbitrary extra data (extensible).
    """
    # ── Core required fields ───────────────────────────────────────────────
    prompt:              str
    seed:                int                     = 0
    model_version:       str                     = ""
    clip_score:          Optional[float]          = None
    fid_score:           Optional[float]          = None
    generation_time:     float                    = 0.0
    # ── Identity ────────────────────────────────────────────────────────────
    experiment_id:       str                     = field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "")[:12]
    )
    run_name:            str                     = ""
    timestamp:           str                     = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # ── Style / metadata ────────────────────────────────────────────────────
    style:               str                     = ""
    pipeline_config:     Dict[str, Any]          = field(default_factory=dict)
    # ── Image quality scores ─────────────────────────────────────────────────
    quality_score:       Optional[float]          = None
    quality_rating:      str                     = ""
    sharpness:           Optional[float]          = None
    brightness:          Optional[float]          = None
    contrast:            Optional[float]          = None
    color_distribution:  Optional[float]          = None
    noise_level:         Optional[float]          = None
    # ── Generation parameters ────────────────────────────────────────────────
    image_width:         Optional[int]            = None
    image_height:        Optional[int]            = None
    num_inference_steps: Optional[int]            = None
    guidance_scale:      Optional[float]          = None
    # ── Provenance ───────────────────────────────────────────────────────────
    tags:                List[str]               = field(default_factory=list)
    notes:               str                     = ""
    status:              str                     = "completed"
    extra:               Dict[str, Any]          = field(default_factory=dict)

    # =========================================================================
    # ── Computed Properties
    # =========================================================================

    @property
    def composite_score(self) -> Optional[float]:
        """
        Weighted composite score combining CLIP and quality dimensions.

        Formula (all normalised to [0, 1] before weighting):
          0.35 × clip_score  +  0.25 × quality_score/100
          + 0.20 × (1 - fid_score/200 clamped)
          + 0.10 × sharpness/100  +  0.10 × (1 - noise_level_inv)
        """
        scores: List[Tuple[float, float]] = []
        if self.clip_score is not None:
            scores.append((self.clip_score, 0.35))
        if self.quality_score is not None:
            scores.append((self.quality_score / 100.0, 0.25))
        if self.fid_score is not None:
            fid_component = max(0.0, 1.0 - min(self.fid_score, 200.0) / 200.0)
            scores.append((fid_component, 0.20))
        if self.sharpness is not None:
            scores.append((self.sharpness / 100.0, 0.10))
        if self.noise_level is not None:
            scores.append((self.noise_level / 100.0, 0.10))
        if not scores:
            return None
        total_w = sum(w for _, w in scores)
        return round(sum(s * w for s, w in scores) / total_w, 4)

    @property
    def passed(self) -> bool:
        """True if CLIP score ≥ 0.20 and status is completed."""
        clip_ok = self.clip_score is None or self.clip_score >= 0.20
        return clip_ok and self.status == "completed"

    # =========================================================================
    # ── Serialisation
    # =========================================================================

    def to_dict(self, include_composite: bool = True) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        d: Dict[str, Any] = {
            "experiment_id":      self.experiment_id,
            "run_name":           self.run_name,
            "timestamp":          self.timestamp,
            "prompt":             self.prompt,
            "style":              self.style,
            "seed":               self.seed,
            "model_version":      self.model_version,
            "pipeline_config":    self.pipeline_config,
            "clip_score":         self.clip_score,
            "fid_score":          self.fid_score,
            "generation_time":    self.generation_time,
            "quality_score":      self.quality_score,
            "quality_rating":     self.quality_rating,
            "sharpness":          self.sharpness,
            "brightness":         self.brightness,
            "contrast":           self.contrast,
            "color_distribution": self.color_distribution,
            "noise_level":        self.noise_level,
            "image_width":        self.image_width,
            "image_height":       self.image_height,
            "num_inference_steps":self.num_inference_steps,
            "guidance_scale":     self.guidance_scale,
            "tags":               list(self.tags),
            "notes":              self.notes,
            "status":             self.status,
            "extra":              self.extra,
            "passed":             self.passed,
        }
        if include_composite:
            d["composite_score"] = self.composite_score
        return d

    def to_csv_row(self) -> Dict[str, str]:
        """Return a flat dict suitable for CSV writing."""
        return {
            "experiment_id":      self.experiment_id,
            "run_name":           self.run_name,
            "timestamp":          self.timestamp,
            "prompt":             self.prompt,
            "style":              self.style,
            "seed":               str(self.seed),
            "model_version":      self.model_version,
            "pipeline_config":    json.dumps(self.pipeline_config),
            "clip_score":         str(self.clip_score) if self.clip_score is not None else "",
            "fid_score":          str(self.fid_score)  if self.fid_score  is not None else "",
            "generation_time":    str(round(self.generation_time, 4)),
            "quality_score":      str(self.quality_score) if self.quality_score is not None else "",
            "quality_rating":     self.quality_rating,
            "sharpness":          str(self.sharpness) if self.sharpness is not None else "",
            "brightness":         str(self.brightness) if self.brightness is not None else "",
            "contrast":           str(self.contrast) if self.contrast is not None else "",
            "color_distribution": str(self.color_distribution) if self.color_distribution is not None else "",
            "noise_level":        str(self.noise_level) if self.noise_level is not None else "",
            "image_width":        str(self.image_width) if self.image_width is not None else "",
            "image_height":       str(self.image_height) if self.image_height is not None else "",
            "num_inference_steps":str(self.num_inference_steps) if self.num_inference_steps is not None else "",
            "guidance_scale":     str(self.guidance_scale) if self.guidance_scale is not None else "",
            "tags":               json.dumps(self.tags),
            "notes":              self.notes,
            "status":             self.status,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        clip  = f"CLIP={self.clip_score:.3f}" if self.clip_score is not None else "CLIP=N/A"
        fid   = f"FID={self.fid_score:.1f}"   if self.fid_score  is not None else "FID=N/A"
        prompt_preview = (self.prompt[:50] + "…") if len(self.prompt) > 50 else self.prompt
        return (
            f"[{self.experiment_id}] {self.run_name or 'unnamed'} | "
            f"{clip} | {fid} | time={self.generation_time:.2f}s | "
            f"seed={self.seed} | model={self.model_version[:30] if self.model_version else 'N/A'!r} | "
            f"prompt={prompt_preview!r}"
        )

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# ── ExperimentTracker
# =============================================================================

class ExperimentTracker:
    """
    Production-grade experiment tracking for fashion image generation.

    Features
    --------
    • Persistent SQLite storage — survives restarts, thread-safe.
    • Tracks 6 required fields plus rich optional metadata.
    • Query / filter / sort experiments.
    • Find best experiment by any metric.
    • Side-by-side comparison of two experiments.
    • Export to CSV and JSON (full or filtered).
    • Rich text summary report.
    • Context manager for run timing.

    Parameters
    ----------
    experiments_dir : Path, optional
        Directory for the SQLite database and exports.
        Defaults to ``week2/outputs/experiments/``.
    db_name : str
        SQLite filename. Defaults to ``experiments.db``.
    auto_export : bool
        Automatically export JSON on each log. Default False.

    Example
    -------
        tracker = ExperimentTracker()

        exp_id = tracker.log_experiment(
            prompt          = "a black oversized streetwear hoodie",
            seed            = 42,
            model_version   = "stabilityai/stable-diffusion-xl-base-1.0",
            clip_score      = 0.91,
            fid_score       = 28.4,
            generation_time = 4.32,
            style           = "streetwear",
            tags            = ["batch_01"],
        )

        tracker.export_csv()
        tracker.export_json()
        print(tracker.generate_report())
    """

    def __init__(
        self,
        experiments_dir: Optional[Union[str, Path]] = None,
        db_name:         str                         = DEFAULT_DB_NAME,
        auto_export:     bool                        = False,
    ) -> None:
        self.experiments_dir = Path(experiments_dir or DEFAULT_EXPERIMENTS_DIR)
        self.db_name         = db_name
        self.auto_export     = auto_export
        self._db_path        = self.experiments_dir / db_name
        self._lock           = threading.Lock()

        self._init_storage()

        logger.info(
            "ExperimentTracker initialised | db={} | auto_export={}",
            self._db_path, auto_export,
        )

    # =========================================================================
    # ── Primary Logging API
    # =========================================================================

    def log_experiment(
        self,
        prompt:              str,
        *,
        seed:                int                      = 0,
        model_version:       str                      = "",
        clip_score:          Optional[float]           = None,
        fid_score:           Optional[float]           = None,
        generation_time:     float                     = 0.0,
        run_name:            str                      = "",
        style:               str                      = "",
        pipeline_config:     Optional[Dict[str, Any]] = None,
        quality_score:       Optional[float]           = None,
        quality_rating:      str                      = "",
        sharpness:           Optional[float]           = None,
        brightness:          Optional[float]           = None,
        contrast:            Optional[float]           = None,
        color_distribution:  Optional[float]           = None,
        noise_level:         Optional[float]           = None,
        image_width:         Optional[int]             = None,
        image_height:        Optional[int]             = None,
        num_inference_steps: Optional[int]             = None,
        guidance_scale:      Optional[float]           = None,
        tags:                Optional[List[str]]       = None,
        notes:               str                      = "",
        status:              str                      = "completed",
        extra:               Optional[Dict[str, Any]] = None,
        experiment_id:       Optional[str]             = None,
    ) -> str:
        """
        Log a single experiment run.

        Parameters
        ----------
        prompt          : str    Full generation prompt (required).
        seed            : int    RNG seed for reproducibility.
        model_version   : str    Model ID (e.g. "stabilityai/stable-diffusion-xl-base-1.0").
        clip_score      : float  CLIP cosine similarity [0, 1].
        fid_score       : float  FID score (lower = better).
        generation_time : float  Wall-clock seconds for generation.
        run_name        : str    Human-readable run label.
        style           : str    Fashion style.
        pipeline_config : dict   Full pipeline configuration snapshot.
        quality_score   : float  Composite quality score [0, 100].
        quality_rating  : str    Quality label.
        sharpness       : float  Sharpness score [0, 100].
        brightness      : float  Brightness score [0, 100].
        contrast        : float  Contrast score [0, 100].
        color_distribution : float  Colour distribution score [0, 100].
        noise_level     : float  Noise level score [0, 100].
        image_width / height : int  Output image dimensions.
        num_inference_steps : int   SDXL denoising steps.
        guidance_scale  : float  CFG scale.
        tags            : list   String tags for grouping/filtering.
        notes           : str    Free-text notes.
        status          : str    "completed" | "failed" | "pending".
        extra           : dict   Arbitrary extensible data.
        experiment_id   : str    Override auto-generated ID.

        Returns
        -------
        str — experiment_id

        Example
        -------
            exp_id = tracker.log_experiment(
                prompt          = "a black oversized streetwear hoodie",
                seed            = 42,
                model_version   = "stabilityai/stable-diffusion-xl-base-1.0",
                clip_score      = 0.91,
                fid_score       = 28.4,
                generation_time = 4.32,
            )
        """
        record = ExperimentRecord(
            prompt               = prompt,
            seed                 = seed,
            model_version        = model_version,
            clip_score           = clip_score,
            fid_score            = fid_score,
            generation_time      = generation_time,
            run_name             = run_name,
            style                = style,
            pipeline_config      = pipeline_config or {},
            quality_score        = quality_score,
            quality_rating       = quality_rating,
            sharpness            = sharpness,
            brightness           = brightness,
            contrast             = contrast,
            color_distribution   = color_distribution,
            noise_level          = noise_level,
            image_width          = image_width,
            image_height         = image_height,
            num_inference_steps  = num_inference_steps,
            guidance_scale       = guidance_scale,
            tags                 = list(tags or []),
            notes                = notes,
            status               = status,
            extra                = extra or {},
        )
        if experiment_id:
            record.experiment_id = experiment_id

        self._insert(record)
        logger.info("Experiment logged | id={} | {}", record.experiment_id, record.summary())

        if self.auto_export:
            self.export_json()

        return record.experiment_id

    def log_batch(
        self,
        experiments: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Log multiple experiments in a single call.

        Parameters
        ----------
        experiments : list of dict
            Each dict is the same keyword arguments as ``log_experiment()``.

        Returns
        -------
        list of str — experiment IDs

        Example
        -------
            ids = tracker.log_batch([
                {"prompt": "hoodie", "seed": 1, "clip_score": 0.88},
                {"prompt": "dress",  "seed": 2, "clip_score": 0.91},
            ])
        """
        ids = []
        with self._db_connection() as conn:
            for kwargs in experiments:
                record = ExperimentRecord(
                    prompt             = kwargs.get("prompt", ""),
                    seed               = kwargs.get("seed", 0),
                    model_version      = kwargs.get("model_version", ""),
                    clip_score         = kwargs.get("clip_score"),
                    fid_score          = kwargs.get("fid_score"),
                    generation_time    = kwargs.get("generation_time", 0.0),
                    run_name           = kwargs.get("run_name", ""),
                    style              = kwargs.get("style", ""),
                    pipeline_config    = kwargs.get("pipeline_config") or {},
                    quality_score      = kwargs.get("quality_score"),
                    quality_rating     = kwargs.get("quality_rating", ""),
                    sharpness          = kwargs.get("sharpness"),
                    brightness         = kwargs.get("brightness"),
                    contrast           = kwargs.get("contrast"),
                    color_distribution = kwargs.get("color_distribution"),
                    noise_level        = kwargs.get("noise_level"),
                    image_width        = kwargs.get("image_width"),
                    image_height       = kwargs.get("image_height"),
                    num_inference_steps= kwargs.get("num_inference_steps"),
                    guidance_scale     = kwargs.get("guidance_scale"),
                    tags               = list(kwargs.get("tags") or []),
                    notes              = kwargs.get("notes", ""),
                    status             = kwargs.get("status", "completed"),
                    extra              = kwargs.get("extra") or {},
                )
                if "experiment_id" in kwargs:
                    record.experiment_id = kwargs["experiment_id"]
                self._insert_with_conn(conn, record)
                ids.append(record.experiment_id)

        logger.info("Batch logged | n={} experiments", len(ids))
        return ids

    # =========================================================================
    # ── Retrieval API
    # =========================================================================

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """
        Retrieve a single experiment by ID.

        Parameters
        ----------
        experiment_id : str

        Returns
        -------
        ExperimentRecord or None
        """
        with self._db_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_experiments(
        self,
        *,
        style:           Optional[str]   = None,
        run_name:        Optional[str]   = None,
        status:          Optional[str]   = None,
        tags:            Optional[List[str]] = None,
        min_clip_score:  Optional[float] = None,
        max_fid_score:   Optional[float] = None,
        model_version:   Optional[str]   = None,
        limit:           Optional[int]   = None,
        order_by:        str             = "timestamp",
        descending:      bool            = True,
    ) -> List[ExperimentRecord]:
        """
        List / filter experiments with optional sorting and pagination.

        Parameters
        ----------
        style          : str    Filter by fashion style.
        run_name       : str    Filter by run name (exact match).
        status         : str    Filter by status.
        tags           : list   Return only experiments containing ALL given tags.
        min_clip_score : float  Lower bound on CLIP score.
        max_fid_score  : float  Upper bound on FID score.
        model_version  : str    Filter by model version.
        limit          : int    Max results.
        order_by       : str    Column name to sort by.
        descending     : bool   Sort direction.

        Returns
        -------
        list of ExperimentRecord

        Example
        -------
            top_clip = tracker.list_experiments(
                min_clip_score = 0.85,
                order_by       = "clip_score",
                descending     = True,
                limit          = 10,
            )
        """
        where_clauses: List[str] = []
        params: List[Any] = []

        if style:
            where_clauses.append("style = ?")
            params.append(style)
        if run_name:
            where_clauses.append("run_name = ?")
            params.append(run_name)
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if min_clip_score is not None:
            where_clauses.append("clip_score >= ?")
            params.append(min_clip_score)
        if max_fid_score is not None:
            where_clauses.append("fid_score <= ?")
            params.append(max_fid_score)
        if model_version:
            where_clauses.append("model_version = ?")
            params.append(model_version)

        sql  = "SELECT * FROM experiments"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # Safe column name for ORDER BY (whitelist)
        safe_cols = {
            "timestamp", "clip_score", "fid_score", "generation_time",
            "quality_score", "sharpness", "experiment_id", "run_name",
        }
        sort_col = order_by if order_by in safe_cols else "timestamp"
        direction = "DESC" if descending else "ASC"
        sql += f" ORDER BY {sort_col} {direction}"

        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        with self._db_connection() as conn:
            cur  = conn.execute(sql, params)
            rows = cur.fetchall()

        records = [self._row_to_record(r) for r in rows]

        # Post-filter by tags (SQLite JSON contains check)
        if tags:
            records = [
                r for r in records
                if all(t in r.tags for t in tags)
            ]

        return records

    def best_experiment(
        self,
        metric:     str  = "clip_score",
        *,
        minimize:   bool = False,
        style:      Optional[str] = None,
        run_name:   Optional[str] = None,
    ) -> Optional[ExperimentRecord]:
        """
        Find the best experiment by a given metric.

        Parameters
        ----------
        metric    : str   Column name (e.g. "clip_score", "fid_score", "composite_score").
        minimize  : bool  True for metrics where lower is better (e.g. FID, generation_time).
        style     : str   Optional style filter.
        run_name  : str   Optional run filter.

        Returns
        -------
        ExperimentRecord or None

        Example
        -------
            best_clip = tracker.best_experiment("clip_score")
            best_fid  = tracker.best_experiment("fid_score", minimize=True)
            fastest   = tracker.best_experiment("generation_time", minimize=True)
        """
        if metric == "composite_score":
            # Compute in Python (not stored as SQL column)
            records = self.list_experiments(style=style, run_name=run_name)
            records = [r for r in records if r.composite_score is not None]
            if not records:
                return None
            return min(records, key=lambda r: r.composite_score) if minimize \
                   else max(records, key=lambda r: r.composite_score)

        safe_cols = {
            "clip_score", "fid_score", "generation_time", "quality_score",
            "sharpness", "brightness", "contrast", "noise_level",
        }
        col = metric if metric in safe_cols else "clip_score"
        fn  = "MIN" if minimize else "MAX"

        where_clauses = [f"{col} IS NOT NULL"]
        params: List[Any] = []
        if style:
            where_clauses.append("style = ?")
            params.append(style)
        if run_name:
            where_clauses.append("run_name = ?")
            params.append(run_name)

        sql = (
            f"SELECT * FROM experiments WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY {col} {'ASC' if minimize else 'DESC'} LIMIT 1"
        )

        with self._db_connection() as conn:
            cur = conn.execute(sql, params)
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def compare_experiments(
        self,
        id_a: str,
        id_b: str,
    ) -> Dict[str, Any]:
        """
        Side-by-side comparison of two experiments.

        Parameters
        ----------
        id_a, id_b : str   Experiment IDs.

        Returns
        -------
        dict with keys:
            "a", "b" — full experiment dicts
            "delta"  — metric differences (a - b)
            "winner" — which experiment wins on each metric

        Example
        -------
            diff = tracker.compare_experiments(id_a="abc123", id_b="def456")
            print(diff["winner"]["clip_score"])  # "a" or "b"
        """
        rec_a = self.get_experiment(id_a)
        rec_b = self.get_experiment(id_b)

        if rec_a is None or rec_b is None:
            missing = id_a if rec_a is None else id_b
            return {"error": f"Experiment {missing!r} not found"}

        metrics_compare = [
            ("clip_score",       False),   # higher = better
            ("fid_score",        True),    # lower = better
            ("generation_time",  True),    # lower = better
            ("quality_score",    False),
            ("sharpness",        False),
            ("brightness",       False),
            ("contrast",         False),
            ("noise_level",      False),
        ]

        delta:  Dict[str, Any] = {}
        winner: Dict[str, str] = {}

        for metric, minimize in metrics_compare:
            va = getattr(rec_a, metric)
            vb = getattr(rec_b, metric)
            if va is None or vb is None:
                delta[metric]  = None
                winner[metric] = "tie"
                continue
            diff = float(va) - float(vb)
            delta[metric] = round(diff, 4)
            if abs(diff) < 1e-6:
                winner[metric] = "tie"
            elif minimize:
                winner[metric] = "a" if va < vb else "b"
            else:
                winner[metric] = "a" if va > vb else "b"

        a_wins = sum(1 for v in winner.values() if v == "a")
        b_wins = sum(1 for v in winner.values() if v == "b")
        overall_winner = "a" if a_wins > b_wins else ("b" if b_wins > a_wins else "tie")

        return {
            "a":               rec_a.to_dict(),
            "b":               rec_b.to_dict(),
            "delta":           delta,
            "winner":          winner,
            "overall_winner":  overall_winner,
            "a_wins":          a_wins,
            "b_wins":          b_wins,
        }

    def statistics(
        self,
        *,
        style:      Optional[str] = None,
        run_name:   Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics across tracked experiments.

        Returns
        -------
        dict with keys:
            total, completed, failed, pass_rate,
            mean/min/max/std for clip_score, fid_score, generation_time,
            style_breakdown, top_models.
        """
        records = self.list_experiments(style=style, run_name=run_name)
        if not records:
            return {"total": 0}

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            if _NUMPY:
                arr = np.array(values)
                return {
                    "mean": round(float(arr.mean()), 4),
                    "min":  round(float(arr.min()),  4),
                    "max":  round(float(arr.max()),  4),
                    "std":  round(float(arr.std()),  4),
                }
            n    = len(values)
            mean = sum(values) / n
            var  = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
            return {
                "mean": round(mean, 4),
                "min":  round(min(values), 4),
                "max":  round(max(values), 4),
                "std":  round(var ** 0.5, 4),
            }

        clip_scores    = [r.clip_score    for r in records if r.clip_score    is not None]
        fid_scores     = [r.fid_score     for r in records if r.fid_score     is not None]
        gen_times      = [r.generation_time for r in records]
        quality_scores = [r.quality_score for r in records if r.quality_score is not None]

        style_breakdown: Dict[str, int] = {}
        model_counts:    Dict[str, int] = {}
        for r in records:
            if r.style:
                style_breakdown[r.style] = style_breakdown.get(r.style, 0) + 1
            if r.model_version:
                model_counts[r.model_version] = model_counts.get(r.model_version, 0) + 1

        completed = sum(1 for r in records if r.status == "completed")
        failed    = sum(1 for r in records if r.status == "failed")
        passed    = sum(1 for r in records if r.passed)

        return {
            "total":           len(records),
            "completed":       completed,
            "failed":          failed,
            "passed":          passed,
            "pass_rate":       round(passed / len(records), 4),
            "clip_score":      _stats(clip_scores),
            "fid_score":       _stats(fid_scores),
            "generation_time": _stats(gen_times),
            "quality_score":   _stats(quality_scores),
            "style_breakdown": style_breakdown,
            "top_models":      dict(sorted(model_counts.items(),
                                           key=lambda kv: kv[1], reverse=True)[:5]),
        }

    # =========================================================================
    # ── Export API
    # =========================================================================

    def export_csv(
        self,
        path:    Optional[Union[str, Path]] = None,
        *,
        style:   Optional[str]  = None,
        run_name:Optional[str]  = None,
        limit:   Optional[int]  = None,
    ) -> Path:
        """
        Export experiments to a CSV file.

        Parameters
        ----------
        path     : str or Path, optional
            Output file path. Defaults to
            ``outputs/experiments/export_<ts>.csv``.
        style    : str   Filter by style.
        run_name : str   Filter by run name.
        limit    : int   Max rows.

        Returns
        -------
        Path — absolute path of the written CSV file.

        Example
        -------
            csv_path = tracker.export_csv("my_experiments.csv")
            print(f"CSV saved to {csv_path}")
        """
        records = self.list_experiments(style=style, run_name=run_name, limit=limit)
        out_path = self._resolve_export_path(path, "csv")

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_csv_row())

        logger.info("CSV export | {} rows → {}", len(records), out_path)
        return out_path

    def export_json(
        self,
        path:    Optional[Union[str, Path]] = None,
        *,
        style:   Optional[str]  = None,
        run_name:Optional[str]  = None,
        limit:   Optional[int]  = None,
        indent:  int             = 2,
        include_stats: bool      = True,
    ) -> Path:
        """
        Export experiments to a JSON file.

        Parameters
        ----------
        path     : str or Path, optional
        style    : str   Filter by style.
        run_name : str   Filter by run name.
        limit    : int   Max records.
        indent   : int   JSON indentation.
        include_stats : bool  Include aggregate statistics block.

        Returns
        -------
        Path — absolute path of the written JSON file.

        Example
        -------
            json_path = tracker.export_json("my_experiments.json")
        """
        records = self.list_experiments(style=style, run_name=run_name, limit=limit)
        out_path = self._resolve_export_path(path, "json")

        payload: Dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total":       len(records),
            "filters": {
                "style":    style,
                "run_name": run_name,
                "limit":    limit,
            },
            "experiments": [r.to_dict() for r in records],
        }
        if include_stats:
            payload["statistics"] = self.statistics(style=style, run_name=run_name)

        out_path.write_text(
            json.dumps(payload, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("JSON export | {} records → {}", len(records), out_path)
        return out_path

    def to_csv_string(
        self,
        records: Optional[List[ExperimentRecord]] = None,
    ) -> str:
        """
        Return CSV content as a string (in-memory, no file written).

        Parameters
        ----------
        records : list, optional   If None, exports all experiments.

        Returns
        -------
        str — CSV content
        """
        if records is None:
            records = self.list_experiments()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_csv_row())
        return buf.getvalue()

    def to_json_string(
        self,
        records: Optional[List[ExperimentRecord]] = None,
        indent:  int = 2,
    ) -> str:
        """
        Return JSON content as a string (in-memory, no file written).

        Parameters
        ----------
        records : list, optional
        indent  : int

        Returns
        -------
        str — JSON content
        """
        if records is None:
            records = self.list_experiments()
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total":       len(records),
            "experiments": [r.to_dict() for r in records],
        }
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    # =========================================================================
    # ── Reporting
    # =========================================================================

    def generate_report(
        self,
        *,
        style:    Optional[str] = None,
        run_name: Optional[str] = None,
        top_n:    int           = 5,
    ) -> str:
        """
        Generate a rich, human-readable summary report.

        Parameters
        ----------
        style    : str   Filter by style.
        run_name : str   Filter by run name.
        top_n    : int   Number of top experiments to list.

        Returns
        -------
        str — multi-line formatted text report

        Example
        -------
            print(tracker.generate_report())
        """
        stats    = self.statistics(style=style, run_name=run_name)
        records  = self.list_experiments(
            style=style, run_name=run_name,
            order_by="clip_score", descending=True, limit=top_n,
        )

        sep  = "=" * 66
        thin = "-" * 66

        lines = [
            sep,
            "  FASHION GENERATION EXPERIMENT REPORT",
            sep,
            f"  Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"  Database   : {self._db_path}",
        ]
        if style:
            lines.append(f"  Style      : {style}")
        if run_name:
            lines.append(f"  Run        : {run_name}")

        lines += [
            "",
            "  SUMMARY",
            thin,
            f"  Total experiments : {stats.get('total', 0)}",
            f"  Completed         : {stats.get('completed', 0)}",
            f"  Failed            : {stats.get('failed', 0)}",
            f"  Pass rate         : {stats.get('pass_rate', 0):.1%}",
            "",
        ]

        # Per-metric stats table
        for metric, label in [
            ("clip_score",      "CLIP Score     "),
            ("fid_score",       "FID Score      "),
            ("generation_time", "Gen Time (s)   "),
            ("quality_score",   "Quality Score  "),
        ]:
            ms = stats.get(metric, {})
            if ms:
                lines.append(
                    f"  {label}: "
                    f"mean={ms.get('mean', 'N/A'):>8}  "
                    f"min={ms.get('min', 'N/A'):>8}  "
                    f"max={ms.get('max', 'N/A'):>8}  "
                    f"std={ms.get('std', 'N/A'):>8}"
                )

        # Style breakdown
        breakdown = stats.get("style_breakdown", {})
        if breakdown:
            lines += ["", "  STYLE BREAKDOWN", thin]
            for sty, cnt in sorted(breakdown.items(), key=lambda kv: -kv[1]):
                bar = "#" * min(cnt, 30)
                lines.append(f"  {sty:<18} {cnt:>4}  {bar}")

        # Top models
        top_models = stats.get("top_models", {})
        if top_models:
            lines += ["", "  TOP MODELS", thin]
            for m, cnt in top_models.items():
                lines.append(f"  {m[:40]:<42} {cnt:>4} runs")

        # Top experiments
        if records:
            lines += [
                "",
                f"  TOP {top_n} EXPERIMENTS  (by CLIP score)",
                thin,
                f"  {'ID':<14} {'Run':<15} {'CLIP':>6} {'FID':>8} {'Time(s)':>8} {'Style':<12} {'Seed':>6}",
                "  " + thin,
            ]
            for r in records:
                clip_s = f"{r.clip_score:.3f}" if r.clip_score is not None else "  N/A"
                fid_s  = f"{r.fid_score:.1f}"  if r.fid_score  is not None else "   N/A"
                lines.append(
                    f"  {r.experiment_id:<14} {(r.run_name or '-'):<15} "
                    f"{clip_s:>6} {fid_s:>8} {r.generation_time:>8.2f} "
                    f"{(r.style or '-'):<12} {r.seed:>6}"
                )

        lines.append(sep)
        return "\n".join(lines)

    # =========================================================================
    # ── Maintenance API
    # =========================================================================

    def update_experiment(
        self,
        experiment_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Update fields on an existing experiment.

        Parameters
        ----------
        experiment_id : str
        **kwargs      : field=value pairs to update.

        Returns
        -------
        bool — True if a record was updated.

        Example
        -------
            tracker.update_experiment("abc123", notes="re-evaluated", clip_score=0.93)
        """
        allowed = {
            "clip_score", "fid_score", "generation_time", "quality_score",
            "quality_rating", "sharpness", "brightness", "contrast",
            "color_distribution", "noise_level", "notes", "status",
            "run_name", "tags", "extra",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        set_clauses = []
        params: List[Any] = []
        for col, val in updates.items():
            set_clauses.append(f"{col} = ?")
            if col in ("tags", "extra", "pipeline_config"):
                params.append(json.dumps(val) if not isinstance(val, str) else val)
            else:
                params.append(val)
        params.append(experiment_id)

        sql = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE experiment_id = ?"
        with self._db_connection() as conn:
            cur = conn.execute(sql, params)
            return cur.rowcount > 0

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete a single experiment record.

        Returns
        -------
        bool — True if a record was deleted.
        """
        with self._db_connection() as conn:
            cur = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("Experiment deleted | id={}", experiment_id)
        return deleted

    def clear_all(self, *, confirm: bool = False) -> int:
        """
        Delete ALL experiments from the database.

        Parameters
        ----------
        confirm : bool   Must be True to proceed (safety gate).

        Returns
        -------
        int — number of records deleted.
        """
        if not confirm:
            raise ValueError(
                "clear_all requires confirm=True to prevent accidental deletion."
            )
        with self._db_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM experiments")
            count = cur.fetchone()[0]
            conn.execute("DELETE FROM experiments")
        logger.warning("clear_all | {} experiments deleted", count)
        return count

    def count(self) -> int:
        """Return total number of experiments in the database."""
        with self._db_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM experiments")
            return cur.fetchone()[0]

    # =========================================================================
    # ── Context Manager (timed run)
    # =========================================================================

    @contextmanager
    def track_run(
        self,
        prompt:        str,
        *,
        run_name:      str                      = "",
        seed:          int                      = 0,
        model_version: str                      = "",
        style:         str                      = "",
        tags:          Optional[List[str]]      = None,
        extra:         Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for automatically timing a generation run.

        Usage
        -----
            with tracker.track_run("a black hoodie", seed=42) as run:
                image = generator.generate(...)
                run["clip_score"] = evaluator.evaluate(image, "a black hoodie").clip_score
                run["fid_score"]  = 28.4

        The ``run`` dict can be populated inside the ``with`` block with
        any fields from ``log_experiment()``. ``generation_time`` is set
        automatically.

        Yields
        ------
        dict — mutable run metadata dict (populated, then logged on exit).
        """
        run_data: Dict[str, Any] = {
            "prompt":        prompt,
            "run_name":      run_name,
            "seed":          seed,
            "model_version": model_version,
            "style":         style,
            "tags":          list(tags or []),
            "extra":         extra or {},
            "status":        "completed",
        }
        t0 = time.perf_counter()
        try:
            yield run_data
        except Exception as exc:
            run_data["status"] = "failed"
            run_data["notes"]  = str(exc)
            logger.error("track_run failed: {}", exc)
        finally:
            run_data["generation_time"] = round(time.perf_counter() - t0, 4)
            self.log_experiment(**run_data)

    # =========================================================================
    # ── SQLite Internals
    # =========================================================================

    def _init_storage(self) -> None:
        """Create experiments directory and SQLite schema."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        with self._db_connection() as conn:
            conn.executescript(_SQL_CREATE_TABLE)
            conn.executescript(_SQL_CREATE_INDEX)
        logger.debug("SQLite schema initialised | {}", self._db_path)

    @contextmanager
    def _db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Thread-safe SQLite connection context manager."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # Write-Ahead Logging for concurrency
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _insert(self, record: ExperimentRecord) -> None:
        """Insert a single record (acquires lock internally)."""
        with self._db_connection() as conn:
            self._insert_with_conn(conn, record)

    def _insert_with_conn(
        self,
        conn:   sqlite3.Connection,
        record: ExperimentRecord,
    ) -> None:
        """Insert a record using an existing connection (no lock)."""
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id, run_name, timestamp,
                prompt, style, seed, model_version, pipeline_config,
                clip_score, fid_score, generation_time,
                quality_score, quality_rating,
                sharpness, brightness, contrast, color_distribution, noise_level,
                image_width, image_height, num_inference_steps, guidance_scale,
                tags, notes, status, extra
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                record.experiment_id,
                record.run_name,
                record.timestamp,
                record.prompt,
                record.style,
                record.seed,
                record.model_version,
                json.dumps(record.pipeline_config),
                record.clip_score,
                record.fid_score,
                record.generation_time,
                record.quality_score,
                record.quality_rating,
                record.sharpness,
                record.brightness,
                record.contrast,
                record.color_distribution,
                record.noise_level,
                record.image_width,
                record.image_height,
                record.num_inference_steps,
                record.guidance_scale,
                json.dumps(record.tags),
                record.notes,
                record.status,
                json.dumps(record.extra),
            ),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        """Convert a SQLite Row to an ExperimentRecord dataclass."""
        d = dict(row)
        # Deserialise JSON blobs
        for col in ("pipeline_config", "tags", "extra"):
            raw = d.get(col)
            if raw:
                try:
                    d[col] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[col] = {} if col != "tags" else []
            else:
                d[col] = {} if col != "tags" else []

        return ExperimentRecord(
            experiment_id        = d.get("experiment_id", ""),
            run_name             = d.get("run_name", "") or "",
            timestamp            = d.get("timestamp", ""),
            prompt               = d.get("prompt", ""),
            style                = d.get("style", "") or "",
            seed                 = d.get("seed", 0) or 0,
            model_version        = d.get("model_version", "") or "",
            pipeline_config      = d.get("pipeline_config", {}),
            clip_score           = d.get("clip_score"),
            fid_score            = d.get("fid_score"),
            generation_time      = d.get("generation_time", 0.0) or 0.0,
            quality_score        = d.get("quality_score"),
            quality_rating       = d.get("quality_rating", "") or "",
            sharpness            = d.get("sharpness"),
            brightness           = d.get("brightness"),
            contrast             = d.get("contrast"),
            color_distribution   = d.get("color_distribution"),
            noise_level          = d.get("noise_level"),
            image_width          = d.get("image_width"),
            image_height         = d.get("image_height"),
            num_inference_steps  = d.get("num_inference_steps"),
            guidance_scale       = d.get("guidance_scale"),
            tags                 = d.get("tags", []),
            notes                = d.get("notes", "") or "",
            status               = d.get("status", "completed") or "completed",
            extra                = d.get("extra", {}),
        )

    # =========================================================================
    # ── Utilities
    # =========================================================================

    def _resolve_export_path(
        self,
        path:      Optional[Union[str, Path]],
        extension: str,
    ) -> Path:
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = self.experiments_dir / f"export_{ts}.{extension}"
        return out

    def __repr__(self) -> str:
        n = self.count()
        return (
            f"ExperimentTracker("
            f"db={self._db_path.name!r} | "
            f"experiments={n} | "
            f"dir={self.experiments_dir})"
        )

    def __len__(self) -> int:
        return self.count()
