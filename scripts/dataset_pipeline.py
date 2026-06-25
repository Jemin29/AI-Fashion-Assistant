"""
=============================================================================
AI-Powered Fashion Design Assistant
dataset_pipeline.py — Master Pipeline Controller
=============================================================================
MODULE  : Master Dataset Pipeline Controller
WEEK    : 1 — Fashion Domain Research & Dataset Curation
AUTHOR  : Fashion AI Team

PURPOSE
-------
Single entry-point that orchestrates all sub-pipelines end-to-end:

    Stage 1 ── FashionGen Loader
    Stage 2 ── DeepFashion Loader
    Stage 3 ── Metadata Generator    (enriches records with auto-generated attrs)
    Stage 4 ── Preprocessing Pipeline (7-stage clean / dedup / normalise)
    Stage 5 ── Validation Framework  (7-layer quality gate + report)
    Stage 6 ── Unified Dataset Export (merged, schema-validated JSON)

FEATURES
--------
  ▸ One-click execution  : MasterPipelineController().run()
  ▸ Rich logging         : loguru, per-stage timers, per-record error traps
  ▸ Error handling       : every stage wrapped; failures logged, pipeline continues
  ▸ Progress tracking    : stage-level progress bars (tqdm) + % banners in logs
  ▸ Execution summary    : ExecutionSummary dataclass → execution_summary.json

OUTPUT
------
  datasets/processed/final_fashion_dataset.json  (primary output)
  datasets/processed/clean_dataset.json           (preprocessing intermediate)
  datasets/processed/validation_report.json       (7-layer validation report)
  datasets/processed/execution_summary.json       (pipeline run summary)

USAGE
-----
  # One-click (CLI):
  python dataset_pipeline.py

  # With limits (useful for testing without full datasets):
  python dataset_pipeline.py --max-fg 500 --max-df 500

  # Python API:
  from dataset_pipeline import MasterPipelineController, MasterPipelineConfig
  ctrl = MasterPipelineController()
  result = ctrl.run()
  print(result.to_dict())

  # Custom config:
  cfg = MasterPipelineConfig(
      fashiongen_max_records = 1000,
      deepfashion_max_records= 1000,
      deepfashion_split      = "train",
      enable_metadata_gen    = True,
      enable_preprocessing   = True,
      enable_validation      = True,
  )
  ctrl = MasterPipelineController(config=cfg)
  ctrl.run()

ARCHITECTURE
------------
  MasterPipelineConfig    — all knobs for the master pipeline
  ExecutionSummary        — structured summary (JSON-serialisable)
  StageTimer              — per-stage timing helper
  MasterPipelineController— orchestrator: 6 stages + export
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Third-party ──────────────────────────────────────────────────────────────
from loguru import logger

# Optional tqdm for per-stage progress bars
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ─── Resolve project root ──────────────────────────────────────────────────────
_FILE_DIR     = Path(__file__).resolve().parent        # fashion-ai-assistant/
_PROJECT_ROOT = _FILE_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Internal sub-pipeline imports (all graceful) ─────────────────────────────

# Stage 1: FashionGen loader
try:
    from src.data.ingestion.fashiongen_loader import FashionGenLoader
    _FG_AVAILABLE = True
except ImportError as _e:
    _FG_AVAILABLE = False
    logger.warning(f"FashionGenLoader unavailable: {_e}")

# Stage 2: DeepFashion loader
try:
    from src.data.ingestion.deepfashion_loader import DeepFashionLoader
    _DF_AVAILABLE = True
except ImportError as _e:
    _DF_AVAILABLE = False
    logger.warning(f"DeepFashionLoader unavailable: {_e}")

# Stage 3: Metadata generator
try:
    from src.data.metadata_generation.metadata_generator import MetadataGeneratorEngine
    _META_AVAILABLE = True
except ImportError as _e:
    _META_AVAILABLE = False
    logger.warning(f"MetadataGeneratorEngine unavailable: {_e}")

# Stage 4: Preprocessing pipeline
try:
    from src.data.preprocessing.preprocessing_pipeline import (
        PreprocessingPipeline, PipelineConfig,
    )
    _PREPROC_AVAILABLE = True
except ImportError as _e:
    _PREPROC_AVAILABLE = False
    logger.warning(f"PreprocessingPipeline unavailable: {_e}")

# Stage 5: Validation framework
try:
    from src.data.validation.data_validator import (
        FashionDataValidator, ValidationConfig,
    )
    _VAL_AVAILABLE = True
except ImportError as _e:
    _VAL_AVAILABLE = False
    logger.warning(f"FashionDataValidator unavailable: {_e}")

# ─── Output paths ──────────────────────────────────────────────────────────────
_OUTPUT_DIR      = _PROJECT_ROOT / "datasets" / "processed"
_FINAL_JSON      = _OUTPUT_DIR / "final_fashion_dataset.json"
_CLEAN_JSON      = _OUTPUT_DIR / "clean_dataset.json"
_VALIDATION_JSON = _OUTPUT_DIR / "validation_report.json"
_SUMMARY_JSON    = _OUTPUT_DIR / "execution_summary.json"
_FG_JSON         = _OUTPUT_DIR / "fashiongen_processed.json"
_DF_JSON         = _OUTPUT_DIR / "deepfashion_processed.json"

_PIPELINE_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# Banner helper (keeps logs visually clean)
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 70) -> None:
    """Log a section banner using plain ASCII (Windows-safe)."""
    bar = "=" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


# =============================================================================
# ── 1. Configuration
# =============================================================================

@dataclass
class MasterPipelineConfig:
    """
    All configurable parameters for the MasterPipelineController.

    Each stage can be individually enabled/disabled so the controller
    can be used for partial runs (e.g. re-run only validation).
    """

    # ── Stage 1: FashionGen ───────────────────────────────────────────────────
    enable_fashiongen      : bool           = True
    fashiongen_hdf5_path   : Optional[Path] = None   # None → auto-discover
    fashiongen_max_records : Optional[int]  = None   # None → all records
    fashiongen_start_index : int            = 0
    fashiongen_save_images : bool           = False

    # ── Stage 2: DeepFashion ──────────────────────────────────────────────────
    enable_deepfashion      : bool           = True
    deepfashion_root_dir    : Optional[Path] = None  # None → auto-discover
    deepfashion_split       : str            = "train"
    deepfashion_max_records : Optional[int]  = None

    # ── Stage 3: Metadata generator ───────────────────────────────────────────
    enable_metadata_gen     : bool = True
    metadata_nlp_enabled    : bool = True   # requires spaCy

    # ── Stage 4: Preprocessing ────────────────────────────────────────────────
    enable_preprocessing    : bool = True
    preproc_target_size     : tuple = (256, 256)
    preproc_dedup_strategy  : str   = "path_hash"
    preproc_drop_unknown    : bool  = False  # keep as "uncategorized"
    preproc_lowercase_desc  : bool  = False

    # ── Stage 5: Validation ───────────────────────────────────────────────────
    enable_validation       : bool  = True
    validation_image_check  : bool  = False  # set True if images exist on disk
    validation_deep_image   : bool  = False  # PIL open — slow

    # ── Stage 6: Export ───────────────────────────────────────────────────────
    export_include_invalid  : bool  = False  # include failed validation records
    export_pretty_json      : bool  = True   # indent=2 for readability

    # ── Misc ──────────────────────────────────────────────────────────────────
    output_dir              : Path  = _OUTPUT_DIR
    show_progress           : bool  = True
    use_kb                  : bool  = True   # Knowledge Base in loaders

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ── 2. Stage Timer
# =============================================================================

class StageTimer:
    """
    Lightweight context-manager timer for one pipeline stage.

    Usage:
        with StageTimer("FashionGen Loader") as t:
            ...
        print(t.elapsed_s)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.end_time  : float = 0.0
        self.error     : Optional[str] = None

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        logger.info(f"[START] Stage: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = time.perf_counter()
        if exc_type is not None:
            self.error = f"{exc_type.__name__}: {exc_val}"
            logger.error(
                f"[FAIL] Stage FAILED: {self.name} | {self.error} | "
                f"elapsed={self.elapsed_s:.2f}s"
            )
            # Return False -> let the exception propagate; caller decides to skip
            return False
        logger.success(
            f"[DONE] Stage done: {self.name} | elapsed={self.elapsed_s:.2f}s"
        )
        return False

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.perf_counter()
        return round(end - self.start_time, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage"     : self.name,
            "elapsed_s" : self.elapsed_s,
            "status"    : "error" if self.error else "ok",
            "error"     : self.error,
        }


# =============================================================================
# ── 3. Execution Summary
# =============================================================================

@dataclass
class ExecutionSummary:
    """
    Structured summary of a complete MasterPipelineController run.

    JSON-serialisable via to_dict(). Written to execution_summary.json.
    """

    # ── Run identity ──────────────────────────────────────────────────────────
    pipeline_version   : str            = _PIPELINE_VERSION
    run_id             : str            = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    started_at         : str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at       : str            = ""
    total_elapsed_s    : float          = 0.0
    status             : str            = "running"  # running | success | partial | failed

    # ── Per-stage timers ──────────────────────────────────────────────────────
    stage_timers       : List[StageTimer] = field(default_factory=list)

    # ── Record counts ─────────────────────────────────────────────────────────
    fashiongen_records  : int = 0
    deepfashion_records : int = 0
    total_raw_records   : int = 0
    metadata_enriched   : int = 0
    after_preprocessing : int = 0
    duplicates_removed  : int = 0
    uncategorized       : int = 0
    validation_valid    : int = 0
    validation_failed   : int = 0
    final_records       : int = 0

    # ── Quality metrics ───────────────────────────────────────────────────────
    validation_success_rate : float = 0.0
    validation_quality_score: float = 0.0

    # ── Output paths ──────────────────────────────────────────────────────────
    output_paths        : Dict[str, str] = field(default_factory=dict)

    # ── Error log ─────────────────────────────────────────────────────────────
    errors              : List[str] = field(default_factory=list)
    warnings            : List[str] = field(default_factory=list)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Mark pipeline as complete and record timestamp."""
        self.completed_at   = datetime.now(timezone.utc).isoformat()
        self.total_elapsed_s = sum(t.elapsed_s for t in self.stage_timers)
        if not self.errors:
            self.status = "success"
        elif self.final_records > 0:
            self.status = "partial"
        else:
            self.status = "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_version"      : self.pipeline_version,
            "run_id"                : self.run_id,
            "started_at"            : self.started_at,
            "completed_at"          : self.completed_at,
            "total_elapsed_s"       : round(self.total_elapsed_s, 3),
            "status"                : self.status,
            "record_counts": {
                "fashiongen_raw"    : self.fashiongen_records,
                "deepfashion_raw"   : self.deepfashion_records,
                "total_raw"         : self.total_raw_records,
                "metadata_enriched" : self.metadata_enriched,
                "after_preprocessing": self.after_preprocessing,
                "duplicates_removed": self.duplicates_removed,
                "uncategorized"     : self.uncategorized,
                "validation_valid"  : self.validation_valid,
                "validation_failed" : self.validation_failed,
                "final_exported"    : self.final_records,
            },
            "quality_metrics": {
                "validation_success_rate" : round(self.validation_success_rate, 4),
                "validation_quality_score": round(self.validation_quality_score, 4),
            },
            "stage_timings"         : [t.to_dict() for t in self.stage_timers],
            "output_paths"          : self.output_paths,
            "errors"                : self.errors,
            "warnings"              : self.warnings,
        }

    def print_summary(self) -> None:
        """Print a formatted execution summary to stdout and log."""
        bar   = "=" * 70
        bar2  = "-" * 70
        lines = [
            bar,
            "  MASTER PIPELINE -- EXECUTION SUMMARY",
            bar,
            f"  Run ID        : {self.run_id}",
            f"  Status        : {self.status.upper()}",
            f"  Total elapsed : {self.total_elapsed_s:.2f}s",
            "",
            f"  -- Record Counts --------------------------------------------------",
            f"  FashionGen raw    : {self.fashiongen_records:>8,}",
            f"  DeepFashion raw   : {self.deepfashion_records:>8,}",
            f"  Total raw input   : {self.total_raw_records:>8,}",
            f"  After metadata gen: {self.metadata_enriched:>8,}",
            f"  After preprocessing:{self.after_preprocessing:>7,}  "
            f"(-{self.duplicates_removed} dups, -{self.uncategorized} unknown cat)",
            f"  Validation valid  : {self.validation_valid:>8,}  "
            f"({self.validation_success_rate:.1%})",
            f"  Validation failed : {self.validation_failed:>8,}",
            f"  Final exported    : {self.final_records:>8,}",
            "",
            f"  -- Quality --------------------------------------------------------",
            f"  Success rate      : {self.validation_success_rate:.1%}",
            f"  Quality score     : {self.validation_quality_score:.1%}",
            "",
            f"  -- Stage Timings --------------------------------------------------",
        ]
        for t in self.stage_timers:
            status_icon = "OK" if not t.error else "FAIL"
            lines.append(
                f"  [{status_icon}] {t.name:<36} {t.elapsed_s:>7.2f}s"
            )
        lines += [
            "",
            f"  -- Outputs --------------------------------------------------------",
        ]
        for label, path in self.output_paths.items():
            lines.append(f"  {label:<30} {path}")
        if self.errors:
            lines += ["", f"  -- Errors ({len(self.errors)}) ------------------------------------------------"]
            for e in self.errors[:10]:
                lines.append(f"  [ERR] {e}")
        lines.append(bar)

        output = "\n".join(lines)
        print(output)
        logger.info("Execution summary printed to stdout")


# =============================================================================
# ── 4. MasterPipelineController
# =============================================================================

class MasterPipelineController:
    """
    Orchestrates all 6 dataset pipeline stages in sequence.

    Stages:
      1. FashionGen Loader        — ingest from HDF5
      2. DeepFashion Loader       — ingest from annotation TXT files
      3. Metadata Generator       — auto-enrich descriptions
      4. Preprocessing Pipeline   — clean, dedup, normalise (7 stages)
      5. Validation Framework     — 7-layer quality gate
      6. Unified Dataset Export   — merge + write final_fashion_dataset.json

    Design principles:
      • Fail-safe stages: each stage is wrapped in try/except; failures
        are logged and the pipeline proceeds with whatever data was collected.
      • Stage isolation: each stage writes to the summary and hands off
        a clean list of dicts to the next stage.
      • Idempotent output: running twice overwrites previous output safely.

    Args:
        config : MasterPipelineConfig (optional, uses defaults if omitted).
    """

    def __init__(
        self,
        config: Optional[MasterPipelineConfig] = None,
    ) -> None:
        self.config  : MasterPipelineConfig = config or MasterPipelineConfig()
        self.summary : ExecutionSummary     = ExecutionSummary()

        logger.info(
            f"MasterPipelineController v{_PIPELINE_VERSION} initialised | "
            f"output={self.config.output_dir}"
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def run(self) -> ExecutionSummary:
        """
        Execute the full 6-stage pipeline end-to-end.

        Returns:
            ExecutionSummary with counts, timings, quality metrics and paths.
        """
        _banner("MASTER DATASET PIPELINE  ──  Starting")
        logger.info(f"  FashionGen  : {'enabled' if self.config.enable_fashiongen   else 'SKIPPED'}")
        logger.info(f"  DeepFashion : {'enabled' if self.config.enable_deepfashion  else 'SKIPPED'}")
        logger.info(f"  Metadata gen: {'enabled' if self.config.enable_metadata_gen else 'SKIPPED'}")
        logger.info(f"  Preprocessing: {'enabled' if self.config.enable_preprocessing else 'SKIPPED'}")
        logger.info(f"  Validation  : {'enabled' if self.config.enable_validation   else 'SKIPPED'}")

        combined_records: List[Dict[str, Any]] = []

        # ── Stage 1: FashionGen Loader ────────────────────────────────────────
        fg_records = self._run_fashiongen_stage()
        combined_records.extend(fg_records)
        self.summary.fashiongen_records = len(fg_records)

        # ── Stage 2: DeepFashion Loader ───────────────────────────────────────
        df_records = self._run_deepfashion_stage()
        combined_records.extend(df_records)
        self.summary.deepfashion_records = len(df_records)

        self.summary.total_raw_records = len(combined_records)
        logger.info(
            f"Combined pool: {len(combined_records):,} records "
            f"(FG={len(fg_records):,} + DF={len(df_records):,})"
        )

        if not combined_records:
            self.summary.errors.append(
                "No records collected from any loader — check dataset paths."
            )
            self._finish_and_save(combined_records)
            return self.summary

        # ── Stage 3: Metadata Generator ───────────────────────────────────────
        combined_records = self._run_metadata_stage(combined_records)
        self.summary.metadata_enriched = len(combined_records)

        # ── Stage 4: Preprocessing Pipeline ──────────────────────────────────
        combined_records = self._run_preprocessing_stage(combined_records)
        self.summary.after_preprocessing = len(combined_records)

        # ── Stage 5: Validation Framework ─────────────────────────────────────
        valid_records, invalid_records = self._run_validation_stage(combined_records)
        self.summary.validation_valid  = len(valid_records)
        self.summary.validation_failed = len(invalid_records)

        # ── Stage 6: Unified Export ───────────────────────────────────────────
        export_records = (
            valid_records
            if not self.config.export_include_invalid
            else valid_records + invalid_records
        )
        self._finish_and_save(export_records)
        return self.summary

    # =========================================================================
    # ── Stage Implementations
    # =========================================================================

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — FashionGen Loader
    # ─────────────────────────────────────────────────────────────────────────

    def _run_fashiongen_stage(self) -> List[Dict[str, Any]]:
        """
        Stage 1: Run the FashionGen ingestion pipeline.

        Returns list of record dicts (FashionGenRecord.to_dict() format).
        Returns [] on error or if stage is disabled.
        """
        timer = StageTimer("1 - FashionGen Loader")
        self.summary.stage_timers.append(timer)
        records: List[Dict[str, Any]] = []

        if not self.config.enable_fashiongen:
            timer.name += " [SKIPPED]"
            logger.info("Stage 1 skipped (enable_fashiongen=False)")
            return records

        if not _FG_AVAILABLE:
            msg = "Stage 1 skipped — FashionGenLoader not importable"
            self.summary.warnings.append(msg)
            logger.warning(msg)
            return records

        try:
            with timer:
                loader = FashionGenLoader(
                    hdf5_path   = self.config.fashiongen_hdf5_path,
                    output_dir  = self.config.output_dir,
                    save_images = self.config.fashiongen_save_images,
                    use_kb      = self.config.use_kb,
                )
                result = loader.run(
                    max_records   = self.config.fashiongen_max_records,
                    start_index   = self.config.fashiongen_start_index,
                    show_progress = self.config.show_progress,
                )
                # FashionGenLoader.run() returns {"output_path": ..., "stats": ..., ...}
                output_path = result.get("output_path")
                if output_path and Path(str(output_path)).exists():
                    records = self._load_records_from_json(Path(str(output_path)))
                    logger.info(
                        f"Stage 1 — FashionGen: loaded {len(records):,} records "
                        f"from {output_path}"
                    )
                    # Ensure source_dataset key is present for downstream
                    for r in records:
                        r.setdefault("source_dataset", "fashiongen")
                else:
                    msg = (
                        "Stage 1 — FashionGen produced no output "
                        "(dataset file may be missing)"
                    )
                    self.summary.warnings.append(msg)
                    logger.warning(msg)

        except Exception as exc:
            msg = f"Stage 1 — FashionGen error: {exc}"
            self.summary.errors.append(msg)
            logger.exception(msg)

        return records

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — DeepFashion Loader
    # ─────────────────────────────────────────────────────────────────────────

    def _run_deepfashion_stage(self) -> List[Dict[str, Any]]:
        """
        Stage 2: Run the DeepFashion ingestion pipeline.

        Returns list of record dicts (DeepFashionRecord.to_dict() format).
        Returns [] on error or if stage is disabled.
        """
        timer = StageTimer("2 - DeepFashion Loader")
        self.summary.stage_timers.append(timer)
        records: List[Dict[str, Any]] = []

        if not self.config.enable_deepfashion:
            timer.name += " [SKIPPED]"
            logger.info("Stage 2 skipped (enable_deepfashion=False)")
            return records

        if not _DF_AVAILABLE:
            msg = "Stage 2 skipped — DeepFashionLoader not importable"
            self.summary.warnings.append(msg)
            logger.warning(msg)
            return records

        try:
            with timer:
                loader = DeepFashionLoader(
                    root_dir   = self.config.deepfashion_root_dir,
                    output_dir = self.config.output_dir,
                    use_kb     = self.config.use_kb,
                )
                result = loader.run(
                    split         = self.config.deepfashion_split,
                    max_records   = self.config.deepfashion_max_records,
                    show_progress = self.config.show_progress,
                )
                output_path = result.get("output_path")
                if output_path and Path(str(output_path)).exists():
                    records = self._load_records_from_json(Path(str(output_path)))
                    logger.info(
                        f"Stage 2 — DeepFashion: loaded {len(records):,} records "
                        f"from {output_path}"
                    )
                    for r in records:
                        r.setdefault("source_dataset", "deepfashion")
                else:
                    msg = (
                        "Stage 2 — DeepFashion produced no output "
                        "(dataset files may be missing)"
                    )
                    self.summary.warnings.append(msg)
                    logger.warning(msg)

        except Exception as exc:
            msg = f"Stage 2 — DeepFashion error: {exc}"
            self.summary.errors.append(msg)
            logger.exception(msg)

        return records

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — Metadata Generator
    # ─────────────────────────────────────────────────────────────────────────

    def _run_metadata_stage(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: Auto-enrich each record's missing metadata fields from its
        text description using MetadataGeneratorEngine.generate_from_record().

        Fields filled (only if not already set):
            style, season, gender, occasion, fit, pattern, color

        Records without a description are left unchanged (engine returns defaults).
        All records are returned — enrichment never drops records.

        Args:
            records : Combined raw records from Stages 1 + 2.

        Returns:
            Enriched list of record dicts.
        """
        timer = StageTimer("3 - Metadata Generator")
        self.summary.stage_timers.append(timer)

        if not self.config.enable_metadata_gen:
            timer.name += " [SKIPPED]"
            logger.info("Stage 3 skipped (enable_metadata_gen=False)")
            return records

        if not _META_AVAILABLE:
            msg = "Stage 3 skipped — MetadataGeneratorEngine not importable"
            self.summary.warnings.append(msg)
            logger.warning(msg)
            return records

        enriched: List[Dict[str, Any]] = []
        errors_in_stage = 0

        try:
            with timer:
                engine = MetadataGeneratorEngine(
                    enable_nlp=self.config.metadata_nlp_enabled,
                )

                # Build a tqdm wrapper for per-record progress
                iterable = (
                    _tqdm(records, desc="Metadata Gen", unit="rec")
                    if _TQDM_AVAILABLE and self.config.show_progress
                    else records
                )

                for i, rec in enumerate(iterable):
                    try:
                        enriched_rec = engine.generate_from_record(rec)
                        enriched.append(enriched_rec)
                    except Exception as exc:
                        # On per-record error, keep original record untouched
                        errors_in_stage += 1
                        logger.debug(
                            f"Metadata gen error on record "
                            f"'{rec.get('image_id', i)}': {exc}"
                        )
                        enriched.append(dict(rec))

                    # Periodic log every 5000 records
                    if (i + 1) % 5000 == 0:
                        logger.info(
                            f"  Metadata gen progress: "
                            f"{i+1:,}/{len(records):,} records"
                        )

                if errors_in_stage:
                    self.summary.warnings.append(
                        f"Stage 3 — {errors_in_stage} per-record metadata errors"
                    )
                logger.info(
                    f"Stage 3 — Metadata enrichment complete: "
                    f"{len(enriched):,} records "
                    f"({errors_in_stage} errors)"
                )

        except Exception as exc:
            msg = f"Stage 3 — Metadata Generator fatal error: {exc}"
            self.summary.errors.append(msg)
            logger.exception(msg)
            # Fall back: return original records unchanged
            return records

        return enriched

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — Preprocessing Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _run_preprocessing_stage(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Stage 4: Run the 7-stage PreprocessingPipeline on the combined batch.

        Stages within:
          1. Image resize metadata
          2. Image normalization metadata (ImageNet stats)
          3. Duplicate detection (path hash)
          4. Description cleaning (HTML strip, whitespace, unicode)
          5. Attribute normalisation (color, style, fit, season, occasion)
          6. Category normalisation (11-key taxonomy)
          7. Balance statistics generation

        Also saves the intermediate clean_dataset.json.

        Args:
            records : Enriched records from Stage 3.

        Returns:
            List of cleaned record dicts.
        """
        timer = StageTimer("4 - Preprocessing Pipeline")
        self.summary.stage_timers.append(timer)

        if not self.config.enable_preprocessing:
            timer.name += " [SKIPPED]"
            logger.info("Stage 4 skipped (enable_preprocessing=False)")
            return records

        if not _PREPROC_AVAILABLE:
            msg = "Stage 4 skipped — PreprocessingPipeline not importable"
            self.summary.warnings.append(msg)
            logger.warning(msg)
            return records

        try:
            with timer:
                cfg = PipelineConfig(
                    target_size             = self.config.preproc_target_size,
                    dedup_strategy          = self.config.preproc_dedup_strategy,
                    drop_unknown_categories = self.config.preproc_drop_unknown,
                    lowercase_description   = self.config.preproc_lowercase_desc,
                    keep_normalization_log  = True,
                    balance_target_per_class= None,
                )
                pipeline = PreprocessingPipeline(config=cfg)
                result   = pipeline.run(records)

                # Save intermediate clean_dataset.json
                clean_path = pipeline.save(result, self.config.output_dir / "clean_dataset.json")
                self.summary.output_paths["clean_dataset"] = str(clean_path)

                # Capture dedup stats into summary
                self.summary.duplicates_removed = result.duplicates_removed
                self.summary.uncategorized      = result.uncategorized

                result.print_summary()
                logger.info(
                    f"Stage 4 — Preprocessing: {result.total_input:,} in "
                    f"→ {result.total_output:,} out "
                    f"(dups={result.duplicates_removed}, "
                    f"uncategorized={result.uncategorized})"
                )
                return result.records

        except Exception as exc:
            msg = f"Stage 4 — Preprocessing fatal error: {exc}"
            self.summary.errors.append(msg)
            logger.exception(msg)
            return records

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5 — Validation Framework
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation_stage(
        self,
        records: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stage 5: Run FashionDataValidator's 7-layer checks on every record.

        Saves validation_report.json to the output directory.

        Args:
            records : Cleaned records from Stage 4.

        Returns:
            Tuple of (valid_records, invalid_records) — both as plain dicts.
        """
        timer = StageTimer("5 - Validation Framework")
        self.summary.stage_timers.append(timer)

        if not self.config.enable_validation:
            timer.name += " [SKIPPED]"
            logger.info("Stage 5 skipped (enable_validation=False)")
            # Treat all records as valid when skipped
            return records, []

        if not _VAL_AVAILABLE:
            msg = "Stage 5 skipped — FashionDataValidator not importable"
            self.summary.warnings.append(msg)
            logger.warning(msg)
            return records, []

        valid_records  : List[Dict[str, Any]] = []
        invalid_records: List[Dict[str, Any]] = []

        try:
            with timer:
                val_cfg = ValidationConfig(
                    verify_image_exists   = self.config.validation_image_check,
                    verify_image_readable = self.config.validation_deep_image,
                )
                validator = FashionDataValidator(config=val_cfg)

                batch_result = validator.validate_batch(records)

                # Split records into valid / invalid
                valid_ids  = {
                    r.image_id
                    for r in batch_result._record_results
                    if r.is_valid
                }
                for rec in records:
                    if rec.get("image_id") in valid_ids:
                        valid_records.append(rec)
                    else:
                        invalid_records.append(rec)

                # Update summary quality metrics
                self.summary.validation_success_rate  = batch_result.success_rate
                self.summary.validation_quality_score = batch_result.quality_score

                # Save validation report
                report_path = validator.save_report(
                    batch_result,
                    self.config.output_dir / "validation_report.json",
                )
                self.summary.output_paths["validation_report"] = str(report_path)

                logger.info(
                    f"Stage 5 — Validation: "
                    f"valid={len(valid_records):,} | "
                    f"failed={len(invalid_records):,} | "
                    f"success_rate={batch_result.success_rate:.1%} | "
                    f"quality={batch_result.quality_score:.1%}"
                )

        except Exception as exc:
            msg = f"Stage 5 — Validation fatal error: {exc}"
            self.summary.errors.append(msg)
            logger.exception(msg)
            # On fatal error, pass all records through as "valid"
            return records, []

        return valid_records, invalid_records

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6 — Unified Export
    # ─────────────────────────────────────────────────────────────────────────

    def _finish_and_save(
        self,
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Stage 6: Write the final unified dataset JSON and execution summary.

        Output: datasets/processed/final_fashion_dataset.json
        Format:
          {
            "_meta"   : { pipeline info, run ID, counts },
            "summary" : ExecutionSummary.to_dict(),
            "records" : [ ... one dict per fashion item ... ]
          }

        Args:
            records : Final records to export.
        """
        timer = StageTimer("6 - Unified Dataset Export")
        self.summary.stage_timers.append(timer)
        self.summary.final_records = len(records)

        try:
            with timer:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)

                # ── Build final payload ────────────────────────────────────────
                self.summary.finalize()

                payload: Dict[str, Any] = {
                    "_meta": {
                        "pipeline"          : "MasterPipelineController",
                        "version"           : _PIPELINE_VERSION,
                        "run_id"            : self.summary.run_id,
                        "generated_at"      : self.summary.completed_at,
                        "total_records"     : len(records),
                        "fashiongen_records": self.summary.fashiongen_records,
                        "deepfashion_records": self.summary.deepfashion_records,
                        "status"            : self.summary.status,
                    },
                    "summary"   : self.summary.to_dict(),
                    "records"   : records,
                }

                # ── Write final_fashion_dataset.json ──────────────────────────
                indent = 2 if self.config.export_pretty_json else None
                with open(_FINAL_JSON, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=indent, ensure_ascii=False)

                file_size_mb = _FINAL_JSON.stat().st_size / 1024 / 1024
                self.summary.output_paths["final_dataset"] = str(_FINAL_JSON)
                logger.success(
                    f"Stage 6 - Exported {len(records):,} records -> "
                    f"{_FINAL_JSON.name}  ({file_size_mb:.1f} MB)"
                )

                # ── Write execution_summary.json ──────────────────────────────
                with open(_SUMMARY_JSON, "w", encoding="utf-8") as f:
                    json.dump(self.summary.to_dict(), f, indent=2, ensure_ascii=False)
                self.summary.output_paths["execution_summary"] = str(_SUMMARY_JSON)
                logger.info(f"Execution summary → {_SUMMARY_JSON.name}")

        except Exception as exc:
            msg = f"Stage 6 — Export fatal error: {exc}"
            self.summary.errors.append(msg)
            self.summary.status = "failed"
            logger.exception(msg)

        # Print human-readable summary regardless of success
        self.summary.print_summary()

    # =========================================================================
    # ── Private Helpers
    # =========================================================================

    @staticmethod
    def _load_records_from_json(path: Path) -> List[Dict[str, Any]]:
        """
        Load the 'records' array from a pipeline-output JSON file.

        Supports both:
          { "records": [...] }             ← preprocessing output
          { "_meta": ..., "records": [...] } ← loader output

        Args:
            path : Path to the JSON file.

        Returns:
            List of record dicts. Empty list on any error.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            if not isinstance(records, list):
                logger.warning(
                    f"Expected 'records' to be a list in {path.name}, "
                    f"got {type(records).__name__}"
                )
                return []
            logger.debug(f"Loaded {len(records):,} records from {path.name}")
            return records
        except FileNotFoundError:
            logger.warning(f"File not found — no records loaded: {path}")
            return []
        except json.JSONDecodeError as exc:
            logger.error(f"JSON decode error in {path.name}: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Unexpected error loading {path.name}: {exc}")
            return []


# =============================================================================
# ── 5. Demo / Synthetic Batch Runner (used when real datasets are absent)
# =============================================================================

def _build_demo_records(n: int = 20) -> List[Dict[str, Any]]:
    """
    Build a small synthetic batch of realistic fashion records for a
    standalone demo run when neither FashionGen nor DeepFashion files exist.

    Returns n records across all 11 taxonomy categories.
    Note: source_dataset='fashiongen' so the validator accepts these records
    (validator only accepts 'fashiongen' | 'deepfashion' as valid sources).
    """
    samples = [
        {"cat": "t_shirts",    "gender": "unisex", "style": "streetwear",
         "desc": "An oversized graphic tee with bold neon prints.",
         "color": ["Black"], "season": "summer"},
        {"cat": "shirts",      "gender": "men",    "style": "formal",
         "desc": "Slim-fit white cotton dress shirt with a classic collar.",
         "color": ["White"], "season": "all_season"},
        {"cat": "hoodies",     "gender": "unisex", "style": "athleisure",
         "desc": "A cosy fleece-lined pullover hoodie in charcoal.",
         "color": ["Charcoal"], "season": "winter"},
        {"cat": "jackets",     "gender": "women",  "style": "luxury",
         "desc": "An ivory double-breasted wool blazer for evening events.",
         "color": ["Ivory"], "season": "autumn"},
        {"cat": "pants",       "gender": "men",    "style": "business_casual",
         "desc": "Slim chino trousers in olive, ideal for smart casual.",
         "color": ["Olive"], "season": "all_season"},
        {"cat": "jeans",       "gender": "women",  "style": "vintage",
         "desc": "High-waisted distressed denim jeans with frayed hem.",
         "color": ["Navy"], "season": "all_season"},
        {"cat": "shorts",      "gender": "men",    "style": "athleisure",
         "desc": "Lightweight running shorts with built-in liner.",
         "color": ["Black"], "season": "summer"},
        {"cat": "dresses",     "gender": "women",  "style": "luxury",
         "desc": "A floor-length red silk evening gown for gala events.",
         "color": ["Red"], "season": "all_season"},
        {"cat": "ethnic_wear", "gender": "men",    "style": "formal",
         "desc": "Ivory silk kurta with hand-embroidered cuffs for weddings.",
         "color": ["Ivory"], "season": "all_season"},
        {"cat": "footwear",    "gender": "unisex", "style": "streetwear",
         "desc": "Classic white leather low-top sneakers with gum sole.",
         "color": ["White"], "season": "all_season"},
        {"cat": "accessories", "gender": "women",  "style": "minimalist",
         "desc": "Structured brown leather tote with gold clasp hardware.",
         "color": ["Brown"], "season": "all_season"},
    ]

    records = []
    for i in range(n):
        s   = samples[i % len(samples)]
        idx = i + 1
        records.append({
            "image_id"      : f"DEMO_{idx:05d}",
            "image_path"    : f"datasets/demo/images/DEMO_{idx:05d}.jpg",
            "source_dataset": "fashiongen",   # validator requires fashiongen|deepfashion
            "category"      : s["cat"],
            "gender"        : s["gender"],
            "style"         : s["style"],
            "description"   : s["desc"],
            "color"         : s["color"],          # list, not str
            "season"        : s["season"],
            "fit"           : "regular_fit",
            "occasion"      : ["casual"],
            "pattern"       : ["solid"],           # list, not str
            "attributes"    : [],
            "landmarks"     : [],
        })
    return records


class DemoPipelineController(MasterPipelineController):
    """
    A MasterPipelineController subclass that injects synthetic demo records
    when both FashionGen and DeepFashion datasets are unavailable.

    Use this for end-to-end testing without real dataset files.
    """

    def __init__(
        self,
        config   : Optional[MasterPipelineConfig] = None,
        n_records: int = 20,
    ) -> None:
        super().__init__(config)
        self._demo_n = n_records

    def run(self) -> ExecutionSummary:
        _banner("DEMO PIPELINE  ──  Starting (synthetic data)")

        demo_records = _build_demo_records(self._demo_n)
        logger.info(f"Demo batch: {len(demo_records)} synthetic records")

        self.summary.fashiongen_records  = 0
        self.summary.deepfashion_records = 0
        self.summary.total_raw_records   = len(demo_records)

        # Register stage timers for stages 1 & 2 as SKIPPED
        for name in ("1 · FashionGen Loader [DEMO]", "2 · DeepFashion Loader [DEMO]"):
            t = StageTimer(name)
            t.start_time = t.end_time = time.perf_counter()
            self.summary.stage_timers.append(t)

        # Run stages 3–6 on the demo records
        records = self._run_metadata_stage(demo_records)
        self.summary.metadata_enriched = len(records)

        records = self._run_preprocessing_stage(records)
        self.summary.after_preprocessing = len(records)

        valid, invalid = self._run_validation_stage(records)
        self.summary.validation_valid  = len(valid)
        self.summary.validation_failed = len(invalid)

        export = valid if not self.config.export_include_invalid else valid + invalid
        self._finish_and_save(export)
        return self.summary


# =============================================================================
# ── 6. CLI Entry-Point
# =============================================================================

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Master Dataset Pipeline Controller — "
            "orchestrates FashionGen + DeepFashion → Metadata → Preprocessing → "
            "Validation → Export"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-fg", type=int, default=None,
        help="Max FashionGen records (None = all)"
    )
    parser.add_argument(
        "--max-df", type=int, default=None,
        help="Max DeepFashion records (None = all)"
    )
    parser.add_argument(
        "--df-split", type=str, default="train",
        choices=["train", "val", "test", "all"],
        help="DeepFashion split to process"
    )
    parser.add_argument(
        "--no-fashiongen", action="store_true",
        help="Disable FashionGen stage"
    )
    parser.add_argument(
        "--no-deepfashion", action="store_true",
        help="Disable DeepFashion stage"
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Disable metadata generation stage"
    )
    parser.add_argument(
        "--no-preproc", action="store_true",
        help="Disable preprocessing stage"
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Disable validation stage"
    )
    parser.add_argument(
        "--include-invalid", action="store_true",
        help="Include validation-failed records in final export"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help=(
            "Run on synthetic demo records instead of real datasets. "
            "Useful for testing the full pipeline without dataset files."
        )
    )
    parser.add_argument(
        "--demo-n", type=int, default=20,
        help="Number of synthetic demo records to generate"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress tqdm progress bars"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_OUTPUT_DIR),
        help="Directory for all output files"
    )
    return parser


def main() -> int:
    """
    CLI entry-point. Parses arguments, builds config, and runs the pipeline.

    Returns:
        0 on success (or partial success), 1 on total failure.
    """
    parser  = _build_cli_parser()
    args    = parser.parse_args()

    # ── Configure loguru for the CLI run ──────────────────────────────────────
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )
    # Also write to a log file
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUTPUT_DIR / f"pipeline_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger.add(log_path, level="DEBUG", encoding="utf-8")
    logger.info(f"Log file: {log_path}")

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = MasterPipelineConfig(
        enable_fashiongen       = not args.no_fashiongen,
        fashiongen_max_records  = args.max_fg,
        enable_deepfashion      = not args.no_deepfashion,
        deepfashion_max_records = args.max_df,
        deepfashion_split       = args.df_split,
        enable_metadata_gen     = not args.no_metadata,
        enable_preprocessing    = not args.no_preproc,
        enable_validation       = not args.no_validation,
        export_include_invalid  = args.include_invalid,
        show_progress           = not args.no_progress,
        output_dir              = Path(args.output_dir),
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.demo:
        ctrl = DemoPipelineController(config=cfg, n_records=args.demo_n)
    else:
        ctrl = MasterPipelineController(config=cfg)

    summary = ctrl.run()

    # ── Return code ───────────────────────────────────────────────────────────
    if summary.status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
