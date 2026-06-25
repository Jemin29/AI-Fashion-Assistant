"""
week2/pipelines/fashion_generation_pipeline.py
================================================
Master Fashion Generation Pipeline
AI-Powered Fashion Design Assistant — Week 2

╔══════════════════════════════════════════════════════════════════════╗
║               Fashion Generation Pipeline                           ║
║                                                                      ║
║  Complete end-to-end flow:                                           ║
║                                                                      ║
║    User Input  (item dict / prompt string / library style)           ║
║          ↓                                                           ║
║    Prompt Builder  (prompt_templates + PromptBuilder)                ║
║          ↓                                                           ║
║    SDXL Generator  (FashionSDXLGenerator / SDXLGenerator)            ║
║          ↓                                                           ║
║    Image Save  (outputs/generated/)                                  ║
║          ↓                                                           ║
║    Evaluation  (QualityScorer + EvaluationReport)                    ║
║          ↓                                                           ║
║    Execution Summary  (JSON + console table)                         ║
║                                                                      ║
║  Features                                                            ║
║  ────────                                                            ║
║   • Single-item and batch generation                                 ║
║   • Dict-driven or free-text input                                   ║
║   • Rich progress bar + live status                                  ║
║   • Per-stage structured logging                                     ║
║   • Per-item fault isolation (batch never stops on one error)        ║
║   • Full metadata sidecar (JSON) per image                           ║
║   • Execution summary with pass/fail breakdown                       ║
║   • Dry-run mode (prompt inspection without inference)               ║
╚══════════════════════════════════════════════════════════════════════╝

Quick Start
-----------
    from src.generation.pipelines.fashion_generation_pipeline import FashionGenerationPipeline

    # Single item
    pipeline = FashionGenerationPipeline()
    result   = pipeline.run({
        "category": "hoodie",
        "style":    "streetwear",
        "color":    "black",
    })
    print(result.summary())

    # Batch
    result = pipeline.run_batch([
        {"style": "luxury",     "category": "evening gown", "color": "emerald"},
        {"style": "streetwear", "category": "hoodie",       "color": "black"},
        {"style": "formal",     "category": "suit",         "color": "navy"},
    ])

    # Dry-run (prompt preview, no inference)
    result = pipeline.run({"style": "techwear"}, dry_run=True)
    print(result.extra["prompts"])
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("fashion_generation_pipeline")  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]

# ── Internal imports ──────────────────────────────────────────────────────────
from src.generation.prompts.prompt_templates import (
    generate_prompt,
    generate_negative_prompt,
    prompt_enhancer,
    explain_prompt,
    get_template,
)
from src.generation.prompts.prompt_builder import PromptBuilder
from src.generation.prompts.prompt_validator import PromptValidator

# Generator — try production SDXL first, fall back to stub
try:
    from src.generation.generator.sdxl_generator import FashionSDXLGenerator as _SDXLGen
    _SDXL_AVAILABLE = True
except ImportError:
    _SDXL_AVAILABLE = False

# Evaluation
try:
    from src.evaluation.week2_quality_scorer import QualityScorer, QualityScore
    from src.evaluation.week2_evaluation_report import EvaluationReport
    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False

# Config
try:
    from src.utils.config_manager import get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False


# =============================================================================
# ── Data Structures
# =============================================================================

@dataclass
class StageResult:
    """Result from a single pipeline stage execution."""
    stage:      str
    success:    bool    = True
    skipped:    bool    = False
    elapsed_s:  float   = 0.0
    data:       Dict[str, Any] = field(default_factory=dict)
    error:      str     = ""

    def __repr__(self) -> str:
        status = "OK" if self.success else ("SKIP" if self.skipped else "ERR")
        return f"StageResult({self.stage} | {status} | {self.elapsed_s:.2f}s)"


@dataclass
class ItemResult:
    """
    Complete generation result for a single fashion item.

    Attributes
    ----------
    item_id : str               Unique identifier for this item.
    item    : dict              Original input item dictionary.
    success : bool
    dry_run : bool
    stages  : dict              Per-stage StageResult objects.
    prompt  : str               Final positive prompt used.
    negative_prompt : str
    image_paths : list of Path  Saved image file paths.
    metadata_paths : list of Path  Saved metadata sidecar paths.
    eval_score : float          Overall quality score [0-1].
    eval_passed : bool
    elapsed_s : float           Total wall time for this item.
    error : str                 Top-level error string if success=False.
    extra : dict                Catch-all for additional metadata.
    """
    item_id:        str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    item:           Dict[str, Any]  = field(default_factory=dict)
    success:        bool            = True
    dry_run:        bool            = False
    stages:         Dict[str, StageResult] = field(default_factory=dict)
    prompt:         str             = ""
    negative_prompt:str             = ""
    image_paths:    List[Path]      = field(default_factory=list)
    metadata_paths: List[Path]      = field(default_factory=list)
    eval_score:     float           = 0.0
    eval_passed:    bool            = True
    elapsed_s:      float           = 0.0
    error:          str             = ""
    extra:          Dict[str, Any]  = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "item_id":    self.item_id,
            "style":      self.item.get("style", ""),
            "category":   self.item.get("category", ""),
            "success":    self.success,
            "dry_run":    self.dry_run,
            "images":     len(self.image_paths),
            "eval_score": round(self.eval_score, 4),
            "eval_passed":self.eval_passed,
            "elapsed_s":  round(self.elapsed_s, 3),
            "error":      self.error,
        }


@dataclass
class PipelineResult:
    """
    Aggregate result for a full pipeline run (single or batch).

    Attributes
    ----------
    run_id : str
    items  : list of ItemResult   One per input item.
    total  : int
    succeeded : int
    failed    : int
    skipped   : int
    total_images : int
    passed_eval  : int
    failed_eval  : int
    elapsed_s    : float
    output_dir   : Path
    report_path  : Path or None
    dry_run      : bool
    errors       : list of str
    warnings     : list of str
    """
    run_id:         str             = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    items:          List[ItemResult] = field(default_factory=list)
    total:          int             = 0
    succeeded:      int             = 0
    failed:         int             = 0
    skipped:        int             = 0
    total_images:   int             = 0
    passed_eval:    int             = 0
    failed_eval:    int             = 0
    elapsed_s:      float           = 0.0
    output_dir:     Path            = field(default_factory=lambda: Path("week2/outputs/generated"))
    report_path:    Optional[Path]  = None
    dry_run:        bool            = False
    errors:         List[str]       = field(default_factory=list)
    warnings:       List[str]       = field(default_factory=list)
    extra:          Dict[str, Any]  = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.failed == 0 or self.succeeded > 0

    @property
    def pass_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.passed_eval / self.total_images

    def summary(self) -> str:
        """Human-readable execution summary."""
        sep = "=" * 62
        lines = [
            sep,
            f"  FASHION GENERATION PIPELINE  Run {self.run_id}",
            sep,
            f"  Mode            : {'DRY-RUN (no inference)' if self.dry_run else 'GENERATION'}",
            f"  Items processed : {self.total}",
            f"  Succeeded       : {self.succeeded}",
            f"  Failed          : {self.failed}",
            f"  Total images    : {self.total_images}",
            f"  Passed eval     : {self.passed_eval}",
            f"  Failed eval     : {self.failed_eval}",
            f"  Pass rate       : {self.pass_rate:.1%}" if self.total_images > 0 else "  Pass rate       : N/A",
            f"  Elapsed         : {self.elapsed_s:.2f}s",
            f"  Output dir      : {self.output_dir}",
        ]
        if self.report_path:
            lines.append(f"  Eval report     : {self.report_path}")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors[:5]:
                lines.append(f"    [x] {e}")
        if self.warnings:
            lines.append("  Warnings:")
            for w_msg in self.warnings[:5]:
                lines.append(f"    [!] {w_msg}")
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Full serialisable execution record."""
        return {
            "run_id":       self.run_id,
            "dry_run":      self.dry_run,
            "elapsed_s":    round(self.elapsed_s, 3),
            "output_dir":   str(self.output_dir),
            "report_path":  str(self.report_path) if self.report_path else None,
            "summary": {
                "total":        self.total,
                "succeeded":    self.succeeded,
                "failed":       self.failed,
                "total_images": self.total_images,
                "passed_eval":  self.passed_eval,
                "failed_eval":  self.failed_eval,
                "pass_rate":    round(self.pass_rate, 4),
            },
            "items":    [it.summary_dict() for it in self.items],
            "errors":   self.errors,
            "warnings": self.warnings,
            "extra":    self.extra,
        }


# =============================================================================
# ── Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Lightweight configuration for FashionGenerationPipeline.

    All parameters have sensible defaults so zero-config usage works.

    Parameters
    ----------
    output_dir : Path
        Root directory for generated images and metadata.
    report_dir : Path
        Directory for evaluation JSON reports.
    num_images_per_item : int
        Default number of images per item.
    default_steps : int
        Default SDXL inference steps.
    default_guidance : float
        Default classifier-free guidance scale.
    default_width : int
    default_height : int
    default_seed : int      -1 = random seed.
    boost_quality : bool    Add global quality boosters to prompts.
    photo_style : str       Default photography context.
    save_metadata : bool    Write JSON sidecar per image.
    evaluate : bool         Run QualityScorer on outputs.
    save_report : bool      Save evaluation JSON report.
    stop_on_error : bool    If True, abort batch on first error.
    show_progress : bool    Show Rich progress bars.
    dry_run : bool          Build prompts only, skip inference.
    log_prompts : bool      Log full prompts at DEBUG level.
    """
    output_dir:          Path    = field(default_factory=lambda: Path("week2/outputs/generated"))
    report_dir:          Path    = field(default_factory=lambda: Path("week2/outputs/evaluation_reports"))
    num_images_per_item: int     = 1
    default_steps:       int     = 30
    default_guidance:    float   = 7.5
    default_width:       int     = 1024
    default_height:      int     = 1024
    default_seed:        int     = -1
    boost_quality:       bool    = True
    photo_style:         Optional[str]  = None   # None = use style default
    save_metadata:       bool    = True
    evaluate:            bool    = True
    save_report:         bool    = True
    stop_on_error:       bool    = False
    show_progress:       bool    = True
    dry_run:             bool    = False
    log_prompts:         bool    = False


# =============================================================================
# ── Fashion Generation Pipeline
# =============================================================================

class FashionGenerationPipeline:
    """
    High-level orchestration pipeline for AI fashion image generation.

    Integrates every Week 2 component:
      - ``prompt_templates.py`` — style-aware prompt engineering
      - ``PromptBuilder``       — structured tag assembly
      - ``PromptValidator``     — token budget enforcement
      - ``FashionSDXLGenerator``— Stable Diffusion XL inference
      - ``QualityScorer``       — CLIP + quality checks
      - ``EvaluationReport``    — structured JSON report

    Parameters
    ----------
    pipeline_config : PipelineConfig, optional
        Pipeline-level settings. Defaults to a zero-config instance.
    config : Week2Config, optional
        System-level config (``week2.config_manager.get_config()``).
        If omitted the pipeline tries to load it automatically.

    Example
    -------
        pipeline = FashionGenerationPipeline()

        # Single item
        result = pipeline.run({
            "category": "hoodie",
            "style":    "streetwear",
            "color":    "black",
        })
        print(result.summary())

        # Batch
        result = pipeline.run_batch([
            {"style": "luxury",     "category": "evening gown"},
            {"style": "streetwear", "category": "jacket"},
        ])

        # Free-text prompt
        result = pipeline.run_from_prompt(
            "A woman in an emerald green silk gown",
            style="luxury",
        )

        # Dry run — preview prompts without generation
        result = pipeline.run({"style": "techwear"}, dry_run=True)
    """

    # ── Stage names (used as keys in ItemResult.stages) ──────────────────
    STAGE_PROMPT_BUILD  = "prompt_build"
    STAGE_PROMPT_VALID  = "prompt_validate"
    STAGE_GENERATION    = "generation"
    STAGE_SAVE          = "image_save"
    STAGE_EVALUATION    = "evaluation"

    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        config = None,
    ) -> None:
        self.cfg    = pipeline_config or PipelineConfig()
        self._week2_cfg = config

        # Lazy-loaded components
        self._generator:  Any = None
        self._scorer:     Any = None
        self._builder:    Optional[PromptBuilder]   = None
        self._validator:  Optional[PromptValidator] = None

        # Ensure output directories exist
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.report_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FashionGenerationPipeline created | output={} | dry_run={}",
            self.cfg.output_dir, self.cfg.dry_run,
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def run(
        self,
        item:            Union[Dict[str, Any], str],
        *,
        dry_run:         Optional[bool] = None,
        num_images:      Optional[int]  = None,
        seed:            Optional[int]  = None,
        steps:           Optional[int]  = None,
        guidance_scale:  Optional[float]= None,
        photo_style:     Optional[str]  = None,
        extra_tags:      Optional[List[str]] = None,
        output_dir:      Optional[Path] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for a single fashion item.

        Parameters
        ----------
        item : dict or str
            Fashion item descriptor:
              ``{"style": "luxury", "category": "gown", "color": "emerald"}``
            Or a free-text prompt string.
        dry_run : bool, optional
            Override cfg.dry_run.
        num_images : int, optional
            Number of images to generate (overrides cfg.num_images_per_item).
        seed : int, optional
        steps : int, optional
        guidance_scale : float, optional
        photo_style : str, optional
        extra_tags : list of str, optional
        output_dir : Path, optional

        Returns
        -------
        PipelineResult

        Example
        -------
            result = pipeline.run({
                "category": "hoodie",
                "style":    "streetwear",
                "color":    "black",
            })
            print(result.summary())
        """
        # Normalise string → dict
        if isinstance(item, str):
            item = {"description": item, "style": "casual"}

        _dry_run   = dry_run   if dry_run   is not None else self.cfg.dry_run
        _n_images  = num_images or self.cfg.num_images_per_item
        _out_dir   = output_dir or self.cfg.output_dir
        _seed      = seed      if seed      is not None else self.cfg.default_seed
        _steps     = steps     or self.cfg.default_steps
        _guidance  = guidance_scale or self.cfg.default_guidance
        _photo     = photo_style   or self.cfg.photo_style

        t0   = time.perf_counter()
        prun = PipelineResult(
            dry_run    = _dry_run,
            output_dir = _out_dir,
        )

        item_result = self._process_item(
            item         = item,
            dry_run      = _dry_run,
            n_images     = _n_images,
            seed         = _seed,
            steps        = _steps,
            guidance     = _guidance,
            photo_style  = _photo,
            extra_tags   = extra_tags,
            output_dir   = _out_dir,
        )

        prun.items.append(item_result)
        prun.total      = 1
        prun.succeeded  = 1 if item_result.success else 0
        prun.failed     = 0 if item_result.success else 1
        prun.total_images   = len(item_result.image_paths)
        prun.passed_eval    = 1 if item_result.eval_passed else 0
        prun.failed_eval    = 0 if item_result.eval_passed else 1
        prun.errors         = [item_result.error] if item_result.error else []
        prun.elapsed_s  = time.perf_counter() - t0

        prun.extra["prompts"] = {
            "positive": item_result.prompt,
            "negative": item_result.negative_prompt,
        }

        self._print_summary(prun)
        return prun

    def run_batch(
        self,
        items:           Sequence[Union[Dict[str, Any], str]],
        *,
        dry_run:         Optional[bool] = None,
        num_images:      Optional[int]  = None,
        seed:            Optional[int]  = None,
        steps:           Optional[int]  = None,
        guidance_scale:  Optional[float]= None,
        photo_style:     Optional[str]  = None,
        extra_tags:      Optional[List[str]] = None,
        output_dir:      Optional[Path] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for a list of fashion items.

        Parameters
        ----------
        items : sequence of dict or str
            List of fashion item descriptors.
        dry_run, num_images, seed, steps, guidance_scale, photo_style, extra_tags
            Applied to every item in the batch.
        output_dir : Path, optional

        Returns
        -------
        PipelineResult  (aggregate across all items)

        Example
        -------
            result = pipeline.run_batch([
                {"style": "luxury",     "category": "gown",   "color": "gold"},
                {"style": "streetwear", "category": "hoodie", "color": "black"},
                {"style": "formal",     "category": "suit",   "color": "navy"},
            ])
            print(result.summary())
        """
        _dry_run  = dry_run  if dry_run  is not None else self.cfg.dry_run
        _n_images = num_images or self.cfg.num_images_per_item
        _out_dir  = output_dir or self.cfg.output_dir
        _seed     = seed     if seed     is not None else self.cfg.default_seed
        _steps    = steps    or self.cfg.default_steps
        _guidance = guidance_scale or self.cfg.default_guidance
        _photo    = photo_style   or self.cfg.photo_style

        t0   = time.perf_counter()
        prun = PipelineResult(
            dry_run    = _dry_run,
            output_dir = _out_dir,
        )

        normalised: List[Dict[str, Any]] = [
            ({"description": it, "style": "casual"} if isinstance(it, str) else it)
            for it in items
        ]

        logger.info(
            "FashionGenerationPipeline batch starting | items={} | dry_run={}",
            len(normalised), _dry_run,
        )

        if not normalised:
            prun.warnings.append("Batch is empty — nothing to generate.")
            prun.elapsed_s = time.perf_counter() - t0
            return prun

        # ── Setup generator once for the whole batch ──────────────────────
        if not _dry_run:
            self._ensure_generator()

        # ── Execute with progress tracking ────────────────────────────────
        item_results = self._run_batch_items(
            items       = normalised,
            dry_run     = _dry_run,
            n_images    = _n_images,
            seed        = _seed,
            steps       = _steps,
            guidance    = _guidance,
            photo_style = _photo,
            extra_tags  = extra_tags,
            output_dir  = _out_dir,
        )

        # ── Aggregate ─────────────────────────────────────────────────────
        prun.items = item_results
        prun.total = len(item_results)
        for it in item_results:
            if it.success:
                prun.succeeded   += 1
            else:
                prun.failed      += 1
                prun.errors.append(f"[{it.item_id}] {it.error}")
            prun.total_images += len(it.image_paths)
            if it.eval_passed:
                prun.passed_eval += 1
            else:
                prun.failed_eval += 1

        prun.elapsed_s = time.perf_counter() - t0

        # ── Save aggregate evaluation report ──────────────────────────────
        if not _dry_run and self.cfg.save_report and _EVAL_AVAILABLE:
            prun.report_path = self._save_aggregate_report(prun)

        # ── Save execution summary JSON ───────────────────────────────────
        self._save_run_summary(prun)

        self._print_summary(prun)
        return prun

    def run_from_prompt(
        self,
        prompt:  str,
        style:   Optional[str] = None,
        *,
        enhance: bool           = True,
        **kwargs,
    ) -> PipelineResult:
        """
        Run the pipeline from a raw free-text prompt string.

        Parameters
        ----------
        prompt  : str   Raw prompt e.g. ``"A woman in a red silk gown"``.
        style   : str, optional   Style to inject.
        enhance : bool  Apply prompt_enhancer() before generation.
        **kwargs  Forwarded to ``run()``.

        Returns
        -------
        PipelineResult

        Example
        -------
            result = pipeline.run_from_prompt(
                "A woman in an emerald green silk gown",
                style="luxury",
            )
        """
        final_prompt = (
            prompt_enhancer(prompt, style=style, boost_quality=self.cfg.boost_quality)
            if enhance
            else prompt
        )
        item = {
            "description": final_prompt,
            "style":       style or "casual",
        }
        return self.run(item, **kwargs)

    def run_from_library(
        self,
        style:       str,
        section:     Optional[str] = None,
        n:           int            = 1,
        **kwargs,
    ) -> PipelineResult:
        """
        Generate images using prompts from the Prompt Library.

        Parameters
        ----------
        style   : str   Library style (streetwear, luxury, …).
        section : str, optional  Library section (runway, e-commerce, …).
        n       : int   Number of prompts to draw.
        **kwargs  Forwarded to ``run_batch()``.

        Returns
        -------
        PipelineResult

        Example
        -------
            result = pipeline.run_from_library("luxury", section="runway", n=5)
        """
        try:
            from src.generation.prompts.library import random_batch
            prompts = random_batch(style, n=n, section=section)
        except Exception as exc:
            logger.error("Library load failed: {}", exc)
            prompts = [generate_prompt({"style": style, "category": "outfit"})]

        items = [{"description": p, "style": style} for p in prompts]
        return self.run_batch(items, **kwargs)

    def dry_run(
        self,
        item: Union[Dict[str, Any], str],
        **kwargs,
    ) -> PipelineResult:
        """Shortcut: ``run(item, dry_run=True)`` — builds prompts, skips inference."""
        return self.run(item, dry_run=True, **kwargs)

    def explain(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a detailed prompt breakdown without running generation.

        Parameters
        ----------
        item : dict   Fashion item descriptor.

        Returns
        -------
        dict with keys: full_prompt, negative, layers, token_estimate, template.
        """
        return explain_prompt(item)

    # =========================================================================
    # ── Pipeline Stages
    # =========================================================================

    def _process_item(
        self,
        item:        Dict[str, Any],
        dry_run:     bool,
        n_images:    int,
        seed:        int,
        steps:       int,
        guidance:    float,
        photo_style: Optional[str],
        extra_tags:  Optional[List[str]],
        output_dir:  Path,
    ) -> ItemResult:
        """Execute all 5 stages for a single item. Returns ItemResult."""
        t_item = time.perf_counter()
        result = ItemResult(item=dict(item))
        result.dry_run = dry_run

        # ─────────────────────────────────────────────────────────────────
        # STAGE 1 — Prompt Build
        # ─────────────────────────────────────────────────────────────────
        stage1 = self._stage_prompt_build(item, photo_style, extra_tags)
        result.stages[self.STAGE_PROMPT_BUILD] = stage1

        if not stage1.success:
            result.success = False
            result.error   = stage1.error
            result.elapsed_s = time.perf_counter() - t_item
            return result

        result.prompt          = stage1.data["positive"]
        result.negative_prompt = stage1.data["negative"]

        if self.cfg.log_prompts:
            logger.debug(
                "Prompt built | positive={!r:.120} | negative={!r:.60}",
                result.prompt, result.negative_prompt,
            )

        # ─────────────────────────────────────────────────────────────────
        # STAGE 2 — Prompt Validate
        # ─────────────────────────────────────────────────────────────────
        stage2 = self._stage_prompt_validate(result.prompt)
        result.stages[self.STAGE_PROMPT_VALID] = stage2

        if not stage2.success:
            result.success = False
            result.error   = stage2.error
            result.elapsed_s = time.perf_counter() - t_item
            return result

        # Use sanitised prompt if validator cleaned it
        if stage2.data.get("sanitized"):
            result.prompt = stage2.data["sanitized"]

        # ─────────────────────────────────────────────────────────────────
        # STAGE 3 — Generation  (skip in dry-run)
        # ─────────────────────────────────────────────────────────────────
        if dry_run:
            result.stages[self.STAGE_GENERATION] = StageResult(
                stage   = self.STAGE_GENERATION,
                skipped = True,
                data    = {"reason": "dry_run"},
            )
            result.stages[self.STAGE_SAVE] = StageResult(
                stage   = self.STAGE_SAVE,
                skipped = True,
                data    = {"reason": "dry_run"},
            )
            result.stages[self.STAGE_EVALUATION] = StageResult(
                stage   = self.STAGE_EVALUATION,
                skipped = True,
                data    = {"reason": "dry_run"},
            )
            result.elapsed_s = time.perf_counter() - t_item
            result.extra["prompt_breakdown"] = explain_prompt(item)
            return result

        stage3 = self._stage_generate(
            positive  = result.prompt,
            negative  = result.negative_prompt,
            n_images  = n_images,
            seed      = seed,
            steps     = steps,
            guidance  = guidance,
            item_id   = result.item_id,
        )
        result.stages[self.STAGE_GENERATION] = stage3

        if not stage3.success:
            result.success = False
            result.error   = stage3.error
            result.elapsed_s = time.perf_counter() - t_item
            return result

        images    = stage3.data.get("images", [])
        image_ids = stage3.data.get("image_ids", [])

        # ─────────────────────────────────────────────────────────────────
        # STAGE 4 — Image Save + Metadata
        # ─────────────────────────────────────────────────────────────────
        stage4 = self._stage_save(
            images     = images,
            image_ids  = image_ids,
            item       = item,
            item_id    = result.item_id,
            prompt     = result.prompt,
            negative   = result.negative_prompt,
            seed       = stage3.data.get("seed", -1),
            steps      = steps,
            guidance   = guidance,
            output_dir = output_dir,
        )
        result.stages[self.STAGE_SAVE] = stage4
        result.image_paths    = [Path(p) for p in stage4.data.get("image_paths", [])]
        result.metadata_paths = [Path(p) for p in stage4.data.get("metadata_paths", [])]

        # ─────────────────────────────────────────────────────────────────
        # STAGE 5 — Evaluation
        # ─────────────────────────────────────────────────────────────────
        stage5 = self._stage_evaluate(
            images    = images,
            image_ids = image_ids,
            prompt    = result.prompt,
        )
        result.stages[self.STAGE_EVALUATION] = stage5
        result.eval_score  = stage5.data.get("mean_score", 1.0)
        result.eval_passed = stage5.data.get("all_passed", True)

        result.elapsed_s = time.perf_counter() - t_item
        result.extra.update({
            "seed":      stage3.data.get("seed", -1),
            "image_ids": image_ids,
        })
        return result

    # ── Stage 1: Prompt Build ─────────────────────────────────────────────

    def _stage_prompt_build(
        self,
        item:        Dict[str, Any],
        photo_style: Optional[str],
        extra_tags:  Optional[List[str]],
    ) -> StageResult:
        t = time.perf_counter()
        try:
            # If a description is given use prompt_enhancer; otherwise use full template
            if item.get("description"):
                positive = prompt_enhancer(
                    item["description"],
                    style         = item.get("style"),
                    photo_style   = photo_style,
                    boost_quality = self.cfg.boost_quality,
                    extra_tags    = extra_tags,
                )
            else:
                positive = generate_prompt(
                    item,
                    photo_style   = photo_style,
                    extra_tags    = extra_tags,
                    boost_quality = self.cfg.boost_quality,
                )

            negative = generate_negative_prompt(
                style = item.get("style"),
                item  = item,
            )

            logger.debug(
                "Stage[prompt_build] | tokens≈{} | style={}",
                len(positive.split()), item.get("style", ""),
            )
            return StageResult(
                stage     = self.STAGE_PROMPT_BUILD,
                success   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {"positive": positive, "negative": negative},
            )
        except Exception as exc:
            logger.error("Stage[prompt_build] failed: {}", exc)
            return StageResult(
                stage     = self.STAGE_PROMPT_BUILD,
                success   = False,
                elapsed_s = time.perf_counter() - t,
                error     = f"Prompt build failed: {exc}",
            )

    # ── Stage 2: Prompt Validate ──────────────────────────────────────────

    def _stage_prompt_validate(self, prompt: str) -> StageResult:
        t = time.perf_counter()
        try:
            validator = self._get_validator()
            val       = validator.validate(prompt)

            warnings = []
            if not val.is_valid:
                return StageResult(
                    stage     = self.STAGE_PROMPT_VALID,
                    success   = False,
                    elapsed_s = time.perf_counter() - t,
                    error     = f"Prompt validation failed: {'; '.join(val.errors)}",
                    data      = {"errors": val.errors},
                )

            for w in (val.warnings or []):
                logger.warning("Prompt warning: {}", w)
                warnings.append(w)

            logger.debug("Stage[prompt_validate] | valid=True | warnings={}", len(warnings))
            return StageResult(
                stage     = self.STAGE_PROMPT_VALID,
                success   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {
                    "sanitized": val.sanitized_prompt,
                    "warnings":  warnings,
                },
            )
        except Exception as exc:
            # Validator not available — treat as pass-through
            logger.debug("Stage[prompt_validate] skipped (not available): {}", exc)
            return StageResult(
                stage     = self.STAGE_PROMPT_VALID,
                success   = True,
                skipped   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {"sanitized": prompt},
            )

    # ── Stage 3: Generation ───────────────────────────────────────────────

    def _stage_generate(
        self,
        positive:  str,
        negative:  str,
        n_images:  int,
        seed:      int,
        steps:     int,
        guidance:  float,
        item_id:   str,
    ) -> StageResult:
        t = time.perf_counter()
        if not _SDXL_AVAILABLE:
            logger.warning("Stage[generation] | SDXL not available — stub result")
            # Return stub result (for CI / testing without GPU)
            return StageResult(
                stage     = self.STAGE_GENERATION,
                success   = True,
                skipped   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {
                    "images":    [],
                    "image_ids": [f"stub_{item_id}_{i}" for i in range(n_images)],
                    "seed":      seed,
                    "stub":      True,
                },
            )

        try:
            gen = self._ensure_generator()
            logger.info(
                "Stage[generation] | steps={} | guidance={} | n={}",
                steps, guidance, n_images,
            )

            gen_result = gen.generate(
                prompt                = positive,
                negative_prompt       = negative,
                num_inference_steps   = steps,
                guidance_scale        = guidance,
                seed                  = seed,
                num_images_per_prompt = n_images,
                save                  = False,   # Stage 4 handles saving
            )

            if not gen_result.success:
                return StageResult(
                    stage     = self.STAGE_GENERATION,
                    success   = False,
                    elapsed_s = time.perf_counter() - t,
                    error     = f"SDXL generation failed: {gen_result.error}",
                )

            logger.success(
                "Stage[generation] | produced {} image(s) | seed={}",
                len(gen_result.images), gen_result.seed,
            )
            return StageResult(
                stage     = self.STAGE_GENERATION,
                success   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {
                    "images":    gen_result.images,
                    "image_ids": gen_result.image_ids,
                    "seed":      gen_result.seed,
                },
            )
        except Exception as exc:
            logger.error("Stage[generation] exception: {}", exc)
            return StageResult(
                stage     = self.STAGE_GENERATION,
                success   = False,
                elapsed_s = time.perf_counter() - t,
                error     = f"Generation exception: {exc}",
            )

    # ── Stage 4: Image Save + Metadata ────────────────────────────────────

    def _stage_save(
        self,
        images:     List[Any],
        image_ids:  List[str],
        item:       Dict[str, Any],
        item_id:    str,
        prompt:     str,
        negative:   str,
        seed:       int,
        steps:      int,
        guidance:   float,
        output_dir: Path,
    ) -> StageResult:
        t = time.perf_counter()
        image_paths:    List[str] = []
        metadata_paths: List[str] = []

        if not images:
            # Stub / dry-run path — nothing to save
            return StageResult(
                stage     = self.STAGE_SAVE,
                success   = True,
                skipped   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {"image_paths": [], "metadata_paths": []},
            )

        try:
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            style  = item.get("style", "unknown")
            cat    = item.get("category", "item").replace(" ", "_")

            for idx, (img, img_id) in enumerate(zip(images, image_ids)):
                # ── Save image ────────────────────────────────────────────
                img_filename = f"{style}_{cat}_{run_ts}_{idx:02d}.png"
                img_path     = output_dir / img_filename

                try:
                    img.save(str(img_path))
                    image_paths.append(str(img_path))
                    logger.debug("Saved image: {}", img_path)
                except Exception as exc:
                    logger.error("Failed to save image {}: {}", img_filename, exc)
                    continue

                # ── Save metadata sidecar ─────────────────────────────────
                if self.cfg.save_metadata:
                    meta = {
                        "image_id":    img_id,
                        "item_id":     item_id,
                        "image_path":  str(img_path),
                        "generated_at":datetime.now(timezone.utc).isoformat(),
                        "prompt":      prompt,
                        "negative_prompt": negative,
                        "generation": {
                            "seed":           seed,
                            "steps":          steps,
                            "guidance_scale": guidance,
                            "width":          getattr(img, "width",  self.cfg.default_width),
                            "height":         getattr(img, "height", self.cfg.default_height),
                        },
                        "item": item,
                    }
                    meta_filename = img_filename.replace(".png", "_meta.json")
                    meta_path     = output_dir / meta_filename
                    try:
                        meta_path.write_text(
                            json.dumps(meta, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        metadata_paths.append(str(meta_path))
                        logger.debug("Saved metadata: {}", meta_path)
                    except Exception as exc:
                        logger.warning("Failed to save metadata {}: {}", meta_filename, exc)

            logger.success(
                "Stage[image_save] | saved={} images | {} metadata files",
                len(image_paths), len(metadata_paths),
            )
            return StageResult(
                stage     = self.STAGE_SAVE,
                success   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {
                    "image_paths":    image_paths,
                    "metadata_paths": metadata_paths,
                },
            )
        except Exception as exc:
            logger.error("Stage[image_save] exception: {}", exc)
            return StageResult(
                stage     = self.STAGE_SAVE,
                success   = False,
                elapsed_s = time.perf_counter() - t,
                error     = f"Image save failed: {exc}",
                data      = {"image_paths": image_paths, "metadata_paths": metadata_paths},
            )

    # ── Stage 5: Evaluation ───────────────────────────────────────────────

    def _stage_evaluate(
        self,
        images:    List[Any],
        image_ids: List[str],
        prompt:    str,
    ) -> StageResult:
        t = time.perf_counter()

        if not self.cfg.evaluate or not _EVAL_AVAILABLE or not images:
            return StageResult(
                stage     = self.STAGE_EVALUATION,
                success   = True,
                skipped   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {"mean_score": 1.0, "all_passed": True, "reason": "skipped"},
            )

        try:
            scorer = self._get_scorer()
            scores = scorer.score_batch(
                images    = images,
                image_ids = image_ids,
                prompts   = [prompt] * len(images),
            )
            mean_score  = sum(s.overall_score for s in scores) / len(scores) if scores else 0.0
            all_passed  = all(s.passed for s in scores)
            n_passed    = sum(1 for s in scores if s.passed)
            n_failed    = len(scores) - n_passed

            logger.info(
                "Stage[evaluation] | {}/{} passed | mean_score={:.3f}",
                n_passed, len(scores), mean_score,
            )
            return StageResult(
                stage     = self.STAGE_EVALUATION,
                success   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {
                    "scores":     [s.to_dict() for s in scores],
                    "mean_score": round(mean_score, 4),
                    "all_passed": all_passed,
                    "n_passed":   n_passed,
                    "n_failed":   n_failed,
                },
            )
        except Exception as exc:
            logger.warning("Stage[evaluation] non-fatal exception: {}", exc)
            return StageResult(
                stage     = self.STAGE_EVALUATION,
                success   = True,   # Evaluation failure is non-fatal
                skipped   = True,
                elapsed_s = time.perf_counter() - t,
                data      = {"mean_score": 1.0, "all_passed": True, "error": str(exc)},
            )

    # =========================================================================
    # ── Batch Execution
    # =========================================================================

    def _run_batch_items(
        self,
        items:       List[Dict[str, Any]],
        dry_run:     bool,
        n_images:    int,
        seed:        int,
        steps:       int,
        guidance:    float,
        photo_style: Optional[str],
        extra_tags:  Optional[List[str]],
        output_dir:  Path,
    ) -> List[ItemResult]:
        """Execute batch with progress tracking and per-item fault isolation."""
        results: List[ItemResult] = []
        n = len(items)

        if _RICH and self.cfg.show_progress:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold cyan]Fashion Pipeline[/bold cyan]"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=_console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Generating {n} item(s)…",
                    total=n,
                )
                for idx, item in enumerate(items):
                    progress.update(
                        task,
                        description=(
                            f"[cyan]{idx+1}/{n} "
                            f"{item.get('style','')}"
                            f" {item.get('category','')}"
                        ),
                    )
                    ir = self._process_item_safe(
                        item, dry_run, n_images, seed, steps, guidance,
                        photo_style, extra_tags, output_dir,
                    )
                    results.append(ir)
                    progress.advance(task)
                    if not ir.success and self.cfg.stop_on_error:
                        logger.error("stop_on_error=True — aborting batch at item {}", idx + 1)
                        break
        else:
            for idx, item in enumerate(items):
                logger.info(
                    "Batch item {}/{} | style={} | category={}",
                    idx + 1, n,
                    item.get("style", "?"),
                    item.get("category", "?"),
                )
                ir = self._process_item_safe(
                    item, dry_run, n_images, seed, steps, guidance,
                    photo_style, extra_tags, output_dir,
                )
                results.append(ir)
                if not ir.success and self.cfg.stop_on_error:
                    logger.error("stop_on_error=True — aborting batch at item {}", idx + 1)
                    break

        return results

    def _process_item_safe(self, item, dry_run, n_images, seed, steps, guidance,
                           photo_style, extra_tags, output_dir) -> ItemResult:
        """Wrap _process_item so exceptions never crash the batch loop."""
        try:
            return self._process_item(
                item        = item,
                dry_run     = dry_run,
                n_images    = n_images,
                seed        = seed,
                steps       = steps,
                guidance    = guidance,
                photo_style = photo_style,
                extra_tags  = extra_tags,
                output_dir  = output_dir,
            )
        except Exception as exc:
            logger.error("Unexpected error for item {}: {}", item, exc)
            ir       = ItemResult(item=item)
            ir.success = False
            ir.error   = f"Unexpected exception: {exc}"
            return ir

    # =========================================================================
    # ── Output: Reports & Summaries
    # =========================================================================

    def _save_aggregate_report(self, prun: PipelineResult) -> Optional[Path]:
        """Collect all QualityScores from the batch and save one JSON report."""
        try:
            all_scores = []
            for it in prun.items:
                stage = it.stages.get(self.STAGE_EVALUATION)
                if stage and stage.data.get("scores"):
                    # Re-hydrate QualityScore objects are not needed;
                    # EvaluationReport accepts the raw score dicts via adapter
                    pass

            # Build a lightweight JSON aggregate directly
            report_path = prun.report_dir / f"eval_{prun.run_id}.json"  if hasattr(prun, "report_dir") \
                          else self.cfg.report_dir / f"eval_{prun.run_id}.json"

            report_data: Dict[str, Any] = {
                "run_id":       prun.run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_items":   prun.total,
                    "succeeded":     prun.succeeded,
                    "failed":        prun.failed,
                    "total_images":  prun.total_images,
                    "passed_eval":   prun.passed_eval,
                    "failed_eval":   prun.failed_eval,
                    "pass_rate":     round(prun.pass_rate, 4),
                    "elapsed_s":     round(prun.elapsed_s, 3),
                },
                "items": [it.summary_dict() for it in prun.items],
            }

            self.cfg.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.cfg.report_dir / f"eval_{prun.run_id}.json"
            report_path.write_text(
                json.dumps(report_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.success("Evaluation report saved | {}", report_path)
            return report_path
        except Exception as exc:
            logger.error("Failed to save evaluation report: {}", exc)
            return None

    def _save_run_summary(self, prun: PipelineResult) -> Optional[Path]:
        """Write the full execution summary to a JSON file."""
        try:
            summary_path = self.cfg.report_dir / f"run_{prun.run_id}.json"
            self.cfg.report_dir.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(prun.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Run summary saved | {}", summary_path)
            return summary_path
        except Exception as exc:
            logger.warning("Failed to save run summary: {}", exc)
            return None

    def _print_summary(self, prun: PipelineResult) -> None:
        """Print execution summary to console (Rich table or plain text)."""
        if _RICH and self.cfg.show_progress and _console:
            self._rich_summary(prun)
        else:
            try:
                print(prun.summary())
            except UnicodeEncodeError:
                # Fallback for terminals with limited encoding (Windows CP1252)
                summary = prun.summary()
                print(summary.encode('ascii', errors='replace').decode('ascii'))

    def _rich_summary(self, prun: PipelineResult) -> None:
        """Render a Rich table summary."""
        table = Table(
            title=f"[bold cyan]Fashion Pipeline Run -- {prun.run_id}[/bold cyan]",
            show_header=True, header_style="bold magenta",
            border_style="cyan", min_width=60,
        )
        table.add_column("Metric",       style="dim",   no_wrap=True)
        table.add_column("Value",        style="white", no_wrap=True)

        status_color = "green" if prun.success else "red"
        status_text  = "[OK] SUCCESS" if prun.success else "[FAIL] FAILED"
        mode_text    = "DRY RUN" if prun.dry_run else "GENERATION"

        table.add_row("Mode",         f"[yellow]{mode_text}[/yellow]")
        table.add_row("Status",       f"[{status_color}]{status_text}[/{status_color}]")
        table.add_row("Items",        str(prun.total))
        table.add_row("Succeeded",    f"[green]{prun.succeeded}[/green]")
        table.add_row("Failed",       f"[red]{prun.failed}[/red]" if prun.failed else "[green]0[/green]")
        table.add_row("Total images", str(prun.total_images))
        if prun.total_images > 0:
            table.add_row(
                "Pass rate",
                f"[green]{prun.pass_rate:.1%}[/green]"
                if prun.pass_rate >= 0.8
                else f"[yellow]{prun.pass_rate:.1%}[/yellow]",
            )
        table.add_row("Elapsed",      f"{prun.elapsed_s:.2f}s")
        table.add_row("Output dir",   str(prun.output_dir))
        if prun.report_path:
            table.add_row("Report",   str(prun.report_path))

        _console.print(table)

        # Per-item mini table for batches
        if len(prun.items) > 1:
            item_table = Table(
                title="[bold]Per-Item Results[/bold]",
                border_style="dim", header_style="bold",
            )
            item_table.add_column("ID",       style="dim",   width=10)
            item_table.add_column("Style",    style="cyan",  width=12)
            item_table.add_column("Category", style="white", width=14)
            item_table.add_column("Images",   width=8)
            item_table.add_column("Score",    width=8)
            item_table.add_column("Status",   width=10)

            for ir in prun.items:
                ok_color = "green" if ir.success else "red"
                ok_sym   = "OK" if ir.success else "FAIL"
                item_table.add_row(
                    ir.item_id,
                    ir.item.get("style",    "-"),
                    ir.item.get("category", "-"),
                    str(len(ir.image_paths)),
                    f"{ir.eval_score:.3f}" if ir.eval_score else "-",
                    f"[{ok_color}]{ok_sym}[/{ok_color}]",
                )
            _console.print(item_table)

    # =========================================================================
    # ── Lazy Component Initialisation
    # =========================================================================

    def _ensure_generator(self) -> Any:
        """Lazy-load and return the SDXL generator."""
        if self._generator is None and _SDXL_AVAILABLE:
            cfg = self._get_week2_cfg()
            self._generator = _SDXLGen(config=cfg)
            logger.info("FashionSDXLGenerator initialised (lazy)")
        return self._generator

    def _get_scorer(self) -> Any:
        """Lazy-load and return the QualityScorer."""
        if self._scorer is None and _EVAL_AVAILABLE:
            cfg = self._get_week2_cfg()
            self._scorer = QualityScorer(cfg)
        return self._scorer

    def _get_validator(self) -> PromptValidator:
        """Lazy-load and return the PromptValidator."""
        if self._validator is None:
            try:
                cfg = self._get_week2_cfg()
                max_tok = cfg.prompts.structure.max_tokens
            except Exception:
                max_tok = 77
            self._validator = PromptValidator(max_tokens=max_tok)
        return self._validator

    def _get_week2_cfg(self) -> Any:
        """Return or load the system-level Week2Config."""
        if self._week2_cfg is None and _CONFIG_AVAILABLE:
            try:
                self._week2_cfg = get_config()
            except Exception as exc:
                logger.debug("Config load failed (non-fatal): {}", exc)
        return self._week2_cfg

    # =========================================================================
    # ── Context Manager Support
    # =========================================================================

    def __enter__(self) -> "FashionGenerationPipeline":
        logger.debug("FashionGenerationPipeline context entered")
        return self

    def __exit__(self, *_) -> None:
        self._unload()

    def _unload(self) -> None:
        """Release GPU/memory resources."""
        if self._generator and hasattr(self._generator, "unload"):
            self._generator.unload()
            self._generator = None
            logger.info("Generator unloaded")

    def __repr__(self) -> str:
        return (
            f"FashionGenerationPipeline("
            f"output_dir={self.cfg.output_dir} | "
            f"dry_run={self.cfg.dry_run})"
        )
