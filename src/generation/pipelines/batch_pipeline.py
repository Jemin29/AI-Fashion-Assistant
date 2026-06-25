"""
week2/pipelines/batch_pipeline.py
====================================
Batch generation pipeline — generates multiple prompts in sequence
with progress tracking and per-item error isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.generation.pipelines.base_pipeline import BasePipeline, PipelineRunResult
from src.generation.pipelines.text2image_pipeline import Text2ImagePipeline


# =============================================================================
# ── Batch Item
# =============================================================================

@dataclass
class BatchItem:
    """
    A single item in a batch generation job.

    Attributes
    ----------
    prompt : str             Main positive prompt.
    style : str, optional    Style preset name.
    gender : str, optional
    season : str, optional
    seed : int               -1 for random.
    num_images : int         Images to generate for this item.
    metadata : dict          Extra metadata to attach to output.
    """
    prompt:     str
    style:      Optional[str]       = None
    gender:     Optional[str]       = None
    season:     Optional[str]       = None
    occasion:   Optional[str]       = None
    seed:       int                 = -1
    num_images: int                 = 1
    metadata:   Dict[str, Any]      = field(default_factory=dict)


@dataclass
class BatchResult:
    """
    Result for a single item in a batch.

    Attributes
    ----------
    item : BatchItem
    result : PipelineRunResult
    index : int     Position in batch.
    """
    item:   BatchItem
    result: PipelineRunResult
    index:  int

    @property
    def success(self) -> bool:
        return self.result.success


# =============================================================================
# ── Batch Pipeline
# =============================================================================

class BatchPipeline(BasePipeline):
    """
    Run multiple text-to-image generation requests in sequence.

    Features
    --------
    - Rich progress bar (if ``rich`` is installed)
    - Per-item error isolation (one failure does not stop the batch)
    - Aggregate summary across all items
    - Configurable concurrency (sequential only in Week 2)

    Parameters
    ----------
    config : Week2Config
    stop_on_error : bool   If True, stop batch on first failure.
    show_progress : bool   Show Rich progress bar.

    Example
    -------
        items = [
            BatchItem(prompt="A red summer dress", style="casual"),
            BatchItem(prompt="A tailored navy suit", style="formal"),
        ]
        batch = BatchPipeline(config=get_config())
        results = batch.run(items)
        print(f"{results.passed_evaluation}/{results.total_generated} passed")
    """

    def __init__(
        self,
        config,
        stop_on_error:  bool = False,
        show_progress:  bool = True,
    ) -> None:
        super().__init__(config, name="BatchPipeline")
        self._stop_on_error = stop_on_error
        self._show_progress = show_progress and RICH_AVAILABLE
        # Shared sub-pipeline — models loaded once for the whole batch
        self._t2i: Optional[Text2ImagePipeline] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        super().setup()
        self._t2i = Text2ImagePipeline(config=self._cfg)
        self._t2i.setup()

    def teardown(self) -> None:
        if self._t2i:
            self._t2i.teardown()
            self._t2i = None
        super().teardown()

    # ── Main Run ──────────────────────────────────────────────────────────

    def run(
        self,
        items:      List[BatchItem],
        output_dir: Optional[Path] = None,
        save_report: bool          = True,
    ) -> PipelineRunResult:
        """
        Generate images for all items in the batch.

        Parameters
        ----------
        items : list of BatchItem
        output_dir : Path, optional
        save_report : bool   Save a combined evaluation report.

        Returns
        -------
        PipelineRunResult   Aggregate result across all items.
        """
        aggregate = PipelineRunResult()
        t_start   = self._log_start(n_items=len(items))

        if not items:
            aggregate.warnings.append("Batch is empty — nothing to generate.")
            self._log_end(t_start, aggregate)
            return aggregate

        if self._t2i is None:
            self._t2i = Text2ImagePipeline(config=self._cfg)
            self._t2i.setup()

        batch_results: List[BatchResult] = []

        def _run_item(idx: int, item: BatchItem) -> BatchResult:
            logger.info("Batch item {}/{} | prompt={!r:.60}", idx + 1, len(items), item.prompt)
            r = self._t2i.run(
                prompt      = item.prompt,
                style       = item.style,
                gender      = item.gender,
                season      = item.season,
                occasion    = item.occasion,
                seed        = item.seed,
                num_images  = item.num_images,
                save        = True,
                evaluate    = True,
                save_report = False,    # Aggregate report below
                output_dir  = output_dir,
            )
            return BatchResult(item=item, result=r, index=idx)

        # ── Execute with or without progress bar ──────────────────────────
        if self._show_progress and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Generating…", total=len(items))
                for idx, item in enumerate(items):
                    br = _run_item(idx, item)
                    batch_results.append(br)
                    progress.advance(task)
                    if not br.success and self._stop_on_error:
                        aggregate.errors.append(
                            f"Stopped at item {idx + 1}: {br.result.errors}"
                        )
                        break
        else:
            for idx, item in enumerate(items):
                br = _run_item(idx, item)
                batch_results.append(br)
                if not br.success and self._stop_on_error:
                    aggregate.errors.append(
                        f"Stopped at item {idx + 1}: {br.result.errors}"
                    )
                    break

        # ── Aggregate ─────────────────────────────────────────────────────
        for br in batch_results:
            aggregate.total_generated    += br.result.total_generated
            aggregate.passed_evaluation  += br.result.passed_evaluation
            aggregate.failed_evaluation  += br.result.failed_evaluation
            aggregate.output_paths       += br.result.output_paths
            aggregate.errors             += br.result.errors

        aggregate.success = all(br.success for br in batch_results)
        aggregate.extra["batch_results"] = [
            {
                "index":    br.index,
                "prompt":   br.item.prompt[:80],
                "success":  br.success,
                "generated":br.result.total_generated,
            }
            for br in batch_results
        ]

        self._log_end(t_start, aggregate)
        return aggregate

    # ── Convenience class method ───────────────────────────────────────────

    @classmethod
    def from_prompts(
        cls,
        prompts: List[str],
        config,
        style:   Optional[str] = None,
        **item_kwargs,
    ) -> "BatchPipeline":
        """
        Create a BatchPipeline and attach a list of string prompts as BatchItems.

        Returns the pipeline (not results) — call ``.run(items)`` separately,
        or use ``BatchPipeline.run_prompts()`` for a one-liner.
        """
        return cls(config=config)

    @classmethod
    def run_prompts(
        cls,
        prompts:    List[str],
        config,
        style:      Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> PipelineRunResult:
        """
        One-liner: generate images for a list of text prompts.

        Example
        -------
            result = BatchPipeline.run_prompts(
                ["A red dress", "A blue hoodie"],
                config=get_config(),
                style="casual",
            )
        """
        items = [
            BatchItem(prompt=p, style=style)
            for p in prompts
        ]
        pipeline = cls(config=config)
        with pipeline:
            return pipeline.run(items, output_dir=output_dir)
