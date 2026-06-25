"""
week2/pipelines/base_pipeline.py
==================================
Abstract base class for all Week 2 generation pipelines.

Defines the contract that every pipeline must implement and provides
shared infrastructure: logging, config access, result dataclass.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# =============================================================================
# ── Pipeline Run Result
# =============================================================================

@dataclass
class PipelineRunResult:
    """
    Standardised result returned by every pipeline's `run()` method.

    Attributes
    ----------
    success : bool
    total_generated : int
    passed_evaluation : int
    failed_evaluation : int
    output_paths : list of Path
    metadata_paths : list of Path
    report_path : Path or None
    elapsed_s : float
    errors : list of str
    warnings : list of str
    extra : dict  — Any extra pipeline-specific data.
    """
    success:            bool            = True
    total_generated:    int             = 0
    passed_evaluation:  int             = 0
    failed_evaluation:  int             = 0
    output_paths:       List[Path]      = field(default_factory=list)
    metadata_paths:     List[Path]      = field(default_factory=list)
    report_path:        Optional[Path]  = None
    elapsed_s:          float           = 0.0
    errors:             List[str]       = field(default_factory=list)
    warnings:           List[str]       = field(default_factory=list)
    extra:              Dict[str, Any]  = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"PipelineRunResult({status} | "
            f"generated={self.total_generated} | "
            f"passed={self.passed_evaluation} | "
            f"time={self.elapsed_s:.1f}s)"
        )

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"  Pipeline Result",
            "=" * 55,
            f"  Status         : {'SUCCESS' if self.success else 'FAILED'}",
            f"  Generated      : {self.total_generated}",
            f"  Passed eval    : {self.passed_evaluation}",
            f"  Failed eval    : {self.failed_evaluation}",
            f"  Elapsed        : {self.elapsed_s:.2f}s",
        ]
        if self.output_paths:
            lines.append(f"  Output dir     : {self.output_paths[0].parent}")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("=" * 55)
        return "\n".join(lines)


# =============================================================================
# ── Abstract Base Pipeline
# =============================================================================

class BasePipeline(ABC):
    """
    Abstract base class for Week 2 generation pipelines.

    Subclasses must implement:
    - ``run()``   — Execute the full pipeline and return PipelineRunResult.

    Subclasses may override:
    - ``setup()``    — One-time initialisation (model loading etc.).
    - ``teardown()`` — Cleanup after run (release VRAM etc.).
    - ``validate_inputs()`` — Pre-run input validation.

    Parameters
    ----------
    config : Week2Config
        Loaded Week 2 configuration.
    name : str
        Human-readable pipeline name (used in logs).
    """

    def __init__(self, config, name: str = "BasePipeline") -> None:
        self._cfg      = config
        self._name     = name
        self._is_setup = False
        logger.info("Pipeline created | name={}", name)

    # ── Contract ──────────────────────────────────────────────────────────

    @abstractmethod
    def run(self, *args, **kwargs) -> PipelineRunResult:
        """Execute the pipeline and return a PipelineRunResult."""
        ...

    # ── Lifecycle hooks (override as needed) ──────────────────────────────

    def setup(self) -> None:
        """
        One-time setup: load models, warm up caches, etc.
        Called automatically by ``__call__`` before the first ``run()``.
        """
        self._is_setup = True
        logger.debug("{}: setup complete", self._name)

    def teardown(self) -> None:
        """Release resources (VRAM, file handles, etc.)."""
        self._is_setup = False
        logger.debug("{}: teardown complete", self._name)

    def validate_inputs(self, **kwargs) -> List[str]:
        """
        Validate inputs before running. Returns a list of error strings.
        Empty list means inputs are valid.
        """
        return []

    # ── Convenience ───────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs) -> PipelineRunResult:
        """
        Call syntax: ``result = pipeline(prompt="…")``.

        Automatically calls ``setup()`` on first use.
        """
        if not self._is_setup:
            self.setup()
        return self.run(*args, **kwargs)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    @staticmethod
    def _timer() -> float:
        return time.perf_counter()

    def _log_start(self, **context) -> float:
        ctx_str = " | ".join(f"{k}={v!r}" for k, v in context.items())
        logger.info("{} starting | {}", self._name, ctx_str)
        return self._timer()

    def _log_end(self, t_start: float, result: PipelineRunResult) -> None:
        result.elapsed_s = self._timer() - t_start
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            "{} {} | generated={} | time={:.2f}s",
            self._name, status,
            result.total_generated,
            result.elapsed_s,
        )
