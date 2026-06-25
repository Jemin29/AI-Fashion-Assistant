"""
week2/evaluation/evaluation_report.py
========================================
Structured evaluation report generator.

Produces JSON reports summarising a batch of quality scores,
with per-image details and aggregate statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.evaluation.week2_quality_scorer import QualityScore
from src.evaluation.week2_metrics import aggregate_metrics


# =============================================================================
# ── Report Builder
# =============================================================================

class EvaluationReport:
    """
    Builds and saves a structured evaluation report from a list of QualityScores.

    Usage
    -----
        report = EvaluationReport(scores, config=get_config())
        path   = report.save()
        print(report.summary())
    """

    def __init__(
        self,
        scores: List[QualityScore],
        config  = None,
        batch_id: str = "",
    ) -> None:
        self._scores   = scores
        self._cfg      = config
        self._batch_id = batch_id or self._ts()
        self._generated_at = datetime.now(timezone.utc).isoformat()

    # ── Public API ────────────────────────────────────────────────────────

    def to_dict(
        self,
        include_per_image: bool = True,
        max_records: int        = 500,
    ) -> Dict[str, Any]:
        """
        Build the full report as a Python dict (JSON-serialisable).

        Parameters
        ----------
        include_per_image : bool   Include per-image score dicts.
        max_records : int          Cap on per-image records in output.

        Returns
        -------
        dict
        """
        metrics_list = [s.metrics for s in self._scores if s.metrics]
        agg          = aggregate_metrics(metrics_list)

        passed = [s for s in self._scores if s.passed]
        failed = [s for s in self._scores if not s.passed]

        report: Dict[str, Any] = {
            "batch_id":     self._batch_id,
            "generated_at": self._generated_at,
            "summary": {
                "total_images":    len(self._scores),
                "passed":          len(passed),
                "failed":          len(failed),
                "pass_rate":       round(len(passed) / len(self._scores), 4) if self._scores else 0,
                "mean_score":      round(
                    sum(s.overall_score for s in self._scores) / len(self._scores), 4
                ) if self._scores else 0,
                "aggregate_metrics": agg,
            },
            "rejection_breakdown": self._rejection_breakdown(failed),
        }

        if include_per_image:
            report["images"] = [
                s.to_dict() for s in self._scores[:max_records]
            ]

        return report

    def save(
        self,
        output_dir: Optional[Path] = None,
        filename:   Optional[str]  = None,
    ) -> Path:
        """
        Serialise the report to a JSON file.

        Parameters
        ----------
        output_dir : Path, optional   Defaults to config output dir.
        filename : str, optional      Defaults to ``eval_{batch_id}.json``.

        Returns
        -------
        Path  to the saved report file.
        """
        if output_dir is None and self._cfg:
            output_dir = Path(self._cfg.evaluation.report.output_dir)
        elif output_dir is None:
            output_dir = Path("week2/outputs/evaluation_reports")

        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"eval_{self._batch_id}.json"

        path = output_dir / filename
        data = self.to_dict(
            include_per_image = getattr(
                self._cfg.evaluation.report, "include_per_image_scores", True
            ) if self._cfg else True
        )

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.success(
            "Evaluation report saved | path={} | {}/{} passed",
            path,
            data["summary"]["passed"],
            data["summary"]["total_images"],
        )
        return path

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        d = self.to_dict(include_per_image=False)
        s = d["summary"]
        lines = [
            "=" * 60,
            f"  EVALUATION REPORT — {self._batch_id}",
            "=" * 60,
            f"  Total images   : {s['total_images']}",
            f"  Passed         : {s['passed']}",
            f"  Failed         : {s['failed']}",
            f"  Pass rate      : {s['pass_rate']:.1%}",
            f"  Mean score     : {s['mean_score']:.3f}",
        ]
        if s["aggregate_metrics"].get("mean_clip"):
            lines.append(f"  Mean CLIP      : {s['aggregate_metrics']['mean_clip']:.4f}")
        reject_bkd = d.get("rejection_breakdown", {})
        if reject_bkd:
            lines.append("  Rejection reasons:")
            for reason, count in reject_bkd.items():
                lines.append(f"    [{count:>3}x] {reason}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _rejection_breakdown(failed: List[QualityScore]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for s in failed:
            reason = s.reject_reason or "unknown"
            breakdown[reason] = breakdown.get(reason, 0) + 1
        return breakdown
