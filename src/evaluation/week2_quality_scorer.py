"""
week2/evaluation/quality_scorer.py
=====================================
Composite quality scorer that runs all enabled evaluation checks
and returns a structured per-image score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.evaluation.week2_metrics import (
    ImageMetrics,
    check_image_quality,
    compute_clip_similarity,
    compute_ssim,
    compute_psnr,
)


# =============================================================================
# ── Score Result
# =============================================================================

@dataclass
class QualityScore:
    """
    Composite quality score for a single generated image.

    Attributes
    ----------
    image_id : str
    overall_score : float   Weighted composite [0.0 – 1.0].
    metrics : ImageMetrics  Raw per-check metrics.
    passed : bool           True if image meets minimum thresholds.
    reject_reason : str     Non-empty if image should be rejected.
    eval_time_s : float     Time taken to evaluate this image.
    """
    image_id:       str
    overall_score:  float           = 0.0
    metrics:        Optional[ImageMetrics] = None
    passed:         bool            = True
    reject_reason:  str             = ""
    eval_time_s:    float           = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id":     self.image_id,
            "overall_score":round(self.overall_score, 4),
            "passed":       self.passed,
            "reject_reason":self.reject_reason,
            "eval_time_s":  round(self.eval_time_s, 3),
            "metrics":      self.metrics.to_dict() if self.metrics else {},
        }


# =============================================================================
# ── Quality Scorer
# =============================================================================

class QualityScorer:
    """
    Runs all enabled quality checks on a generated image and returns
    a composite QualityScore.

    Parameters
    ----------
    config : Week2Config
        Config (evaluation section used).

    Example
    -------
        scorer = QualityScorer(config=get_config())
        score  = scorer.score(image, prompt="a red dress", image_id="GEN_001")
        print(score.overall_score)
    """

    def __init__(self, config=None) -> None:
        if config is not None:
            eval_cfg = config.evaluation
            self._quality_cfg   = eval_cfg.quality
            self._clip_cfg      = eval_cfg.clip
            self._device        = config.model.runtime.device
        else:
            self._quality_cfg   = None
            self._clip_cfg      = None
            self._device        = "cpu"

    # ── Public API ────────────────────────────────────────────────────────

    def score(
        self,
        image,
        image_id: str,
        prompt:   str           = "",
        reference             = None,
    ) -> QualityScore:
        """
        Evaluate a single image.

        Parameters
        ----------
        image : PIL.Image.Image
        image_id : str
        prompt : str   The text prompt used to generate this image.
        reference : PIL.Image.Image, optional   For SSIM/PSNR.

        Returns
        -------
        QualityScore
        """
        t_start = time.perf_counter()

        # ── 1. Quality checks ─────────────────────────────────────────────
        min_w = getattr(self._quality_cfg, "min_width",  512) if self._quality_cfg else 512
        min_h = getattr(self._quality_cfg, "min_height", 512) if self._quality_cfg else 512
        black_t = getattr(self._quality_cfg, "black_threshold", 5) if self._quality_cfg else 5
        white_t = getattr(self._quality_cfg, "white_threshold", 250) if self._quality_cfg else 250

        metrics = check_image_quality(
            image, image_id,
            min_width       = min_w,
            min_height      = min_h,
            black_threshold = black_t,
            white_threshold = white_t,
        )

        # ── 2. CLIP similarity ────────────────────────────────────────────
        clip_enabled = getattr(self._clip_cfg, "enabled", True) if self._clip_cfg else False
        if clip_enabled and prompt:
            model_name = getattr(self._clip_cfg, "model_name", "ViT-L-14")
            pretrained = getattr(self._clip_cfg, "pretrained", "openai")
            metrics.clip_similarity = compute_clip_similarity(
                image, prompt,
                model_name = model_name,
                pretrained = pretrained,
                device     = self._device if self._device != "auto" else "cpu",
            )

        # ── 3. SSIM / PSNR ────────────────────────────────────────────────
        quality_enabled = getattr(self._quality_cfg, "enabled", True) if self._quality_cfg else True
        if quality_enabled and reference is not None:
            if getattr(self._quality_cfg, "compute_ssim", False):
                metrics.ssim = compute_ssim(image, reference)
            if getattr(self._quality_cfg, "compute_psnr", False):
                metrics.psnr = compute_psnr(image, reference)

        # ── 4. Compute composite score ────────────────────────────────────
        overall, reject_reason = self._compute_composite(metrics)

        eval_time = time.perf_counter() - t_start

        qs = QualityScore(
            image_id      = image_id,
            overall_score = overall,
            metrics       = metrics,
            passed        = not bool(reject_reason),
            reject_reason = reject_reason,
            eval_time_s   = eval_time,
        )

        if qs.passed:
            logger.debug("Quality OK | {} | score={:.3f}", image_id, overall)
        else:
            logger.warning("Quality FAIL | {} | reason={}", image_id, reject_reason)

        return qs

    def score_batch(
        self,
        images:    List[Any],
        image_ids: List[str],
        prompts:   Optional[List[str]] = None,
    ) -> List[QualityScore]:
        """
        Score a batch of images.

        Parameters
        ----------
        images : list of PIL.Image.Image
        image_ids : list of str
        prompts : list of str, optional

        Returns
        -------
        list of QualityScore
        """
        if prompts is None:
            prompts = [""] * len(images)

        scores = []
        for img, img_id, prompt in zip(images, image_ids, prompts):
            qs = self.score(img, img_id, prompt=prompt)
            scores.append(qs)

        passed = sum(1 for s in scores if s.passed)
        logger.info(
            "Batch evaluation complete | {}/{} passed",
            passed, len(scores),
        )
        return scores

    # ── Private ───────────────────────────────────────────────────────────

    def _compute_composite(self, metrics: ImageMetrics) -> tuple[float, str]:
        """
        Compute a composite quality score [0, 1] and optional reject reason.

        Scoring weights:
        - resolution_ok : 0.30
        - not black     : 0.25
        - not white     : 0.25
        - clip (if avail): 0.20
        """
        score = 0.0
        reject_reason = ""

        # Resolution (30%)
        if metrics.resolution_ok:
            score += 0.30
        else:
            reject_reason = "resolution below minimum"

        # Not black (25%)
        if not metrics.is_black:
            score += 0.25
        else:
            reject_reason = reject_reason or "black image"

        # Not white (25%)
        if not metrics.is_white:
            score += 0.25
        else:
            reject_reason = reject_reason or "white/saturated image"

        # CLIP similarity (20%)
        if metrics.clip_similarity is not None:
            clip_min = 0.20
            if self._clip_cfg:
                clip_min = getattr(self._clip_cfg, "min_similarity_threshold", 0.20)
            if metrics.clip_similarity >= clip_min:
                score += 0.20
            else:
                reject_reason = reject_reason or (
                    f"low CLIP similarity ({metrics.clip_similarity:.3f} < {clip_min})"
                )
        else:
            # CLIP not available → give partial credit
            score += 0.20

        return round(score, 4), reject_reason
