"""
week2/evaluation/clip_evaluator.py
=====================================
Production-grade CLIP-based Image Evaluation Engine
AI-Powered Fashion Design Assistant — Week 2

Uses HuggingFace Transformers (CLIPModel + CLIPProcessor) to compute:

  1. Prompt-Image Similarity  — cosine similarity in CLIP embedding space
  2. Semantic Alignment       — bucketed label: "very high" / "high" / "medium" / "low"
  3. Ranking Score            — normalised rank across a candidate set

API
---
    from src.evaluation.week2_clip_evaluator import CLIPEvaluator

    evaluator = CLIPEvaluator()
    evaluator.load_clip()

    # Single image
    result = evaluator.evaluate(image=pil_img, prompt="a black hoodie")
    # {"clip_score": 0.92, "alignment": "high", ...}

    # Rank a list of candidate images
    ranked = evaluator.rank_images(images=[...], prompt="a red evening gown")
    # [{"rank": 1, "clip_score": 0.91, ...}, ...]

    # Compare two generation runs
    comparison = evaluator.compare_generations(
        images_a=[...], images_b=[...], prompt="a navy suit"
    )

Module-level convenience functions
-----------------------------------
    load_clip()
    evaluate(image, prompt, ...)
    rank_images(images, prompt, ...)
    compare_generations(images_a, images_b, prompt, ...)

Output schema
-------------
    {
        "clip_score":          float,   # cosine similarity [0.0 – 1.0]
        "alignment":           str,     # "very high" | "high" | "medium" | "low"
        "semantic_score":      float,   # alignment score normalised to [0, 1]
        "ranking_score":       float,   # relative rank in candidate set [0, 1]
        "image_id":            str,
        "prompt_preview":      str,
        "model":               str,
        "device":              str,
        "eval_ms":             float,
    }
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging as _log
    logger = _log.getLogger("clip_evaluator")  # type: ignore[assignment]

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# ── HuggingFace Transformers (CLIPModel + CLIPProcessor) ─────────────────────
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    _HF_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore[assignment]
    CLIPModel = None      # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]
    _HF_AVAILABLE = False

# ── PIL ───────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    PILImage = None   # type: ignore[assignment]
    _PIL_AVAILABLE = False


# =============================================================================
# ── Constants
# =============================================================================

#: Default HuggingFace model ID for CLIP
DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"

#: Alignment thresholds (cosine similarity → label)
#: Tuned for fashion image–prompt pairs
ALIGNMENT_THRESHOLDS: Dict[str, float] = {
    "very high": 0.30,   # ≥ 0.30
    "high":      0.25,   # ≥ 0.25
    "medium":    0.20,   # ≥ 0.20
    "low":       0.00,   # below 0.20
}

#: Minimum similarity score required to "pass" evaluation
DEFAULT_MIN_SIMILARITY = 0.20


# =============================================================================
# ── Result Dataclasses
# =============================================================================

@dataclass
class CLIPScore:
    """
    Complete CLIP evaluation result for a single image.

    Attributes
    ----------
    image_id         : str
    clip_score       : float   Cosine similarity [0, 1].
    alignment        : str     "very high" | "high" | "medium" | "low".
    semantic_score   : float   Alignment label normalised to [0, 1].
    ranking_score    : float   Relative rank in candidate set [0, 1].
    passed           : bool    clip_score >= min_similarity_threshold.
    prompt_preview   : str     First 80 chars of the evaluated prompt.
    model            : str     Model ID used.
    device           : str
    eval_ms          : float   Milliseconds taken.
    raw_logit        : float   Raw CLIP logit (before softmax) if available.
    error            : str     Non-empty if evaluation failed.
    """
    image_id:       str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    clip_score:     float           = 0.0
    alignment:      str             = "low"
    semantic_score: float           = 0.0
    ranking_score:  float           = 0.0
    passed:         bool            = False
    prompt_preview: str             = ""
    model:          str             = DEFAULT_MODEL_ID
    device:         str             = "cpu"
    eval_ms:        float           = 0.0
    raw_logit:      Optional[float] = None
    error:          str             = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a JSON-compatible dict.

        Returns
        -------
        dict — matches the output schema specified in the module docstring:
            {"clip_score": 0.92, "alignment": "high", ...}
        """
        return {
            "clip_score":    round(self.clip_score, 4),
            "alignment":     self.alignment,
            "semantic_score":round(self.semantic_score, 4),
            "ranking_score": round(self.ranking_score, 4),
            "passed":        self.passed,
            "image_id":      self.image_id,
            "prompt_preview":self.prompt_preview,
            "model":         self.model,
            "device":        self.device,
            "eval_ms":       round(self.eval_ms, 2),
            "error":         self.error,
        }

    def __repr__(self) -> str:
        return (
            f"CLIPScore(id={self.image_id} | score={self.clip_score:.4f} | "
            f"alignment={self.alignment!r} | passed={self.passed})"
        )


@dataclass
class RankedImage:
    """Single entry in a ranked list of candidate images."""
    rank:        int
    image_id:    str
    clip_score:  float
    alignment:   str
    ranking_score: float
    passed:      bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank":          self.rank,
            "image_id":      self.image_id,
            "clip_score":    round(self.clip_score, 4),
            "alignment":     self.alignment,
            "ranking_score": round(self.ranking_score, 4),
            "passed":        self.passed,
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing two batches of generated images (A vs B).

    Attributes
    ----------
    prompt_preview : str
    mean_score_a   : float   Mean CLIP score for set A.
    mean_score_b   : float   Mean CLIP score for set B.
    winner         : str     "A" | "B" | "tie"
    delta          : float   mean_score_a - mean_score_b.
    scores_a       : list of CLIPScore
    scores_b       : list of CLIPScore
    """
    prompt_preview: str
    mean_score_a:   float
    mean_score_b:   float
    winner:         str
    delta:          float
    scores_a:       List[CLIPScore] = field(default_factory=list)
    scores_b:       List[CLIPScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_preview": self.prompt_preview,
            "mean_score_a":   round(self.mean_score_a, 4),
            "mean_score_b":   round(self.mean_score_b, 4),
            "winner":         self.winner,
            "delta":          round(self.delta, 4),
            "scores_a":       [s.to_dict() for s in self.scores_a],
            "scores_b":       [s.to_dict() for s in self.scores_b],
        }


# =============================================================================
# ── CLIPEvaluator
# =============================================================================

class CLIPEvaluator:
    """
    Production-grade CLIP-based image evaluation engine.

    Uses ``openai/clip-vit-large-patch14`` (HuggingFace Transformers) by
    default. Gracefully degrades to a stub when ``transformers`` or
    ``torch`` are not installed, returning neutral scores with an error
    message instead of raising exceptions.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Defaults to ``openai/clip-vit-large-patch14``.
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (auto-detects GPU).
    min_similarity : float
        Minimum CLIP cosine similarity to mark an image as ``passed``.
    cache_dir : Path, optional
        Local model cache directory (passed to HuggingFace as ``cache_dir``).
    half_precision : bool
        Use fp16 on GPU to reduce VRAM. Ignored on CPU.

    Example
    -------
        evaluator = CLIPEvaluator()
        evaluator.load_clip()

        result = evaluator.evaluate(pil_image, "a black oversized hoodie")
        print(result)      # {"clip_score": 0.28, "alignment": "high", ...}

        ranked = evaluator.rank_images([img1, img2, img3], "a red silk gown")
        print(ranked[0])   # {"rank": 1, "clip_score": 0.31, ...}
    """

    def __init__(
        self,
        model_id:       str            = DEFAULT_MODEL_ID,
        device:         str            = "auto",
        min_similarity: float          = DEFAULT_MIN_SIMILARITY,
        cache_dir:      Optional[Path] = None,
        half_precision: bool           = False,
    ) -> None:
        self.model_id       = model_id
        self.min_similarity = min_similarity
        self.cache_dir      = cache_dir
        self.half_precision = half_precision

        # Resolve device
        self.device = self._resolve_device(device)

        # Model state
        self._model:     Any = None
        self._processor: Any = None
        self._loaded:    bool = False

        logger.info(
            "CLIPEvaluator initialised | model={} | device={} | hf_available={}",
            model_id, self.device, _HF_AVAILABLE,
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def load_clip(
        self,
        model_id:  Optional[str]  = None,
        device:    Optional[str]  = None,
        force:     bool           = False,
    ) -> bool:
        """
        Load the CLIP model and processor into memory.

        Idempotent — calling it multiple times is safe (uses cache unless
        ``force=True``).

        Parameters
        ----------
        model_id : str, optional   Override the instance model_id.
        device   : str, optional   Override the instance device.
        force    : bool            Reload even if already loaded.

        Returns
        -------
        bool — True if model loaded successfully, False if unavailable.

        Raises
        ------
        Does NOT raise. Returns False on failure and logs the error.

        Example
        -------
            evaluator = CLIPEvaluator()
            ok = evaluator.load_clip()
            if not ok:
                print("CLIP not available — running in stub mode")
        """
        if self._loaded and not force:
            logger.debug("CLIP already loaded (use force=True to reload)")
            return True

        if not _HF_AVAILABLE:
            logger.warning(
                "HuggingFace Transformers / PyTorch not installed. "
                "Install with: pip install transformers torch Pillow. "
                "CLIPEvaluator will run in stub mode."
            )
            return False

        _model_id = model_id or self.model_id
        _device   = self._resolve_device(device or self.device)

        try:
            t0 = time.perf_counter()
            logger.info("Loading CLIP model: {} on {}", _model_id, _device)

            cache_str = str(self.cache_dir) if self.cache_dir else None

            self._processor = CLIPProcessor.from_pretrained(
                _model_id,
                cache_dir = cache_str,
            )
            self._model = CLIPModel.from_pretrained(
                _model_id,
                cache_dir = cache_str,
            ).to(_device)

            # fp16 on GPU
            if self.half_precision and _device != "cpu":
                self._model = self._model.half()
                logger.debug("Model cast to fp16")

            self._model.eval()
            self.device  = _device
            self._loaded = True

            elapsed = time.perf_counter() - t0
            logger.success(
                "CLIP model loaded | model={} | device={} | {:.2f}s",
                _model_id, _device, elapsed,
            )
            return True

        except Exception as exc:
            logger.error("Failed to load CLIP model {!r}: {}", _model_id, exc)
            self._loaded = False
            return False

    def evaluate(
        self,
        image,
        prompt:     str,
        image_id:   Optional[str] = None,
        *,
        ranking_score: float = 1.0,
    ) -> CLIPScore:
        """
        Evaluate a single PIL image against a text prompt.

        Parameters
        ----------
        image       : PIL.Image.Image   The generated image.
        prompt      : str               The text prompt used for generation.
        image_id    : str, optional     Identifier for this image.
        ranking_score : float           Pre-computed ranking score [0, 1]
                                        (set by rank_images; default 1.0).

        Returns
        -------
        CLIPScore

        Output dict shape::

            {
                "clip_score":    0.92,
                "alignment":     "high",
                "semantic_score":0.75,
                "ranking_score": 1.0,
                "passed":        True,
                ...
            }

        Example
        -------
            result = evaluator.evaluate(pil_img, "a black streetwear hoodie")
            print(result.to_dict())
        """
        t0      = time.perf_counter()
        img_id  = image_id or str(uuid.uuid4())[:8]
        preview = prompt[:80]

        score = CLIPScore(
            image_id      = img_id,
            prompt_preview= preview,
            model         = self.model_id,
            device        = self.device,
            ranking_score = ranking_score,
        )

        # ── Try to compute real CLIP score ────────────────────────────────
        clip_sim = self._compute_similarity(image, prompt)

        if clip_sim is None:
            # Stub mode — no model available
            score.clip_score    = 0.0
            score.alignment     = "low"
            score.semantic_score= 0.0
            score.passed        = False
            score.error         = "CLIP model not available (stub mode)"
            score.eval_ms       = (time.perf_counter() - t0) * 1000
            logger.debug("CLIPEvaluator stub | image_id={}", img_id)
            return score

        # ── Compute derived metrics ───────────────────────────────────────
        alignment        = self._classify_alignment(clip_sim)
        semantic_score   = self._alignment_to_score(alignment)
        passed           = clip_sim >= self.min_similarity

        score.clip_score    = round(float(clip_sim), 4)
        score.alignment     = alignment
        score.semantic_score= round(semantic_score, 4)
        score.passed        = passed
        score.eval_ms       = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "CLIP evaluate | id={} | score={:.4f} | alignment={!r} | passed={}",
            img_id, clip_sim, alignment, passed,
        )
        return score

    def rank_images(
        self,
        images:     List[Any],
        prompt:     str,
        image_ids:  Optional[List[str]] = None,
        *,
        descending: bool = True,
    ) -> List[RankedImage]:
        """
        Score and rank a list of candidate images by CLIP similarity.

        Parameters
        ----------
        images    : list of PIL.Image.Image   Candidate images.
        prompt    : str                       Text prompt.
        image_ids : list of str, optional     IDs for each image.
        descending : bool                     Highest score = rank 1 (default).

        Returns
        -------
        list of RankedImage — sorted by rank (best first).

        Example
        -------
            ranked = evaluator.rank_images(
                [img1, img2, img3],
                "a luxury evening gown in emerald green"
            )
            best = ranked[0]
            print(best.to_dict())  # {"rank": 1, "clip_score": 0.31, ...}
        """
        if not images:
            return []

        n      = len(images)
        ids    = image_ids or [str(uuid.uuid4())[:8] for _ in range(n)]

        # ── Score all images ──────────────────────────────────────────────
        scores: List[Tuple[int, float, CLIPScore]] = []
        for idx, (img, img_id) in enumerate(zip(images, ids)):
            sc = self.evaluate(img, prompt, image_id=img_id)
            scores.append((idx, sc.clip_score, sc))

        # ── Sort ──────────────────────────────────────────────────────────
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=descending)

        # ── Compute normalised ranking scores ─────────────────────────────
        raw_scores = [s[1] for s in sorted_scores]
        min_s      = min(raw_scores) if raw_scores else 0.0
        max_s      = max(raw_scores) if raw_scores else 1.0
        score_range= max_s - min_s if max_s != min_s else 1.0

        ranked: List[RankedImage] = []
        for rank_pos, (orig_idx, clip_sc, clip_score_obj) in enumerate(sorted_scores):
            norm_rank = 1.0 - (rank_pos / max(n - 1, 1))  # 1.0 = best, 0.0 = worst
            norm_clip = (clip_sc - min_s) / score_range if score_range > 0 else 1.0

            ranked.append(RankedImage(
                rank          = rank_pos + 1,
                image_id      = clip_score_obj.image_id,
                clip_score    = clip_sc,
                alignment     = clip_score_obj.alignment,
                ranking_score = round(norm_rank, 4),
                passed        = clip_score_obj.passed,
            ))

        logger.info(
            "rank_images | n={} | best={:.4f} | worst={:.4f} | prompt={!r:.50}",
            n,
            sorted_scores[0][1] if sorted_scores else 0.0,
            sorted_scores[-1][1] if sorted_scores else 0.0,
            prompt,
        )
        return ranked

    def compare_generations(
        self,
        images_a: List[Any],
        images_b: List[Any],
        prompt:   str,
        *,
        ids_a:    Optional[List[str]] = None,
        ids_b:    Optional[List[str]] = None,
        tie_threshold: float           = 0.005,
    ) -> ComparisonResult:
        """
        Compare two sets of generated images (e.g. two model variants or
        two prompt strategies) against the same target prompt.

        Parameters
        ----------
        images_a : list of PIL.Image.Image   Set A (e.g. baseline generation).
        images_b : list of PIL.Image.Image   Set B (e.g. new generation).
        prompt   : str                       Text prompt used for both sets.
        ids_a    : list of str, optional
        ids_b    : list of str, optional
        tie_threshold : float   |delta| below this → "tie".

        Returns
        -------
        ComparisonResult

        Example
        -------
            comparison = evaluator.compare_generations(
                images_a = [img_baseline],
                images_b = [img_new],
                prompt   = "a tailored navy suit on a fashion runway",
            )
            print(comparison.winner)   # "A" | "B" | "tie"
            print(comparison.to_dict())
        """
        n_a = len(images_a)
        n_b = len(images_b)
        _ids_a = ids_a or [f"A_{i:02d}" for i in range(n_a)]
        _ids_b = ids_b or [f"B_{i:02d}" for i in range(n_b)]

        logger.info(
            "compare_generations | n_a={} | n_b={} | prompt={!r:.50}",
            n_a, n_b, prompt,
        )

        # ── Score both sets ───────────────────────────────────────────────
        scores_a = [
            self.evaluate(img, prompt, image_id=img_id)
            for img, img_id in zip(images_a, _ids_a)
        ]
        scores_b = [
            self.evaluate(img, prompt, image_id=img_id)
            for img, img_id in zip(images_b, _ids_b)
        ]

        mean_a = sum(s.clip_score for s in scores_a) / len(scores_a) if scores_a else 0.0
        mean_b = sum(s.clip_score for s in scores_b) / len(scores_b) if scores_b else 0.0
        delta  = mean_a - mean_b

        if abs(delta) < tie_threshold:
            winner = "tie"
        elif delta > 0:
            winner = "A"
        else:
            winner = "B"

        logger.info(
            "compare_generations | mean_a={:.4f} | mean_b={:.4f} | winner={}",
            mean_a, mean_b, winner,
        )

        return ComparisonResult(
            prompt_preview = prompt[:80],
            mean_score_a   = round(mean_a, 4),
            mean_score_b   = round(mean_b, 4),
            winner         = winner,
            delta          = round(delta, 4),
            scores_a       = scores_a,
            scores_b       = scores_b,
        )

    # =========================================================================
    # ── Batch Evaluation
    # =========================================================================

    def evaluate_batch(
        self,
        images:    List[Any],
        prompts:   Union[str, List[str]],
        image_ids: Optional[List[str]] = None,
    ) -> List[CLIPScore]:
        """
        Evaluate a batch of images, each against its own (or a shared) prompt.

        Parameters
        ----------
        images    : list of PIL.Image.Image
        prompts   : str or list of str   If str, used for all images.
        image_ids : list of str, optional

        Returns
        -------
        list of CLIPScore

        Example
        -------
            scores = evaluator.evaluate_batch(
                [img1, img2],
                prompts=["a red hoodie", "a blue suit"],
            )
        """
        n      = len(images)
        ids    = image_ids or [str(uuid.uuid4())[:8] for _ in range(n)]
        _prompts: List[str] = (
            [prompts] * n
            if isinstance(prompts, str)
            else list(prompts)
        )

        if len(_prompts) != n:
            raise ValueError(
                f"prompts list length ({len(_prompts)}) must match images ({n})"
            )

        results: List[CLIPScore] = []
        for img, prompt, img_id in zip(images, _prompts, ids):
            results.append(self.evaluate(img, prompt, image_id=img_id))

        passed = sum(1 for r in results if r.passed)
        logger.info(
            "evaluate_batch | n={} | passed={} | mean_score={:.4f}",
            n, passed,
            sum(r.clip_score for r in results) / n if n else 0.0,
        )
        return results

    def aggregate(self, scores: List[CLIPScore]) -> Dict[str, Any]:
        """
        Compute aggregate statistics across a list of CLIPScores.

        Parameters
        ----------
        scores : list of CLIPScore

        Returns
        -------
        dict with keys: total, passed, failed, pass_rate, mean_clip_score,
                        min_clip_score, max_clip_score, alignment_breakdown.

        Example
        -------
            scores = evaluator.evaluate_batch(images, "a streetwear look")
            stats  = evaluator.aggregate(scores)
            print(stats["mean_clip_score"])
        """
        if not scores:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        n        = len(scores)
        passed   = sum(1 for s in scores if s.passed)
        raw      = [s.clip_score for s in scores]
        align_counts: Dict[str, int] = {}
        for s in scores:
            align_counts[s.alignment] = align_counts.get(s.alignment, 0) + 1

        return {
            "total":             n,
            "passed":            passed,
            "failed":            n - passed,
            "pass_rate":         round(passed / n, 4),
            "mean_clip_score":   round(sum(raw) / n, 4),
            "min_clip_score":    round(min(raw), 4),
            "max_clip_score":    round(max(raw), 4),
            "std_clip_score":    round(self._std(raw), 4),
            "alignment_breakdown": align_counts,
            "model":             self.model_id,
            "device":            self.device,
        }

    # =========================================================================
    # ── Private Helpers
    # =========================================================================

    def _compute_similarity(
        self,
        image: Any,
        prompt: str,
    ) -> Optional[float]:
        """
        Core CLIP forward pass — returns cosine similarity in [0, 1].

        Returns None if:
          - CLIP is not loaded / available
          - image is None or not a PIL Image
          - Any exception occurs (logged as warning)
        """
        if not self._loaded:
            # Try auto-load silently
            if _HF_AVAILABLE and not self._loaded:
                ok = self.load_clip()
                if not ok:
                    return None
            else:
                return None

        if image is None:
            logger.warning("_compute_similarity: image is None")
            return None

        if not _PIL_AVAILABLE:
            logger.warning("Pillow not installed — cannot encode image")
            return None

        try:
            # Ensure image is PIL
            if not isinstance(image, PILImage.Image):
                logger.warning(
                    "Image is not a PIL.Image (type={}). Skipping CLIP.",
                    type(image).__name__,
                )
                return None

            # Ensure RGB
            image = image.convert("RGB")

            with torch.no_grad():
                inputs = self._processor(
                    text   = [prompt],
                    images = [image],
                    return_tensors = "pt",
                    padding        = True,
                    truncation     = True,
                    max_length     = 77,
                ).to(self.device)

                # fp16 input if model is fp16
                if self.half_precision and self.device != "cpu":
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].half()

                outputs = self._model(**inputs)

                # CLIP logits: model outputs logits_per_image [1 × 1]
                # Normalised embeddings → cosine similarity
                img_emb  = outputs.image_embeds  # [1, D]
                txt_emb  = outputs.text_embeds   # [1, D]

                img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

                cosine   = (img_norm @ txt_norm.T).squeeze().item()

            # Clip to [0, 1] — raw cosine can be slightly negative
            return float(max(0.0, min(1.0, cosine)))

        except Exception as exc:
            logger.warning("CLIP similarity computation failed: {}", exc)
            return None

    @staticmethod
    def _classify_alignment(score: float) -> str:
        """
        Map a cosine similarity score to an alignment label.

        Thresholds (tuned for fashion image–text pairs):

        +-----------+----------+
        | Score     | Label    |
        +===========+==========+
        | >= 0.30   | very high|
        | >= 0.25   | high     |
        | >= 0.20   | medium   |
        | <  0.20   | low      |
        +-----------+----------+
        """
        if score >= ALIGNMENT_THRESHOLDS["very high"]:
            return "very high"
        if score >= ALIGNMENT_THRESHOLDS["high"]:
            return "high"
        if score >= ALIGNMENT_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    @staticmethod
    def _alignment_to_score(alignment: str) -> float:
        """Map alignment label → normalised [0, 1] semantic score."""
        return {
            "very high": 1.00,
            "high":      0.75,
            "medium":    0.50,
            "low":       0.25,
        }.get(alignment, 0.0)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu'."""
        if device == "auto":
            if _HF_AVAILABLE and torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    @staticmethod
    def _std(values: List[float]) -> float:
        """Compute standard deviation without numpy dependency."""
        if _NUMPY:
            return float(np.std(values))
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    # =========================================================================
    # ── Context Manager / Lifecycle
    # =========================================================================

    def unload(self) -> None:
        """
        Release model from memory (GPU VRAM or RAM).

        Useful in batch pipelines where the CLIP model is only needed for
        a short evaluation window.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        if _HF_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("CLIPEvaluator: model unloaded from {}", self.device)

    def __enter__(self) -> "CLIPEvaluator":
        self.load_clip()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"CLIPEvaluator(model={self.model_id!r} | "
            f"device={self.device} | {status})"
        )

    # =========================================================================
    # ── Properties
    # =========================================================================

    @property
    def is_loaded(self) -> bool:
        """True if the CLIP model is loaded and ready."""
        return self._loaded

    @property
    def model(self) -> Any:
        """The underlying CLIPModel (or None if not loaded)."""
        return self._model

    @property
    def processor(self) -> Any:
        """The underlying CLIPProcessor (or None if not loaded)."""
        return self._processor


# =============================================================================
# ── Module-Level Convenience API
# =============================================================================

_DEFAULT_EVALUATOR: Optional[CLIPEvaluator] = None


def _get_evaluator() -> CLIPEvaluator:
    """Return or create the module-level singleton evaluator."""
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = CLIPEvaluator()
    return _DEFAULT_EVALUATOR


def load_clip(
    model_id:   str  = DEFAULT_MODEL_ID,
    device:     str  = "auto",
    force:      bool = False,
) -> bool:
    """
    Module-level shortcut — load the CLIP model into the shared evaluator.

    Parameters
    ----------
    model_id : str
    device   : str  ``"cuda"`` | ``"cpu"`` | ``"auto"``
    force    : bool  Reload even if already loaded.

    Returns
    -------
    bool  True if loaded successfully.

    Example
    -------
        from src.evaluation.week2_clip_evaluator import load_clip
        ok = load_clip()
    """
    ev = _get_evaluator()
    ev.model_id = model_id
    ev.device   = ev._resolve_device(device)
    return ev.load_clip(force=force)


def evaluate(
    image,
    prompt:     str,
    image_id:   Optional[str] = None,
    *,
    auto_load:  bool = True,
) -> Dict[str, Any]:
    """
    Module-level shortcut — evaluate a single image.

    Parameters
    ----------
    image     : PIL.Image.Image
    prompt    : str
    image_id  : str, optional
    auto_load : bool  Auto-load model if not already loaded.

    Returns
    -------
    dict — ``{"clip_score": 0.92, "alignment": "high", ...}``

    Example
    -------
        from src.evaluation.week2_clip_evaluator import evaluate
        result = evaluate(pil_img, "a black oversized hoodie")
        print(result["clip_score"])
    """
    ev = _get_evaluator()
    if auto_load and not ev.is_loaded:
        ev.load_clip()
    return ev.evaluate(image, prompt, image_id=image_id).to_dict()


def rank_images(
    images:     List[Any],
    prompt:     str,
    image_ids:  Optional[List[str]] = None,
    *,
    descending: bool = True,
    auto_load:  bool = True,
) -> List[Dict[str, Any]]:
    """
    Module-level shortcut — rank images by CLIP similarity.

    Parameters
    ----------
    images     : list of PIL.Image.Image
    prompt     : str
    image_ids  : list of str, optional
    descending : bool  Highest score = rank 1 (default True).
    auto_load  : bool

    Returns
    -------
    list of dict — ``[{"rank": 1, "clip_score": 0.31, ...}, ...]``

    Example
    -------
        from src.evaluation.week2_clip_evaluator import rank_images
        ranked = rank_images([img1, img2, img3], "a red silk gown")
    """
    ev = _get_evaluator()
    if auto_load and not ev.is_loaded:
        ev.load_clip()
    return [r.to_dict() for r in ev.rank_images(images, prompt, image_ids, descending=descending)]


def compare_generations(
    images_a:  List[Any],
    images_b:  List[Any],
    prompt:    str,
    *,
    ids_a:     Optional[List[str]] = None,
    ids_b:     Optional[List[str]] = None,
    auto_load: bool = True,
) -> Dict[str, Any]:
    """
    Module-level shortcut — compare two image sets.

    Parameters
    ----------
    images_a  : list of PIL.Image.Image   Baseline / set A.
    images_b  : list of PIL.Image.Image   New / set B.
    prompt    : str
    ids_a, ids_b : list of str, optional
    auto_load : bool

    Returns
    -------
    dict — comparison result with winner, delta, per-set scores.

    Example
    -------
        from src.evaluation.week2_clip_evaluator import compare_generations
        result = compare_generations([baseline_img], [new_img], "a navy suit")
        print(result["winner"])  # "A" | "B" | "tie"
    """
    ev = _get_evaluator()
    if auto_load and not ev.is_loaded:
        ev.load_clip()
    return ev.compare_generations(
        images_a, images_b, prompt,
        ids_a=ids_a, ids_b=ids_b,
    ).to_dict()
