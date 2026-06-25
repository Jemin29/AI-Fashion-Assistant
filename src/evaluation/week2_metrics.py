"""
week2/evaluation/metrics.py
==============================
Image quality and semantic similarity metrics for Week 2.

Metrics Implemented
-------------------
- CLIP cosine similarity (prompt ↔ image alignment)
- Basic image quality checks (black/white detection, resolution)
- SSIM / PSNR (reference-based, optional)
- Histogram entropy (diversity proxy)

All metrics gracefully degrade — unavailable libraries are skipped
with a warning rather than crashing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# ── Per-Image Metric Result
# =============================================================================

@dataclass
class ImageMetrics:
    """
    All computed metrics for a single generated image.

    Attributes
    ----------
    image_id : str
    clip_similarity : float or None   Range [0, 1].
    is_black : bool                   Mean pixel < threshold.
    is_white : bool                   Mean pixel > threshold.
    resolution_ok : bool              Width & height ≥ minimums.
    width, height : int
    mean_pixel : float                Mean pixel value [0, 255].
    histogram_entropy : float         Shannon entropy of pixel histogram.
    ssim : float or None              SSIM vs reference (if provided).
    psnr : float or None              PSNR vs reference (if provided).
    passed_quality : bool             Summary flag.
    """
    image_id:           str
    clip_similarity:    Optional[float] = None
    is_black:           bool            = False
    is_white:           bool            = False
    resolution_ok:      bool            = True
    width:              int             = 0
    height:             int             = 0
    mean_pixel:         float           = 0.0
    histogram_entropy:  float           = 0.0
    ssim:               Optional[float] = None
    psnr:               Optional[float] = None
    passed_quality:     bool            = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id":          self.image_id,
            "clip_similarity":   self.clip_similarity,
            "is_black":          self.is_black,
            "is_white":          self.is_white,
            "resolution_ok":     self.resolution_ok,
            "width":             self.width,
            "height":            self.height,
            "mean_pixel":        round(self.mean_pixel, 2),
            "histogram_entropy": round(self.histogram_entropy, 4),
            "ssim":              self.ssim,
            "psnr":              self.psnr,
            "passed_quality":    self.passed_quality,
        }


# =============================================================================
# ── Basic Image Quality Checks (no ML required)
# =============================================================================

def check_image_quality(
    image,
    image_id:        str,
    min_width:       int   = 512,
    min_height:      int   = 512,
    black_threshold: int   = 5,
    white_threshold: int   = 250,
) -> ImageMetrics:
    """
    Run fast, heuristic quality checks on a PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
    image_id : str
    min_width, min_height : int   Minimum acceptable resolution.
    black_threshold : int         Mean pixel below this → black image.
    white_threshold : int         Mean pixel above this → white image.

    Returns
    -------
    ImageMetrics  with quality flags set.
    """
    if not PIL_AVAILABLE:
        logger.warning("Pillow not available — skipping quality checks")
        return ImageMetrics(image_id=image_id, passed_quality=True)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    mean_px   = float(arr.mean())
    w, h      = image.size
    res_ok    = (w >= min_width) and (h >= min_height)
    is_black  = mean_px < black_threshold
    is_white  = mean_px > white_threshold
    entropy   = _histogram_entropy(arr)

    passed = res_ok and not is_black and not is_white

    if is_black:
        logger.warning("Image {} appears all-black (mean={:.1f})", image_id, mean_px)
    if is_white:
        logger.warning("Image {} appears all-white (mean={:.1f})", image_id, mean_px)
    if not res_ok:
        logger.warning("Image {} resolution {}x{} below minimum {}x{}", image_id, w, h, min_width, min_height)

    return ImageMetrics(
        image_id          = image_id,
        is_black          = is_black,
        is_white          = is_white,
        resolution_ok     = res_ok,
        width             = w,
        height            = h,
        mean_pixel        = mean_px,
        histogram_entropy = entropy,
        passed_quality    = passed,
    )


def _histogram_entropy(arr: np.ndarray) -> float:
    """
    Compute Shannon entropy of the flattened pixel histogram.

    Higher entropy → more visual diversity (less likelihood of degenerate output).
    """
    flat = arr.flatten().astype(np.uint8)
    counts = np.bincount(flat, minlength=256).astype(np.float64)
    probs  = counts / (counts.sum() + 1e-9)
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# =============================================================================
# ── CLIP Similarity
# =============================================================================

def compute_clip_similarity(
    image,
    prompt: str,
    model_name: str   = "ViT-L-14",
    pretrained: str   = "openai",
    device: str       = "cpu",
) -> Optional[float]:
    """
    Compute CLIP cosine similarity between an image and a text prompt.

    Requires ``open-clip-torch`` to be installed.

    Parameters
    ----------
    image : PIL.Image.Image
    prompt : str
    model_name : str   CLIP model variant.
    pretrained : str   Pretrained weights key.
    device : str

    Returns
    -------
    float in [0, 1] or None if CLIP is unavailable.
    """
    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model     = model.to(device).eval()

        with torch.no_grad():
            img_tensor   = preprocess(image).unsqueeze(0).to(device)
            text_tokens  = tokenizer([prompt]).to(device)
            img_features = model.encode_image(img_tensor)
            txt_features = model.encode_text(text_tokens)

            # Normalise
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

            similarity = (img_features @ txt_features.T).item()

        logger.debug("CLIP similarity | prompt={:.40}… | score={:.4f}", prompt, similarity)
        return float(similarity)

    except ImportError:
        logger.debug("open-clip-torch not installed — CLIP similarity skipped")
        return None
    except Exception as exc:
        logger.warning("CLIP similarity computation failed: {}", exc)
        return None


# =============================================================================
# ── SSIM / PSNR (Reference-Based)
# =============================================================================

def compute_ssim(image, reference) -> Optional[float]:
    """
    Compute SSIM between two PIL images.

    Requires ``scikit-image`` or ``torchmetrics``.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        arr1 = np.array(image.convert("L"))
        arr2 = np.array(reference.convert("L").resize(image.size))
        score = float(ssim_fn(arr1, arr2, data_range=255))
        return score
    except ImportError:
        pass
    try:
        import torch
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        import torchvision.transforms.functional as TF

        t1 = TF.to_tensor(image).unsqueeze(0)
        t2 = TF.to_tensor(reference.resize(image.size)).unsqueeze(0)
        return float(ssim_fn(t1, t2).item())
    except Exception:
        return None


def compute_psnr(image, reference) -> Optional[float]:
    """Compute PSNR (dB) between two PIL images."""
    try:
        arr1 = np.array(image.convert("RGB")).astype(np.float64)
        arr2 = np.array(reference.convert("RGB").resize(image.size)).astype(np.float64)
        mse  = np.mean((arr1 - arr2) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * math.log10(255.0 / math.sqrt(mse)))
    except Exception as exc:
        logger.debug("PSNR computation failed: {}", exc)
        return None


# =============================================================================
# ── Batch Aggregation
# =============================================================================

def aggregate_metrics(metrics_list: List[ImageMetrics]) -> Dict[str, Any]:
    """
    Compute summary statistics across a list of ImageMetrics.

    Returns
    -------
    dict with keys: total, passed, failed, pass_rate,
                    mean_clip, mean_entropy, black_count, white_count
    """
    n          = len(metrics_list)
    passed     = sum(1 for m in metrics_list if m.passed_quality)
    clips      = [m.clip_similarity for m in metrics_list if m.clip_similarity is not None]
    entropies  = [m.histogram_entropy for m in metrics_list]
    black_ct   = sum(1 for m in metrics_list if m.is_black)
    white_ct   = sum(1 for m in metrics_list if m.is_white)

    return {
        "total":        n,
        "passed":       passed,
        "failed":       n - passed,
        "pass_rate":    round(passed / n, 4) if n else 0.0,
        "mean_clip":    round(float(np.mean(clips)), 4) if clips else None,
        "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "black_count":  black_ct,
        "white_count":  white_ct,
    }
