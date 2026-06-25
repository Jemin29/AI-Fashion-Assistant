"""
week2/generator/image_processor.py
====================================
Post-processing utilities for SDXL-generated images.

Responsibilities
----------------
- Save PIL images to disk with metadata sidecar
- Resize / upscale output images
- Add optional watermark
- Convert between formats
- Generate unique image IDs
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed — image processing disabled")


# =============================================================================
# ── Image ID Generation
# =============================================================================

def generate_image_id(prefix: str = "GEN") -> str:
    """
    Generate a unique image ID with timestamp and UUID suffix.

    Example: ``GEN_20260611_103045_a3f2``
    """
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{uid}"


# =============================================================================
# ── Sidecar Metadata
# =============================================================================

def build_metadata(
    image_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    scheduler: str,
    seed: int,
    model_id: str,
    generation_time_s: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structured metadata dict to accompany a generated image.

    Returns
    -------
    dict  — JSON-serialisable metadata record.
    """
    meta = {
        "image_id":          image_id,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "model_id":          model_id,
        "prompt":            prompt,
        "negative_prompt":   negative_prompt,
        "generation": {
            "width":          width,
            "height":         height,
            "steps":          steps,
            "guidance_scale": guidance_scale,
            "scheduler":      scheduler,
            "seed":           seed,
        },
        "timing": {
            "generation_time_s": round(generation_time_s, 3),
        },
    }
    if extra:
        meta.update(extra)
    return meta


# =============================================================================
# ── Saving
# =============================================================================

def save_image(
    image,                          # PIL.Image.Image
    image_id: str,
    output_dir: Path,
    fmt: str                = "png",
    quality: int            = 95,
) -> Path:
    """
    Save a PIL image to ``output_dir / {image_id}.{fmt}``.

    Parameters
    ----------
    image : PIL.Image.Image
    image_id : str
    output_dir : Path
    fmt : str   png | jpg | webp
    quality : int  JPEG/WebP quality (ignored for PNG)

    Returns
    -------
    Path  to the saved image file.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required to save images")

    output_dir.mkdir(parents=True, exist_ok=True)
    ext  = "jpg" if fmt == "jpeg" else fmt
    path = output_dir / f"{image_id}.{ext}"

    save_kwargs: Dict[str, Any] = {}
    if fmt in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    if fmt == "png":
        save_kwargs["optimize"] = True

    image.save(str(path), **save_kwargs)
    logger.debug("Image saved | path={} | size={}x{}", path, image.width, image.height)
    return path


def save_metadata_sidecar(
    metadata: Dict[str, Any],
    image_path: Path,
) -> Path:
    """
    Save a JSON sidecar file next to the image with the same stem.

    E.g.  ``outputs/images/GEN_001.png``  →  ``outputs/images/GEN_001.json``
    """
    sidecar = image_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Metadata sidecar saved | path={}", sidecar)
    return sidecar


# =============================================================================
# ── Resizing / Upscaling
# =============================================================================

def resize_image(
    image,
    target_size: Tuple[int, int],
    resample=None,
) -> "Image.Image":
    """
    Resize a PIL image to ``target_size`` (width, height).

    Uses LANCZOS for upscale, BICUBIC for downscale.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required")
    from PIL import Image as PILImage

    w, h       = target_size
    orig_w, orig_h = image.size
    is_upscale = (w * h) > (orig_w * orig_h)

    if resample is None:
        resample = PILImage.LANCZOS if is_upscale else PILImage.BICUBIC

    resized = image.resize((w, h), resample=resample)
    logger.debug(
        "Image resized | {}x{} → {}x{}",
        orig_w, orig_h, w, h,
    )
    return resized


def crop_to_square(image) -> "Image.Image":
    """Centre-crop a PIL image to a square using the shorter dimension."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required")
    w, h   = image.size
    side   = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


# =============================================================================
# ── Watermark
# =============================================================================

def add_watermark(
    image,
    text: str           = "AI Fashion · Week 2",
    opacity: int        = 40,
    position: str       = "bottom_right",
) -> "Image.Image":
    """
    Overlay a semi-transparent text watermark on the image.

    Parameters
    ----------
    image : PIL.Image.Image
    text : str
    opacity : int   0 = invisible, 255 = fully opaque
    position : str  bottom_right | bottom_left | top_right | top_left | center

    Returns
    -------
    PIL.Image.Image  with watermark composited.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required")
    from PIL import Image as PILImage, ImageDraw as PILDraw

    img   = image.convert("RGBA")
    layer = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw  = PILDraw.Draw(layer)

    # Simple font (no custom font needed)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    margin = 12
    w, h = img.size

    pos_map = {
        "bottom_right": (w - tw - margin, h - th - margin),
        "bottom_left":  (margin, h - th - margin),
        "top_right":    (w - tw - margin, margin),
        "top_left":     (margin, margin),
        "center":       ((w - tw) // 2, (h - th) // 2),
    }
    pos = pos_map.get(position, pos_map["bottom_right"])

    draw.text(pos, text, font=font, fill=(255, 255, 255, opacity))
    result = PILImage.alpha_composite(img, layer)
    return result.convert("RGB")


# =============================================================================
# ── Batch utilities
# =============================================================================

def images_to_grid(
    images: List,
    cols: int = 2,
    padding: int = 8,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
) -> "Image.Image":
    """
    Arrange a list of PIL images into a grid layout.

    Parameters
    ----------
    images : list of PIL.Image.Image
    cols : int   Number of columns in the grid
    padding : int   Pixels of padding between cells
    bg_color : tuple   RGB background colour

    Returns
    -------
    PIL.Image.Image  — composite grid image.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required")
    if not images:
        raise ValueError("images list is empty")

    from PIL import Image as PILImage

    rows      = (len(images) + cols - 1) // cols
    cell_w, cell_h = images[0].size
    grid_w    = cols * cell_w + (cols + 1) * padding
    grid_h    = rows * cell_h + (rows + 1) * padding

    grid = PILImage.new("RGB", (grid_w, grid_h), bg_color)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx  % cols
        x   = padding + col * (cell_w + padding)
        y   = padding + row * (cell_h + padding)
        grid.paste(img.resize((cell_w, cell_h)), (x, y))

    logger.debug(
        "Image grid created | {}×{} cells | {}x{}px",
        rows, cols, grid_w, grid_h,
    )
    return grid
