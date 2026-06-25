"""
week3/preprocessors/sketch_processor.py
========================================
Sketch Preprocessing Engine for Week 3.
Transforms raw fashion designs/sketches into normalized ControlNet inputs.

Pipeline:
Image -> Grayscale -> Edge Detection -> Normalization -> ControlNet Input
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

# ── Lazy imports to allow importing without cv2 / controlnet_aux ──────────────
cv2 = None
controlnet_aux = None
np = None


# =============================================================================
# ── SketchProcessor Class
# =============================================================================

class SketchProcessor:
    """
    Preprocesses sketches and design images using Canny, HED, or Lineart.
    Normalizes images for direct injection into ControlNet SDXL models.
    """

    def __init__(self, config=None) -> None:
        """
        Initialize the SketchProcessor.
        """
        self.config = config
        self._hed_detector: Any = None
        self._lineart_detector: Any = None
        self._load_numpy()

    # ── Public APIs: Core Methods ─────────────────────────────────────────────

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """
        Load an image from disk and correct its rotation if EXIF orientation tag is set.
        """
        filepath = Path(path).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")

        try:
            img = Image.open(filepath)
            # Correct orientation using EXIF tags
            img = ImageOps.exif_transpose(img)
            logger.debug(f"Loaded image from {filepath} | size={img.size} | mode={img.mode}")
            return img.convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to load image from {filepath}: {exc}")
            raise IOError(f"Could not load image: {exc}") from exc

    def preprocess_sketch(
        self,
        image: Image.Image,
        method: str = "canny",
        **kwargs
    ) -> Image.Image:
        """
        Preprocess sketch image into standardized binary/edge outlines.

        Parameters
        ----------
        image : PIL.Image.Image
            Input sketch or fashion design image.
        method : str
            Edge detection strategy: "canny" | "hed" | "lineart".
        **kwargs
            Configurable parameters for detectors (e.g. low_threshold, high_threshold).
        """
        method = method.lower()
        t0 = ImageChops.duplicate(image) # Keeps copy of original reference
        
        logger.info(f"Preprocessing sketch using method: '{method}'...")

        # 1. Pipeline Stage: Grayscale
        gray_img = image.convert("L")

        # 2. Pipeline Stage: Edge Detection
        if method == "canny":
            processed = self._apply_canny(gray_img, **kwargs)
        elif method == "hed":
            processed = self._apply_hed(gray_img, **kwargs)
        elif method == "lineart":
            processed = self._apply_lineart(gray_img, **kwargs)
        else:
            logger.warning(f"Unknown edge method '{method}'. Defaulting to Canny.")
            processed = self._apply_canny(gray_img, **kwargs)

        # 3. Pipeline Stage: Normalization (Min-Max contrast correction)
        normalized = ImageOps.autocontrast(processed, cutoff=2)

        # 4. Pipeline Stage: ControlNet Format Input (Standardizes to 3-channel RGB)
        controlnet_input = normalized.convert("RGB")
        logger.success(f"Sketch preprocessed successfully | method={method} | size={controlnet_input.size}")
        return controlnet_input

    def save_processed_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save the preprocessed image to disk.
        """
        filepath = Path(output_path).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            image.save(filepath, format="PNG")
            logger.info(f"Saved preprocessed edge map to: {filepath}")
            return filepath
        except Exception as exc:
            logger.error(f"Failed to save processed image: {exc}")
            raise IOError(f"Failed to write image to disk: {exc}") from exc

    def create_comparison_grid(
        self,
        original: Image.Image,
        processed: Image.Image,
        title: str = "Sketch Preprocessing Comparison"
    ) -> Image.Image:
        """
        Creates a side-by-side comparison canvas.
        
        Parameters
        ----------
        original : PIL.Image.Image
            Original design sketch.
        processed : PIL.Image.Image
            Preprocessed ControlNet outline map.
        title : str
            Header label text.
        """
        w, h = original.size
        # Resize processed to match original size if necessary
        if processed.size != (w, h):
            processed = processed.resize((w, h), Image.Resampling.LANCZOS)

        # Define grid layout (Double width + margin space + header space)
        header_h = 60
        margin = 15
        grid_w = (w * 2) + (margin * 3)
        grid_h = h + header_h + (margin * 2)

        # Create canvas
        grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 35))
        draw = ImageDraw.Draw(grid)

        # Draw Header Title
        draw.text((grid_w // 2, header_h // 2), title.upper(), fill=(255, 255, 255), anchor="mm")

        # Paste images side-by-side
        x1 = margin
        y1 = header_h + margin
        grid.paste(original, (x1, y1))
        
        x2 = w + (margin * 2)
        grid.paste(processed, (x2, y1))

        # Draw labels below images
        draw.text((x1 + w // 2, y1 + h + 5), "ORIGINAL SKETCH", fill=(180, 180, 180), anchor="mt")
        draw.text((x2 + w // 2, y1 + h + 5), "CONTROLNET OUTLINE MAP", fill=(180, 180, 180), anchor="mt")

        return grid

    # ── Private Utility & Fallback Methods ────────────────────────────────────

    def _load_numpy(self) -> None:
        """Attempt to load numpy for fallback Sobel operations."""
        global np
        try:
            import numpy as _np
            np = _np
        except ImportError:
            np = None

    def _apply_canny(self, gray_img: Image.Image, **kwargs) -> Image.Image:
        """Apply Canny edge detection, using OpenCV or pure NumPy Sobel fallback."""
        global cv2
        low = kwargs.get("low_threshold", 100)
        high = kwargs.get("high_threshold", 200)

        # Try OpenCV Canny
        try:
            import cv2 as _cv2
            cv2 = _cv2
            img_arr = np.array(gray_img)
            edges = cv2.Canny(img_arr, low, high)
            return Image.fromarray(edges)
        except (ImportError, AttributeError):
            # Degrading fallback using custom Sobel-like filter in NumPy/PIL
            logger.warning("OpenCV/NumPy unavailable or failed. Using pure NumPy/PIL Canny fallback.")
            if np is not None:
                return self._sobel_edge_detection(gray_img, threshold=40)
            else:
                # PIL FIND_EDGES fallback
                edges = gray_img.filter(ImageFilter.FIND_EDGES)
                # Binarize to sharpen outlines
                return edges.point(lambda p: 255 if p > 35 else 0)

    def _apply_hed(self, gray_img: Image.Image, **kwargs) -> Image.Image:
        """Apply Holistically-Nested Edge Detection (HED) or PIL fallback."""
        global controlnet_aux
        
        try:
            from controlnet_aux import HEDdetector
            if self._hed_detector is None:
                self._hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
            
            # HED expects RGB PIL Image
            rgb_img = gray_img.convert("RGB")
            hed_out = self._hed_detector(rgb_img)
            return hed_out.convert("L")
        except Exception as err:
            logger.warning(f"HED detector unavailable. Using PIL mock filter fallback: {err}")
            # Fallback HED: Smoothed/blurred FIND_EDGES representation
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            hed_sim = edges.filter(ImageFilter.GaussianBlur(radius=1.5))
            return hed_sim

    def _apply_lineart(self, gray_img: Image.Image, **kwargs) -> Image.Image:
        """Apply Lineart Edge Detection or PIL fallback."""
        try:
            from controlnet_aux import LineartDetector
            if self._lineart_detector is None:
                self._lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
            
            rgb_img = gray_img.convert("RGB")
            line_out = self._lineart_detector(rgb_img, coarse=kwargs.get("coarse", False))
            return line_out.convert("L")
        except Exception as err:
            logger.warning(f"Lineart detector unavailable. Using PIL adaptive thresholding fallback: {err}")
            # Fallback Lineart: Sharpened edge contours (Finding Edges -> Inverting -> Dilating)
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            # Thresholding and dilating using PIL ops
            line_sim = ImageOps.invert(edges.point(lambda p: 255 if p > 25 else 0))
            return ImageOps.invert(line_sim)

    def _sobel_edge_detection(self, gray_img: Image.Image, threshold: int = 40) -> Image.Image:
        """Custom Sobel implementation using NumPy/SciPy operations."""
        if np is None:
            return gray_img

        arr = np.array(gray_img, dtype=np.float32)
        h, w = arr.shape

        # Initialize Sobel Kernels
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)

        # Sobel operations over indices
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Horizontal gradients
                gx[i, j] = (
                    -1 * arr[i-1, j-1] + 1 * arr[i-1, j+1] +
                    -2 * arr[i,   j-1] + 2 * arr[i,   j+1] +
                    -1 * arr[i+1, j-1] + 1 * arr[i+1, j+1]
                )
                # Vertical gradients
                gy[i, j] = (
                    -1 * arr[i-1, j-1] - 2 * arr[i-1, j] - 1 * arr[i-1, j+1] +
                    1 * arr[i+1, j-1] + 2 * arr[i+1, j] + 1 * arr[i+1, j+1]
                )

        # Calculate absolute magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Binarize output
        binary = np.where(magnitude > threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(binary)
