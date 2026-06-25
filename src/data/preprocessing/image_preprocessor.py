"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/preprocessing/image_preprocessor.py
=============================================================================
PURPOSE:
    Transforms raw fashion images into a clean, standardized format ready
    for model training and metadata generation.

OPERATIONS PERFORMED (in order):
    1. Colour mode normalisation  — convert to RGB
    2. Resize                     — to target resolution (default 256×256)
    3. Quality filtering          — reject blurry/low-contrast images
    4. Normalization              — ImageNet mean/std scaling (for model input)
    5. Export                     — save processed image to disk
    6. Metadata update            — return updated record dict

DESIGN DECISIONS:
    - Pure Pillow for resizing/saving (no OpenCV dependency for basic ops).
    - All transforms are stateless — the preprocessor is thread-safe.
    - Quality checks use Laplacian variance (blur) & histogram entropy (contrast).
    - Normalization is optional (raw uint8 arrays also saved for flexibility).

USAGE:
    >>> from src.data.preprocessing import FashionPreprocessor
    >>> pp = FashionPreprocessor(output_dir="datasets/processed", target_size=(256, 256))
    >>> processed_record = pp.process(raw_record)
=============================================================================
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError as exc:
    raise ImportError(
        "Pillow is required. Install with: pip install Pillow"
    ) from exc

try:
    import cv2  # OpenCV for Laplacian blur detection
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False
    logger.warning(
        "opencv-python not found — blur detection will be disabled. "
        "Install with: pip install opencv-python"
    )


# ─── Configuration dataclass ──────────────────────────────────────────────────

@dataclass
class PreprocessorConfig:
    """
    All tuneable parameters for the image preprocessor.
    Mirrors the values in configs/settings.yaml → pipeline.preprocessing.
    """
    # Output dimensions (height, width)
    target_size: Tuple[int, int] = (256, 256)

    # PIL resampling filter: LANCZOS for downscaling (best quality)
    resample_filter: int = Image.LANCZOS  # type: ignore[attr-defined]

    # Output format and JPEG quality
    image_format: str = "JPEG"
    jpeg_quality: int = 95

    # ImageNet normalization parameters
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Quality filtering thresholds (set to 0 to disable)
    blur_threshold: float = 50.0     # Laplacian variance; below = too blurry
    min_entropy: float = 3.0         # Image entropy; below = too flat/solid

    # Minimum acceptable dimensions (reject smaller images)
    min_width: int = 64
    min_height: int = 64


# ─── Processing result dataclass ──────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """Encapsulates the outcome of processing one image record."""
    success: bool
    image_id: str
    processed_path: Optional[str] = None
    normalized_array: Optional[np.ndarray] = None  # float32, shape (3, H, W)
    rejection_reason: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Main class ───────────────────────────────────────────────────────────────

class FashionPreprocessor:
    """
    Transforms raw fashion image records into standardized processed outputs.

    Thread-safe: all methods are stateless (no mutable state after __init__).
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: Optional[PreprocessorConfig] = None,
    ) -> None:
        """
        Args:
            output_dir : Directory where processed images will be saved.
            config     : PreprocessorConfig instance. Defaults if not provided.
        """
        self.output_dir = Path(output_dir)
        self.config = config or PreprocessorConfig()

        # Create output directory structure upfront
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"FashionPreprocessor initialised | "
            f"target_size={self.config.target_size} | "
            f"output_dir={self.output_dir}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, record: Dict[str, Any]) -> ProcessingResult:
        """
        Process a single raw FashionRecord and return the result.

        Pipeline:
            raw uint8 array
            → colour normalisation (→ RGB)
            → quality check (blur, contrast)
            → resize (Lanczos)
            → save to disk
            → normalize (float32, channel-first)
            → return ProcessingResult

        Args:
            record : Dict from FashionGenIngester or DeepFashionIngester.

        Returns:
            ProcessingResult with success status and all artefacts.
        """
        start_time = time.perf_counter()
        image_id   = record.get("image_id", "unknown")

        try:
            # ── Step 1: obtain PIL image ─────────────────────────────────────
            pil_image = self._to_pil(record)
            if pil_image is None:
                return ProcessingResult(
                    success=False,
                    image_id=image_id,
                    rejection_reason="Could not convert source to PIL Image",
                )

            # ── Step 2: ensure RGB ────────────────────────────────────────────
            pil_image = pil_image.convert("RGB")

            # ── Step 3: dimension check ───────────────────────────────────────
            w, h = pil_image.size
            if w < self.config.min_width or h < self.config.min_height:
                return ProcessingResult(
                    success=False,
                    image_id=image_id,
                    rejection_reason=(
                        f"Image too small: {w}×{h} "
                        f"(min {self.config.min_width}×{self.config.min_height})"
                    ),
                )

            # ── Step 4: quality checks ─────────────────────────────────────
            rejection = self._quality_check(pil_image)
            if rejection:
                return ProcessingResult(
                    success=False,
                    image_id=image_id,
                    rejection_reason=rejection,
                )

            # ── Step 5: resize ────────────────────────────────────────────────
            resized = pil_image.resize(
                self.config.target_size,
                resample=self.config.resample_filter,
            )

            # ── Step 6: save processed image to disk ─────────────────────────
            out_path = self._build_output_path(record)
            self._save_image(resized, out_path)

            # ── Step 7: build float32 normalized tensor (for model input) ────
            raw_array = np.array(resized, dtype=np.float32) / 255.0  # [0,1]
            if self.config.normalize:
                normalized = self._normalize(raw_array)
            else:
                normalized = raw_array.transpose(2, 0, 1)  # HWC → CHW

            # ── Step 8: build result metadata ────────────────────────────────
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result_meta = {
                "width"          : self.config.target_size[1],
                "height"         : self.config.target_size[0],
                "color_mode"     : "RGB",
                "file_size_bytes": out_path.stat().st_size,
                "image_format"   : self.config.image_format,
                "md5_hash"       : self._md5(out_path),
                "processing_time_ms": round(elapsed_ms, 2),
            }
            # Carry over original metadata from record
            result_meta.update({
                k: v for k, v in record.items()
                if k not in ("image_array",)  # exclude raw array
            })

            return ProcessingResult(
                success=True,
                image_id=image_id,
                processed_path=str(out_path),
                normalized_array=normalized,
                processing_time_ms=round(elapsed_ms, 2),
                metadata=result_meta,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Preprocessing failed for {image_id}: {exc}")
            return ProcessingResult(
                success=False,
                image_id=image_id,
                rejection_reason=f"Unexpected error: {exc}",
                processing_time_ms=round(elapsed_ms, 2),
            )

    def process_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[ProcessingResult]:
        """
        Process a list of records sequentially.

        For parallel processing, wrap this with a ThreadPoolExecutor.

        Args:
            records : List of raw FashionRecord dicts.

        Returns:
            List of ProcessingResult objects (one per record).
        """
        results = []
        for record in records:
            result = self.process(record)
            results.append(result)
            if not result.success:
                logger.debug(
                    f"Rejected {result.image_id}: {result.rejection_reason}"
                )
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Batch complete | {success_count}/{len(records)} succeeded"
        )
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(record: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Convert a record's image source to a PIL Image.

        Supports:
          - record["image_array"] : np.ndarray (from FashionGen HDF5)
          - record["original_path"]: str path to image file (DeepFashion)

        Returns None if conversion fails.
        """
        if "image_array" in record and record["image_array"] is not None:
            arr = record["image_array"]
            if isinstance(arr, np.ndarray):
                return Image.fromarray(arr.astype(np.uint8))

        if "original_path" in record and record["original_path"]:
            try:
                return Image.open(record["original_path"])
            except Exception as exc:
                logger.debug(f"Cannot open {record['original_path']}: {exc}")

        return None

    def _quality_check(self, image: Image.Image) -> Optional[str]:
        """
        Run quality filters. Returns a rejection reason string, or None if OK.

        Checks:
          1. Blur (Laplacian variance via OpenCV) — skipped if cv2 unavailable.
          2. Entropy (contrast proxy via PIL ImageStat).
        """
        # ── Blur check (requires OpenCV) ──────────────────────────────────────
        if _OPENCV_AVAILABLE and self.config.blur_threshold > 0:
            gray = np.array(image.convert("L"))
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.config.blur_threshold:
                return (
                    f"Image too blurry "
                    f"(Laplacian={laplacian_var:.1f} < {self.config.blur_threshold})"
                )

        # ── Contrast check via entropy proxy ─────────────────────────────────
        if self.config.min_entropy > 0:
            stat = ImageStat.Stat(image)
            # Use mean of per-channel std as a simple contrast proxy
            mean_std = sum(stat.stddev) / len(stat.stddev)
            if mean_std < self.config.min_entropy:
                return (
                    f"Image too low contrast "
                    f"(mean_std={mean_std:.2f} < {self.config.min_entropy})"
                )

        return None  # Passed all checks

    def _build_output_path(self, record: Dict[str, Any]) -> Path:
        """
        Determine the output file path for a processed image.

        Organises by: output_dir / dataset_source / split / image_id.ext
        """
        source = record.get("dataset_source", "unknown")
        split  = record.get("split", "train")
        img_id = record.get("image_id", "img_unknown")

        ext = ".jpg" if self.config.image_format == "JPEG" else f".{self.config.image_format.lower()}"
        out_subdir = self.output_dir / source / split
        out_subdir.mkdir(parents=True, exist_ok=True)

        return out_subdir / f"{img_id}{ext}"

    def _save_image(self, image: Image.Image, path: Path) -> None:
        """Save a PIL image to disk with appropriate format settings."""
        save_kwargs: Dict[str, Any] = {}
        if self.config.image_format == "JPEG":
            save_kwargs["quality"] = self.config.jpeg_quality
            save_kwargs["optimize"] = True
        image.save(path, format=self.config.image_format, **save_kwargs)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet mean/std normalization and convert HWC → CHW.

        Args:
            arr : Float32 array, shape (H, W, 3), values in [0, 1].

        Returns:
            Float32 array, shape (3, H, W), normalized per channel.
        """
        mean = np.array(self.config.mean, dtype=np.float32)  # (3,)
        std  = np.array(self.config.std,  dtype=np.float32)  # (3,)
        # Broadcast: (H, W, 3) - (3,) / (3,) → normalize each channel
        normalized = (arr - mean) / std
        # Convert HWC → CHW for PyTorch compatibility
        return normalized.transpose(2, 0, 1)  # (3, H, W)

    @staticmethod
    def _md5(path: Path) -> str:
        """Compute MD5 hex digest of a file for integrity verification."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
