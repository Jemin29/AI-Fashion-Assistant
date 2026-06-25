"""
week3/evaluation/structure_evaluator.py
========================================
Structural Evaluation Engine.
AI-Powered Fashion Design Assistant — Week 3.

Evaluates spatial layout, shape preservation, contour matching, and pixel-level structural similarity
between generated fashion images and reference conditions (sketches, pose sticks, depth maps).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from loguru import logger
from PIL import Image, ImageFilter


class StructureEvaluator:
    """
    Evaluates spatial structures, shape contours, and layouts
    between generated clothing designs and conditioning inputs.
    """

    def __init__(self, config: Any = None, mock: bool = False) -> None:
        """
        Initialize the StructureEvaluator.
        """
        self.config = config
        self.mock = mock

    def evaluate_structure(self, generated_img: Image.Image, reference_img: Image.Image) -> Dict[str, float]:
        """
        Computes structural metrics between the generated design and reference condition.

        Returns
        -------
        Dict[str, float]
            Dictionary containing ssim, edge_preservation, layout_consistency, and shape_preservation.
        """
        t0 = time.perf_counter()

        # Compute metrics
        ssim = self._compute_ssim(generated_img, reference_img)
        edge_pres = self._compute_edge_preservation(generated_img, reference_img)
        layout_const = self._compute_layout_consistency(generated_img, reference_img)
        shape_pres = self._compute_shape_preservation(generated_img, reference_img)

        elapsed = round(time.perf_counter() - t0, 4)
        logger.debug(f"Computed structural evaluation in {elapsed}s | ssim={ssim:.4f} | edge={edge_pres:.4f}")

        return {
            "ssim": round(ssim, 4),
            "edge_preservation": round(edge_pres, 4),
            "layout_consistency": round(layout_const, 4),
            "shape_preservation": round(shape_pres, 4)
        }

    def compare_images(
        self,
        standard_img: Image.Image,
        controlnet_img: Image.Image,
        reference_img: Image.Image
    ) -> Dict[str, Any]:
        """
        Compares unconditioned standard SDXL output vs conditioned ControlNet output
        against the reference template.
        """
        logger.info("Running structural comparison (Standard vs ControlNet)...")

        standard_metrics = self.evaluate_structure(standard_img, reference_img)
        controlnet_metrics = self.evaluate_structure(controlnet_img, reference_img)

        # Compute improvements
        improvements = {}
        for key in standard_metrics.keys():
            diff = controlnet_metrics[key] - standard_metrics[key]
            improvements[key] = round(diff, 4)

        return {
            "metrics": {
                "standard": standard_metrics,
                "controlnet": controlnet_metrics
            },
            "improvements": improvements,
            "controlnet_advantage": all(controlnet_metrics[k] >= standard_metrics[k] for k in standard_metrics)
        }

    # ── Private Mathematical Metric Implementations ───────────────────────────

    def _compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Grayscale SSIM formula."""
        size = (256, 256)
        im1 = np.array(img1.convert("L").resize(size), dtype=np.float32)
        im2 = np.array(img2.convert("L").resize(size), dtype=np.float32)

        K1 = 0.01
        K2 = 0.03
        L = 255.0
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        mu1 = im1.mean()
        mu2 = im2.mean()

        var1 = im1.var()
        var2 = im2.var()
        cov = np.mean((im1 - mu1) * (im2 - mu2))

        num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
        den = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
        
        if den == 0.0:
            return 0.0

        return float(num / den)

    def _compute_edge_preservation(self, img1: Image.Image, img2: Image.Image) -> float:
        """Computes IoU of binarized edge contours (Sobel/FIND_EDGES)."""
        size = (128, 128)
        
        # Apply edge detection
        edges1 = img1.convert("L").resize(size).filter(ImageFilter.FIND_EDGES)
        edges2 = img2.convert("L").resize(size).filter(ImageFilter.FIND_EDGES)
        
        arr1 = np.array(edges1, dtype=np.float32)
        arr2 = np.array(edges2, dtype=np.float32)
        
        # Binarize edge maps
        threshold = 40.0
        mask1 = arr1 > threshold
        mask2 = arr2 > threshold
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return float(intersection / union)

    def _compute_layout_consistency(self, img1: Image.Image, img2: Image.Image) -> float:
        """Pearson correlation of downsampled layout grids."""
        size = (16, 16)
        grid1 = np.array(img1.convert("L").resize(size), dtype=np.float32).flatten()
        grid2 = np.array(img2.convert("L").resize(size), dtype=np.float32).flatten()
        
        # Subtract mean
        g1 = grid1 - grid1.mean()
        g2 = grid2 - grid2.mean()
        
        norm1 = np.linalg.norm(g1)
        norm2 = np.linalg.norm(g2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        corr = np.dot(g1, g2) / (norm1 * norm2)
        # Normalize/clip correlation to [0.0, 1.0] range
        return float(max(0.0, corr))

    def _compute_shape_preservation(self, img1: Image.Image, img2: Image.Image) -> float:
        """Dice coefficient of foreground silhouette masks."""
        size = (128, 128)
        arr1 = np.array(img1.convert("L").resize(size), dtype=np.float32)
        arr2 = np.array(img2.convert("L").resize(size), dtype=np.float32)
        
        # Threshold to get silhouette mask (ignore black backgrounds)
        threshold = 15.0
        mask1 = arr1 > threshold
        mask2 = arr2 > threshold
        
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return float((2.0 * intersection) / total)
