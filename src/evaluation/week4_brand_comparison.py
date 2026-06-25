"""
week4/evaluation/brand_comparison.py
====================================
Brand Comparison Framework.
Supports comparing generated designs across brand styles (Nike, Gucci, Zara, H&M)
measuring style similarity, visual differences, and CLIP scores, and generating reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from loguru import logger

from src.evaluation.week4_style_evaluator import (
    compute_clip_similarity,
    compute_structural_similarity,
)


class BrandComparisonFramework:
    """
    Framework to compare design styles, prompt alignments, and visual features
    between different fashion brands (e.g. Nike vs Gucci, Gucci vs Zara).
    """

    SUPPORTED_COMPARISONS = [
        ("nike", "gucci"),
        ("nike", "zara"),
        ("gucci", "zara"),
        ("h&m", "nike")
    ]

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the framework.

        Parameters
        ----------
        output_dir : Path or str, optional
            Where to save comparison reports.
        """
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = Path("outputs/evaluation").resolve()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BrandComparisonFramework | output_dir={self.output_dir}")

    def compute_visual_differences(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculate visual L1 pixel difference between two images.

        Parameters
        ----------
        image1 : PIL.Image.Image
        image2 : PIL.Image.Image

        Returns
        -------
        float
            L1 difference score (0.0 to 1.0, where 0.0 means identical).
        """
        img1 = np.array(image1.resize((128, 128)).convert("RGB"), dtype=np.float64)
        img2 = np.array(image2.resize((128, 128)).convert("RGB"), dtype=np.float64)
        
        # Absolute difference normalized to 0.0 - 1.0 range
        diff = np.mean(np.abs(img1 - img2)) / 255.0
        return float(diff)

    def compare_pair(
        self,
        image1: Image.Image,
        image2: Image.Image,
        prompt: str,
        brand1: str,
        brand2: str
    ) -> Dict[str, Any]:
        """
        Compare generated design pair of two brands.

        Parameters
        ----------
        image1 : PIL.Image.Image
            Generated design for brand1.
        image2 : PIL.Image.Image
            Generated design for brand2.
        prompt : str
            Base generation prompt.
        brand1 : str
        brand2 : str

        Returns
        -------
        dict
            Calculated comparison metrics.
        """
        b1_key = brand1.lower().strip()
        b2_key = brand2.lower().strip()

        # Check if the brand comparison is in our supported list
        is_supported = False
        for sb1, sb2 in self.SUPPORTED_COMPARISONS:
            if (b1_key == sb1 and b2_key == sb2) or (b1_key == sb2 and b2_key == sb1):
                is_supported = True
                break

        if not is_supported:
            logger.warning(f"Comparison between '{brand1}' and '{brand2}' is not explicitly listed in requirements.")

        # 1. Style Similarity (via structural SSIM alignment)
        style_sim = compute_structural_similarity(image1, image2)

        # 2. Visual Differences (L1 pixel diff)
        vis_diff = self.compute_visual_differences(image1, image2)

        # 3. CLIP Prompt Alignment Scores
        clip1 = compute_clip_similarity(image1, prompt)
        clip2 = compute_clip_similarity(image2, prompt)

        return {
            "comparison": f"{brand1.upper()} vs {brand2.upper()}",
            "brand1": b1_key,
            "brand2": b2_key,
            "style_similarity": round(style_sim, 4),
            "visual_differences": round(vis_diff, 4),
            "metrics": {
                f"{b1_key}_clip_score": round(clip1, 4),
                f"{b2_key}_clip_score": round(clip2, 4)
            }
        }

    def generate_comparison_report(
        self,
        brand_images: Dict[str, Image.Image],
        prompt: str,
        report_name: str = "brand_comparison_report.json"
    ) -> Dict[str, Any]:
        """
        Compare all listed brand pairs and generate a structured JSON report.

        Parameters
        ----------
        brand_images : dict of str to PIL.Image.Image
            Dictionary containing brand name keys mapped to their design PIL images.
            Must contain keys: "nike", "gucci", "zara", "h&m".
        prompt : str
            Generation prompt.
        report_name : str
            Name of the JSON file to save.

        Returns
        -------
        dict
            Full compiled report data.
        """
        # Ensure all required brands are present
        required_brands = {"nike", "gucci", "zara", "h&m"}
        available_brands = {k.lower().strip() for k in brand_images.keys()}
        missing = required_brands - available_brands
        if missing:
            raise ValueError(f"Missing images for required brands: {missing}")

        logger.info(f"Generating brand comparison report for prompt: '{prompt}'")
        
        comparisons = []
        for b1, b2 in self.SUPPORTED_COMPARISONS:
            img1 = brand_images[b1]
            img2 = brand_images[b2]
            res = self.compare_pair(img1, img2, prompt, b1, b2)
            comparisons.append(res)

        report = {
            "prompt": prompt,
            "timestamp": int(time.time()),
            "comparisons": comparisons,
            "summary": {
                "total_comparisons": len(comparisons),
                "mean_style_similarity": round(float(np.mean([c["style_similarity"] for c in comparisons])), 4),
                "mean_visual_differences": round(float(np.mean([c["visual_differences"] for c in comparisons])), 4)
            }
        }

        report_path = self.output_dir / report_name
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

        logger.success(f"Brand comparison report saved to: {report_path}")
        return report
