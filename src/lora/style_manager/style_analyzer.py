"""
week4/style_manager/style_analyzer.py
=====================================
Brand Style Analysis Engine.
Extracts color palettes, silhouettes, design patterns, aesthetics,
and garment categories to build a serialized brand style profile registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image

try:
    from src.lora.datasets.brand_dataset_manager import BrandDatasetManager
    _HAS_DATASET_MANAGER = True
except ImportError:
    BrandDatasetManager = None
    _HAS_DATASET_MANAGER = False


# =============================================================================
# ── Brand Style Analyzer Class
# =============================================================================

class BrandStyleAnalyzer:
    """
    Analyzes fashion brand datasets to identify stylistic signatures.
    Compiles profiles containing palettes, design patterns, aesthetics, fits, and silhouettes.
    """

    COLOR_MAP = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "grey": (128, 128, 128),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 128, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "brown": (139, 69, 19),
        "pink": (255, 192, 203),
        "olive": (128, 128, 0),
        "navy": (0, 0, 128),
        "cream": (255, 253, 208),
        "beige": (245, 245, 220)
    }

    # Brand defaults in case dataset is empty or metadata keywords are absent
    BRAND_DEFAULTS = {
        "nike": {
            "style": "sportswear",
            "dominant_colors": ["black", "white", "grey"],
            "fit": "athletic",
            "design_language": "performance",
            "silhouette": "relaxed",
            "aesthetic": "techwear",
            "pattern": "graphic",
            "style_signature": "performance fabrics with screen-printed brand graphic logos"
        },
        "gucci": {
            "style": "luxury",
            "dominant_colors": ["red", "green", "brown"],
            "fit": "tailored",
            "design_language": "opulence",
            "silhouette": "fitted",
            "aesthetic": "haute-couture",
            "pattern": "monogram",
            "style_signature": "bold monogram prints, embroidery, and gold metal accents"
        },
        "zara": {
            "style": "smart-casual",
            "dominant_colors": ["beige", "black", "cream"],
            "fit": "relaxed",
            "design_language": "trend-driven",
            "silhouette": "oversized",
            "aesthetic": "streetwear",
            "pattern": "solid",
            "style_signature": "oversized structured shoulders and lightweight minimalist linens"
        },
        "h&m": {
            "style": "basic-casual",
            "dominant_colors": ["white", "grey", "navy"],
            "fit": "regular",
            "design_language": "essential",
            "silhouette": "standard",
            "aesthetic": "minimalist",
            "pattern": "plain",
            "style_signature": "solid organic cottons, ribbing, and standard basic cuts"
        }
    }

    def __init__(
        self,
        config: Any = None,
        dataset_manager: Optional[BrandDatasetManager] = None,
        profile_output_path: Union[str, Path, None] = None
    ) -> None:
        """
        Initialize the BrandStyleAnalyzer.

        Parameters
        ----------
        config : Week4Config, optional
        dataset_manager : BrandDatasetManager, optional
            Manager to load manifests and images.
        profile_output_path : Path or str, optional
            Path to save brand_style_profile.json (default: outputs/datasets/brand_style_profile.json).
        """
        self.config = config
        self.dataset_manager = dataset_manager or (
            BrandDatasetManager(config) if _HAS_DATASET_MANAGER else None
        )
        
        # Determine output path
        if profile_output_path:
            self.profile_output_path = Path(profile_output_path).resolve()
        elif dataset_manager:
            self.profile_output_path = dataset_manager.dataset_root / "brand_style_profile.json"
        elif config and getattr(config, "output_root", None):
            self.profile_output_path = Path(config.output_root).resolve() / "datasets" / "brand_style_profile.json"
        else:
            self.profile_output_path = Path("outputs/datasets/brand_style_profile.json").resolve()

        self.profile_output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BrandStyleAnalyzer | output={self.profile_output_path}")

    # ── Public APIs: Core Analysis Methods ────────────────────────────────────

    def analyze_brand(self, brand: str) -> Dict[str, Any]:
        """
        Analyze a brand's dataset manifest to compile its style profile.

        Parameters
        ----------
        brand : str
            Brand to analyze (nike, gucci, zara, h&m).

        Returns
        -------
        dict
            Brand style profile.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.BRAND_DEFAULTS:
            raise ValueError(f"Brand '{brand}' not supported by style analyzer.")

        defaults = self.BRAND_DEFAULTS[brand_key]

        # Return defaults if manager is absent or manifest is empty
        if not self.dataset_manager:
            logger.debug(f"Dataset manager absent. Returning style defaults for {brand_key}.")
            return defaults

        manifest_path = self.dataset_manager.dataset_root / f"{brand_key}_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found for '{brand_key}'. Returning defaults.")
            return defaults

        # Load manifest
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as err:
            logger.error(f"Failed to read manifest {manifest_path}: {err}. Returning defaults.")
            return defaults

        if not manifest:
            logger.info(f"Manifest for '{brand_key}' is empty. Returning style defaults.")
            return defaults

        # ── 1. Analyze categories, silhouettes, design patterns, aesthetics ───────
        categories: Dict[str, int] = {}
        silhouettes: Dict[str, int] = {}
        patterns: Dict[str, int] = {}
        aesthetics: Dict[str, int] = {}
        colors_meta: Dict[str, int] = {}

        for record in manifest.values():
            # Categories
            cat = record.get("category", "apparel")
            categories[cat] = categories.get(cat, 0) + 1

            # Scrape prompt strings for style details
            prompt = record.get("prompt", "").lower()
            
            # Silhouettes
            for sil in ["oversized", "fitted", "relaxed", "tailored", "a-line", "baggy", "cropped"]:
                if sil in prompt:
                    silhouettes[sil] = silhouettes.get(sil, 0) + 1
            
            # Design Patterns
            for pat in ["solid", "print", "monogram", "striped", "check", "graphic", "plain"]:
                if pat in prompt:
                    patterns[pat] = patterns.get(pat, 0) + 1

            # Aesthetics
            for aes in ["minimalist", "techwear", "luxury", "haute-couture", "casual", "streetwear", "retro"]:
                if aes in prompt:
                    aesthetics[aes] = aesthetics.get(aes, 0) + 1

            # Style keywords (extracted color metadata if specified)
            for tag in record.get("style_tags", []):
                aesthetics[tag] = aesthetics.get(tag, 0) + 1

        # ── 2. Analyze dominant colors from files ─────────────────────────────
        dominant_colors = self._extract_dominant_colors_from_files(manifest)

        # ── 3. Compile Profile fields ─────────────────────────────────────────
        style = defaults["style"]
        fit = defaults["fit"]
        design_lang = defaults["design_language"]

        # Extract top categories/aesthetics to determine signatures
        top_category = self._get_most_frequent(categories, "apparel")
        top_silhouette = self._get_most_frequent(silhouettes, defaults["silhouette"])
        top_aesthetic = self._get_most_frequent(aesthetics, defaults["aesthetic"])
        top_pattern = self._get_most_frequent(patterns, defaults["pattern"])

        # Construct style signature description
        sig = f"{top_silhouette} {top_aesthetic} silhouettes using predominantly {top_pattern} patterns"

        profile = {
            "style": style,
            "dominant_colors": dominant_colors if dominant_colors else defaults["dominant_colors"],
            "fit": fit,
            "design_language": design_lang,
            "silhouette": top_silhouette,
            "aesthetic": top_aesthetic,
            "pattern": top_pattern,
            "style_signature": sig
        }

        logger.success(f"Successfully analyzed style profile for brand: '{brand_key}'")
        return profile

    def generate_profile_registry(self, brands: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compile style profiles for all brands and save a centralized profile json registry.

        Parameters
        ----------
        brands : list of str
            Brands to compile.

        Returns
        -------
        dict
            Combined registry mapping brand keys to compiled profiles.
        """
        registry = {}
        for brand in brands:
            brand_key = brand.lower().strip()
            if brand_key in self.BRAND_DEFAULTS:
                registry[brand_key] = self.analyze_brand(brand_key)

        try:
            with open(self.profile_output_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            logger.success(f"Centralized brand style profile registry saved to: {self.profile_output_path}")
        except Exception as err:
            logger.error(f"Failed to serialize profile registry to {self.profile_output_path}: {err}")

        return registry

    # ── Internal Helper Methods ──────────────────────────────────────────────

    def _get_most_frequent(self, frequencies: Dict[str, int], fallback: str) -> str:
        """Helper to find the key with maximum value in a frequencies dict."""
        if not frequencies:
            return fallback
        return max(frequencies, key=frequencies.get) # type: ignore[arg-type]

    def _extract_dominant_colors_from_files(self, manifest: Dict[str, Dict[str, Any]]) -> List[str]:
        """Read files listed in manifest and quantize colors to extract top binned hues."""
        color_counts: Dict[str, int] = {}
        processed_count = 0

        # Scan up to 5 images to avoid heavy I/O loops in large datasets
        for record in list(manifest.values())[:5]:
            img_path = self.dataset_manager.dataset_root / record["image_path"] # type: ignore[union-attr]
            if not img_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    # Quantize colors
                    colors = self._get_image_dominant_colors(img)
                    for col in colors:
                        color_counts[col] = color_counts.get(col, 0) + 1
                processed_count += 1
            except Exception as err:
                logger.debug(f"Could not extract colors from {img_path}: {err}")

        if not color_counts:
            return []

        # Sort binned colors by frequency
        sorted_colors = sorted(color_counts, key=color_counts.get, reverse=True) # type: ignore[arg-type]
        return sorted_colors[:3]

    def _get_image_dominant_colors(self, image: Image.Image) -> List[str]:
        """Resize image and map pixel colors to standard binned color hues."""
        img_small = image.resize((32, 32)).convert("RGB")
        pixels = list(img_small.getdata())

        binned_counts: Dict[str, int] = {}
        for r, g, b in pixels:
            # Map pixel to standard COLOR_MAP key using closest Euclidean distance
            closest_color = "grey"
            min_dist = float("inf")
            for color_name, (cr, cg, cb) in self.COLOR_MAP.items():
                dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_color = color_name
            binned_counts[closest_color] = binned_counts.get(closest_color, 0) + 1

        # Return top 2 colors of this image
        top_colors = sorted(binned_counts, key=binned_counts.get, reverse=True) # type: ignore[arg-type]
        return top_colors[:2]
