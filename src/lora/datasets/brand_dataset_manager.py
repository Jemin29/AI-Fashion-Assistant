"""
week4/datasets/brand_dataset_manager.py
=======================================
Fashion Brand Dataset Management System.
Manages ingestion, duplicate detection, image validation, metadata tagging,
and statistics compilation for brand-specific styling datasets (Nike, Gucci, Zara, H&M).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image


# =============================================================================
# ── Brand Dataset Manager Class
# =============================================================================

class BrandDatasetManager:
    """
    Manages brand fashion image datasets (Nike, Gucci, Zara, H&M) for LoRA fine-tuning.
    Provides image ingestion, validation, duplicate auditing, metadata generation, and statistics.
    """

    SUPPORTED_BRANDS = {"nike", "gucci", "zara", "h&m"}

    BRAND_STYLE_PRESETS = {
        "nike": ["sportswear", "techwear", "athletic"],
        "gucci": ["luxury", "haute-couture", "avant-garde", "formal"],
        "zara": ["casual", "contemporary", "streetwear", "smart-casual"],
        "h&m": ["basic", "casual", "minimalist", "affordable-chic"]
    }

    def __init__(self, config: Any = None, dataset_root: Union[str, Path, None] = None) -> None:
        """
        Initialize the BrandDatasetManager.

        Parameters
        ----------
        config : Week4Config, optional
        dataset_root : Path or str, optional
            Root path to store manifests and ingested images (default: outputs/datasets).
        """
        self.config = config
        
        # Determine dataset root folder
        if dataset_root:
            self.dataset_root = Path(dataset_root).resolve()
        elif config and getattr(config, "output_root", None):
            self.dataset_root = Path(config.output_root).resolve() / "datasets"
        else:
            self.dataset_root = Path("outputs/datasets").resolve()

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BrandDatasetManager | root={self.dataset_root}")

    # ── Public APIs: Core Ingestion & Auditing Methods ────────────────────────

    def ingest_image(
        self,
        brand: str,
        image: Image.Image,
        filename: str,
        raw_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate, tag, and ingest a single image into the brand's dataset.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        image : PIL.Image.Image
            PIL image object to ingest.
        filename : str
            Desired save filename.
        raw_metadata : dict, optional
            Manual tags or metadata context (e.g. categories, prompts).

        Returns
        -------
        dict
            The ingested record detailing image metadata and file paths.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.SUPPORTED_BRANDS:
            raise ValueError(f"Brand '{brand}' not supported. Choose from: {self.SUPPORTED_BRANDS}")

        # 1. Image Validation
        is_valid, reason = self.validate_image(image)
        if not is_valid:
            logger.error(f"Image validation failed for file {filename}: {reason}")
            raise ValueError(f"Invalid image file: {reason}")

        # Create brand folder structure
        brand_dir = self.dataset_root / brand_key
        brand_dir.mkdir(parents=True, exist_ok=True)

        # 2. Pixel Hashing for Duplicate Detection
        pixel_hash = self._compute_pixel_hash(image)

        # 3. Metadata Generation
        metadata = self._generate_metadata(brand_key, image, filename, raw_metadata)
        metadata["pixel_hash"] = pixel_hash

        # Check for duplicates in current registry
        manifest_path = self.dataset_root / f"{brand_key}_manifest.json"
        manifest = self._load_manifest(manifest_path)

        is_duplicate = any(record["pixel_hash"] == pixel_hash for record in manifest.values())
        metadata["is_duplicate"] = is_duplicate
        if is_duplicate:
            logger.warning(f"Duplicate image hash detected for {filename} in brand '{brand_key}' manifest.")

        # Save Image File
        dest_path = brand_dir / filename
        image.save(dest_path)
        metadata["image_path"] = str(dest_path.relative_to(self.dataset_root))

        # Update Manifest File
        manifest[filename] = metadata
        self._save_manifest(manifest_path, manifest)

        logger.success(f"Successfully ingested image into '{brand_key}' dataset: {dest_path}")
        return metadata

    def detect_duplicates(self, brand: str) -> List[Tuple[str, str]]:
        """
        Identify duplicate images in a brand's manifest.

        Returns
        -------
        list of tuple
            Pairs of duplicate filenames: [("image1.jpg", "image2.jpg"), ...]
        """
        brand_key = brand.lower().strip()
        manifest_path = self.dataset_root / f"{brand_key}_manifest.json"
        manifest = self._load_manifest(manifest_path)

        hash_map: Dict[str, List[str]] = {}
        for filename, record in manifest.items():
            h = record["pixel_hash"]
            hash_map.setdefault(h, []).append(filename)

        duplicates = []
        for filenames in hash_map.values():
            if len(filenames) > 1:
                # Pairwise duplicate matches
                for i in range(len(filenames)):
                    for j in range(i + 1, len(filenames)):
                        duplicates.append((filenames[i], filenames[j]))

        logger.info(f"Duplicate detection completed for brand '{brand_key}' | duplicates_found={len(duplicates)}")
        return duplicates

    def get_statistics(self, brand: str) -> Dict[str, Any]:
        """
        Compile dataset statistics matching the requested output format.

        Returns
        -------
        dict
            Summary format containing brand, total images count, and categories list.
        """
        brand_key = brand.lower().strip()
        manifest_path = self.dataset_root / f"{brand_key}_manifest.json"
        manifest = self._load_manifest(manifest_path)

        categories_set = set()
        for record in manifest.values():
            cat = record.get("category")
            if cat:
                categories_set.add(cat)

        stats = {
            "brand": brand_key,
            "images": len(manifest),
            "categories": sorted(list(categories_set))
        }
        
        logger.info(f"Statistics aggregated for brand '{brand_key}' | count={stats['images']}")
        return stats

    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Validate image dimensions, formats, and corruption status.
        """
        try:
            if not isinstance(image, Image.Image):
                return False, "Input is not a valid PIL Image object."

            w, h = image.size
            
            # Resolution audit: minimum 512x512
            if w < 512 or h < 512:
                return False, f"Resolution {w}x{h} is below minimum requirement of 512x512."

            # Aspect ratio check (ideal fashion ratios: 1:1, 4:5, 3:4)
            aspect_ratio = w / h
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                return False, f"Extreme aspect ratio {aspect_ratio:.2f} is outside normal bounds."

            return True, "Valid"
        except Exception as err:
            return False, f"Image structure corrupted or unreadable: {err}"

    def validate_dataset(self, brand: str) -> Dict[str, Any]:
        """
        Scan and audit all files listed in the brand's manifest.
        """
        brand_key = brand.lower().strip()
        manifest_path = self.dataset_root / f"{brand_key}_manifest.json"
        manifest = self._load_manifest(manifest_path)

        failures = []
        valid_count = 0

        for filename, record in manifest.items():
            img_path = self.dataset_root / record["image_path"]
            if not img_path.exists():
                failures.append({"filename": filename, "reason": "File does not exist on disk."})
                continue

            try:
                with Image.open(img_path) as img:
                    is_ok, reason = self.validate_image(img)
                    if not is_ok:
                        failures.append({"filename": filename, "reason": reason})
                    else:
                        valid_count += 1
            except Exception as err:
                failures.append({"filename": filename, "reason": f"File unreadable: {err}"})

        audit_result = {
            "brand": brand_key,
            "total_manifest_records": len(manifest),
            "valid_files": valid_count,
            "corrupt_or_missing_files": len(failures),
            "failures": failures
        }
        
        logger.info(f"Dataset validation completed for '{brand_key}' | valid={valid_count} | failures={len(failures)}")
        return audit_result

    # ── Internal Helper Methods ──────────────────────────────────────────────

    def _compute_pixel_hash(self, image: Image.Image) -> str:
        """Compute MD5 hash of image pixels for fast exact duplicate detection."""
        # Convert to RGB to standardize pixel format before hash
        img_rgb = image.convert("RGB")
        pixel_bytes = img_rgb.tobytes()
        return hashlib.md5(pixel_bytes).hexdigest()

    def _generate_metadata(
        self,
        brand: str,
        image: Image.Image,
        filename: str,
        raw_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate style descriptors, prompts, and category details."""
        raw = raw_metadata or {}
        
        # 1. Category Resolution
        category = raw.get("category", "apparel").lower().strip()
        # Guess category from filename if generic
        if category == "apparel":
            for cat in ["hoodie", "jacket", "shirt", "dress", "pants", "shoes", "sneakers"]:
                if cat in filename.lower():
                    category = cat + "s" if not cat.endswith("s") else cat
                    break

        # 2. Style Preset Extraction
        presets = self.BRAND_STYLE_PRESETS.get(brand, ["fashion"])
        style_tags = raw.get("style_tags", presets)

        # 3. Create descriptive Prompt
        color_dominant = raw.get("color", ["monochrome"])[0]
        desc = raw.get("description", f"A custom brand {brand} design.")
        
        prompt = (
            f"A high-fidelity fashion photo of a {brand} {category}, "
            f"{', '.join(style_tags)} style, {color_dominant} fabric, {desc}"
        )

        metadata = {
            "image_id": f"img_{int(time.time())}_{hashlib.md5(filename.encode()).hexdigest()[:6]}",
            "filename": filename,
            "brand": brand,
            "category": category,
            "style_tags": style_tags,
            "prompt": prompt,
            "resolution": f"{image.width}x{image.height}",
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "extra_info": raw.get("extra_info", {})
        }
        return metadata

    def _load_manifest(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Load manifest registry dictionary."""
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Could not read manifest registry {path}: {err}. Returning empty.")
            return {}

    def _save_manifest(self, path: Path, manifest: Dict[str, Dict[str, Any]]) -> None:
        """Save updated manifest registry dictionary."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
        except Exception as err:
            logger.error(f"Failed to write manifest registry {path}: {err}")
