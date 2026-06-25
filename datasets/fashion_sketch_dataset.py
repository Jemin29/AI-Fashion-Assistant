"""
datasets/fashion_sketch_dataset.py
==================================
ControlNet training dataset pipeline.
Loads paired fashion design images and sketch outline conditioning maps.
Supports augmentations, train/val splits, and on-the-fly edge extraction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

try:
    from week3.preprocessors.sketch_processor import SketchProcessor
    _HAS_SKETCH_PROCESSOR = True
except ImportError:
    SketchProcessor = None
    _HAS_SKETCH_PROCESSOR = False


class FashionSketchDataset(Dataset):
    """
    PyTorch Dataset for paired fashion designs and sketch condition images.
    Suitable for training and evaluating SDXL ControlNet models.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path, List[Dict[str, Any]], None] = None,
        design_dir: Union[str, Path, None] = None,
        sketch_dir: Union[str, Path, None] = None,
        split: str = "train",
        split_ratio: float = 0.8,
        target_size: Tuple[int, int] = (1024, 1024),
        augment: bool = False,
        edge_method: str = "canny",
        seed: int = 42,
    ) -> None:
        """
        Initialize the FashionSketchDataset.

        Parameters
        ----------
        manifest_path : Path or list of dicts, optional
            Path to final_fashion_dataset.json, or pre-loaded records list.
            If None, scans design_dir directly for images.
        design_dir : Path, optional
            Base directory containing design/garment images.
        sketch_dir : Path, optional
            Base directory containing pre-extracted sketch images.
            If None, sketches will be generated on-the-fly.
        split : str
            Dataset split name: "train" | "val" | "validation".
        split_ratio : float
            Fraction of data for the training split (default: 0.8).
        target_size : Tuple[int, int]
            Resize dimensions (default: 1024x1024).
        augment : bool
            Whether to apply random augmentations (active in train split only).
        edge_method : str
            Method for on-the-fly sketch extraction: "canny" | "hed" | "lineart".
        seed : int
            Deterministic seed for splitting and shuffling.
        """
        super().__init__()
        self.split = split.lower()
        if self.split not in ("train", "val", "validation"):
            raise ValueError(f"Invalid split name '{split}'. Choose from 'train', 'val', or 'validation'.")

        self.design_dir = Path(design_dir).resolve() if design_dir else None
        self.sketch_dir = Path(sketch_dir).resolve() if sketch_dir else None
        self.target_size = target_size
        self.augment = augment and (self.split == "train")
        self.edge_method = edge_method.lower()

        # Try to resolve project root dynamically
        self.project_root = Path(__file__).resolve().parent.parent

        # Initialize edge extractor if on-the-fly generation is required
        self.sketch_processor = None
        if _HAS_SKETCH_PROCESSOR:
            self.sketch_processor = SketchProcessor()
        else:
            logger.warning("SketchProcessor import failed. Grayscale edge detection fallbacks will be used.")

        # Load raw records
        raw_records = self._load_raw_records(manifest_path)

        # Apply deterministic split
        self.records = self._split_dataset(raw_records, split_ratio, seed)
        logger.info(
            f"Initialized FashionSketchDataset | split={self.split} | size={len(self.records)} "
            f"| target_size={self.target_size} | augment={self.augment} | edge_method={self.edge_method}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._get_item_with_retry(idx, retry_count=0)

    # ── Internal Processing Methods ───────────────────────────────────────────

    def _load_raw_records(
        self,
        manifest_path: Union[str, Path, List[Dict[str, Any]], None]
    ) -> List[Dict[str, Any]]:
        """Load manifest list or scan directory directly if manifest is absent."""
        # Case 1: Pre-loaded list of records
        if isinstance(manifest_path, list):
            logger.debug(f"Loaded {len(manifest_path)} records directly from memory list.")
            return manifest_path

        # Case 2: Load manifest path from file
        if manifest_path is not None:
            m_path = Path(manifest_path).resolve()
            if not m_path.exists():
                raise FileNotFoundError(f"Manifest file not found at: {m_path}")
            
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # final_fashion_dataset.json format matches { "_meta": {...}, "records": [...] }
                if isinstance(data, dict) and "records" in data:
                    records = data["records"]
                elif isinstance(data, list):
                    records = data
                else:
                    raise ValueError("Manifest format unrecognized. Must be list of records or contain 'records' key.")
                
                logger.debug(f"Loaded {len(records)} records from manifest JSON file: {m_path}")
                return records
            except Exception as exc:
                logger.error(f"Failed to read manifest file: {exc}")
                raise

        # Case 3: Scan directories directly if no manifest is provided
        if self.design_dir is None:
            raise ValueError("Must specify either manifest_path or design_dir to load records.")

        if not self.design_dir.exists():
            raise DirectoryNotFoundError(f"Design directory does not exist: {self.design_dir}")

        logger.info(f"No manifest provided. Scanning design_dir for images: {self.design_dir}")
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        records = []
        
        # Scan and pair files
        for item_path in self.design_dir.rglob("*"):
            if item_path.is_file() and item_path.suffix.lower() in valid_exts:
                # Store relative paths to resolve later
                rel_path = item_path.relative_to(self.design_dir)
                image_id = item_path.stem
                
                # Check for corresponding sketch
                sketch_rel_path = None
                if self.sketch_dir:
                    # Match filename variations
                    for sketch_stem in (image_id, f"{image_id}_sketch", f"{image_id}_preprocessed"):
                        for ext in valid_exts:
                            test_path = self.sketch_dir / rel_path.parent / f"{sketch_stem}{ext}"
                            if test_path.exists():
                                sketch_rel_path = test_path.relative_to(self.sketch_dir)
                                break
                        if sketch_rel_path:
                            break

                records.append({
                    "image_id": image_id,
                    "image_path": str(rel_path),
                    "sketch_path": str(sketch_rel_path) if sketch_rel_path else None,
                    "description": f"A fashion garment design showing {image_id.replace('_', ' ')}."
                })
        
        logger.info(f"Discovered {len(records)} image pairs/records from directory scan.")
        return records

    def _split_dataset(
        self,
        raw_records: List[Dict[str, Any]],
        split_ratio: float,
        seed: int
    ) -> List[Dict[str, Any]]:
        """Filters by pre-defined splits or splits list deterministically."""
        # 1. Filter if records contain explicit split metadata
        split_keys = ("split", "dataset_split", "source_split")
        explicit_split_records = []
        for r in raw_records:
            record_split = None
            for key in split_keys:
                if key in r and isinstance(r[key], str):
                    record_split = r[key].lower()
                    break
            
            # Match split name
            if record_split is not None:
                is_match = (
                    (self.split == "train" and record_split in ("train", "training")) or
                    (self.split in ("val", "validation") and record_split in ("val", "validation", "test", "testing"))
                )
                if is_match:
                    explicit_split_records.append(r)

        if explicit_split_records:
            logger.debug(f"Loaded {len(explicit_split_records)} records matching split '{self.split}' from explicit metadata.")
            return explicit_split_records

        # 2. Otherwise, split list deterministically
        sorted_records = sorted(
            raw_records,
            key=lambda x: (x.get("image_id") or "", x.get("image_path") or "")
        )
        
        rng = random.Random(seed)
        rng.shuffle(sorted_records)

        split_idx = int(len(sorted_records) * split_ratio)
        if self.split == "train":
            return sorted_records[:split_idx]
        else:
            return sorted_records[split_idx:]

    def _resolve_image_path(self, path_str: str, base_dir: Optional[Path]) -> Path:
        """Resolve image path checking relative to base, project root, or absolute."""
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        
        # 1. Check relative to base directory (direct join)
        if base_dir:
            resolved = base_dir / p
            if resolved.exists():
                return resolved

            # 2. Check direct filename under base directory (e.g. p is designs/img.jpg and base_dir is /path/to/designs)
            resolved = base_dir / p.name
            if resolved.exists():
                return resolved

            # 3. Check subpath overlap (e.g. base_dir ends with first component of p)
            if len(p.parts) > 1:
                try:
                    subpath = Path(*p.parts[1:])
                    resolved = base_dir / subpath
                    if resolved.exists():
                        return resolved
                except Exception:
                    pass

        # 4. Check relative to project root (direct join)
        resolved = self.project_root / p
        if resolved.exists():
            return resolved

        # 5. Check direct filename under project root
        resolved = self.project_root / p.name
        if resolved.exists():
            return resolved

        # 6. Try relative to current working directory
        resolved = Path.cwd() / p
        if resolved.exists():
            return resolved

        raise FileNotFoundError(f"Could not resolve image path: {path_str}")

    def _get_item_with_retry(self, idx: int, retry_count: int = 0) -> Dict[str, Any]:
        """Loads and processes paired items. Retries on load failures to prevent crashes."""
        if retry_count > 15:
            raise RuntimeError(f"Failed to load a valid dataset item after 15 retries. Last index tried: {idx}")

        try:
            record = self.records[idx]
            
            # Resolve Design image
            design_path = self._resolve_image_path(record["image_path"], self.design_dir)
            design_img = Image.open(design_path).convert("RGB")
            
            # Resolve or Extract Sketch image
            sketch_img = None
            sketch_path_str = record.get("sketch_path")
            
            if sketch_path_str:
                try:
                    sketch_path = self._resolve_image_path(sketch_path_str, self.sketch_dir)
                    sketch_img = Image.open(sketch_path).convert("RGB")
                except Exception as err:
                    logger.debug(f"Failed to load pre-extracted sketch from {sketch_path_str}: {err}. Falling back to extraction.")

            if sketch_img is None:
                # Extract dynamically
                if self.sketch_processor:
                    sketch_img = self.sketch_processor.preprocess_sketch(design_img, method=self.edge_method)
                else:
                    # Basic PIL fallback for edge detection
                    gray = design_img.convert("L")
                    from PIL import ImageFilter, ImageOps
                    edges = gray.filter(ImageFilter.FIND_EDGES)
                    # Standardize: invert if sketch is drawn on white, but ControlNet expects white lines on black
                    # Usually, FIND_EDGES creates white lines on black background, so we just auto-contrast it.
                    normalized = ImageOps.autocontrast(edges, cutoff=2)
                    sketch_img = normalized.convert("RGB")

            # Match dimensions
            if sketch_img.size != design_img.size:
                sketch_img = sketch_img.resize(design_img.size, Image.Resampling.LANCZOS)

            # Apply Paired Augmentation (geometric changes must align perfectly)
            design_img, sketch_img = self._apply_augmentations(design_img, sketch_img)

            # Preprocessing resize to final target size
            design_img = F.resize(design_img, self.target_size, Image.Resampling.LANCZOS)
            sketch_img = F.resize(sketch_img, self.target_size, Image.Resampling.LANCZOS)

            # Normalize values
            # Design image normalized to [-1, 1]
            design_tensor = F.to_tensor(design_img)
            design_tensor = (design_tensor * 2.0) - 1.0

            # Sketch image normalized to [0, 1]
            sketch_tensor = F.to_tensor(sketch_img)

            # Build metadata caption
            prompt = record.get("description") or record.get("prompt") or ""
            if not prompt:
                # Build simple fallback prompt from metadata properties
                color_str = ", ".join(record.get("color", []))
                style_str = record.get("style", "")
                cat_str = record.get("category", "garment").replace("_", " ")
                
                parts = []
                if style_str:
                    parts.append(style_str)
                if color_str:
                    parts.append(color_str)
                parts.append(cat_str)
                prompt = f"A photo of a {', '.join(parts)}"

            # Exclude images from returned metadata dictionary to keep it serializable
            clean_meta = {k: v for k, v in record.items() if k not in ("image_path", "sketch_path")}

            return {
                "pixel_values": design_tensor,
                "conditioning_pixel_values": sketch_tensor,
                "prompt": prompt,
                "metadata": clean_meta
            }

        except Exception as exc:
            logger.warning(f"Error loading record index {idx}: {exc}. Fetching random alternative.")
            alternative_idx = random.randint(0, len(self.records) - 1)
            return self._get_item_with_retry(alternative_idx, retry_count + 1)

    def _apply_augmentations(self, design: Image.Image, sketch: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Applies matched geometric transforms to both, and color transforms only to design."""
        if not self.augment:
            return design, sketch

        # 1. Random Horizontal Flip
        if random.random() < 0.5:
            design = F.hflip(design)
            sketch = F.hflip(sketch)

        # 2. Random Rotation (-15 to 15 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-15.0, 15.0)
            design = F.rotate(design, angle, fill=255) # Pad design with white background
            sketch = F.rotate(sketch, angle, fill=0)   # Pad sketch with black background

        # 3. Random Crop & Scale
        if random.random() < 0.4:
            w, h = design.size
            scale = random.uniform(0.8, 0.95)
            new_w, new_h = int(w * scale), int(h * scale)
            
            max_x = w - new_w
            max_y = h - new_h
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0

            design = F.crop(design, y, x, new_h, new_w)
            sketch = F.crop(sketch, y, x, new_h, new_w)

        # 4. Color Jitter (only applied to the target design image)
        if random.random() < 0.5:
            design = F.adjust_brightness(design, random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            design = F.adjust_contrast(design, random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            design = F.adjust_saturation(design, random.uniform(0.85, 1.15))

        return design, sketch
