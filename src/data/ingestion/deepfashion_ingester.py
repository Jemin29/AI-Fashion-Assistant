"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/deepfashion_ingester.py
=============================================================================
PURPOSE:
    Ingests the DeepFashion dataset stored as image folders + annotation files.

DATASET OVERVIEW:
    DeepFashion (Liu et al., CVPR 2016) is a large-scale fashion benchmark
    with 800,000+ annotated images across 50 fine-grained categories.

    Annotation Files (under Anno/):
        - list_category_img.txt  : image → category mapping
        - list_attr_img.txt      : image → 1000-dim binary attribute vector
        - list_bbox.txt          : bounding boxes [x1, y1, x2, y2]
        - list_landmarks.txt     : 6 clothing landmark keypoints
        - list_attr_cloth.txt    : attribute names (1000 entries)
        - list_category_cloth.txt: category names (50 entries)

    Expected folder layout:
        datasets/deepfashion/
            ├── img/               ← all images (JPEGs)
            ├── Anno/
            │   ├── list_category_img.txt
            │   ├── list_attr_img.txt
            │   ├── list_bbox.txt
            │   ├── list_landmarks.txt
            │   ├── list_attr_cloth.txt
            │   └── list_category_cloth.txt
            └── Eval/
                └── list_eval_partition.txt  ← train/val/test split

USAGE:
    >>> from src.data.ingestion import DeepFashionIngester
    >>> ingester = DeepFashionIngester(root_dir="datasets/deepfashion")
    >>> for record in ingester.stream(split="train", max_items=500):
    ...     print(record["image_id"], record["category"])
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Any, Tuple

import numpy as np
from loguru import logger

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required for DeepFashion ingestion. "
        "Install it with: pip install Pillow"
    ) from exc


# ─── Type aliases ─────────────────────────────────────────────────────────────
FashionRecord = Dict[str, Any]


class DeepFashionIngester:
    """
    Streams records from the DeepFashion dataset directory.

    Each yielded record contains:
        - image_id          : str  — relative image path used as ID
        - image_array       : np.ndarray  — shape (H, W, 3), dtype uint8
        - category          : str  — one of 50 fine-grained categories
        - category_id       : int  — integer category index (1-based)
        - attributes        : List[str]  — active attribute labels
        - bbox              : List[int]  — [x1, y1, x2, y2] in image coords
        - split             : str  — "train" | "val" | "test"
        - dataset_source    : str  — always "deepfashion"
    """

    # ── Annotation file names (relative to root_dir/Anno/) ────────────────────
    _CATEGORY_IMG_FILE   = "list_category_img.txt"
    _ATTR_IMG_FILE       = "list_attr_img.txt"
    _BBOX_FILE           = "list_bbox.txt"
    _ATTR_CLOTH_FILE     = "list_attr_cloth.txt"
    _CATEGORY_CLOTH_FILE = "list_category_cloth.txt"
    _EVAL_PARTITION_FILE = "Eval/list_eval_partition.txt"

    def __init__(self, root_dir: str | Path) -> None:
        """
        Initialise the ingester by loading all annotation indices into memory.

        All annotation files are parsed once at __init__ time and held as
        dictionaries so that per-record lookups during streaming are O(1).

        Args:
            root_dir : Path to the DeepFashion dataset root directory.
        """
        self.root_dir = Path(root_dir)
        self.img_dir  = self.root_dir / "img"
        self.anno_dir = self.root_dir / "Anno"

        # Loaded annotation maps (populated lazily if files exist)
        self._split_map: Dict[str, str] = {}          # img_path → split
        self._category_map: Dict[str, Tuple[str, int]] = {}  # img → (name, id)
        self._attr_map: Dict[str, List[str]] = {}     # img → [active attrs]
        self._bbox_map: Dict[str, List[int]] = {}     # img → [x1,y1,x2,y2]
        self._attr_names: List[str] = []              # 1000 attribute names
        self._category_names: List[str] = []          # 50 category names

        if not self.root_dir.exists():
            logger.warning(
                f"DeepFashion root not found: {self.root_dir}\n"
                "Download the dataset from:\n"
                "  https://liuziwei7.github.io/projects/DeepFashion.html\n"
                "Or via Kaggle:\n"
                "  kaggle datasets download -d nguyngiabol/colorful-fashion-dataset-for-object-detection"
            )
        else:
            logger.info(f"DeepFashionIngester initialised → {self.root_dir}")
            self._load_annotations()

    # ── Public API ─────────────────────────────────────────────────────────────

    def stream(
        self,
        split: str = "train",
        max_items: Optional[int] = None,
    ) -> Generator[FashionRecord, None, None]:
        """
        Lazily yield FashionRecord dicts for a given dataset split.

        Images are loaded from disk one at a time (Pillow → numpy). If an
        image file is missing or corrupt, the record is skipped with a warning.

        Args:
            split     : Dataset partition: "train" | "val" | "test".
            max_items : Maximum records to yield (None = unlimited).

        Yields:
            FashionRecord dict (see class docstring).

        Raises:
            RuntimeError : If annotations were not loaded (root_dir missing).
        """
        if not self._split_map:
            raise RuntimeError(
                "Annotation maps are empty. "
                "Check that root_dir exists and contains Anno/ files."
            )

        # Filter image paths belonging to the requested split
        split_images = [
            img_path
            for img_path, s in self._split_map.items()
            if s == split
        ]

        if max_items is not None:
            split_images = split_images[:max_items]

        logger.info(
            f"Starting DeepFashion stream | split={split} "
            f"| total in split={len(split_images)}"
        )

        yielded = 0
        for img_rel_path in split_images:
            try:
                record = self._build_record(img_rel_path, split)
                if record is not None:
                    yield record
                    yielded += 1
            except Exception as exc:
                logger.warning(f"Skipping {img_rel_path}: {exc}")
                continue

        logger.success(
            f"DeepFashion stream complete | yielded {yielded} records "
            f"| split={split}"
        )

    def get_split_sizes(self) -> Dict[str, int]:
        """Return the number of images per split."""
        sizes: Dict[str, int] = {}
        for split in ("train", "val", "test"):
            sizes[split] = sum(
                1 for s in self._split_map.values() if s == split
            )
        return sizes

    def get_category_distribution(self) -> Dict[str, int]:
        """Return a count of images per category (across all splits)."""
        dist: Dict[str, int] = {}
        for img_path, (cat_name, _) in self._category_map.items():
            dist[cat_name] = dist.get(cat_name, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    # ── Private: annotation loading ────────────────────────────────────────────

    def _load_annotations(self) -> None:
        """Parse all annotation files into in-memory dictionaries."""
        logger.debug("Loading DeepFashion annotations…")
        self._load_attribute_names()
        self._load_category_names()
        self._load_eval_partition()
        self._load_category_map()
        self._load_attribute_map()
        self._load_bbox_map()
        logger.info(
            f"Annotations loaded | "
            f"images={len(self._split_map)} | "
            f"categories={len(self._category_names)} | "
            f"attributes={len(self._attr_names)}"
        )

    def _load_eval_partition(self) -> None:
        """Parse Eval/list_eval_partition.txt → {img_path: split}."""
        fpath = self.root_dir / self._EVAL_PARTITION_FILE
        if not fpath.exists():
            logger.warning(f"Eval partition file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: total count; Line 1: header; Lines 2+: data
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]   # e.g., "img/Shawls/img_00000001.jpg"
                split_label = parts[1].lower()  # "train" | "val" | "test"
                self._split_map[img_path] = split_label

    def _load_category_names(self) -> None:
        """Parse Anno/list_category_cloth.txt → list of category name strings."""
        fpath = self.anno_dir / self._CATEGORY_CLOTH_FILE
        if not fpath.exists():
            logger.warning(f"Category cloth file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: count; Line 1: header; Lines 2+: "CategoryName  type_id"
        self._category_names = [
            line.strip().split()[0] for line in lines[2:] if line.strip()
        ]

    def _load_attribute_names(self) -> None:
        """Parse Anno/list_attr_cloth.txt → list of 1000 attribute name strings."""
        fpath = self.anno_dir / self._ATTR_CLOTH_FILE
        if not fpath.exists():
            logger.warning(f"Attribute cloth file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: count; Line 1: header; Lines 2+: "attribute_name  type_id"
        self._attr_names = [
            line.strip().split()[0] for line in lines[2:] if line.strip()
        ]

    def _load_category_map(self) -> None:
        """Parse Anno/list_category_img.txt → {img_path: (cat_name, cat_id)}."""
        fpath = self.anno_dir / self._CATEGORY_IMG_FILE
        if not fpath.exists():
            logger.warning(f"Category image file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: count; Line 1: header; Lines 2+: "img_path  category_id"
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                cat_id   = int(parts[1])  # 1-based index
                cat_name = (
                    self._category_names[cat_id - 1]
                    if 0 < cat_id <= len(self._category_names)
                    else "unknown"
                )
                self._category_map[img_path] = (cat_name, cat_id)

    def _load_attribute_map(self) -> None:
        """Parse Anno/list_attr_img.txt → {img_path: [active_attr_names]}."""
        fpath = self.anno_dir / self._ATTR_IMG_FILE
        if not fpath.exists():
            logger.warning(f"Attribute image file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: total images; Line 1: total attributes; Line 2: header
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path  = parts[0]
            # Remaining values are 1 (present) or -1 (absent) per attribute
            attr_vals = [int(v) for v in parts[1:]]
            active_attrs = [
                self._attr_names[i]
                for i, v in enumerate(attr_vals)
                if v == 1 and i < len(self._attr_names)
            ]
            self._attr_map[img_path] = active_attrs

    def _load_bbox_map(self) -> None:
        """Parse Anno/list_bbox.txt → {img_path: [x1, y1, x2, y2]}."""
        fpath = self.anno_dir / self._BBOX_FILE
        if not fpath.exists():
            logger.warning(f"Bbox file not found: {fpath}")
            return
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Line 0: count; Line 1: header; Lines 2+: "img_path  x1 y1 x2 y2"
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 5:
                img_path = parts[0]
                bbox = [int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4])]
                self._bbox_map[img_path] = bbox

    # ── Private: record construction ───────────────────────────────────────────

    def _build_record(
        self, img_rel_path: str, split: str
    ) -> Optional[FashionRecord]:
        """
        Load one image from disk and assemble its full metadata record.

        Args:
            img_rel_path : Relative image path as stored in annotations.
            split        : Dataset split label.

        Returns:
            FashionRecord dict, or None if the image cannot be loaded.
        """
        img_abs_path = self.root_dir / img_rel_path

        # Attempt to load image with Pillow
        try:
            with Image.open(img_abs_path) as img:
                img_rgb = img.convert("RGB")
                image_array = np.array(img_rgb, dtype=np.uint8)
        except Exception as exc:
            logger.debug(f"Cannot open image {img_abs_path}: {exc}")
            return None

        # Look up metadata from pre-loaded annotation maps
        cat_name, cat_id = self._category_map.get(img_rel_path, ("unknown", -1))
        attributes       = self._attr_map.get(img_rel_path, [])
        bbox             = self._bbox_map.get(img_rel_path, [])

        return {
            "image_id"      : img_rel_path.replace("/", "_").replace("\\", "_"),
            "image_array"   : image_array,
            "category"      : cat_name,
            "category_id"   : cat_id,
            "attributes"    : attributes,
            "bbox"          : bbox,
            "split"         : split,
            "dataset_source": "deepfashion",
            "original_path" : str(img_abs_path),
        }
