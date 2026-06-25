"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/deepfashion_loader.py
=============================================================================
MODULE : DeepFashion Dataset Ingestion Pipeline
WEEK   : 1 — Fashion Domain Research & Dataset Curation
AUTHOR : Fashion AI Team

PURPOSE
-------
A production-grade, clean-architecture ingestion pipeline for the
DeepFashion dataset (Liu et al., CVPR 2016) that:

  1. Discovers and validates all six DeepFashion annotation files.
  2. Parses raw TXT annotations (category, attributes, landmarks, bbox, split).
  3. Extracts per-record fields: image paths, category labels, 1000-dim
     attribute vectors, 6-point clothing landmarks, bounding boxes.
  4. Transforms raw data into canonical DeepFashionRecord objects.
  5. Normalises: 50 raw category strings → 11 taxonomy keys, landmark
     coordinates → normalised [0,1] space, attribute vectors → named lists.
  6. Validates every record against taxonomy + integrity rules.
  7. Saves processed records to datasets/processed/deepfashion_processed.json.

ARCHITECTURE  (Clean / Layered — mirrors FashionGen loader)
--------------------------------------------------------------
  ┌───────────────────────────────────────────────┐
  │ DeepFashionLoader       (Orchestrator)         │  ← Public API
  ├───────────────────────────────────────────────┤
  │ DeepFashionAnnotationParser  (Parse)           │  ← TXT → index dicts
  ├───────────────────────────────────────────────┤
  │ DeepFashionExtractor    (Extract)              │  ← index → raw records
  ├───────────────────────────────────────────────┤
  │ DeepFashionTransformer  (Transform)            │  ← raw → canonical
  ├───────────────────────────────────────────────┤
  │ DeepFashionValidator    (Validate)             │  ← schema + taxonomy
  ├───────────────────────────────────────────────┤
  │ DeepFashionWriter       (Save)                 │  ← JSON output
  ├───────────────────────────────────────────────┤
  │ DFPipelineStats         (Metrics)              │  ← counters & timing
  └───────────────────────────────────────────────┘

DEEPFASHION ANNOTATION FORMAT
------------------------------
Dataset root (datasets/deepfashion/):
  img/           — all images (nested by category)
  Anno/
    list_category_cloth.txt   — 50 category names + type ids
    list_category_img.txt     — image → category id mapping
    list_attr_cloth.txt       — 1000 attribute names + type ids
    list_attr_img.txt         — image → 1000-bit attribute vector
    list_bbox.txt             — image → [x1, y1, x2, y2]
    list_landmarks.txt        — image → 6 keypoints (x, y, visibility)
  Eval/
    list_eval_partition.txt   — image → split (train/val/test)

LANDMARK FORMAT (6 keypoints)
------------------------------
Each landmark row after the header contains:
  img_path  <lm_x1> <lm_y1> <lm_vis1>  <lm_x2> <lm_y2> <lm_vis2>  ... × 6
  visibility: 0 = hidden, 1 = visible

ATTRIBUTE FORMAT (1000 per image)
----------------------------------
Each attribute value: +1 = present, -1 = absent

OUTPUT SCHEMA (per record)
--------------------------
  {
    "image_id"   : "DF_img_Shawls_img_00000001",
    "category"   : "accessories",
    "attributes" : ["floral", "sleeveless", ...],
    "landmarks"  : [
      {"name": "left_collar",  "x": 0.42, "y": 0.18, "visible": true},
      ...
    ],
    "image_path" : "datasets/deepfashion/img/Shawls/img_00000001.jpg"
  }

USAGE
-----
  # Python API
  from src.data.ingestion.deepfashion_loader import DeepFashionLoader
  loader = DeepFashionLoader()
  result = loader.run(split="train", max_records=500)
  print(result["stats"]["total_valid"])
  print(result["output_path"])

  # CLI
  python data_pipeline/ingestion/deepfashion_loader.py --info
  python data_pipeline/ingestion/deepfashion_loader.py --split train --max-records 1000
  python data_pipeline/ingestion/deepfashion_loader.py --split all
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
from loguru import logger

# Optional tqdm — degrades gracefully to a plain iterator
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ─── Internal: resolve project root regardless of cwd ─────────────────────────
_FILE_DIR     = Path(__file__).resolve().parent       # data_pipeline/ingestion/
_PROJECT_ROOT = _FILE_DIR.parent.parent               # fashion-ai-assistant/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Knowledge-base (optional — normalisation degrades gracefully without it)
try:
    from src.data.knowledge_base.fashion_domain_research import (
        FashionDomainResearch,
        normalize_color,
        normalize_fabric,
        normalize_style,
        normalize_occasion,
    )
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    logger.debug("KB unavailable — normalisation operates in standalone mode.")


# =============================================================================
# ── 1. Constants & Default Paths
# =============================================================================

# Default paths (all overridable via constructor)
_DEFAULT_ROOT_DIR  = _PROJECT_ROOT / "datasets" / "deepfashion"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "processed"
_OUTPUT_FILENAME   = "deepfashion_processed.json"

# Annotation file names (relative to root_dir)
_ANNO_FILES = {
    "category_cloth"  : "Anno/list_category_cloth.txt",
    "category_img"    : "Anno/list_category_img.txt",
    "attr_cloth"      : "Anno/list_attr_cloth.txt",
    "attr_img"        : "Anno/list_attr_img.txt",
    "bbox"            : "Anno/list_bbox.txt",
    "landmarks"       : "Anno/list_landmarks.txt",
    "eval_partition"  : "Eval/list_eval_partition.txt",
}

# DeepFashion 50 raw category names → our 11 taxonomy keys.
# Derived from the official DeepFashion category list.
_DF_CATEGORY_MAP: Dict[str, str] = {
    # ── Upper body ────────────────────────────────────────────────────────────
    "Blouse"             : "shirts",
    "Button-Down"        : "shirts",
    "Dress Shirt"        : "shirts",
    "Shirt"              : "shirts",
    "Tank"               : "t_shirts",
    "Tee"                : "t_shirts",
    "Top"                : "t_shirts",
    "Sweater"            : "hoodies",
    "Cardigan"           : "hoodies",
    "Sweatshirt"         : "hoodies",
    "Hoodie"             : "hoodies",
    "Jacket"             : "jackets",
    "Blazer"             : "jackets",
    "Bomber"             : "jackets",
    "Parka"              : "jackets",
    "Peacoat"            : "jackets",
    "Windbreaker"        : "jackets",
    "Vest"               : "jackets",
    "Poncho"             : "jackets",
    "Anorak"             : "jackets",
    # ── Lower body ────────────────────────────────────────────────────────────
    "Chinos"             : "pants",
    "Culottes"           : "pants",
    "Jodhpurs"           : "pants",
    "Joggers"            : "pants",
    "Leggings"           : "pants",
    "Palazzo"            : "pants",
    "Jeans"              : "jeans",
    "Shorts"             : "shorts",
    "Cutoffs"            : "shorts",
    "Cargo Shorts"       : "shorts",
    # ── Full body ────────────────────────────────────────────────────────────
    "Caftan"             : "dresses",
    "Dress"              : "dresses",
    "Jumpsuit"           : "dresses",
    "Romper"             : "dresses",
    "Skirt"              : "dresses",
    # ── Accessories / Outerwear ────────────────────────────────────────────────
    "Shawl"              : "accessories",
    "Cape"               : "jackets",
    "Kimono"             : "ethnic_wear",
    # ── Generic fallbacks ─────────────────────────────────────────────────────
    # (These appear in some DeepFashion variant splits)
    "Tshirt"             : "t_shirts",
    "Coat"               : "jackets",
    "Trousers"           : "pants",
    "Pants"              : "pants",
    "Denim"              : "jeans",
}

# Six standard DeepFashion landmark names (in order of annotation columns)
_LANDMARK_NAMES: List[str] = [
    "left_collar",
    "right_collar",
    "left_sleeve",
    "right_sleeve",
    "left_hem",
    "right_hem",
]

# Valid taxonomy constants (mirrors knowledge_base module)
_VALID_CATEGORIES = frozenset({
    "t_shirts", "shirts", "hoodies", "jackets", "pants",
    "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories",
})
_VALID_SPLITS = frozenset({"train", "val", "test", "all"})


# =============================================================================
# ── 2. Data Models
# =============================================================================

@dataclass
class RawDeepFashionRecord:
    """
    Raw record assembled directly from annotation file lookups.

    All coordinate values are in pixel space (integers). This model lives
    only in the Extraction layer and is never written to disk.
    """
    image_rel_path   : str           # e.g. "img/Shawls/img_00000001.jpg"
    category_raw     : str           # Raw category name from list_category_cloth
    category_id      : int           # 1-based category index
    attr_vector      : List[int]     # 1000-dim: +1 present, -1 absent, 0 missing
    attr_names       : List[str]     # Parallel list of 1000 attribute name strings
    landmarks_raw    : List[Dict[str, Any]]  # [{"name":…,"x":int,"y":int,"vis":int}]
    bbox             : List[int]     # [x1, y1, x2, y2] pixel coords
    split            : str           # "train" | "val" | "test"
    image_width      : int = 0       # From image metadata (0 if unread)
    image_height     : int = 0       # From image metadata (0 if unread)


@dataclass
class DeepFashionRecord:
    """
    Canonical, taxonomy-aligned DeepFashion record ready for persistence.

    All raw pixel coordinates have been normalised to [0,1].
    Attribute vectors have been decoded to named label lists.
    JSON-serialisable via to_dict().
    """
    # ── Spec-required fields ──────────────────────────────────────────────────
    image_id         : str                      # e.g. "DF_img_Shawls_img_00000001"
    category         : str                      # taxonomy key e.g. "accessories"
    attributes       : List[str]                # active attribute name strings
    landmarks        : List[Dict[str, Any]]     # normalised landmark dicts
    image_path       : str                      # relative path from project root

    # ── Extended provenance fields ────────────────────────────────────────────
    dataset_source   : str  = "deepfashion"
    category_raw     : str  = ""               # original category name
    category_id      : int  = -1               # 1-based raw category index
    split            : str  = "train"
    bbox             : List[int] = field(default_factory=list)
    bbox_normalised  : List[float] = field(default_factory=list)
    attr_count       : int  = 0                # number of active attributes
    is_valid         : bool = False
    errors           : List[str] = field(default_factory=list)
    warnings         : List[str] = field(default_factory=list)
    processed_at     : str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain, JSON-safe dict matching the output spec exactly.

        Spec output format:
          {
            "image_id"   : "DF_…",
            "category"   : "accessories",
            "attributes" : ["floral", …],
            "landmarks"  : [{"name": "left_collar", "x": 0.42, …}, …],
            "image_path" : "datasets/deepfashion/img/…"
          }
        """
        return {
            # ── Required spec fields ──────────────────────────────────────────
            "image_id"        : self.image_id,
            "category"        : self.category,
            "attributes"      : self.attributes,
            "landmarks"       : self.landmarks,
            "image_path"      : self.image_path,
            # ── Provenance / extended fields ──────────────────────────────────
            "dataset_source"  : self.dataset_source,
            "category_raw"    : self.category_raw,
            "category_id"     : self.category_id,
            "split"           : self.split,
            "bbox"            : self.bbox,
            "bbox_normalised" : self.bbox_normalised,
            "attr_count"      : self.attr_count,
            "is_valid"        : self.is_valid,
            "errors"          : self.errors,
            "warnings"        : self.warnings,
            "processed_at"    : self.processed_at,
        }


@dataclass
class DFPipelineStats:
    """
    Live counters and timing metrics for the DeepFashion pipeline run.

    Mirrors PipelineStats from fashiongen_loader.py but is independent
    so each pipeline owns its own metric model.
    """
    total_read        : int = 0
    total_processed   : int = 0
    total_valid       : int = 0
    total_invalid     : int = 0
    total_skipped     : int = 0
    total_missing_img : int = 0   # Image file not found on disk
    total_saved       : int = 0

    category_counts   : Dict[str, int] = field(default_factory=dict)
    split_counts      : Dict[str, int] = field(default_factory=dict)
    landmark_coverage : Dict[str, int] = field(default_factory=dict)  # vis counts

    start_time        : float = field(default_factory=time.time)
    end_time          : float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)

    @property
    def records_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return round(self.total_processed / elapsed, 1) if elapsed > 0 else 0.0

    @property
    def valid_rate(self) -> float:
        return (
            round(self.total_valid / self.total_processed, 4)
            if self.total_processed > 0 else 0.0
        )

    def increment_category(self, cat: str) -> None:
        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1

    def increment_split(self, split: str) -> None:
        self.split_counts[split] = self.split_counts.get(split, 0) + 1

    def finalize(self) -> None:
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_read"        : self.total_read,
            "total_processed"   : self.total_processed,
            "total_valid"       : self.total_valid,
            "total_invalid"     : self.total_invalid,
            "total_skipped"     : self.total_skipped,
            "total_missing_img" : self.total_missing_img,
            "total_saved"       : self.total_saved,
            "valid_rate"        : self.valid_rate,
            "records_per_second": self.records_per_second,
            "elapsed_seconds"   : self.elapsed_seconds,
            "category_counts"   : self.category_counts,
            "split_counts"      : self.split_counts,
            "landmark_coverage" : self.landmark_coverage,
        }

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("DEEPFASHION PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total read        : {self.total_read:,}")
        logger.info(f"  Total processed   : {self.total_processed:,}")
        logger.info(f"  Total valid       : {self.total_valid:,}")
        logger.info(f"  Total invalid     : {self.total_invalid:,}")
        logger.info(f"  Total skipped     : {self.total_skipped:,}")
        logger.info(f"  Missing images    : {self.total_missing_img:,}")
        logger.info(f"  Total saved       : {self.total_saved:,}")
        logger.info(f"  Valid rate        : {self.valid_rate:.1%}")
        logger.info(f"  Throughput        : {self.records_per_second} rec/s")
        logger.info(f"  Elapsed           : {self.elapsed_seconds}s")
        logger.info(f"  Category counts   : {self.category_counts}")
        logger.info(f"  Split counts      : {self.split_counts}")
        logger.info("=" * 60)


# =============================================================================
# ── 3. DeepFashionAnnotationParser — Annotation Parsing Layer
# =============================================================================

class DeepFashionAnnotationParser:
    """
    Parsing Layer: reads all six DeepFashion TXT annotation files into
    fast in-memory lookup dictionaries.

    This layer owns all file I/O and TXT parsing logic. It is called once
    at pipeline start and the resulting indexes are passed to the Extractor.

    Annotation files parsed:
      • list_category_cloth.txt  → category_names[]: list of 50 names
      • list_category_img.txt    → category_map: {img_path → (name, id)}
      • list_attr_cloth.txt      → attr_names[]: list of 1000 names
      • list_attr_img.txt        → attr_map: {img_path → [±1 ints]}
      • list_bbox.txt            → bbox_map: {img_path → [x1,y1,x2,y2]}
      • list_landmarks.txt       → landmark_map: {img_path → [lm dicts]}
      • list_eval_partition.txt  → split_map: {img_path → "train"|"val"|"test"}

    Error handling:
      • Missing files → warning logged, empty index used (pipeline continues).
      • Malformed lines → skipped with debug log.
      • Truncated attribute vector → padded with 0s.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """
        Initialise and immediately parse all annotation files.

        Args:
            root_dir : Path to the DeepFashion dataset root directory.
        """
        self.root_dir = Path(root_dir)

        # ── Parsed indexes ────────────────────────────────────────────────────
        self.category_names : List[str]                      = []
        self.attr_names     : List[str]                      = []
        self.split_map      : Dict[str, str]                 = {}
        self.category_map   : Dict[str, Tuple[str, int]]     = {}
        self.attr_map       : Dict[str, List[int]]           = {}
        self.bbox_map       : Dict[str, List[int]]           = {}
        self.landmark_map   : Dict[str, List[Dict[str, Any]]]= {}

        # ── Parse all files ───────────────────────────────────────────────────
        self._parse_all()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_all_image_paths(self) -> List[str]:
        """
        Return all image paths that appear in the split partition file.

        Returns:
            Sorted list of image paths (relative to root_dir).
        """
        return sorted(self.split_map.keys())

    def get_image_paths_for_split(self, split: str) -> List[str]:
        """
        Return image paths for a specific split.

        Args:
            split : "train" | "val" | "test" | "all"

        Returns:
            List of image paths (relative to root_dir).
        """
        if split == "all":
            return self.get_all_image_paths()
        return [p for p, s in self.split_map.items() if s == split]

    def is_available(self) -> bool:
        """Return True if at least the split map was successfully populated."""
        return bool(self.split_map)

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of what was parsed."""
        return {
            "root_dir"             : str(self.root_dir),
            "available"            : self.is_available(),
            "total_images"         : len(self.split_map),
            "total_category_names" : len(self.category_names),
            "total_attr_names"     : len(self.attr_names),
            "total_category_map"   : len(self.category_map),
            "total_attr_map"       : len(self.attr_map),
            "total_bbox_map"       : len(self.bbox_map),
            "total_landmark_map"   : len(self.landmark_map),
            "split_distribution"   : {
                split: sum(1 for s in self.split_map.values() if s == split)
                for split in ("train", "val", "test")
            },
        }

    # ── Private: file parsers ──────────────────────────────────────────────────

    def _parse_all(self) -> None:
        """
        Parse all six annotation files in dependency order.

        category_cloth must come before category_img (need names to resolve ids).
        attr_cloth must come before attr_img (need names for vector decode).
        All others are independent.
        """
        if not self.root_dir.exists():
            logger.warning(
                f"DeepFashion root not found: {self.root_dir}\n"
                f"Download dataset from: https://liuziwei7.github.io/projects/DeepFashion.html"
            )
            return

        logger.info(f"DeepFashionAnnotationParser: parsing annotations in {self.root_dir}")

        # Dependency order matters for category and attr
        self._parse_category_cloth()
        self._parse_attr_cloth()
        self._parse_eval_partition()
        self._parse_category_img()
        self._parse_attr_img()
        self._parse_bbox()
        self._parse_landmarks()

        logger.success(
            f"Annotations parsed | images={len(self.split_map):,} | "
            f"categories={len(self.category_names)} | "
            f"attrs={len(self.attr_names)} | "
            f"landmarks={len(self.landmark_map):,} | "
            f"bbox={len(self.bbox_map):,}"
        )

    def _resolve_path(self, key: str) -> Path:
        """Resolve a named annotation file to its absolute path."""
        return self.root_dir / _ANNO_FILES[key]

    def _safe_open(self, key: str) -> Optional[List[str]]:
        """
        Safely open and read an annotation file by key name.

        Returns:
            List of raw lines, or None if file does not exist.
        """
        fpath = self._resolve_path(key)
        if not fpath.exists():
            logger.warning(f"Annotation file missing: {fpath}")
            return None
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
        except Exception as exc:
            logger.error(f"Failed to read {fpath}: {exc}")
            return None

    def _parse_category_cloth(self) -> None:
        """
        Parse Anno/list_category_cloth.txt.

        Format:
          Line 0: total count
          Line 1: header ("category_name  category_type")
          Line 2+: "<name>  <type_id>"

        Populates:
          self.category_names — list of 50 category name strings.
        """
        lines = self._safe_open("category_cloth")
        if lines is None:
            return

        names: List[str] = []
        # Lines 0 (count) and 1 (header) are skipped
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            # Split on whitespace — first token is the name
            parts = line.split()
            names.append(parts[0])

        self.category_names = names
        logger.debug(f"Parsed {len(names)} category names")

    def _parse_attr_cloth(self) -> None:
        """
        Parse Anno/list_attr_cloth.txt.

        Format:
          Line 0: total count
          Line 1: header ("attribute_name  attribute_type")
          Line 2+: "<name>  <type_id>"

        Populates:
          self.attr_names — list of 1000 attribute name strings.
        """
        lines = self._safe_open("attr_cloth")
        if lines is None:
            return

        names: List[str] = []
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Attribute names sometimes have multiple tokens (e.g. "floral_pattern")
            # The last token is always the type_id (integer), first token(s) = name
            if len(parts) >= 2:
                # Name is everything except the last column (type_id)
                name = "_".join(parts[:-1])
            else:
                name = parts[0]
            names.append(name)

        self.attr_names = names
        logger.debug(f"Parsed {len(names)} attribute names")

    def _parse_eval_partition(self) -> None:
        """
        Parse Eval/list_eval_partition.txt.

        Format:
          Line 0: total count
          Line 1: header ("image_name  evaluation_status")
          Line 2+: "<img_path>  train|val|test"

        Populates:
          self.split_map — {img_rel_path: "train"|"val"|"test"}
        """
        lines = self._safe_open("eval_partition")
        if lines is None:
            return

        split_map: Dict[str, str] = {}
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path   = parts[0]
            split_label = parts[1].strip().lower()
            if split_label not in ("train", "val", "test"):
                logger.debug(f"Unknown split label: '{split_label}' for {img_path}")
                continue
            split_map[img_path] = split_label

        self.split_map = split_map
        logger.debug(f"Parsed {len(split_map):,} image-split pairs")

    def _parse_category_img(self) -> None:
        """
        Parse Anno/list_category_img.txt.

        Format:
          Line 0: total count
          Line 1: header ("image_name  category_label")
          Line 2+: "<img_path>  <cat_id>"

        Populates:
          self.category_map — {img_rel_path: (category_name, category_id)}

        Missing values: if cat_id is out of range, category_name = "unknown".
        """
        lines = self._safe_open("category_img")
        if lines is None:
            return

        cat_map: Dict[str, Tuple[str, int]] = {}
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path = parts[0]
            try:
                cat_id = int(parts[1])
            except ValueError:
                logger.debug(f"Non-integer category id on line: {line.strip()!r}")
                continue

            # Resolve name from pre-parsed list (1-based → 0-based index)
            if 1 <= cat_id <= len(self.category_names):
                cat_name = self.category_names[cat_id - 1]
            else:
                cat_name = "unknown"
                logger.debug(f"Out-of-range category id {cat_id} for {img_path}")

            cat_map[img_path] = (cat_name, cat_id)

        self.category_map = cat_map
        logger.debug(f"Parsed {len(cat_map):,} category-image mappings")

    def _parse_attr_img(self) -> None:
        """
        Parse Anno/list_attr_img.txt.

        Format:
          Line 0: total images
          Line 1: total attributes (should equal len(attr_names))
          Line 2: header
          Line 3+: "<img_path>  <v1> <v2> ... <v1000>"
          where each v = +1 (present) or -1 (absent)

        Populates:
          self.attr_map — {img_rel_path: List[int]}

        Missing values: vectors shorter than 1000 are zero-padded.
        Extra values beyond 1000 are truncated.
        """
        lines = self._safe_open("attr_img")
        if lines is None:
            return

        n_attrs = len(self.attr_names) if self.attr_names else 1000
        attr_map: Dict[str, List[int]] = {}

        # Format: Line 0 = total_imgs, Line 1 = total_attrs, Line 2 = header
        data_start = 3
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            img_path  = parts[0]
            raw_vals  = parts[1:]

            # Parse integers, replacing non-parseable tokens with 0
            parsed: List[int] = []
            for v in raw_vals[:n_attrs]:
                try:
                    parsed.append(int(v))
                except ValueError:
                    parsed.append(0)

            # Zero-pad if shorter than expected
            if len(parsed) < n_attrs:
                parsed.extend([0] * (n_attrs - len(parsed)))

            attr_map[img_path] = parsed

        self.attr_map = attr_map
        logger.debug(f"Parsed {len(attr_map):,} attribute-image mappings")

    def _parse_bbox(self) -> None:
        """
        Parse Anno/list_bbox.txt.

        Format:
          Line 0: count
          Line 1: header ("image_name  x_1 y_1 x_2 y_2")
          Line 2+: "<img_path>  <x1> <y1> <x2> <y2>"

        Populates:
          self.bbox_map — {img_rel_path: [x1, y1, x2, y2]}

        Missing values: records with fewer than 5 columns are skipped.
        Negative coordinates are clamped to 0.
        """
        lines = self._safe_open("bbox")
        if lines is None:
            return

        bbox_map: Dict[str, List[int]] = {}
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            img_path = parts[0]
            try:
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                # Clamp negative coordinates
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = max(0, x2); y2 = max(0, y2)
                bbox_map[img_path] = [x1, y1, x2, y2]
            except ValueError:
                logger.debug(f"Non-integer bbox values on line: {line.strip()!r}")
                continue

        self.bbox_map = bbox_map
        logger.debug(f"Parsed {len(bbox_map):,} bounding-box annotations")

    def _parse_landmarks(self) -> None:
        """
        Parse Anno/list_landmarks.txt.

        Format:
          Line 0: count
          Line 1: header
          Line 2+: "<img_path>  lm_x1 lm_y1 lm_vis1  lm_x2 lm_y2 lm_vis2  ...×6"
          Each landmark has 3 columns: x (int), y (int), visibility (0 or 1).
          Total landmark columns = 6 × 3 = 18.

        Populates:
          self.landmark_map — {img_rel_path: List[Dict]}
          Each dict: {"name": str, "x": int, "y": int, "visible": bool}

        Missing values:
          • Rows with fewer than 19 columns get only partial landmarks.
          • Remaining landmarks filled with {"name":…, "x":0, "y":0, "visible":False}.
        """
        lines = self._safe_open("landmarks")
        if lines is None:
            return

        landmark_map: Dict[str, List[Dict[str, Any]]] = {}
        n_landmarks = len(_LANDMARK_NAMES)   # 6

        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            img_path = parts[0]
            lm_vals  = parts[1:]  # Expected: 18 values (6 × 3)

            landmarks: List[Dict[str, Any]] = []
            for i, name in enumerate(_LANDMARK_NAMES):
                base = i * 3
                # Extract x, y, visibility — use 0 defaults for missing values
                try:
                    x   = int(lm_vals[base])     if base     < len(lm_vals) else 0
                    y   = int(lm_vals[base + 1]) if base + 1 < len(lm_vals) else 0
                    vis = int(lm_vals[base + 2]) if base + 2 < len(lm_vals) else 0
                except (ValueError, IndexError):
                    x, y, vis = 0, 0, 0

                landmarks.append({
                    "name"   : name,
                    "x"      : x,
                    "y"      : y,
                    "visible": bool(vis),
                })

            landmark_map[img_path] = landmarks

        self.landmark_map = landmark_map
        logger.debug(f"Parsed {len(landmark_map):,} landmark annotations")


# =============================================================================
# ── 4. DeepFashionExtractor — Extraction Layer
# =============================================================================

class DeepFashionExtractor:
    """
    Extraction Layer: assembles RawDeepFashionRecord objects from the
    pre-parsed annotation indexes.

    Responsibilities:
      - Combine split_map, category_map, attr_map, bbox_map, landmark_map
        per image path into a unified RawDeepFashionRecord.
      - Handle missing annotation values gracefully.
      - Stream records for a given split lazily (no loading all into RAM).
      - Report dataset info without full iteration.

    This class does NOT write files and does NOT transform/normalise data.
    """

    def __init__(
        self,
        parser   : DeepFashionAnnotationParser,
        root_dir : Union[str, Path],
    ) -> None:
        """
        Initialise the extractor.

        Args:
            parser   : A fully parsed DeepFashionAnnotationParser.
            root_dir : DeepFashion dataset root directory (for absolute paths).
        """
        self.parser   = parser
        self.root_dir = Path(root_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return summary info about the parsed annotations."""
        stats = self.parser.get_stats()
        stats["img_dir_exists"] = (self.root_dir / "img").exists()
        return stats

    def stream(
        self,
        split      : str = "train",
        max_records: Optional[int] = None,
    ) -> Generator[Tuple[RawDeepFashionRecord, Optional[str]], None, None]:
        """
        Lazily yield RawDeepFashionRecord objects for the requested split.

        Args:
            split       : "train" | "val" | "test" | "all"
            max_records : Maximum records to yield (None = all).

        Yields:
            Tuple[RawDeepFashionRecord, Optional[str]]
              — (record, None)        on success
              — (sentinel, error_msg) on extraction failure

        Raises:
            RuntimeError : If the annotation parser has no data (root missing).
        """
        if not self.parser.is_available():
            raise RuntimeError(
                "DeepFashion annotation parser has no data. "
                "Check that root_dir exists and contains Anno/ files."
            )

        img_paths = self.parser.get_image_paths_for_split(split)
        if max_records is not None:
            img_paths = img_paths[:max_records]

        logger.info(
            f"DeepFashionExtractor streaming | split={split} | "
            f"records={len(img_paths):,}"
        )

        for img_path in img_paths:
            try:
                raw, err = self._build_raw_record(img_path)
                yield raw, err
            except Exception as exc:
                sentinel = self._make_sentinel(img_path)
                yield sentinel, f"Extraction error for {img_path}: {exc}"

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_raw_record(
        self,
        img_rel_path: str,
    ) -> Tuple[RawDeepFashionRecord, Optional[str]]:
        """
        Assemble a RawDeepFashionRecord from all annotation indexes.

        Handling of missing values:
          • category_map miss  → category_raw="unknown", category_id=-1
          • attr_map miss      → empty list (no attributes)
          • bbox_map miss      → empty list []
          • landmark_map miss  → all 6 landmarks at (0,0) invisible

        Args:
            img_rel_path : Relative image path (key in all annotation maps).

        Returns:
            Tuple[RawDeepFashionRecord, Optional[str]]
        """
        error: Optional[str] = None

        # ── Category ──────────────────────────────────────────────────────────
        cat_pair    = self.parser.category_map.get(img_rel_path)
        cat_name    = cat_pair[0] if cat_pair else "unknown"
        cat_id      = cat_pair[1] if cat_pair else -1
        if not cat_pair:
            error = f"No category annotation for {img_rel_path}"

        # ── Split ─────────────────────────────────────────────────────────────
        split = self.parser.split_map.get(img_rel_path, "train")

        # ── Attributes ────────────────────────────────────────────────────────
        attr_vector = self.parser.attr_map.get(img_rel_path, [])
        attr_names  = self.parser.attr_names  # shared reference (read-only)

        # ── Bounding box ──────────────────────────────────────────────────────
        bbox = self.parser.bbox_map.get(img_rel_path, [])

        # ── Landmarks ─────────────────────────────────────────────────────────
        lm_raw = self.parser.landmark_map.get(img_rel_path)
        if lm_raw is None:
            # Build default invisible landmarks
            lm_raw = [
                {"name": name, "x": 0, "y": 0, "visible": False}
                for name in _LANDMARK_NAMES
            ]

        record = RawDeepFashionRecord(
            image_rel_path  = img_rel_path,
            category_raw    = cat_name,
            category_id     = cat_id,
            attr_vector     = list(attr_vector),
            attr_names      = list(attr_names),
            landmarks_raw   = list(lm_raw),
            bbox            = list(bbox),
            split           = split,
        )
        return record, error

    @staticmethod
    def _make_sentinel(img_rel_path: str) -> RawDeepFashionRecord:
        """Return a blank sentinel record on catastrophic failure."""
        return RawDeepFashionRecord(
            image_rel_path  = img_rel_path,
            category_raw    = "unknown",
            category_id     = -1,
            attr_vector     = [],
            attr_names      = [],
            landmarks_raw   = [],
            bbox            = [],
            split           = "train",
        )


# =============================================================================
# ── 5. DeepFashionTransformer — Transformation & Normalisation Layer
# =============================================================================

class DeepFashionTransformer:
    """
    Transformation Layer: converts RawDeepFashionRecord → DeepFashionRecord.

    Responsibilities:
      1. Generate canonical image_id from relative path.
      2. Build absolute image_path string.
      3. Normalise category: 50 raw names → 11 taxonomy keys.
      4. Decode attribute vector → active attribute name list.
      5. Normalise landmark coordinates → [0,1] using bbox dimensions.
      6. Normalise bounding box → [0,1] floats.
      7. Clean and deduplicate attribute names.

    This layer does NOT perform I/O (no file reads or writes).
    """

    def __init__(
        self,
        root_dir  : Union[str, Path] = _DEFAULT_ROOT_DIR,
        kb        : Optional["FashionDomainResearch"] = None,
        max_attrs : int = 50,
    ) -> None:
        """
        Initialise the transformer.

        Args:
            root_dir  : DeepFashion root (used to build image_path strings).
            kb        : Optional KB for attribute normalisation.
            max_attrs : Maximum number of attributes to keep per record.
        """
        self.root_dir  = Path(root_dir)
        self.kb        = kb
        self.max_attrs = max_attrs

    # ── Public API ─────────────────────────────────────────────────────────────

    def transform(self, raw: RawDeepFashionRecord) -> DeepFashionRecord:
        """
        Transform a RawDeepFashionRecord into a canonical DeepFashionRecord.

        Steps:
          1. image_id generation
          2. image_path construction
          3. Category normalisation
          4. Attribute vector → named list
          5. Landmark normalisation
          6. Bbox normalisation

        Args:
            raw : RawDeepFashionRecord from the extraction layer.

        Returns:
            DeepFashionRecord ready for validation and persistence.
        """
        # ── 1. Image identity ──────────────────────────────────────────────────
        image_id   = self._build_image_id(raw.image_rel_path)
        image_path = self._build_image_path(raw.image_rel_path)

        # ── 2. Category normalisation ──────────────────────────────────────────
        category = self._normalise_category(raw.category_raw)

        # ── 3. Attribute decoding ──────────────────────────────────────────────
        attributes = self._decode_attributes(raw.attr_vector, raw.attr_names)

        # ── 4. Landmark normalisation ──────────────────────────────────────────
        landmarks_norm = self._normalise_landmarks(raw.landmarks_raw, raw.bbox)

        # ── 5. Bounding box normalisation ──────────────────────────────────────
        bbox_norm = self._normalise_bbox(raw.bbox)

        return DeepFashionRecord(
            image_id        = image_id,
            category        = category,
            attributes      = attributes,
            landmarks       = landmarks_norm,
            image_path      = image_path,
            dataset_source  = "deepfashion",
            category_raw    = raw.category_raw,
            category_id     = raw.category_id,
            split           = raw.split,
            bbox            = raw.bbox,
            bbox_normalised = bbox_norm,
            attr_count      = len(attributes),
        )

    # ── Private: transformation helpers ───────────────────────────────────────

    def _build_image_id(self, img_rel_path: str) -> str:
        """
        Convert a relative image path to a canonical image ID string.

        Replaces path separators and dots with underscores and prepends "DF_".

        Example:
          "img/Shawls/img_00000001.jpg" → "DF_img_Shawls_img_00000001"

        Args:
            img_rel_path : Relative path (e.g. "img/Shawls/img_00000001.jpg").

        Returns:
            Canonical image ID string.
        """
        # Remove extension, replace separators
        stem = Path(img_rel_path).with_suffix("").as_posix()
        stem = re.sub(r"[/\\]+", "_", stem)
        stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
        return f"DF_{stem}"

    def _build_image_path(self, img_rel_path: str) -> str:
        """
        Build the image path string (forward-slash, from project root if possible).

        When root_dir is inside _PROJECT_ROOT (production use), returns a path
        relative to the project root (e.g. "datasets/deepfashion/img/…").
        When root_dir is outside _PROJECT_ROOT (e.g. pytest tmp_path), returns
        an absolute forward-slash path instead — this prevents a ValueError in tests.

        Args:
            img_rel_path : Path relative to DeepFashion root.

        Returns:
            Forward-slash path string.
        """
        full = self.root_dir / img_rel_path
        try:
            return full.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            # root_dir is outside the project root (e.g. during testing)
            return full.as_posix()


    def _normalise_category(self, raw_category: str) -> str:
        """
        Map a DeepFashion raw category name to our 11-key taxonomy.

        Lookup order:
          1. Exact match in _DF_CATEGORY_MAP.
          2. Case-insensitive exact match.
          3. KB alias lookup (if KB is loaded).
          4. Substring fuzzy match.
          5. Fallback: "accessories" (safest unknown default).

        Args:
            raw_category : e.g. "Blouse", "Jeans", "Tee"

        Returns:
            Taxonomy key e.g. "shirts", "jeans", "t_shirts"
        """
        if not raw_category or raw_category == "unknown":
            return "accessories"

        # 1. Exact match
        if raw_category in _DF_CATEGORY_MAP:
            return _DF_CATEGORY_MAP[raw_category]

        # 2. Case-insensitive
        for raw_key, mapped in _DF_CATEGORY_MAP.items():
            if raw_key.lower() == raw_category.lower():
                return mapped

        # 3. KB alias lookup
        if self.kb is not None and _KB_AVAILABLE:
            alias = raw_category.lower().strip()
            from_kb = self.kb._alias_to_category.get(alias)
            if from_kb:
                return from_kb

        # 4. Fuzzy substring
        raw_lower = raw_category.lower()
        for raw_key, mapped in _DF_CATEGORY_MAP.items():
            if raw_key.lower() in raw_lower or raw_lower in raw_key.lower():
                return mapped

        logger.debug(f"Unmapped category: '{raw_category}' → 'accessories'")
        return "accessories"

    def _decode_attributes(
        self,
        attr_vector : List[int],
        attr_names  : List[str],
    ) -> List[str]:
        """
        Decode a 1000-dim attribute vector into a list of active attribute names.

        An attribute is "active" when its value == +1 (present).
        Values of -1 (absent) and 0 (missing) are excluded.

        Post-processing:
          • Names are lowercased and underscores replaced with spaces.
          • Duplicates are removed (preserve order of first occurrence).
          • Result is capped at self.max_attrs entries.

        Args:
            attr_vector : List of int values (+1/-1/0).
            attr_names  : Parallel list of attribute name strings.

        Returns:
            Sorted list of active, cleaned attribute name strings.
        """
        if not attr_vector or not attr_names:
            return []

        active: List[str] = []
        seen: set = set()

        for i, val in enumerate(attr_vector):
            if val == 1 and i < len(attr_names):
                name = self._clean_attr_name(attr_names[i])
                if name and name not in seen:
                    seen.add(name)
                    active.append(name)

        return active[:self.max_attrs]

    @staticmethod
    def _clean_attr_name(raw_name: str) -> str:
        """
        Normalise an attribute name string.

        Steps:
          1. Lowercase.
          2. Replace underscores with spaces.
          3. Strip leading/trailing whitespace.
          4. Collapse multiple spaces.

        Args:
            raw_name : e.g. "floral_pattern", "Sleeveless"

        Returns:
            Cleaned name e.g. "floral pattern", "sleeveless"
        """
        name = raw_name.lower().replace("_", " ").strip()
        return re.sub(r"\s+", " ", name)

    def _normalise_landmarks(
        self,
        landmarks_raw : List[Dict[str, Any]],
        bbox          : List[int],
    ) -> List[Dict[str, Any]]:
        """
        Normalise landmark pixel coordinates to [0,1] relative to bbox.

        Normalisation formula:
          If bbox = [x1, y1, x2, y2] and bbox width/height > 0:
            x_norm = (x_pixel - x1) / (x2 - x1)
            y_norm = (y_pixel - y1) / (y2 - y1)
          Else (no valid bbox):
            x_norm = x_pixel / image_reference_size  (256 as fallback)
            y_norm = y_pixel / image_reference_size

        Coordinates are clamped to [0.0, 1.0] after normalisation.
        Invisible landmarks (visible=False) keep x=0.0, y=0.0.

        Args:
            landmarks_raw : List of raw landmark dicts from parser.
            bbox          : [x1, y1, x2, y2] pixel bbox, or [] if missing.

        Returns:
            List of normalised landmark dicts with float x, y.
        """
        _REF_SIZE = 256.0  # DeepFashion images are nominally 256×256

        # Determine normalisation anchors from bbox
        if len(bbox) == 4:
            bx1, by1, bx2, by2 = bbox
            bw = max(float(bx2 - bx1), 1.0)
            bh = max(float(by2 - by1), 1.0)
            has_bbox = True
        else:
            bx1, by1, bw, bh = 0, 0, _REF_SIZE, _REF_SIZE
            has_bbox = False

        normalised: List[Dict[str, Any]] = []
        for lm in landmarks_raw:
            x_px  = float(lm.get("x", 0))
            y_px  = float(lm.get("y", 0))
            vis   = bool(lm.get("visible", False))
            name  = lm.get("name", "unknown")

            if vis:
                if has_bbox:
                    x_n = (x_px - bx1) / bw
                    y_n = (y_px - by1) / bh
                else:
                    x_n = x_px / _REF_SIZE
                    y_n = y_px / _REF_SIZE

                # Clamp and round to 4 decimal places
                x_n = round(max(0.0, min(1.0, x_n)), 4)
                y_n = round(max(0.0, min(1.0, y_n)), 4)
            else:
                # Invisible landmarks get sentinel values
                x_n = 0.0
                y_n = 0.0

            normalised.append({
                "name"   : name,
                "x"      : x_n,
                "y"      : y_n,
                "visible": vis,
            })

        return normalised

    def _normalise_bbox(self, bbox: List[int]) -> List[float]:
        """
        Normalise a pixel bounding box to [0,1] relative to reference size.

        Uses 256×256 as the reference since DeepFashion images are 256 px.
        Returns [] if bbox is missing or malformed.

        Args:
            bbox : [x1, y1, x2, y2] in pixel coordinates, or [].

        Returns:
            [x1_n, y1_n, x2_n, y2_n] all in [0,1], or [].
        """
        if len(bbox) != 4:
            return []

        ref = 256.0
        x1_n = round(max(0.0, min(1.0, bbox[0] / ref)), 4)
        y1_n = round(max(0.0, min(1.0, bbox[1] / ref)), 4)
        x2_n = round(max(0.0, min(1.0, bbox[2] / ref)), 4)
        y2_n = round(max(0.0, min(1.0, bbox[3] / ref)), 4)
        return [x1_n, y1_n, x2_n, y2_n]


# =============================================================================
# ── 6. DeepFashionValidator — Validation Layer
# =============================================================================

class DeepFashionValidator:
    """
    Validation Layer: validates DeepFashionRecord against schema + taxonomy rules.

    Validation layers:
      1. Required fields  — image_id, category, image_path, dataset_source
      2. Category validity — must be one of 11 taxonomy keys
      3. Attributes type  — must be a list of strings
      4. Landmarks count  — must have exactly 6 entries
      5. Landmark schema  — each entry must have name, x, y, visible
      6. Normalised coords— x, y in [0,1]; warnings for out-of-range
      7. Split validity   — must be train/val/test
    """

    _REQUIRED_FIELDS = ["image_id", "category", "image_path", "dataset_source"]
    _VALID_CATEGORIES = _VALID_CATEGORIES
    _EXPECTED_LANDMARKS = len(_LANDMARK_NAMES)   # 6

    def __init__(self, kb: Optional["FashionDomainResearch"] = None) -> None:
        self.kb = kb

    def validate(self, record: DeepFashionRecord) -> DeepFashionRecord:
        """
        Run all validation layers and return the record with results populated.

        Args:
            record : DeepFashionRecord from the transformation layer.

        Returns:
            The same record with is_valid, errors, warnings updated.
        """
        errors:   List[str] = []
        warnings: List[str] = []

        # ── Layer 1: Required fields ───────────────────────────────────────────
        for req in self._REQUIRED_FIELDS:
            val = getattr(record, req, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"Missing required field: '{req}'")

        # ── Layer 2: Category validity ─────────────────────────────────────────
        if record.category not in self._VALID_CATEGORIES:
            errors.append(
                f"Invalid category: '{record.category}'. "
                f"Valid: {sorted(self._VALID_CATEGORIES)}"
            )

        # ── Layer 3: Attributes type ───────────────────────────────────────────
        if not isinstance(record.attributes, list):
            errors.append("'attributes' must be a list")
        else:
            non_str = [a for a in record.attributes if not isinstance(a, str)]
            if non_str:
                errors.append(
                    f"'attributes' contains non-string values: {non_str[:3]}"
                )

        # ── Layer 4: Landmark count ────────────────────────────────────────────
        if not isinstance(record.landmarks, list):
            errors.append("'landmarks' must be a list")
        elif len(record.landmarks) != self._EXPECTED_LANDMARKS:
            warnings.append(
                f"Expected {self._EXPECTED_LANDMARKS} landmarks, "
                f"got {len(record.landmarks)}"
            )

        # ── Layer 5: Landmark schema ───────────────────────────────────────────
        if isinstance(record.landmarks, list):
            for i, lm in enumerate(record.landmarks):
                if not isinstance(lm, dict):
                    errors.append(f"Landmark[{i}] is not a dict")
                    continue
                for key in ("name", "x", "y", "visible"):
                    if key not in lm:
                        errors.append(f"Landmark[{i}] missing key '{key}'")

        # ── Layer 6: Normalised coordinate range ──────────────────────────────
        if isinstance(record.landmarks, list):
            for i, lm in enumerate(record.landmarks):
                if not isinstance(lm, dict):
                    continue
                if lm.get("visible", False):
                    x, y = lm.get("x", 0), lm.get("y", 0)
                    if not (0.0 <= x <= 1.0):
                        warnings.append(
                            f"Landmark[{i}].x={x} out of [0,1] range"
                        )
                    if not (0.0 <= y <= 1.0):
                        warnings.append(
                            f"Landmark[{i}].y={y} out of [0,1] range"
                        )

        # ── Layer 7: Split validity ────────────────────────────────────────────
        if record.split not in ("train", "val", "test"):
            warnings.append(
                f"Unrecognised split: '{record.split}'. "
                f"Expected: train/val/test"
            )

        record.errors   = errors
        record.warnings = warnings
        record.is_valid = len(errors) == 0
        return record


# =============================================================================
# ── 7. DeepFashionWriter — Write / Save Layer
# =============================================================================

class DeepFashionWriter:
    """
    Write Layer: persists processed DeepFashion records to JSON.

    Responsibilities:
      - Create output directory if required.
      - Write deepfashion_processed.json with _meta + records sections.
      - Write a run report JSON file.
    """

    def __init__(self, output_dir: Union[str, Path] = _DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_records(
        self,
        records : List[DeepFashionRecord],
        stats   : DFPipelineStats,
    ) -> Path:
        """
        Write all processed records to deepfashion_processed.json.

        Output structure:
          {
            "_meta"  : { pipeline info, stats, record count },
            "records": [ { ...DeepFashionRecord.to_dict() }, ... ]
          }

        Args:
            records : List of DeepFashionRecord objects.
            stats   : Final pipeline statistics.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self.output_dir / _OUTPUT_FILENAME

        payload = {
            "_meta": {
                "dataset"          : "deepfashion",
                "pipeline_version" : "1.0.0",
                "generated_at"     : datetime.now(timezone.utc).isoformat(),
                "total_records"    : len(records),
                "valid_records"    : sum(1 for r in records if r.is_valid),
                "invalid_records"  : sum(1 for r in records if not r.is_valid),
                "output_file"      : str(output_path),
                "stats"            : stats.to_dict(),
            },
            "records": [r.to_dict() for r in records],
        }

        logger.info(f"Writing {len(records):,} records → {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.success(
            f"Saved {len(records):,} records | "
            f"file={output_path.name} | "
            f"size={output_path.stat().st_size / 1024:.1f} KB"
        )
        return output_path

    def save_run_report(
        self,
        stats    : DFPipelineStats,
        root_dir : Path,
    ) -> Path:
        """
        Write a machine-readable pipeline run report to JSON.

        Args:
            stats    : Final pipeline stats.
            root_dir : Source DeepFashion root directory.

        Returns:
            Path to the saved report file.
        """
        report_path = self.output_dir / "deepfashion_run_report.json"
        report = {
            "run_info": {
                "pipeline"        : "DeepFashionLoader",
                "version"         : "1.0.0",
                "run_at"          : datetime.now(timezone.utc).isoformat(),
                "source_root"     : str(root_dir),
                "output_file"     : str(self.output_dir / _OUTPUT_FILENAME),
            },
            "pipeline_stats": stats.to_dict(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Run report saved → {report_path}")
        return report_path


# =============================================================================
# ── 8. DeepFashionLoader — Orchestrator (Main Public API)
# =============================================================================

class DeepFashionLoader:
    """
    Pipeline Orchestrator: coordinates all 6 layers into a single .run() call.

    Usage:
        loader = DeepFashionLoader()
        result = loader.run(split="train", max_records=500)
        print(result["stats"]["total_valid"])
        print(result["output_path"])
    """

    def __init__(
        self,
        root_dir   : Union[str, Path, None] = None,
        output_dir : Union[str, Path, None] = None,
        use_kb     : bool = True,
        max_attrs  : int  = 50,
    ) -> None:
        """
        Initialise the DeepFashion pipeline orchestrator.

        Args:
            root_dir   : DeepFashion dataset root.
                         Defaults to datasets/deepfashion/
            output_dir : JSON output directory.
                         Defaults to datasets/processed/
            use_kb     : Whether to load the Knowledge Base for normalisation.
            max_attrs  : Maximum active attributes to keep per record.
        """
        self.root_dir   = Path(root_dir)   if root_dir   else _DEFAULT_ROOT_DIR
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

        # ── Knowledge Base ─────────────────────────────────────────────────────
        self.kb: Optional[FashionDomainResearch] = None
        if use_kb and _KB_AVAILABLE:
            try:
                self.kb = FashionDomainResearch()
                logger.info("Knowledge Base attached to DeepFashion pipeline")
            except Exception as exc:
                logger.warning(f"KB load failed ({exc}) — proceeding without KB")

        # ── Instantiate pipeline layers ────────────────────────────────────────
        self.parser      = DeepFashionAnnotationParser(root_dir=self.root_dir)
        self.extractor   = DeepFashionExtractor(parser=self.parser, root_dir=self.root_dir)
        self.transformer = DeepFashionTransformer(
            root_dir  = self.root_dir,
            kb        = self.kb,
            max_attrs = max_attrs,
        )
        self.validator   = DeepFashionValidator(kb=self.kb)
        self.writer      = DeepFashionWriter(output_dir=self.output_dir)

        logger.info(
            f"DeepFashionLoader ready | "
            f"root={self.root_dir.name} | "
            f"kb={'yes' if self.kb else 'no'} | "
            f"parser_available={self.parser.is_available()}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        split        : str = "train",
        max_records  : Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full DeepFashion ingestion pipeline.

        Flow: Parser → Extractor → Transformer → Validator → Writer

        Args:
            split         : "train" | "val" | "test" | "all"
            max_records   : Maximum records to process. None = all.
            show_progress : Whether to show tqdm progress bar.

        Returns:
            Dict with:
              "output_path"   : Path to deepfashion_processed.json
              "report_path"   : Path to deepfashion_run_report.json
              "stats"         : DFPipelineStats.to_dict()
              "total_records" : int number of records saved
        """
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Invalid split: '{split}'. Choose from {sorted(_VALID_SPLITS)}"
            )

        logger.info("=" * 60)
        logger.info("DeepFashion Pipeline Starting")
        logger.info(f"  Root    : {self.root_dir}")
        logger.info(f"  Split   : {split}")
        logger.info(f"  Max     : {max_records or 'all'}")
        logger.info(f"  Output  : {self.output_dir / _OUTPUT_FILENAME}")
        logger.info("=" * 60)

        stats   = DFPipelineStats()
        records : List[DeepFashionRecord] = []

        # ── Determine total for progress bar ───────────────────────────────────
        total = len(self.parser.get_image_paths_for_split(split))
        if max_records is not None:
            total = min(total, max_records)

        # ── Build stream with optional tqdm ────────────────────────────────────
        raw_stream = self.extractor.stream(split=split, max_records=max_records)
        iterable   = self._wrap_progress(
            raw_stream,
            total = total,
            desc  = f"DeepFashion [{split}]",
            show  = show_progress and _TQDM_AVAILABLE,
        )

        log_interval = max(1, min(500, total // 10)) if total > 0 else 200

        # ── Main processing loop ───────────────────────────────────────────────
        for raw_record, extraction_err in iterable:
            stats.total_read += 1

            # ── Log extraction errors (missing annotations) ────────────────────
            if extraction_err:
                logger.debug(f"Extraction note: {extraction_err}")
                # Don't skip — we still process with default values

            try:
                # ── Transform ─────────────────────────────────────────────────
                record = self.transformer.transform(raw_record)

                # ── Validate ──────────────────────────────────────────────────
                record = self.validator.validate(record)

                # ── Update stats ──────────────────────────────────────────────
                stats.total_processed += 1
                if record.is_valid:
                    stats.total_valid += 1
                else:
                    stats.total_invalid += 1

                stats.increment_category(record.category or "unknown")
                stats.increment_split(record.split or "train")

                records.append(record)

                # ── Periodic log ──────────────────────────────────────────────
                if not (_TQDM_AVAILABLE and show_progress):
                    if stats.total_processed % log_interval == 0:
                        logger.info(
                            f"Progress: {stats.total_processed:,} / {total:,} | "
                            f"valid={stats.total_valid:,} | "
                            f"{stats.records_per_second} rec/s"
                        )

            except Exception as exc:
                logger.error(
                    f"Pipeline error for {raw_record.image_rel_path}: {exc}",
                    exc_info=True,
                )
                stats.total_skipped += 1
                continue

        # ── Finalise ───────────────────────────────────────────────────────────
        stats.finalize()
        stats.total_saved = len(records)
        stats.log_summary()

        output_path = self.writer.save_records(records, stats)
        report_path = self.writer.save_run_report(stats, self.root_dir)

        return {
            "output_path"  : output_path,
            "report_path"  : report_path,
            "stats"        : stats.to_dict(),
            "total_records": len(records),
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the DeepFashion annotations without processing."""
        return self.extractor.get_dataset_info()

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_progress(
        iterable: Iterator,
        total   : int,
        desc    : str,
        show    : bool,
    ) -> Iterator:
        """Wrap iterator with tqdm if available and requested."""
        if show and _TQDM_AVAILABLE:
            return _tqdm(
                iterable,
                total     = total,
                desc      = desc,
                unit      = "rec",
                dynamic_ncols = True,
                bar_format = (
                    "{desc}: {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
        return iterable


# =============================================================================
# ── 9. CLI Entry Point
# =============================================================================

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfashion_loader",
        description=(
            "DeepFashion Dataset Ingestion Pipeline\n"
            "AI-Powered Fashion Design Assistant — Week 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python deepfashion_loader.py --info\n"
            "  python deepfashion_loader.py --split train --max-records 500\n"
            "  python deepfashion_loader.py --split all --no-kb\n"
        ),
    )
    parser.add_argument(
        "--root-dir", type=str, default=None, metavar="DIR",
        help=f"DeepFashion dataset root (default: {_DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, metavar="DIR",
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--max-records", type=int, default=None, metavar="N",
        help="Maximum records to process (default: all)",
    )
    parser.add_argument(
        "--max-attrs", type=int, default=50, metavar="N",
        help="Maximum attributes per record (default: 50)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--no-kb", action="store_true",
        help="Disable Knowledge Base normalisation",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print dataset info and exit",
    )
    return parser


def main() -> int:
    """CLI main entry point. Returns exit code (0 = success, 1 = error)."""
    parser = _build_cli_parser()
    args   = parser.parse_args()

    loader = DeepFashionLoader(
        root_dir   = args.root_dir,
        output_dir = args.output_dir,
        use_kb     = not args.no_kb,
        max_attrs  = args.max_attrs,
    )

    if args.info:
        info = loader.get_dataset_info()
        print("\nDeepFashion Dataset Info")
        print("=" * 50)
        for k, v in info.items():
            print(f"  {k:<30} : {v}")
        return 0

    try:
        result = loader.run(
            split         = args.split,
            max_records   = args.max_records,
            show_progress = not args.no_progress,
        )
        print("\nPipeline Complete")
        print(f"  Records saved : {result['total_records']:,}")
        print(f"  Output        : {result['output_path']}")
        print(f"  Report        : {result['report_path']}")
        return 0
    except RuntimeError as exc:
        logger.error(str(exc))
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
