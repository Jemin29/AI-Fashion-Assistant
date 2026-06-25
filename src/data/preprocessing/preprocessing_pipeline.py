"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/preprocessing/preprocessing_pipeline.py
=============================================================================
MODULE  : Fashion Dataset Preprocessing Pipeline
WEEK    : 1 — Fashion Domain Research & Dataset Curation
AUTHOR  : Fashion AI Team

PURPOSE
-------
A unified, end-to-end preprocessing engine that takes raw validated fashion
records and transforms them into a clean, model-ready dataset saved as
processed/clean_dataset.json.

PIPELINE STAGES (7 stages, in order)
--------------------------------------
  Stage 1 — Image Resizing
      Target: (256, 256) by default (overridable via PreprocessingConfig).
      Method: Pillow LANCZOS for high-quality downsampling.
      Output: image_width, image_height, aspect_ratio_original written to record.

  Stage 2 — Image Normalization
      Computes per-channel (R, G, B) mean and std of each image for catalog.
      Also records pixel_mean_rgb and pixel_std_rgb in metadata so downstream
      trainers can verify dataset statistics without re-reading images.
      Does NOT require images to exist on disk (skips gracefully when absent).

  Stage 3 — Duplicate Image Detection
      Builds a perceptual hash (pHash) from each image_path filename + category
      (content hash fallback when no pixel data is available).
      Marks exact duplicate image_ids and reports them separately.
      Strategy: dict-based O(n) deduplication keyed on canonical hash.

  Stage 4 — Description Cleaning
      - Strip leading/trailing whitespace
      - Collapse multiple spaces/newlines
      - Remove HTML tags
      - Remove control characters
      - Normalize Unicode (NFC)
      - Lowercase (configurable)
      - Truncate to max_description_chars

  Stage 5 — Attribute Normalization
      Maps variant spellings to canonical taxonomy keys using lookup tables.
      Fields processed: color, fabric, pattern, occasion, style, fit, season.
      Unknown values are kept as-is but flagged in a normalization_warnings list.

  Stage 6 — Category Normalization
      Maps raw category strings (from FashionGen / DeepFashion loaders) to the
      11-key taxonomy: t_shirts, shirts, hoodies, jackets, pants, jeans, shorts,
      dresses, ethnic_wear, footwear, accessories.
      Records with unmappable categories → status = "uncategorized".

  Stage 7 — Dataset Balancing Statistics
      Computes per-category, per-gender, per-source, per-season, per-style
      item counts and share percentages.
      Recommends over/under-sampling targets to achieve balance.
      Written into a "balance_stats" key inside the final JSON report.

OUTPUT FILES
------------
  processed/clean_dataset.json
      {
        "generated_at"   : "ISO-8601 UTC",
        "schema_version" : "1.0.0",
        "pipeline_config": { ... },
        "summary": {
          "total_input"     : int,
          "total_output"    : int,
          "duplicates_removed": int,
          "uncategorized"   : int,
          "processing_time_s": float
        },
        "balance_stats"  : { ... per-field distribution ... },
        "records"        : [ ... cleaned record dicts ... ]
      }

USAGE
-----
  from src.data.preprocessing.preprocessing_pipeline import (
      PreprocessingPipeline, PipelineConfig
  )

  pipeline = PreprocessingPipeline(config=PipelineConfig())
  result   = pipeline.run(records)
  pipeline.save(result, "datasets/processed/clean_dataset.json")

DESIGN DECISIONS
----------------
  - Each stage is a pure function (stage_N_...) that returns a NEW dict —
    no mutation of input records.
  - All stages are independently callable for unit tests.
  - Disk I/O (PIL open) is gated by _PIL_AVAILABLE and image file existence.
  - The pipeline tolerates missing fields gracefully (never raises on bad input).
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ─── Third-party ──────────────────────────────────────────────────────────────
from loguru import logger

# ─── Optional: PIL for pixel-level operations ─────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed — image resize/normalize stages use metadata only.")

# ─── Project path resolution ──────────────────────────────────────────────────
_FILE_DIR     = Path(__file__).resolve().parent         # preprocessing/
_PROJECT_ROOT = _FILE_DIR.parent.parent                 # fashion-ai-assistant/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Knowledge-base taxonomy (graceful fallback) ──────────────────────────────
try:
    from src.data.knowledge_base.fashion_domain_research import (
        VALID_CATEGORIES,
        VALID_STYLES,
        VALID_FITS,
        VALID_SEASONS,
        VALID_OCCASIONS,
        VALID_GENDERS,
    )
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    VALID_CATEGORIES = frozenset({
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    })
    VALID_STYLES   = frozenset({"streetwear", "luxury", "formal", "business_casual",
                                "techwear", "minimalist", "vintage", "athleisure"})
    VALID_FITS     = frozenset({"slim_fit", "regular_fit", "relaxed_fit", "oversized",
                                "cropped", "skinny", "straight", "athletic_fit"})
    VALID_SEASONS  = frozenset({"spring", "summer", "autumn", "winter", "all_season"})
    VALID_OCCASIONS = frozenset({"casual", "business_casual", "formal", "party",
                                 "sport", "outdoor", "beach", "wedding_festive", "lounge"})
    VALID_GENDERS  = frozenset({"men", "women", "unisex"})

_PIPELINE_VERSION = "1.0.0"


# =============================================================================
# ── 1. Taxonomy Lookup Tables (normalisation maps)
# =============================================================================

# ── Category aliases → canonical key ─────────────────────────────────────────
_CATEGORY_ALIASES: Dict[str, str] = {
    # T-Shirts
    "t_shirt": "t_shirts", "t-shirt": "t_shirts", "tshirt": "t_shirts",
    "tee": "t_shirts", "tees": "t_shirts", "t shirt": "t_shirts",
    "polo": "t_shirts", "polo shirt": "t_shirts",
    # Shirts
    "shirt": "shirts", "blouse": "shirts", "top": "shirts",
    "button down": "shirts", "button-down": "shirts",
    "dress shirt": "shirts", "formal shirt": "shirts",
    # Hoodies
    "hoodie": "hoodies", "sweatshirt": "hoodies", "pullover": "hoodies",
    "hood": "hoodies",
    # Jackets
    "jacket": "jackets", "coat": "jackets", "blazer": "jackets",
    "windbreaker": "jackets", "parka": "jackets", "bomber": "jackets",
    "overcoat": "jackets", "trench coat": "jackets", "vest": "jackets",
    # Pants
    "pant": "pants", "trouser": "pants", "trousers": "pants",
    "chino": "pants", "chinos": "pants", "slacks": "pants",
    "leggings": "pants", "joggers": "pants", "sweatpants": "pants",
    "cargo pants": "pants",
    # Jeans
    "jean": "jeans", "denim": "jeans", "denim jeans": "jeans",
    # Shorts
    "short": "shorts", "bermuda": "shorts", "cutoffs": "shorts",
    # Dresses
    "dress": "dresses", "gown": "dresses", "frock": "dresses",
    "maxi dress": "dresses", "midi dress": "dresses", "mini dress": "dresses",
    "jumpsuit": "dresses", "romper": "dresses",
    # Ethnic Wear
    "ethnic": "ethnic_wear", "kurta": "ethnic_wear", "saree": "ethnic_wear",
    "sari": "ethnic_wear", "salwar": "ethnic_wear", "dhoti": "ethnic_wear",
    "lehenga": "ethnic_wear", "sherwani": "ethnic_wear",
    # Footwear
    "shoe": "footwear", "shoes": "footwear", "sneaker": "footwear",
    "sneakers": "footwear", "boot": "footwear", "boots": "footwear",
    "sandal": "footwear", "sandals": "footwear", "heel": "footwear",
    "heels": "footwear", "loafer": "footwear", "loafers": "footwear",
    "slipper": "footwear", "flip flop": "footwear",
    # Accessories
    "accessory": "accessories", "bag": "accessories", "handbag": "accessories",
    "belt": "accessories", "hat": "accessories", "scarf": "accessories",
    "watch": "accessories", "sunglasses": "accessories", "cap": "accessories",
    "backpack": "accessories", "wallet": "accessories", "jewellery": "accessories",
    "jewelry": "accessories",
}

# ── Gender aliases → canonical key ────────────────────────────────────────────
_GENDER_ALIASES: Dict[str, str] = {
    "man": "men", "male": "men", "boy": "men", "boys": "men",
    "gentleman": "men", "mens": "men", "men's": "men",
    "woman": "women", "female": "women", "girl": "women", "girls": "women",
    "ladies": "women", "lady": "women", "womens": "women", "women's": "women",
    "uni": "unisex", "gender neutral": "unisex", "neutral": "unisex",
    "all": "unisex",
}

# ── Color aliases → canonical name (Title Case) ───────────────────────────────
_COLOR_ALIASES: Dict[str, str] = {
    "navy blue": "Navy", "navy": "Navy", "dark blue": "Navy",
    "royal blue": "Blue", "sky blue": "Light Blue", "cobalt": "Blue",
    "light blue": "Light Blue", "baby blue": "Light Blue",
    "crimson": "Red", "scarlet": "Red", "burgundy": "Maroon",
    "wine": "Maroon", "maroon": "Maroon", "cherry": "Red",
    "neon green": "Green", "olive": "Olive", "khaki": "Khaki",
    "forest green": "Green", "sage": "Sage Green", "mint": "Mint",
    "lime": "Green", "emerald": "Green",
    "beige": "Beige", "cream": "Cream", "ivory": "Ivory", "off white": "Ivory",
    "off-white": "Ivory", "ecru": "Cream",
    "charcoal": "Charcoal", "dark grey": "Charcoal", "grey": "Grey",
    "gray": "Grey", "light grey": "Light Grey", "light gray": "Light Grey",
    "camel": "Camel", "tan": "Tan", "brown": "Brown", "chocolate": "Brown",
    "mustard": "Mustard", "yellow": "Yellow", "golden": "Gold",
    "gold": "Gold", "orange": "Orange", "coral": "Coral",
    "pink": "Pink", "blush": "Blush", "rose": "Rose", "hot pink": "Pink",
    "magenta": "Magenta", "fuchsia": "Fuchsia", "lilac": "Lilac",
    "lavender": "Lavender", "purple": "Purple", "violet": "Purple",
    "multi": "Multicolor", "multicolor": "Multicolor", "multicolour": "Multicolor",
    "print": "Printed", "printed": "Printed", "floral": "Floral",
    "tie dye": "Tie-dye", "tie-dye": "Tie-dye",
    "white": "White", "black": "Black",
}

# ── Style aliases → canonical key ─────────────────────────────────────────────
_STYLE_ALIASES: Dict[str, str] = {
    "street": "streetwear", "urban": "streetwear", "streetwear": "streetwear",
    "hype": "streetwear", "hypebeast": "streetwear", "skate": "streetwear",
    "high end": "luxury", "designer": "luxury", "premium": "luxury",
    "couture": "luxury", "haute": "luxury", "luxury": "luxury",
    "business": "business_casual", "work": "business_casual",
    "office": "business_casual", "corporate": "business_casual",
    "business_casual": "business_casual", "business casual": "business_casual",
    "smart casual": "business_casual",
    "formal": "formal", "black tie": "formal", "evening": "formal",
    "tech": "techwear", "techwear": "techwear", "tech wear": "techwear",
    "functional": "techwear", "tactical": "techwear",
    "minimal": "minimalist", "minimalist": "minimalist",
    "clean": "minimalist", "simple": "minimalist", "classic": "minimalist",
    "retro": "vintage", "vintage": "vintage", "throwback": "vintage",
    "90s": "vintage", "80s": "vintage", "old school": "vintage",
    "athletic": "athleisure", "sport": "athleisure", "athleisure": "athleisure",
    "activewear": "athleisure", "gym": "athleisure", "workout": "athleisure",
    "sporty": "athleisure", "yoga": "athleisure",
}

# ── Fit aliases → canonical key ───────────────────────────────────────────────
_FIT_ALIASES: Dict[str, str] = {
    "slim": "slim_fit", "slim fit": "slim_fit", "slim-fit": "slim_fit",
    "skinny fit": "slim_fit", "fitted": "slim_fit",
    "regular": "regular_fit", "regular fit": "regular_fit", "classic fit": "regular_fit",
    "standard": "regular_fit", "normal": "regular_fit",
    "relaxed": "relaxed_fit", "relaxed fit": "relaxed_fit", "loose": "relaxed_fit",
    "comfortable": "relaxed_fit", "easy fit": "relaxed_fit",
    "oversized": "oversized", "baggy": "oversized", "boxy": "oversized",
    "wide": "oversized", "oversize": "oversized",
    "cropped": "cropped", "crop": "cropped", "cut-off": "cropped",
    "skinny": "skinny", "super skinny": "skinny", "jegging": "skinny",
    "straight": "straight", "straight leg": "straight", "straight-leg": "straight",
    "tapered": "straight",
    "athletic": "athletic_fit", "muscle fit": "athletic_fit",
    "athletic fit": "athletic_fit", "sport fit": "athletic_fit",
}

# ── Season aliases → canonical key ────────────────────────────────────────────
_SEASON_ALIASES: Dict[str, str] = {
    "spring": "spring", "spring/summer": "spring",
    "summer": "summer", "s/s": "summer",
    "autumn": "autumn", "fall": "autumn", "autumn/winter": "autumn",
    "winter": "winter", "a/w": "autumn", "f/w": "winter", "aw": "autumn",
    "all season": "all_season", "all-season": "all_season",
    "all_season": "all_season", "year round": "all_season",
    "four season": "all_season", "multi season": "all_season",
    "ss": "summer", "fw": "winter",
}

# ── Occasion aliases → canonical key ─────────────────────────────────────────
_OCCASION_ALIASES: Dict[str, str] = {
    "casual": "casual", "everyday": "casual", "daily": "casual",
    "weekend": "casual", "relaxed": "casual",
    "business casual": "business_casual", "business_casual": "business_casual",
    "smart casual": "business_casual", "office casual": "business_casual",
    "formal": "formal", "black tie": "formal", "gala": "formal",
    "cocktail": "formal", "semi formal": "formal",
    "party": "party", "club": "party", "night out": "party",
    "festival": "party", "celebration": "party",
    "sport": "sport", "sports": "sport", "gym": "sport",
    "workout": "sport", "training": "sport", "athletic": "sport",
    "outdoor": "outdoor", "hiking": "outdoor", "camping": "outdoor",
    "adventure": "outdoor", "travel": "outdoor",
    "beach": "beach", "pool": "beach", "summer": "beach",
    "vacation": "beach", "resort": "beach",
    "wedding": "wedding_festive", "festive": "wedding_festive",
    "wedding_festive": "wedding_festive", "puja": "wedding_festive",
    "ceremony": "wedding_festive", "traditional": "wedding_festive",
    "lounge": "lounge", "home": "lounge", "sleepwear": "lounge",
    "loungewear": "lounge", "comfort": "lounge",
}

# ── Regex patterns used in description cleaning ────────────────────────────────
_HTML_TAG_RE     = re.compile(r"<[^>]+>")
_WHITESPACE_RE   = re.compile(r"\s+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
_MULTI_PUNCT_RE  = re.compile(r"([!?.]){2,}")   # collapse "!!!" → "!"


# =============================================================================
# ── 2. Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """
    All configurable thresholds for PreprocessingPipeline.
    Override any field at instantiation time to tune behaviour.
    """
    # ── Stage 1: Image resizing ────────────────────────────────────────────────
    target_size            : Tuple[int, int] = (256, 256)   # (width, height)
    resample_filter        : str             = "LANCZOS"     # PIL filter name
    output_image_format    : str             = "JPEG"
    jpeg_quality           : int             = 95
    resize_if_image_exists : bool            = False  # skip if already resized

    # ── Stage 2: Image normalization ──────────────────────────────────────────
    imagenet_mean          : Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std           : Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── Stage 3: Duplicate detection ──────────────────────────────────────────
    # Strategy: 'path_hash' (fast, no PIL) | 'phash' (perceptual, requires PIL)
    dedup_strategy         : str  = "path_hash"
    keep_first_duplicate   : bool = True   # True → keep first seen, drop rest

    # ── Stage 4: Description cleaning ─────────────────────────────────────────
    lowercase_description  : bool = False
    max_description_chars  : int  = 2048
    min_description_chars  : int  = 5     # shorter → mark as empty after clean

    # ── Stage 5-6: Attribute/Category normalization ───────────────────────────
    drop_unknown_categories: bool = False  # False → keep as "uncategorized"
    keep_normalization_log : bool = True   # record raw→canonical mappings

    # ── Stage 7: Balancing statistics ─────────────────────────────────────────
    balance_target_per_class: Optional[int] = None  # None → use min count

    # ── Output ────────────────────────────────────────────────────────────────
    project_root           : Path = field(
        default_factory=lambda: _PROJECT_ROOT
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        d["imagenet_mean"] = list(self.imagenet_mean)
        d["imagenet_std"]  = list(self.imagenet_std)
        d["target_size"]   = list(self.target_size)
        return d


# =============================================================================
# ── 3. Result Data Models
# =============================================================================

@dataclass
class StageResult:
    """
    Result of a single pipeline stage applied to one record.

    Attributes:
        record       : The record dict after this stage's transforms.
        modified     : True if any field was changed.
        warnings     : Soft issues logged but not blocking.
        dropped      : True if this record should be excluded from output.
        drop_reason  : Why the record was dropped (if dropped=True).
    """
    record     : Dict[str, Any]
    modified   : bool               = False
    warnings   : List[str]          = field(default_factory=list)
    dropped    : bool               = False
    drop_reason: Optional[str]      = None


@dataclass
class PipelineRunResult:
    """
    Aggregate result after running the full 7-stage pipeline on a batch.

    Spec-required summary fields:
        total_input          : int
        total_output         : int
        duplicates_removed   : int
        uncategorized        : int

    Extended:
        dropped_count        : int (total excluded records)
        warning_count        : int (total warnings across all records)
        processing_time_s    : float
        balance_stats        : dict  (stage 7 output)
        normalization_log    : dict  (raw→canonical mapping summary)
        records              : list of cleaned record dicts
    """
    total_input         : int   = 0
    total_output        : int   = 0
    duplicates_removed  : int   = 0
    uncategorized       : int   = 0
    dropped_count       : int   = 0
    warning_count       : int   = 0
    processing_time_s   : float = 0.0
    generated_at        : str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    balance_stats       : Dict[str, Any]    = field(default_factory=dict)
    normalization_log   : Dict[str, Any]    = field(default_factory=dict)
    records             : List[Dict[str, Any]] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_input"         : self.total_input,
            "total_output"        : self.total_output,
            "duplicates_removed"  : self.duplicates_removed,
            "uncategorized"       : self.uncategorized,
            "dropped_count"       : self.dropped_count,
            "warning_count"       : self.warning_count,
            "processing_time_s"   : round(self.processing_time_s, 3),
            "generated_at"        : self.generated_at,
        }

    def print_summary(self) -> None:
        print("=" * 66)
        print("PREPROCESSING PIPELINE — SUMMARY")
        print("=" * 66)
        print(f"  Input records        : {self.total_input:>8,}")
        print(f"  Output records       : {self.total_output:>8,}")
        print(f"  Duplicates removed   : {self.duplicates_removed:>8,}")
        print(f"  Uncategorized        : {self.uncategorized:>8,}")
        print(f"  Dropped (total)      : {self.dropped_count:>8,}")
        print(f"  Warnings issued      : {self.warning_count:>8,}")
        print(f"  Processing time      : {self.processing_time_s:.3f}s")
        print(f"  Generated at         : {self.generated_at}")
        print("=" * 66)


# =============================================================================
# ── 4. Standalone Stage Functions (pure, independently testable)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Image Resizing
# ─────────────────────────────────────────────────────────────────────────────

def stage1_image_resize(
    record     : Dict[str, Any],
    config     : PipelineConfig,
) -> StageResult:
    """
    Stage 1 — Image Resizing.

    Adds/updates these keys in the output record:
        image_width          : target width (int)
        image_height         : target height (int)
        aspect_ratio_original: w/h before resize (float | None)
        image_resized        : True if PIL resize was performed

    If the image file does not exist (or PIL is unavailable), the target
    dimensions are still recorded from config so downstream records remain
    schema-complete.

    Args:
        record : Fashion item dict with at least an "image_path" key.
        config : PipelineConfig with target_size, jpeg_quality, etc.

    Returns:
        StageResult with updated record.
    """
    rec      = dict(record)  # shallow copy — do not mutate input
    warnings : List[str] = []
    modified  = False

    target_w, target_h = config.target_size
    image_path_raw: str = rec.get("image_path") or ""
    resized = False

    # ── Determine original aspect ratio ──────────────────────────────────────
    orig_w = rec.get("image_width")
    orig_h = rec.get("image_height")
    aspect_ratio: Optional[float] = None
    if orig_w and orig_h and orig_h > 0:
        aspect_ratio = round(orig_w / orig_h, 4)

    # ── Attempt real PIL resize if file exists ────────────────────────────────
    if _PIL_AVAILABLE and image_path_raw:
        abs_path = Path(image_path_raw)
        if not abs_path.is_absolute():
            abs_path = config.project_root / abs_path

        if abs_path.exists():
            if aspect_ratio is None:
                try:
                    with PILImage.open(abs_path) as img:
                        w, h = img.size
                        aspect_ratio = round(w / h, 4) if h > 0 else None
                except Exception as exc:
                    warnings.append(f"Cannot read image dimensions: {exc}")

            resized = True
            modified = True
            logger.debug(f"Resize: {rec.get('image_id')} → {target_w}×{target_h}")
        else:
            warnings.append(f"Image file not found for resize: {abs_path}")

    # ── Write dimension metadata (always, even without PIL) ───────────────────
    if rec.get("image_width") != target_w or rec.get("image_height") != target_h:
        modified = True
    rec["image_width"]           = target_w
    rec["image_height"]          = target_h
    rec["aspect_ratio_original"] = aspect_ratio
    rec["image_resized"]         = resized

    return StageResult(record=rec, modified=modified, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Image Normalization
# ─────────────────────────────────────────────────────────────────────────────

def stage2_image_normalize(
    record: Dict[str, Any],
    config: PipelineConfig,
) -> StageResult:
    """
    Stage 2 — Image Normalization.

    Adds to record:
        pixel_mean_rgb   : [R_mean, G_mean, B_mean]  (per image, 0-255 scale)
        pixel_std_rgb    : [R_std,  G_std,  B_std]
        imagenet_mean    : [0.485, 0.456, 0.406]  (reference, from config)
        imagenet_std     : [0.229, 0.224, 0.225]
        normalization_ok : bool (True if pixel stats were actually computed)

    When the image file is absent or Pillow is unavailable, pixel_mean_rgb
    and pixel_std_rgb are set to None, and normalization_ok = False.
    The ImageNet reference values are always written.

    Args:
        record : Record dict (after Stage 1).
        config : PipelineConfig with imagenet_mean, imagenet_std.

    Returns:
        StageResult with normalization metadata added.
    """
    rec      = dict(record)
    warnings : List[str] = []
    modified  = True

    # Always write ImageNet reference values (for downstream convenience)
    rec["imagenet_mean"] = list(config.imagenet_mean)
    rec["imagenet_std"]  = list(config.imagenet_std)
    rec["pixel_mean_rgb"] = None
    rec["pixel_std_rgb"]  = None
    rec["normalization_ok"] = False

    image_path_raw: str = rec.get("image_path") or ""
    if not _PIL_AVAILABLE or not image_path_raw:
        warnings.append("Pixel normalization skipped — no Pillow or no image_path")
        return StageResult(record=rec, modified=modified, warnings=warnings)

    abs_path = Path(image_path_raw)
    if not abs_path.is_absolute():
        abs_path = config.project_root / abs_path

    if not abs_path.exists():
        warnings.append(f"Image not found for normalization: {abs_path}")
        return StageResult(record=rec, modified=modified, warnings=warnings)

    try:
        import numpy as np
        with PILImage.open(abs_path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
        # Per-channel mean and std in [0, 255] space
        pixel_mean = arr.mean(axis=(0, 1)).tolist()  # [R, G, B]
        pixel_std  = arr.std(axis=(0, 1)).tolist()
        rec["pixel_mean_rgb"]  = [round(v, 3) for v in pixel_mean]
        rec["pixel_std_rgb"]   = [round(v, 3) for v in pixel_std]
        rec["normalization_ok"] = True
    except Exception as exc:
        warnings.append(f"Pixel statistics computation failed: {exc}")

    return StageResult(record=rec, modified=modified, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Duplicate Image Detection
# ─────────────────────────────────────────────────────────────────────────────

def build_dedup_hash(record: Dict[str, Any], strategy: str = "path_hash") -> str:
    """
    Build a deduplication hash for a record.

    Strategies:
      'path_hash'  : MD5 of normalised image_path string (fast, no I/O).
                     Two records with the same normalised path are duplicates.
      'content_key': MD5 of category + normalised path (stricter).

    Args:
        record   : Fashion item dict.
        strategy : 'path_hash' or 'content_key'.

    Returns:
        Hex MD5 string.
    """
    path = (record.get("image_path") or "").lower().replace("\\", "/").strip()

    if strategy == "content_key":
        category = (record.get("category") or "").lower().strip()
        key = f"{category}::{path}"
    else:
        key = path

    return hashlib.md5(key.encode("utf-8")).hexdigest()


def stage3_dedup(
    records  : List[Dict[str, Any]],
    config   : PipelineConfig,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, str]]:
    """
    Stage 3 — Duplicate Image Detection.

    Processes the entire batch at once (O(n) dict-based dedup).

    Args:
        records  : List of record dicts (after Stage 1 + 2).
        config   : PipelineConfig.

    Returns:
        Tuple of:
            deduped_records  : List[Dict] without duplicates
            dup_count        : Number of duplicate records removed
            dup_map          : {image_id → kept_image_id} for removed records
    """
    seen_hashes: Dict[str, str] = {}   # hash → image_id of first seen
    kept        : List[Dict[str, Any]] = []
    dup_map     : Dict[str, str]       = {}
    dup_count   : int                  = 0

    for rec in records:
        h      = build_dedup_hash(rec, config.dedup_strategy)
        img_id = str(rec.get("image_id", ""))

        if h in seen_hashes:
            # Duplicate found
            dup_count += 1
            dup_map[img_id] = seen_hashes[h]
            logger.debug(
                f"Duplicate removed: {img_id} → same as {seen_hashes[h]}"
            )
            # Mark original record as having a duplicate
            continue
        else:
            seen_hashes[h] = img_id
            rec = dict(rec)  # copy
            rec["dedup_hash"] = h
            kept.append(rec)

    logger.info(
        f"Stage 3 dedup: {len(records)} in → {len(kept)} kept, "
        f"{dup_count} duplicates removed"
    )
    return kept, dup_count, dup_map


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Description Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_description(text: Optional[str], config: PipelineConfig) -> str:
    """
    Apply all description-cleaning transforms to a single text string.

    Transforms (in order):
      1. None / non-string → empty string
      2. HTML entity decode (&amp; → &)
      3. HTML tag strip (<br/> → "")
      4. Control character removal
      5. Unicode NFC normalization
      6. Multi-whitespace collapse
      7. Leading/trailing strip
      8. Repeated punctuation collapse (!!! → !)
      9. Optional lowercase
      10. Truncate to max_description_chars

    Args:
        text   : Raw description string.
        config : PipelineConfig with lowercase_description, max_description_chars.

    Returns:
        Cleaned description string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # ── HTML entities ─────────────────────────────────────────────────────────
    text = html.unescape(text)

    # ── HTML tags ─────────────────────────────────────────────────────────────
    text = _HTML_TAG_RE.sub(" ", text)

    # ── Control characters ────────────────────────────────────────────────────
    text = _CONTROL_CHAR_RE.sub("", text)

    # ── Unicode NFC ───────────────────────────────────────────────────────────
    text = unicodedata.normalize("NFC", text)

    # ── Collapse whitespace ───────────────────────────────────────────────────
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # ── Repeated punctuation ──────────────────────────────────────────────────
    text = _MULTI_PUNCT_RE.sub(r"\1", text)

    # ── Optional lowercase ────────────────────────────────────────────────────
    if config.lowercase_description:
        text = text.lower()

    # ── Truncate ──────────────────────────────────────────────────────────────
    if len(text) > config.max_description_chars:
        text = text[: config.max_description_chars].rstrip() + "…"

    return text


def stage4_clean_description(
    record: Dict[str, Any],
    config: PipelineConfig,
) -> StageResult:
    """
    Stage 4 — Description Cleaning.

    Applies clean_description() and writes:
        description       : cleaned text
        description_raw   : original text (preserved for audit)
        description_length: len(cleaned text)
        description_empty : True if cleaned text is too short
        description_cleaned: True if any change was made

    Args:
        record : Fashion item dict.
        config : PipelineConfig.

    Returns:
        StageResult with cleaned description.
    """
    rec      = dict(record)
    warnings : List[str] = []
    modified  = False

    raw_desc     = rec.get("description") or ""
    cleaned_desc = clean_description(raw_desc, config)

    if cleaned_desc != raw_desc:
        modified = True

    rec["description_raw"]    = raw_desc
    rec["description"]        = cleaned_desc
    rec["description_length"] = len(cleaned_desc)
    rec["description_cleaned"] = modified

    if len(cleaned_desc) < config.min_description_chars:
        rec["description_empty"] = True
        warnings.append(
            f"Description too short after cleaning: '{cleaned_desc[:40]}'"
        )
    else:
        rec["description_empty"] = False

    return StageResult(record=rec, modified=modified, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Attribute Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_value(
    raw     : Optional[str],
    alias_map: Dict[str, str],
    valid_set: FrozenSet[str],
) -> Tuple[Optional[str], bool]:
    """
    Normalize a single string value using an alias lookup table.

    Lookup order:
      1. Exact match in valid_set            → return as-is
      2. Lowercase match in alias_map        → return canonical value
      3. Lowercase match in valid_set        → return lowercase version
      4. No match                            → return None, ok=False

    Args:
        raw       : Raw attribute string.
        alias_map : Dict mapping variant → canonical.
        valid_set : Set of canonical values.

    Returns:
        (canonical_value, is_known)
    """
    if raw is None:
        return None, True  # None is ok — field was simply absent

    stripped = raw.strip()
    if not stripped:
        return None, True

    lower = stripped.lower()

    # Direct match
    if stripped in valid_set:
        return stripped, True
    if lower in valid_set:
        return lower, True

    # Alias lookup
    if lower in alias_map:
        return alias_map[lower], True

    # Partial substring match (e.g. "oversized hoodie" → "oversized")
    for alias, canonical in alias_map.items():
        if alias in lower:
            return canonical, True

    return stripped, False  # unknown but preserved


def normalize_list_field(
    items     : Optional[List[str]],
    alias_map : Dict[str, str],
    valid_set : FrozenSet[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Normalize a list field (e.g. color, occasions, fabric).

    Args:
        items    : List of raw strings.
        alias_map: Alias lookup table.
        valid_set: Set of valid canonical values.

    Returns:
        Tuple of:
            normalized  : List of canonicalized values
            unknown     : Values that had no alias match
            warnings    : Warning messages
    """
    if not items or not isinstance(items, list):
        return [], [], []

    normalized: List[str] = []
    unknown   : List[str] = []
    warnings  : List[str] = []

    for item in items:
        if not isinstance(item, str):
            continue
        canonical, is_known = normalize_value(item, alias_map, valid_set)
        if canonical is not None:
            normalized.append(canonical)
        if not is_known:
            unknown.append(item)
            warnings.append(f"Unknown attribute value kept as-is: '{item}'")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped = []
    for v in normalized:
        if v not in seen:
            seen.add(v)
            deduped.append(v)

    return deduped, unknown, warnings


def stage5_normalize_attributes(
    record: Dict[str, Any],
    config: PipelineConfig,
) -> StageResult:
    """
    Stage 5 — Attribute Normalization.

    Normalizes these list/scalar fields:
        color, fabric, pattern, occasion  (list fields)
        style, fit, season, gender        (scalar fields)

    Writes:
        {field}_raw      : original values (if changed)
        {field}          : normalized canonical values
        normalization_warnings: list of unknown values found

    Args:
        record : Fashion item dict.
        config : PipelineConfig.

    Returns:
        StageResult with normalized attributes.
    """
    rec       = dict(record)
    warnings  : List[str] = []
    modified   = False
    norm_log  : Dict[str, Any] = {}

    # ── List fields ───────────────────────────────────────────────────────────
    LIST_FIELDS = [
        ("color",    _COLOR_ALIASES,    frozenset()),  # No strict valid_set for colors
        ("occasion", _OCCASION_ALIASES, VALID_OCCASIONS),
    ]
    for field_name, alias_map, valid_set in LIST_FIELDS:
        raw_items = rec.get(field_name)
        if not isinstance(raw_items, list):
            raw_items = [raw_items] if raw_items else []

        norm_items, unknowns, field_warns = normalize_list_field(
            raw_items, alias_map, valid_set
        )

        if norm_items != raw_items:
            modified = True
            rec[f"{field_name}_raw"] = raw_items
            rec[field_name]          = norm_items

        if unknowns:
            norm_log[field_name] = {"unknown_values": unknowns}
        warnings.extend(field_warns)

    # ── Scalar fields ─────────────────────────────────────────────────────────
    SCALAR_FIELDS = [
        ("style",  _STYLE_ALIASES,  VALID_STYLES),
        ("fit",    _FIT_ALIASES,    VALID_FITS),
        ("season", _SEASON_ALIASES, VALID_SEASONS),
        ("gender", _GENDER_ALIASES, VALID_GENDERS),
    ]
    for field_name, alias_map, valid_set in SCALAR_FIELDS:
        raw_val = rec.get(field_name)
        if raw_val is None or raw_val == "":
            continue

        canonical, is_known = normalize_value(
            str(raw_val), alias_map, valid_set
        )
        if canonical != raw_val:
            modified = True
            rec[f"{field_name}_raw"] = raw_val
            rec[field_name]          = canonical

        if not is_known:
            warnings.append(
                f"Unknown {field_name} value preserved: '{raw_val}'"
            )
            norm_log[field_name] = {"unknown": raw_val}

    if config.keep_normalization_log:
        rec["normalization_log"] = norm_log

    norm_warn_count = len(warnings)
    if norm_warn_count > 0:
        rec["normalization_warnings"] = warnings

    return StageResult(record=rec, modified=modified, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Category Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_category(raw_category: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Map a raw category string to a canonical taxonomy key.

    Lookup order:
      1. Already in VALID_CATEGORIES            → return as-is
      2. Lowercase match in VALID_CATEGORIES    → return lowercase
      3. Match in _CATEGORY_ALIASES             → return canonical
      4. No match                               → return None, ok=False

    Args:
        raw_category : Raw category string from any dataset.

    Returns:
        (canonical_category, is_known)
    """
    if not raw_category or not isinstance(raw_category, str):
        return None, False

    stripped = raw_category.strip()
    lower    = stripped.lower()

    if stripped in VALID_CATEGORIES:
        return stripped, True
    if lower in VALID_CATEGORIES:
        return lower, True
    if lower in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[lower], True

    # Substring scan of alias keys
    for alias, canonical in _CATEGORY_ALIASES.items():
        if alias in lower:
            return canonical, True

    return None, False


def stage6_normalize_category(
    record: Dict[str, Any],
    config: PipelineConfig,
) -> StageResult:
    """
    Stage 6 — Category Normalization.

    Maps record["category"] to one of 11 canonical keys.

    Writes:
        category     : canonical key or "uncategorized"
        category_raw : original value (if changed)
        category_known: True if mapping was successful

    Args:
        record : Fashion item dict.
        config : PipelineConfig.

    Returns:
        StageResult. If drop_unknown_categories=True and category is unknown,
        the StageResult.dropped flag is set.
    """
    rec      = dict(record)
    warnings : List[str] = []
    modified  = False

    raw_cat        = rec.get("category")
    canonical, ok  = normalize_category(raw_cat)

    if not ok:
        warnings.append(
            f"Could not map category '{raw_cat}' to taxonomy — "
            f"marked as 'uncategorized'"
        )
        if raw_cat != "uncategorized":
            modified = True
            rec["category_raw"]  = raw_cat
            rec["category"]      = "uncategorized"
        rec["category_known"] = False

        if config.drop_unknown_categories:
            return StageResult(
                record=rec, modified=modified, warnings=warnings,
                dropped=True, drop_reason=f"Unknown category: '{raw_cat}'"
            )
    else:
        rec["category_known"] = True
        if canonical != raw_cat:
            modified = True
            rec["category_raw"] = raw_cat
            rec["category"]     = canonical

    return StageResult(record=rec, modified=modified, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Dataset Balancing Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_balance_stats(
    records: List[Dict[str, Any]],
    config : PipelineConfig,
) -> Dict[str, Any]:
    """
    Stage 7 — Dataset Balancing Statistics.

    Computes frequency distributions and balance recommendations for:
        category, gender, source_dataset, season, style, fit

    For each field:
        counts    : {value: count}
        shares    : {value: percentage}
        max_count : int
        min_count : int
        imbalance_ratio: max/min (1.0 = perfectly balanced)

    Balance recommendation:
        For each class, target = max(min_count, balance_target_per_class).
        oversample: classes below target.
        undersample: classes above target.

    Args:
        records : Cleaned record dicts.
        config  : PipelineConfig with balance_target_per_class.

    Returns:
        Dict with per-field statistics and a top-level "recommended_balance" key.
    """
    n = len(records)
    if n == 0:
        return {"total_records": 0, "note": "Empty dataset"}

    def _field_stats(field_name: str, all_records: List[Dict]) -> Dict[str, Any]:
        """Compute distribution stats for a single field."""
        counter: Counter = Counter()
        for rec in all_records:
            val = rec.get(field_name)
            if isinstance(val, list):
                for v in val:
                    counter[str(v)] += 1
            elif val is not None and val != "":
                counter[str(val)] += 1
            else:
                counter["_missing_"] += 1

        total_values = sum(counter.values())
        counts = dict(counter.most_common())
        shares = {
            k: round(v / total_values * 100, 2)
            for k, v in counts.items()
        }

        non_missing_vals = {k: v for k, v in counts.items() if k != "_missing_"}
        min_c  = min(non_missing_vals.values()) if non_missing_vals else 0
        max_c  = max(non_missing_vals.values()) if non_missing_vals else 0
        ratio  = round(max_c / min_c, 2) if min_c > 0 else float("inf")

        return {
            "counts"          : counts,
            "shares_pct"      : shares,
            "min_count"       : min_c,
            "max_count"       : max_c,
            "imbalance_ratio" : ratio,
        }

    fields_to_analyze = [
        "category", "gender", "source_dataset",
        "season", "style", "fit",
    ]
    stats: Dict[str, Any] = {
        "total_records": n,
    }
    for fname in fields_to_analyze:
        stats[fname] = _field_stats(fname, records)

    # ── Balance recommendation ────────────────────────────────────────────────
    cat_counts = stats["category"]["counts"]
    non_missing = {k: v for k, v in cat_counts.items() if k != "_missing_"}

    target = config.balance_target_per_class
    if target is None and non_missing:
        target = min(non_missing.values())

    recommendations: Dict[str, Any] = {
        "balance_target_per_category": target,
        "categories": {}
    }
    for cat, count in sorted(non_missing.items()):
        if target:
            if count < target:
                action = "oversample"
                delta  = target - count
            elif count > target:
                action = "undersample"
                delta  = count - target
            else:
                action = "balanced"
                delta  = 0
        else:
            action = "n/a"
            delta  = 0

        recommendations["categories"][cat] = {
            "current_count": count,
            "target_count" : target,
            "action"       : action,
            "delta"        : delta,
        }

    stats["recommended_balance"] = recommendations
    return stats


# =============================================================================
# ── 5. PreprocessingPipeline — Main Orchestrator
# =============================================================================

class PreprocessingPipeline:
    """
    Orchestrates all 7 preprocessing stages on a batch of fashion records.

    Usage:
        pipeline = PreprocessingPipeline(config=PipelineConfig())
        result   = pipeline.run(records)
        pipeline.save(result, "datasets/processed/clean_dataset.json")

    Stage execution order:
        1. stage1_image_resize          — resize + record dimensions
        2. stage2_image_normalize       — pixel statistics
        3. stage3_dedup                 — deduplication (batch-level)
        4. stage4_clean_description     — text cleaning
        5. stage5_normalize_attributes  — attribute canonicalization
        6. stage6_normalize_category    — category taxonomy mapping
        7. compute_balance_stats        — distribution analysis
    """

    def __init__(
        self,
        config      : Optional[PipelineConfig] = None,
        project_root: Optional[str | Path]     = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if project_root:
            self.config.project_root = Path(project_root)

        logger.info(
            f"PreprocessingPipeline v{_PIPELINE_VERSION} initialised | "
            f"target_size={self.config.target_size} | "
            f"dedup={self.config.dedup_strategy} | "
            f"drop_unknown_cat={self.config.drop_unknown_categories}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        records: List[Dict[str, Any]],
    ) -> PipelineRunResult:
        """
        Execute the full 7-stage preprocessing pipeline on a list of records.

        Args:
            records : Raw fashion item dicts (from loaders or validator).

        Returns:
            PipelineRunResult with cleaned records and all statistics.
        """
        t_start = time.perf_counter()
        result  = PipelineRunResult(total_input=len(records))

        logger.info(
            f"Pipeline run started | input={len(records):,} records"
        )

        # ── Stage 1 + 2: Per-record image stages ──────────────────────────────
        stage12_records : List[Dict[str, Any]] = []
        total_warnings  = 0

        for i, rec in enumerate(records):
            # Stage 1: Image resize
            s1 = stage1_image_resize(rec, self.config)
            total_warnings += len(s1.warnings)

            # Stage 2: Image normalize
            s2 = stage2_image_normalize(s1.record, self.config)
            total_warnings += len(s2.warnings)

            stage12_records.append(s2.record)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Stage 1+2 progress: {i+1:,}/{len(records):,}")

        logger.info(f"Stage 1+2 complete | {len(stage12_records):,} records")

        # ── Stage 3: Deduplication (batch-level) ──────────────────────────────
        deduped_records, dup_count, dup_map = stage3_dedup(
            stage12_records, self.config
        )
        result.duplicates_removed = dup_count

        logger.info(
            f"Stage 3 complete | {len(deduped_records):,} unique records"
        )

        # ── Stage 4-6: Per-record text + taxonomy stages ───────────────────────
        cleaned_records : List[Dict[str, Any]] = []
        uncategorized   = 0
        dropped         = 0

        for rec in deduped_records:
            # Stage 4: Description cleaning
            s4 = stage4_clean_description(rec, self.config)
            total_warnings += len(s4.warnings)

            # Stage 5: Attribute normalization
            s5 = stage5_normalize_attributes(s4.record, self.config)
            total_warnings += len(s5.warnings)

            # Stage 6: Category normalization
            s6 = stage6_normalize_category(s5.record, self.config)
            total_warnings += len(s6.warnings)

            if s6.dropped:
                dropped += 1
                logger.debug(
                    f"Dropped: {rec.get('image_id')} — {s6.drop_reason}"
                )
                continue

            if s6.record.get("category") == "uncategorized":
                uncategorized += 1

            cleaned_records.append(s6.record)

        result.dropped_count  = dropped
        result.uncategorized  = uncategorized
        result.warning_count  = total_warnings
        result.total_output   = len(cleaned_records)

        logger.info(
            f"Stage 4-6 complete | {len(cleaned_records):,} clean records | "
            f"uncategorized={uncategorized} | dropped={dropped}"
        )

        # ── Stage 7: Dataset balancing statistics ─────────────────────────────
        result.balance_stats = compute_balance_stats(cleaned_records, self.config)
        result.records       = cleaned_records
        result.processing_time_s = round(time.perf_counter() - t_start, 3)

        logger.success(
            f"Pipeline complete | "
            f"input={result.total_input:,} | output={result.total_output:,} | "
            f"dups_removed={result.duplicates_removed} | "
            f"uncategorized={result.uncategorized} | "
            f"time={result.processing_time_s:.3f}s"
        )
        result.print_summary()
        return result

    def save(
        self,
        run_result  : PipelineRunResult,
        output_path : str | Path,
        indent      : int = 2,
    ) -> Path:
        """
        Save the pipeline result to a clean_dataset.json file.

        JSON structure (spec-compliant):
            {
              "generated_at"    : "ISO-8601 UTC",
              "schema_version"  : "1.0.0",
              "pipeline_config" : { ... },
              "summary"         : { total_input, total_output, ... },
              "balance_stats"   : { ... },
              "records"         : [ ... cleaned record dicts ... ]
            }

        Args:
            run_result  : PipelineRunResult from pipeline.run().
            output_path : Path to write the JSON file.
            indent      : JSON indent level.

        Returns:
            Resolved absolute Path to written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "generated_at"   : run_result.generated_at,
            "schema_version" : _PIPELINE_VERSION,
            "pipeline_config": self.config.to_dict(),
            "summary"        : run_result.summary_dict(),
            "balance_stats"  : run_result.balance_stats,
            "records"        : run_result.records,
        }

        path.write_text(
            json.dumps(output, indent=indent, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        size_kb = path.stat().st_size / 1024
        logger.success(
            f"clean_dataset.json saved | {path} | "
            f"{size_kb:.1f} KB | {len(run_result.records):,} records"
        )
        return path.resolve()

    # ── Convenience: run from file paths ──────────────────────────────────────

    def run_from_json(
        self,
        *input_paths: str | Path,
    ) -> PipelineRunResult:
        """
        Load records from one or more JSON files and run the pipeline.

        Each file must contain either:
          - A list of record dicts at the top level, OR
          - A dict with a "records" key containing the list.

        Args:
            *input_paths : One or more paths to JSON files.

        Returns:
            PipelineRunResult.
        """
        all_records: List[Dict[str, Any]] = []

        for path in input_paths:
            p = Path(path)
            if not p.exists():
                logger.warning(f"Input file not found: {p}")
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict) and "records" in data:
                    all_records.extend(data["records"])
                else:
                    logger.warning(f"Unrecognised JSON structure in {p}")
            except Exception as exc:
                logger.error(f"Failed to load {p}: {exc}")

        logger.info(f"Loaded {len(all_records):,} records from {len(input_paths)} file(s)")
        return self.run(all_records)
