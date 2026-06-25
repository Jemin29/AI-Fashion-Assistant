"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/schema/fashion_schema.py
=============================================================================
MODULE  : Unified Fashion Dataset Schema
WEEK    : 1 — Fashion Domain Research & Dataset Curation
AUTHOR  : Fashion AI Team

PURPOSE
-------
A single, authoritative Pydantic schema that can store and validate fashion
data from ANY dataset in the pipeline:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Source Dataset       Native Record          → Unified Schema       │
  ├─────────────────────────────────────────────────────────────────────┤
  │  FashionGen           FashionGenRecord        → UnifiedFashionItem  │
  │  DeepFashion          DeepFashionRecord        → UnifiedFashionItem  │
  │  (future)             Any dataset              → UnifiedFashionItem  │
  └─────────────────────────────────────────────────────────────────────┘

SCHEMA DESIGN DECISIONS
-----------------------
  • Pydantic v2  — model_validator, field_validator, model_config
  • All taxonomy fields use Literal / Enum types derived from the KB constants
    (VALID_CATEGORIES, VALID_GENDERS, VALID_STYLES, etc.) to catch violations
    at construction time, not at query time.
  • Optional fields have safe defaults (None / []) so partially-annotated
    records from DeepFashion (no description, no gender) are still representable.
  • Two factory class-methods: from_fashiongen() and from_deepfashion() to
    create UnifiedFashionItem without manual field mapping.
  • Serialisation: to_dict(), to_json(), to_jsonl_line()
  • Deserialisation: from_dict(), from_json()
  • Schema documentation: schema_json() returns the full JSON Schema string.
  • FashionDatasetBatch: batch container for multi-record operations.

FIELD TAXONOMY (aligned with fashion_domain_research.py)
---------------------------------------------------------
  category       : 11 values (t_shirts, shirts, hoodies, jackets, pants,
                               jeans, shorts, dresses, ethnic_wear,
                               footwear, accessories)
  gender         : men | women | unisex
  style          : streetwear | luxury | formal | business_casual |
                   techwear | minimalist | vintage | athleisure
  fit            : slim_fit | regular_fit | relaxed_fit | oversized |
                   cropped | skinny | straight | athletic_fit
  season         : spring | summer | autumn | winter | all_season
  occasion       : casual | business_casual | formal | party |
                   sport | outdoor | beach | wedding_festive | lounge
  source_dataset : fashiongen | deepfashion

ARCHITECTURE
------------
  UnifiedFashionItem        — The primary schema model (Pydantic BaseModel)
  ├── LandmarkPoint         — Embedded model for landmark coordinates
  ├── BoundingBox           — Embedded model for bounding box
  ├── DatasetSource         — Enum: fashiongen | deepfashion
  ├── GenderEnum            — Enum: men | women | unisex
  ├── CategoryEnum          — Enum: 11 categories
  ├── StyleEnum             — Enum: 8 style hierarchy values
  ├── FitEnum               — Enum: 8 fit types
  ├── SeasonEnum            — Enum: 5 seasons
  ├── OccasionEnum          — Enum: 9 occasions
  ├── SchemaVersion         — Versioned metadata for the schema
  └── FashionDatasetBatch   — Container for a batch of UnifiedFashionItem

USAGE
-----
  # Build from FashionGen
  from src.data.schema.fashion_schema import UnifiedFashionItem, DatasetSource
  item = UnifiedFashionItem.from_fashiongen(fashiongen_record)

  # Build from DeepFashion
  item = UnifiedFashionItem.from_deepfashion(deepfashion_record)

  # Build manually
  item = UnifiedFashionItem(
      image_id    = "MY_001",
      image_path  = "datasets/img/MY_001.jpg",
      category    = "shirts",
      source_dataset = "fashiongen",
  )

  # Serialise
  d    = item.to_dict()
  js   = item.to_json(indent=2)
  line = item.to_jsonl_line()

  # Deserialise
  item2 = UnifiedFashionItem.from_dict(d)
  item3 = UnifiedFashionItem.from_json(js)

  # Schema docs
  print(UnifiedFashionItem.get_schema_doc())
  print(UnifiedFashionItem.model_json_schema())

  # Batch
  batch = FashionDatasetBatch(items=[item, item2])
  batch.save_jsonl("datasets/processed/unified.jsonl")

  # Validation report
  result = item.validate_and_report()
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

# ─── Pydantic v2 ──────────────────────────────────────────────────────────────
try:
    from pydantic import (
        BaseModel,
        Field,
        field_validator,
        model_validator,
        ConfigDict,
    )
    import pydantic
    _PYDANTIC_VERSION = int(pydantic.__version__.split(".")[0])
    if _PYDANTIC_VERSION < 2:
        raise ImportError(
            f"Pydantic v2+ required. Found v{pydantic.__version__}. "
            "Upgrade: pip install 'pydantic>=2.0'"
        )
except ImportError as exc:
    raise ImportError(
        f"Pydantic v2 is required for fashion_schema.py.\n"
        f"Install: pip install 'pydantic>=2.0'\n"
        f"Original error: {exc}"
    ) from exc

# ─── Resolve project root ──────────────────────────────────────────────────────
_FILE_DIR      = Path(__file__).resolve().parent         # data_pipeline/schema/
_PROJECT_ROOT  = _FILE_DIR.parent.parent                 # fashion-ai-assistant/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Loguru (graceful fallback to standard logging) ───────────────────────────
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ─── Schema version constant ───────────────────────────────────────────────────
_SCHEMA_VERSION = "1.0.0"
_SCHEMA_DATE    = "2026-06-03"

# =============================================================================
# ── 1. Enumerations  (taxonomy-locked, aligned with fashion_domain_research.py)
# =============================================================================

class DatasetSource(str, Enum):
    """
    Identifies the originating dataset of a fashion item record.

    Values are lowercase strings matching the dataset_source fields used
    in FashionGenRecord and DeepFashionRecord.

    Examples:
        DatasetSource.FASHIONGEN  → "fashiongen"
        DatasetSource.DEEPFASHION → "deepfashion"
    """
    FASHIONGEN  = "fashiongen"
    DEEPFASHION = "deepfashion"


class GenderEnum(str, Enum):
    """
    Gender taxonomy key. Aligned with VALID_GENDERS in fashion_domain_research.py.

    All dataset-specific gender strings (e.g. "Men", "Women", "Boys") must
    be pre-normalised to these keys before populating a UnifiedFashionItem.
    """
    MEN    = "men"
    WOMEN  = "women"
    UNISEX = "unisex"


class CategoryEnum(str, Enum):
    """
    11-key product category taxonomy. Aligned with VALID_CATEGORIES in
    fashion_domain_research.py.

    Maps to both FashionGen and DeepFashion normalised categories via their
    respective _CATEGORY_MAP / _DF_CATEGORY_MAP constants.
    """
    T_SHIRTS   = "t_shirts"
    SHIRTS     = "shirts"
    HOODIES    = "hoodies"
    JACKETS    = "jackets"
    PANTS      = "pants"
    JEANS      = "jeans"
    SHORTS     = "shorts"
    DRESSES    = "dresses"
    ETHNIC_WEAR = "ethnic_wear"
    FOOTWEAR   = "footwear"
    ACCESSORIES = "accessories"


class StyleEnum(str, Enum):
    """
    8-value style hierarchy. Aligned with VALID_STYLES in fashion_domain_research.py.

    FashionGen infers style from description keywords (_STYLE_KEYWORDS map).
    DeepFashion records may leave this field empty (style is not annotated
    in DeepFashion annotations).
    """
    STREETWEAR      = "streetwear"
    LUXURY          = "luxury"
    FORMAL          = "formal"
    BUSINESS_CASUAL = "business_casual"
    TECHWEAR        = "techwear"
    MINIMALIST      = "minimalist"
    VINTAGE         = "vintage"
    ATHLEISURE      = "athleisure"


class FitEnum(str, Enum):
    """
    8-value fit taxonomy. Aligned with VALID_FITS in fashion_domain_research.py.
    """
    SLIM_FIT     = "slim_fit"
    REGULAR_FIT  = "regular_fit"
    RELAXED_FIT  = "relaxed_fit"
    OVERSIZED    = "oversized"
    CROPPED      = "cropped"
    SKINNY       = "skinny"
    STRAIGHT     = "straight"
    ATHLETIC_FIT = "athletic_fit"


class SeasonEnum(str, Enum):
    """
    5-value season taxonomy. Aligned with VALID_SEASONS in fashion_domain_research.py.

    FashionGen infers season from description text (_SEASON_KEYWORDS).
    DeepFashion does not annotate season; use ALL_SEASON as default.
    """
    SPRING     = "spring"
    SUMMER     = "summer"
    AUTUMN     = "autumn"
    WINTER     = "winter"
    ALL_SEASON = "all_season"


class OccasionEnum(str, Enum):
    """
    9-value occasion taxonomy. Aligned with VALID_OCCASIONS in fashion_domain_research.py.
    """
    CASUAL           = "casual"
    BUSINESS_CASUAL  = "business_casual"
    FORMAL           = "formal"
    PARTY            = "party"
    SPORT            = "sport"
    OUTDOOR          = "outdoor"
    BEACH            = "beach"
    WEDDING_FESTIVE  = "wedding_festive"
    LOUNGE           = "lounge"


# =============================================================================
# ── 2. Embedded Sub-Models
# =============================================================================

class LandmarkPoint(BaseModel):
    """
    A single 2D clothing landmark point with visibility flag.

    Coordinates are normalised to [0, 1] relative to the bounding box
    (or the 256 × 256 reference size when no bbox is available).
    Invisible landmarks have x=0.0, y=0.0, visible=False.

    Used exclusively by DeepFashion records — 6 points per item:
      left_collar, right_collar, left_sleeve, right_sleeve, left_hem, right_hem

    FashionGen records will have an empty landmarks list.
    """

    model_config = ConfigDict(
        frozen         = True,     # Immutable once created
        extra          = "forbid",
        json_schema_extra = {
            "description": "Normalised 2D clothing landmark point",
            "example"    : {"name": "left_collar", "x": 0.32, "y": 0.05, "visible": True},
        },
    )

    name   : str   = Field(..., description="Landmark name, e.g. 'left_collar'")
    x      : float = Field(..., ge=0.0, le=1.0, description="Normalised x-coordinate [0,1]")
    y      : float = Field(..., ge=0.0, le=1.0, description="Normalised y-coordinate [0,1]")
    visible: bool  = Field(..., description="True if landmark is visible in the image")

    @field_validator("name")
    @classmethod
    def name_must_be_known(cls, v: str) -> str:
        """Validate landmark name is one of the 6 standard DeepFashion landmarks."""
        _VALID_LANDMARKS = {
            "left_collar", "right_collar",
            "left_sleeve", "right_sleeve",
            "left_hem",    "right_hem",
        }
        if v not in _VALID_LANDMARKS:
            # Warning only — we allow custom landmarks from future datasets
            logger.debug(
                f"Non-standard landmark name: '{v}'. "
                f"Expected one of {sorted(_VALID_LANDMARKS)}"
            )
        return v


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box for the clothing item in the image.

    Pixel coordinates (x1, y1) are the top-left corner.
    (x2, y2) are the bottom-right corner.
    x2 must be > x1; y2 must be > y1.

    Normalised version (nx1, ny1, nx2, ny2) is in [0,1] space
    relative to the 256 × 256 DeepFashion reference image size.

    FashionGen records will have no bounding box.
    """

    model_config = ConfigDict(
        frozen         = True,
        extra          = "forbid",
        json_schema_extra = {
            "description": "Clothing item bounding box in pixel coordinates",
            "example"    : {"x1": 50, "y1": 30, "x2": 206, "y2": 230},
        },
    )

    x1: int   = Field(..., ge=0, description="Left edge pixel coordinate")
    y1: int   = Field(..., ge=0, description="Top edge pixel coordinate")
    x2: int   = Field(..., ge=0, description="Right edge pixel coordinate")
    y2: int   = Field(..., ge=0, description="Bottom edge pixel coordinate")

    # Normalised [0,1] coordinates (optional — populated when available)
    nx1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalised left edge")
    ny1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalised top edge")
    nx2: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalised right edge")
    ny2: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalised bottom edge")

    @model_validator(mode="after")
    def validate_box_dimensions(self) -> "BoundingBox":
        """Validate that x2 > x1 and y2 > y1 (non-degenerate box)."""
        if self.x2 <= self.x1:
            raise ValueError(
                f"Invalid bounding box: x2 ({self.x2}) must be > x1 ({self.x1})"
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"Invalid bounding box: y2 ({self.y2}) must be > y1 ({self.y1})"
            )
        return self

    @property
    def width(self) -> int:
        """Pixel width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Pixel height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Pixel area of the bounding box."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio of the bounding box."""
        return round(self.width / self.height, 4)


class SchemaVersion(BaseModel):
    """
    Versioned metadata embedded in every UnifiedFashionItem and batch file.

    Enables future schema migrations: downstream consumers can inspect
    the version field and apply appropriate transformation logic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version    : str = Field(default=_SCHEMA_VERSION, description="Schema version string")
    date       : str = Field(default=_SCHEMA_DATE, description="Schema release date")
    description: str = Field(
        default="Unified Fashion Dataset Schema for FashionGen + DeepFashion",
        description="Human-readable schema description",
    )


class ValidationReport(BaseModel):
    """
    Detailed validation report returned by UnifiedFashionItem.validate_and_report().

    Attributes:
        is_valid     : True if no hard errors were raised.
        errors       : List of hard validation failures.
        warnings     : List of soft validation warnings.
        suggestions  : List of improvement suggestions.
        field_coverage : Percentage of optional fields that are populated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid      : bool       = True
    errors        : List[str]  = Field(default_factory=list)
    warnings      : List[str]  = Field(default_factory=list)
    suggestions   : List[str]  = Field(default_factory=list)
    field_coverage: float      = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of optional fields populated (0.0 – 1.0)",
    )


# =============================================================================
# ── 3. UnifiedFashionItem — The Primary Schema Model
# =============================================================================

class UnifiedFashionItem(BaseModel):
    """
    Unified fashion item record. The single canonical schema for all
    fashion data in the AI Fashion Design Assistant pipeline.

    Fields labelled "REQUIRED" must be non-empty for is_valid=True.
    Fields labelled "OPTIONAL" may be None / [] for datasets that do
    not annotate that attribute (e.g. DeepFashion has no gender field).

    ==========================================================================
    FIELD REFERENCE
    ==========================================================================
    ── Identity ──────────────────────────────────────────────────────────────
    image_id       (str)      REQUIRED  Unique ID across all datasets
    image_path     (str)      REQUIRED  Relative path from project root
    source_dataset (str)      REQUIRED  fashiongen | deepfashion

    ── Taxonomy ──────────────────────────────────────────────────────────────
    category       (str)      REQUIRED  11-key product category
    subcategory    (str|None) OPTIONAL  Sub-level category label
    gender         (str|None) OPTIONAL  men | women | unisex

    ── Appearance Attributes ─────────────────────────────────────────────────
    color          (List[str])OPTIONAL  Normalised color names
    fabric         (List[str])OPTIONAL  Normalised fabric names
    pattern        (List[str])OPTIONAL  Pattern type strings
    fit            (str|None) OPTIONAL  slim_fit | regular_fit | ...

    ── Contextual Attributes ─────────────────────────────────────────────────
    style          (str|None) OPTIONAL  8-key style hierarchy value
    season         (str)      OPTIONAL  spring | summer | autumn | winter | all_season
    occasion       (List[str])OPTIONAL  9-key occasion values

    ── Free Text ─────────────────────────────────────────────────────────────
    description    (str|None) OPTIONAL  Human-written or generated text

    ── Raw Attribute List ─────────────────────────────────────────────────────
    attributes     (List[str])OPTIONAL  Free-form attribute label strings
                                        (DeepFashion decoded attribute names)

    ── Spatial Data ──────────────────────────────────────────────────────────
    landmarks      (List[LandmarkPoint]) OPTIONAL  6 clothing landmarks (DeepFashion)
    bounding_box   (BoundingBox|None)    OPTIONAL  Clothing item bbox (DeepFashion)

    ── Pipeline Provenance ────────────────────────────────────────────────────
    is_valid       (bool)     AUTO     Set by validate_and_report()
    errors         (List[str])AUTO     Validation error messages
    warnings       (List[str])AUTO     Validation warning messages
    processed_at   (str)      AUTO     ISO-8601 UTC timestamp
    schema_version (str)      AUTO     "1.0.0"
    ==========================================================================
    """

    # ── Pydantic Configuration ─────────────────────────────────────────────────
    model_config = ConfigDict(
        # Strict validation — unknown fields raise an error (prevents typos)
        extra          = "forbid",
        # Validation runs on assignment too (not just __init__)
        validate_assignment = True,
        # Allow population by field name or alias
        populate_by_name = True,
        # Enable JSON schema with examples
        json_schema_extra = {
            "title"      : "UnifiedFashionItem",
            "description": (
                "Canonical fashion item record for the AI Fashion Design Assistant. "
                "Stores data from FashionGen and DeepFashion in a unified, "
                "taxonomy-aligned format."
            ),
            "example": {
                "image_id"      : "FG_0000042",
                "image_path"    : "datasets/fashiongen/images/FG_0000042.jpg",
                "source_dataset": "fashiongen",
                "category"      : "shirts",
                "subcategory"   : "formal_shirt",
                "gender"        : "men",
                "color"         : ["White", "Blue"],
                "fabric"        : ["Cotton"],
                "pattern"       : ["solid"],
                "fit"           : "slim_fit",
                "style"         : "formal",
                "season"        : "all_season",
                "occasion"      : ["formal", "business_casual"],
                "description"   : "A slim-fit white cotton dress shirt for formal occasions.",
                "attributes"    : [],
                "landmarks"     : [],
                "bounding_box"  : None,
                "is_valid"      : True,
                "errors"        : [],
                "warnings"      : [],
            },
        },
    )

    # ==========================================================================
    # ── Identity Fields (REQUIRED)
    # ==========================================================================

    image_id: str = Field(
        ...,
        min_length = 1,
        description = (
            "Unique identifier for this fashion item across all datasets. "
            "Format: '{PREFIX}_{id}' where PREFIX is FG (FashionGen) or DF (DeepFashion). "
            "Example: 'FG_0000042', 'DF_img_Blouse_img_00000001'"
        ),
        examples = ["FG_0000042", "DF_img_Blouse_img_00000001"],
    )

    image_path: str = Field(
        ...,
        min_length = 1,
        description = (
            "Forward-slash path to the image file, relative to the project root "
            "(fashion-ai-assistant/). "
            "Example: 'datasets/fashiongen/images/FG_0000042.jpg'"
        ),
        examples = [
            "datasets/fashiongen/images/FG_0000042.jpg",
            "datasets/deepfashion/img/Blouse/img_00000001.jpg",
        ],
    )

    source_dataset: DatasetSource = Field(
        ...,
        description = (
            "Originating dataset. Used to apply dataset-specific processing "
            "logic during enrichment and training. "
            "Values: 'fashiongen' | 'deepfashion'"
        ),
    )

    # ==========================================================================
    # ── Taxonomy Fields
    # ==========================================================================

    category: CategoryEnum = Field(
        ...,
        description = (
            "Primary product category. One of 11 taxonomy keys: "
            "t_shirts, shirts, hoodies, jackets, pants, jeans, shorts, "
            "dresses, ethnic_wear, footwear, accessories."
        ),
    )

    subcategory: Optional[str] = Field(
        default = None,
        max_length = 100,
        description = (
            "Optional sub-level category label. "
            "Example: 'graphic_tee' (under t_shirts), 'formal_shirt' (under shirts)."
        ),
    )

    gender: Optional[GenderEnum] = Field(
        default = None,
        description = (
            "Target gender for the clothing item. "
            "None is acceptable when the dataset does not annotate gender "
            "(e.g. DeepFashion category-level annotations). "
            "Values: 'men' | 'women' | 'unisex'"
        ),
    )

    # ==========================================================================
    # ── Appearance Attribute Fields
    # ==========================================================================

    color: List[str] = Field(
        default_factory = list,
        max_length      = 20,
        description     = (
            "List of normalised color names for the item. "
            "Derived from KB alias → canonical color name mapping. "
            "Example: ['Navy', 'White'], ['Olive Green']"
        ),
    )

    fabric: List[str] = Field(
        default_factory = list,
        max_length      = 10,
        description     = (
            "List of normalised fabric names. "
            "Example: ['Cotton', 'Polyester'], ['Denim']"
        ),
    )

    pattern: List[str] = Field(
        default_factory = list,
        max_length      = 10,
        description     = (
            "List of pattern type strings. "
            "Example: ['solid'], ['stripes', 'checked']"
        ),
    )

    fit: Optional[FitEnum] = Field(
        default     = None,
        description = (
            "Fit type. Values: slim_fit | regular_fit | relaxed_fit | "
            "oversized | cropped | skinny | straight | athletic_fit. "
            "FashionGen: inferred from description text. "
            "DeepFashion: inferred from attribute vector."
        ),
    )

    # ==========================================================================
    # ── Contextual Attribute Fields
    # ==========================================================================

    style: Optional[StyleEnum] = Field(
        default     = None,
        description = (
            "Style hierarchy value. "
            "Values: streetwear | luxury | formal | business_casual | "
            "techwear | minimalist | vintage | athleisure. "
            "FashionGen: inferred from _STYLE_KEYWORDS in description. "
            "DeepFashion: typically None (not annotated)."
        ),
    )

    season: SeasonEnum = Field(
        default     = SeasonEnum.ALL_SEASON,
        description = (
            "Season suitability. "
            "Values: spring | summer | autumn | winter | all_season. "
            "FashionGen: inferred from _SEASON_KEYWORDS in description. "
            "DeepFashion: defaults to all_season."
        ),
    )

    occasion: List[OccasionEnum] = Field(
        default_factory = list,
        max_length      = 5,
        description     = (
            "List of suitable occasions. "
            "Values: casual | business_casual | formal | party | "
            "sport | outdoor | beach | wedding_festive | lounge."
        ),
    )

    # ==========================================================================
    # ── Free Text Fields
    # ==========================================================================

    description: Optional[str] = Field(
        default    = None,
        max_length = 4096,
        description = (
            "Human-written product description text. "
            "FashionGen: extracted from HDF5 'input_description' field. "
            "DeepFashion: None (no description annotations)."
        ),
    )

    # ==========================================================================
    # ── Raw Attribute List
    # ==========================================================================

    attributes: List[str] = Field(
        default_factory = list,
        max_length      = 100,
        description     = (
            "Free-form list of attribute label strings. "
            "FashionGen: populated from structured attributes dict. "
            "DeepFashion: decoded from 1000-dim attribute vector "
            "(only +1 = present values included, max 50)."
        ),
    )

    # ==========================================================================
    # ── Spatial Data Fields (DeepFashion-specific)
    # ==========================================================================

    landmarks: List[LandmarkPoint] = Field(
        default_factory = list,
        max_length      = 6,
        description     = (
            "Clothing landmark points. Populated for DeepFashion records. "
            "FashionGen records will always have an empty list. "
            "Each landmark: name, normalised x/y in [0,1], visibility flag."
        ),
    )

    bounding_box: Optional[BoundingBox] = Field(
        default     = None,
        description = (
            "Axis-aligned bounding box for the clothing item. "
            "DeepFashion: from Anno/list_bbox.txt (pixel + normalised). "
            "FashionGen: None (no bbox annotations)."
        ),
    )

    # ==========================================================================
    # ── Pipeline Provenance Fields (AUTO-POPULATED)
    # ==========================================================================

    is_valid: bool = Field(
        default     = False,
        description = (
            "True when the record passes all hard validation rules. "
            "Set automatically by validate_and_report(). "
            "Also pre-set by from_fashiongen() / from_deepfashion() "
            "based on the source record's is_valid flag."
        ),
    )

    errors: List[str] = Field(
        default_factory = list,
        description     = "Hard validation errors from the source pipeline or validate_and_report().",
    )

    warnings: List[str] = Field(
        default_factory = list,
        description     = "Soft validation warnings. Record is kept but flagged.",
    )

    processed_at: str = Field(
        default_factory = lambda: datetime.now(timezone.utc).isoformat(),
        description     = "ISO-8601 UTC timestamp of when this record was created/unified.",
    )

    schema_version: str = Field(
        default     = _SCHEMA_VERSION,
        description = f"Schema version string. Current: {_SCHEMA_VERSION}",
    )

    # ==========================================================================
    # ── Field Validators
    # ==========================================================================

    @field_validator("image_id")
    @classmethod
    def image_id_no_whitespace(cls, v: str) -> str:
        """
        Validate image_id contains no whitespace characters.

        Whitespace in IDs causes problems with file system operations,
        URL construction, and index lookups. Raises ValueError on violation.

        Args:
            v : The image_id string.

        Returns:
            The unchanged image_id string.

        Raises:
            ValueError : If image_id contains whitespace.
        """
        if re.search(r"\s", v):
            raise ValueError(
                f"image_id must not contain whitespace. Got: {v!r}"
            )
        return v

    @field_validator("image_path")
    @classmethod
    def image_path_is_forward_slash(cls, v: str) -> str:
        """
        Normalise image_path to use forward slashes only.

        Converts Windows backslash paths to POSIX-style forward slashes
        for cross-platform consistency. Does NOT check that the file exists
        (the path may refer to a remote or future location).

        Args:
            v : The raw image path string.

        Returns:
            The path with all backslashes replaced by forward slashes.
        """
        return v.replace("\\", "/")

    @field_validator("color", "fabric", "pattern", "attributes", mode="before")
    @classmethod
    def clean_string_lists(cls, v: Any) -> List[str]:
        """
        Sanitise list-of-string fields.

        Ensures that:
          1. Input is always converted to a list (handles None → []).
          2. Each element is a stripped, non-empty string.
          3. Duplicates are removed (preserving first-occurrence order).

        Args:
            v : Raw value from data or constructor.

        Returns:
            Cleaned, deduplicated list of strings.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            # Handle accidental string input e.g. color="White"
            v = [str(v)]
        seen: set = set()
        result: List[str] = []
        for item in v:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result

    @field_validator("description")
    @classmethod
    def description_stripped(cls, v: Optional[str]) -> Optional[str]:
        """
        Strip leading/trailing whitespace from description text.

        Args:
            v : Raw description string or None.

        Returns:
            Stripped string, or None if blank/None.
        """
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("subcategory")
    @classmethod
    def subcategory_lowercase(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalise subcategory to lowercase, underscore-separated.

        Example: "Graphic Tee" → "graphic_tee"

        Args:
            v : Raw subcategory string or None.

        Returns:
            Normalised subcategory or None.
        """
        if v is None:
            return None
        cleaned = v.strip().lower()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
        return cleaned if cleaned else None

    @field_validator("processed_at")
    @classmethod
    def processed_at_is_iso(cls, v: str) -> str:
        """
        Validate processed_at is a valid ISO-8601 datetime string.

        Args:
            v : Timestamp string.

        Returns:
            The unchanged string if valid.

        Raises:
            ValueError : If not a valid ISO-8601 timestamp.
        """
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"processed_at must be a valid ISO-8601 datetime string. Got: {v!r}"
            ) from exc
        return v

    # ==========================================================================
    # ── Model-Level Validators
    # ==========================================================================

    @model_validator(mode="after")
    def cross_field_validation(self) -> "UnifiedFashionItem":
        """
        Cross-field validation rules applied after all field validators pass.

        Rules:
          1. Dresses category → gender should not be 'men' (warning only).
          2. Footwear/accessories with fit set → warning (fit doesn't apply).
          3. DeepFashion with > 6 landmarks → error (format violation).
          4. image_id must match source_dataset prefix convention (warning).

        Returns:
            Self, with warnings updated.
        """
        warnings = list(self.warnings)

        # ── Rule 1: Dresses + men gender ──────────────────────────────────────
        if (self.category == CategoryEnum.DRESSES
                and self.gender == GenderEnum.MEN):
            warnings.append(
                "Category 'dresses' is typically not associated with gender 'men'. "
                "Verify this is correct."
            )

        # ── Rule 2: Fit on accessories/footwear ───────────────────────────────
        if (self.category in (CategoryEnum.ACCESSORIES, CategoryEnum.FOOTWEAR)
                and self.fit is not None):
            warnings.append(
                f"Fit field '{self.fit.value}' is set for category "
                f"'{self.category.value}' where fit is typically not applicable."
            )

        # ── Rule 3: DeepFashion landmark count ────────────────────────────────
        if (self.source_dataset == DatasetSource.DEEPFASHION
                and len(self.landmarks) > 6):
            # This is a hard error — the schema enforces max_length=6
            # but we add an explicit message for diagnostics
            warnings.append(
                f"DeepFashion records should have at most 6 landmarks; "
                f"got {len(self.landmarks)}."
            )

        # ── Rule 4: ID prefix convention ─────────────────────────────────────
        expected_prefix = (
            "FG_" if self.source_dataset == DatasetSource.FASHIONGEN else "DF_"
        )
        if not self.image_id.startswith(expected_prefix):
            warnings.append(
                f"image_id '{self.image_id}' does not follow the expected prefix "
                f"'{expected_prefix}' for dataset '{self.source_dataset.value}'. "
                f"This is a convention warning — IDs are still accepted."
            )

        # Apply updated warnings back (model supports validate_assignment=True)
        object.__setattr__(self, "warnings", warnings)
        return self

    # ==========================================================================
    # ── Serialisation Methods
    # ==========================================================================

    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        """
        Serialise this record to a plain Python dictionary.

        The output is JSON-safe (all values are Python primitives).
        Enum values are serialised as their string value (e.g. "shirts").
        LandmarkPoint and BoundingBox are serialised as nested dicts.

        Args:
            exclude_none : If True, fields with None values are omitted.

        Returns:
            Dict[str, Any] — a flat dict ready for json.dumps().

        Example:
            d = item.to_dict()
            print(d["category"])   # "shirts"
            print(d["landmarks"])  # [{"name": "left_collar", "x": 0.32, ...}]
        """
        data = self.model_dump(
            mode         = "json",
            exclude_none = exclude_none,
        )
        return data

    def to_json(self, indent: int = 2, exclude_none: bool = False) -> str:
        """
        Serialise this record to a formatted JSON string.

        Args:
            indent       : JSON indentation level (default 2).
            exclude_none : If True, None fields are omitted.

        Returns:
            Formatted JSON string.

        Example:
            print(item.to_json(indent=4))
        """
        return json.dumps(
            self.to_dict(exclude_none=exclude_none),
            indent       = indent,
            ensure_ascii = False,
        )

    def to_jsonl_line(self) -> str:
        """
        Serialise this record to a single-line JSON string (JSONL format).

        Suitable for writing to .jsonl files (one record per line).
        Trailing newline is NOT included.

        Returns:
            Compact single-line JSON string.

        Example:
            with open("unified.jsonl", "a") as f:
                f.write(item.to_jsonl_line() + "\\n")
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedFashionItem":
        """
        Deserialise a UnifiedFashionItem from a plain dictionary.

        Applies all Pydantic field validators during construction.

        Args:
            data : Dictionary produced by to_dict() or any compatible source.

        Returns:
            UnifiedFashionItem instance.

        Raises:
            pydantic.ValidationError : On schema violation.

        Example:
            item = UnifiedFashionItem.from_dict({"image_id": "FG_001", ...})
        """
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "UnifiedFashionItem":
        """
        Deserialise a UnifiedFashionItem from a JSON string.

        Args:
            json_str : JSON string (from to_json() or any compatible source).

        Returns:
            UnifiedFashionItem instance.

        Raises:
            json.JSONDecodeError      : On malformed JSON.
            pydantic.ValidationError  : On schema violation.

        Example:
            item = UnifiedFashionItem.from_json('{"image_id": "FG_001", ...}')
        """
        return cls.model_validate_json(json_str)

    # ==========================================================================
    # ── Factory Class Methods: Dataset-Specific Constructors
    # ==========================================================================

    @classmethod
    def from_fashiongen(cls, record: Any) -> "UnifiedFashionItem":
        """
        Create a UnifiedFashionItem from a FashionGenRecord dataclass.

        Mapping:
          FashionGenRecord.image_id       → image_id
          FashionGenRecord.image_path     → image_path
          FashionGenRecord.category       → category (already normalised)
          FashionGenRecord.subcategory    → subcategory
          FashionGenRecord.gender         → gender
          FashionGenRecord.season         → season
          FashionGenRecord.style          → style
          FashionGenRecord.description    → description
          FashionGenRecord.attributes     → attributes (dict → flat list)
          FashionGenRecord.is_valid       → is_valid
          FashionGenRecord.errors         → errors
          FashionGenRecord.warnings       → warnings

        Attributes dict from FashionGen may contain:
          {"colors": [...], "fabrics": [...], "patterns": [...], ...}
        These are expanded into the corresponding top-level fields.

        Args:
            record : FashionGenRecord dataclass instance (duck-typed).

        Returns:
            UnifiedFashionItem

        Raises:
            pydantic.ValidationError : If the mapped values violate schema.

        Example:
            from src.data.ingestion.fashiongen_loader import FashionGenLoader
            result = FashionGenLoader().run(max_records=1)
            # Use the saved JSON instead:
            item = UnifiedFashionItem.from_fashiongen(record)
        """
        attrs: Dict[str, Any] = getattr(record, "attributes", {}) or {}

        # Flatten attributes dict → list of strings for the attributes field
        attr_list: List[str] = []
        if isinstance(attrs, dict):
            for key, val in attrs.items():
                if isinstance(val, list):
                    attr_list.extend(str(v) for v in val if v)
                elif val:
                    attr_list.append(str(val))

        # Extract colors / fabrics / patterns from the attributes dict
        colors  = attrs.get("colors",   []) if isinstance(attrs, dict) else []
        fabrics = attrs.get("fabrics",  []) if isinstance(attrs, dict) else []
        patterns = attrs.get("patterns", []) if isinstance(attrs, dict) else []

        # Map fit string from attributes dict (if present)
        fit_raw = attrs.get("fit") if isinstance(attrs, dict) else None
        fit_val = _safe_fit(fit_raw)

        # Map occasion list from attributes dict (if present)
        occasion_raw = attrs.get("occasions", []) if isinstance(attrs, dict) else []
        occasion_list = _safe_occasion_list(occasion_raw)

        return cls(
            image_id        = str(getattr(record, "image_id",     "")),
            image_path      = str(getattr(record, "image_path",   "")),
            source_dataset  = DatasetSource.FASHIONGEN,
            category        = _safe_category(getattr(record, "category", "accessories")),
            subcategory     = getattr(record, "subcategory", None) or None,
            gender          = _safe_gender(getattr(record, "gender", None)),
            color           = list(colors),
            fabric          = list(fabrics),
            pattern         = list(patterns),
            fit             = fit_val,
            style           = _safe_style(getattr(record, "style", None)),
            season          = _safe_season(getattr(record, "season", "all_season")),
            occasion        = occasion_list,
            description     = getattr(record, "description", None) or None,
            attributes      = attr_list,
            landmarks       = [],           # FashionGen has no landmarks
            bounding_box    = None,         # FashionGen has no bounding box
            is_valid        = bool(getattr(record, "is_valid", False)),
            errors          = list(getattr(record, "errors",   [])),
            warnings        = list(getattr(record, "warnings", [])),
        )

    @classmethod
    def from_deepfashion(cls, record: Any) -> "UnifiedFashionItem":
        """
        Create a UnifiedFashionItem from a DeepFashionRecord dataclass.

        Mapping:
          DeepFashionRecord.image_id          → image_id
          DeepFashionRecord.image_path        → image_path
          DeepFashionRecord.category          → category (already normalised)
          DeepFashionRecord.attributes        → attributes (List[str])
          DeepFashionRecord.landmarks         → landmarks (List[dict] → LandmarkPoint)
          DeepFashionRecord.bbox              → bounding_box (BoundingBox)
          DeepFashionRecord.bbox_normalised   → bounding_box.nx1/ny1/nx2/ny2
          DeepFashionRecord.category_raw      → subcategory (raw name as sub hint)
          DeepFashionRecord.split             → (ignored in unified schema)
          DeepFashionRecord.is_valid          → is_valid
          DeepFashionRecord.errors            → errors
          DeepFashionRecord.warnings          → warnings

        Fields not available in DeepFashion → safe defaults:
          gender      → None (DeepFashion does not annotate gender)
          style       → None (DeepFashion does not annotate style)
          season      → all_season (DeepFashion does not annotate season)
          description → None (DeepFashion has no text descriptions)
          fit         → None (inferred from attributes list if possible)
          color       → extracted from attributes list if present
          fabric      → extracted from attributes list if present

        Args:
            record : DeepFashionRecord dataclass instance (duck-typed).

        Returns:
            UnifiedFashionItem

        Raises:
            pydantic.ValidationError : If the mapped values violate schema.

        Example:
            from src.data.ingestion.deepfashion_loader import DeepFashionLoader
            item = UnifiedFashionItem.from_deepfashion(record)
        """
        raw_attrs: List[str] = list(getattr(record, "attributes", []) or [])

        # ── Extract typed attribute sub-lists from the free text list ──────────
        colors   = _extract_colors_from_attrs(raw_attrs)
        fabrics  = _extract_fabrics_from_attrs(raw_attrs)
        patterns = _extract_patterns_from_attrs(raw_attrs)
        fit_val  = _extract_fit_from_attrs(raw_attrs)

        # ── Convert raw landmark dicts → LandmarkPoint objects ────────────────
        raw_landmarks: List[Dict[str, Any]] = list(getattr(record, "landmarks", []) or [])
        landmark_points: List[LandmarkPoint] = _parse_landmarks(raw_landmarks)

        # ── Convert bbox list → BoundingBox object ────────────────────────────
        raw_bbox: List[int] = list(getattr(record, "bbox", []) or [])
        raw_bbox_norm: List[float] = list(getattr(record, "bbox_normalised", []) or [])
        bbox_obj = _parse_bounding_box(raw_bbox, raw_bbox_norm)

        # ── Use raw category name as subcategory hint ─────────────────────────
        sub = getattr(record, "category_raw", None)
        if sub and sub.lower() in ("unknown", ""):
            sub = None

        return cls(
            image_id        = str(getattr(record, "image_id",    "")),
            image_path      = str(getattr(record, "image_path",  "")),
            source_dataset  = DatasetSource.DEEPFASHION,
            category        = _safe_category(getattr(record, "category", "accessories")),
            subcategory     = sub,
            gender          = None,         # DeepFashion has no gender annotation
            color           = colors,
            fabric          = fabrics,
            pattern         = patterns,
            fit             = fit_val,
            style           = None,         # DeepFashion has no style annotation
            season          = SeasonEnum.ALL_SEASON,
            occasion        = [],           # DeepFashion has no occasion annotation
            description     = None,         # DeepFashion has no text descriptions
            attributes      = raw_attrs,
            landmarks       = landmark_points,
            bounding_box    = bbox_obj,
            is_valid        = bool(getattr(record, "is_valid", False)),
            errors          = list(getattr(record, "errors",   [])),
            warnings        = list(getattr(record, "warnings", [])),
        )

    # ==========================================================================
    # ── Validation Report
    # ==========================================================================

    def validate_and_report(self) -> ValidationReport:
        """
        Run a comprehensive validation and return a structured report.

        Validation Layers:
          1. Required fields (image_id, image_path, category, source_dataset)
          2. Category + gender cross-check (dresses ≠ men)
          3. Landmark count (0 or 6 for DeepFashion, 0 for FashionGen)
          4. Bounding box consistency (nx1 < nx2, ny1 < ny2)
          5. Image path format (forward slashes, known extension)
          6. Field coverage measurement

        Returns:
            ValidationReport with is_valid, errors, warnings, suggestions,
            and field_coverage.

        Example:
            report = item.validate_and_report()
            if not report.is_valid:
                print("Errors:", report.errors)
            print(f"Coverage: {report.field_coverage:.1%}")
        """
        errors:      List[str] = []
        warnings:    List[str] = list(self.warnings)   # carry over existing warnings
        suggestions: List[str] = []

        # ── Layer 1: Required fields ───────────────────────────────────────────
        if not self.image_id.strip():
            errors.append("image_id is empty or whitespace.")
        if not self.image_path.strip():
            errors.append("image_path is empty or whitespace.")

        # ── Layer 2: Category + gender cross-check ────────────────────────────
        if (self.category == CategoryEnum.DRESSES
                and self.gender == GenderEnum.MEN):
            warnings.append(
                "Category 'dresses' assigned to gender='men'. Verify correctness."
            )

        # ── Layer 3: Landmark count ────────────────────────────────────────────
        if self.source_dataset == DatasetSource.DEEPFASHION:
            n_lm = len(self.landmarks)
            if n_lm not in (0, 6):
                warnings.append(
                    f"DeepFashion item has {n_lm} landmarks; "
                    f"expected 0 or 6."
                )
        elif self.source_dataset == DatasetSource.FASHIONGEN:
            if self.landmarks:
                warnings.append(
                    "FashionGen item has non-empty landmarks. "
                    "FashionGen does not annotate landmarks."
                )

        # ── Layer 4: Bounding box consistency ─────────────────────────────────
        if self.bounding_box is not None:
            bb = self.bounding_box
            if (bb.nx1 is not None and bb.nx2 is not None
                    and bb.nx1 >= bb.nx2):
                errors.append(
                    f"Normalised bounding box: nx1 ({bb.nx1}) >= nx2 ({bb.nx2})."
                )
            if (bb.ny1 is not None and bb.ny2 is not None
                    and bb.ny1 >= bb.ny2):
                errors.append(
                    f"Normalised bounding box: ny1 ({bb.ny1}) >= ny2 ({bb.ny2})."
                )

        # ── Layer 5: Image path format ─────────────────────────────────────────
        if "\\" in self.image_path:
            errors.append(
                "image_path contains backslashes. Use forward slashes."
            )
        _valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        suffix = Path(self.image_path).suffix.lower()
        if suffix not in _valid_exts:
            warnings.append(
                f"image_path has uncommon extension '{suffix}'. "
                f"Expected one of {sorted(_valid_exts)}."
            )

        # ── Layer 6: Field completeness suggestions ───────────────────────────
        if not self.color:
            suggestions.append("No color information. Consider adding color attributes.")
        if not self.fabric:
            suggestions.append("No fabric information. Consider adding fabric attributes.")
        if self.gender is None:
            suggestions.append("gender is None. Setting gender improves search recall.")
        if not self.description and self.source_dataset == DatasetSource.FASHIONGEN:
            suggestions.append(
                "No description for a FashionGen record. "
                "Descriptions are typically available in FashionGen."
            )

        # ── Field coverage score ──────────────────────────────────────────────
        optional_fields = [
            self.subcategory,
            self.gender,
            self.color or None,
            self.fabric or None,
            self.pattern or None,
            self.fit,
            self.style,
            self.description,
            self.occasion or None,
            self.attributes or None,
            self.landmarks or None,
            self.bounding_box,
        ]
        populated = sum(1 for f in optional_fields if f is not None)
        coverage  = round(populated / len(optional_fields), 4)

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid       = is_valid,
            errors         = errors,
            warnings       = warnings,
            suggestions    = suggestions,
            field_coverage = coverage,
        )

    # ==========================================================================
    # ── Schema Documentation
    # ==========================================================================

    @classmethod
    def get_schema_doc(cls) -> str:
        """
        Return a human-readable Markdown documentation string for this schema.

        Covers: all fields, their types, valid values, default values,
        and which datasets populate each field.

        Returns:
            Markdown-formatted string.

        Example:
            print(UnifiedFashionItem.get_schema_doc())
        """
        return _SCHEMA_DOCUMENTATION

    @classmethod
    def schema_json(cls, indent: int = 2) -> str:
        """
        Return the full JSON Schema (draft-07) for this model as a string.

        Useful for API documentation, OpenAPI spec generation, and
        external validation tools.

        Args:
            indent : JSON indentation level.

        Returns:
            JSON Schema string.

        Example:
            print(UnifiedFashionItem.schema_json())
        """
        return json.dumps(cls.model_json_schema(), indent=indent, ensure_ascii=False)

    # ==========================================================================
    # ── Utility
    # ==========================================================================

    def __repr__(self) -> str:
        return (
            f"UnifiedFashionItem("
            f"image_id={self.image_id!r}, "
            f"source={self.source_dataset.value}, "
            f"category={self.category.value}, "
            f"valid={self.is_valid}"
            f")"
        )

    def summary(self) -> str:
        """
        Return a compact one-line human-readable summary string.

        Returns:
            Summary string suitable for logging.

        Example:
            logger.info(item.summary())
        """
        status = "✅" if self.is_valid else "❌"
        return (
            f"{status} {self.image_id} | "
            f"src={self.source_dataset.value} | "
            f"cat={self.category.value} | "
            f"gender={self.gender.value if self.gender else 'N/A'} | "
            f"style={self.style.value if self.style else 'N/A'} | "
            f"attrs={len(self.attributes)}"
        )


# =============================================================================
# ── 4. FashionDatasetBatch — Batch Container
# =============================================================================

class FashionDatasetBatch(BaseModel):
    """
    A batch of UnifiedFashionItem records with dataset-level metadata.

    Supports:
      - Iteration:   for item in batch
      - Indexing:    batch[0]
      - Length:      len(batch)
      - Filtering:   batch.filter_valid(), batch.filter_by_category()
      - Serialisation: save_jsonl(), save_json(), to_dict()

    Usage:
        batch = FashionDatasetBatch(items=[item1, item2], source="fashiongen")
        batch.save_jsonl("datasets/processed/unified.jsonl")
        valid = batch.filter_valid()
    """

    model_config = ConfigDict(extra="forbid")

    items          : List[UnifiedFashionItem] = Field(default_factory=list)
    source         : Optional[str]            = Field(default=None)
    created_at     : str                      = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version : str = Field(default=_SCHEMA_VERSION)
    description    : str = Field(
        default="Unified Fashion Dataset Batch",
        description="Human-readable description of this batch.",
    )

    # ── Container protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[UnifiedFashionItem]:
        return iter(self.items)

    def __getitem__(self, index: int) -> UnifiedFashionItem:
        return self.items[index]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """Total number of items in the batch."""
        return len(self.items)

    @property
    def valid_count(self) -> int:
        """Number of items with is_valid=True."""
        return sum(1 for i in self.items if i.is_valid)

    @property
    def invalid_count(self) -> int:
        """Number of items with is_valid=False."""
        return self.total - self.valid_count

    @property
    def valid_rate(self) -> float:
        """Fraction of valid items (0.0 – 1.0)."""
        return round(self.valid_count / self.total, 4) if self.total else 0.0

    def category_distribution(self) -> Dict[str, int]:
        """Return a dict of category → count."""
        dist: Dict[str, int] = {}
        for item in self.items:
            key = item.category.value
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def gender_distribution(self) -> Dict[str, int]:
        """Return a dict of gender → count (including None → 'unknown')."""
        dist: Dict[str, int] = {}
        for item in self.items:
            key = item.gender.value if item.gender else "unknown"
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def source_distribution(self) -> Dict[str, int]:
        """Return a dict of source_dataset → count."""
        dist: Dict[str, int] = {}
        for item in self.items:
            key = item.source_dataset.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    # ── Filtering ─────────────────────────────────────────────────────────────

    def filter_valid(self) -> "FashionDatasetBatch":
        """Return a new batch containing only valid records."""
        return FashionDatasetBatch(
            items      = [i for i in self.items if i.is_valid],
            source     = self.source,
            description= f"{self.description} [valid only]",
        )

    def filter_by_category(self, category: str) -> "FashionDatasetBatch":
        """Return a new batch filtered to the given category key."""
        return FashionDatasetBatch(
            items      = [i for i in self.items if i.category.value == category],
            source     = self.source,
            description= f"{self.description} [category={category}]",
        )

    def filter_by_source(self, source: str) -> "FashionDatasetBatch":
        """Return a new batch filtered to the given source_dataset."""
        return FashionDatasetBatch(
            items      = [i for i in self.items if i.source_dataset.value == source],
            source     = source,
            description= f"{self.description} [source={source}]",
        )

    def filter_by_gender(self, gender: str) -> "FashionDatasetBatch":
        """Return a new batch filtered to the given gender key."""
        return FashionDatasetBatch(
            items      = [i for i in self.items
                          if i.gender is not None and i.gender.value == gender],
            source     = self.source,
            description= f"{self.description} [gender={gender}]",
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the batch to a plain Python dictionary.

        Output structure:
          {
            "_meta"  : { total, valid_count, source, created_at, schema_version },
            "records": [ { ...UnifiedFashionItem.to_dict() }, ... ]
          }

        Returns:
            JSON-safe dict.
        """
        return {
            "_meta": {
                "total"            : self.total,
                "valid_count"      : self.valid_count,
                "invalid_count"    : self.invalid_count,
                "valid_rate"       : self.valid_rate,
                "source"           : self.source,
                "created_at"       : self.created_at,
                "schema_version"   : self.schema_version,
                "description"      : self.description,
                "category_distribution": self.category_distribution(),
                "gender_distribution"  : self.gender_distribution(),
                "source_distribution"  : self.source_distribution(),
            },
            "records": [item.to_dict() for item in self.items],
        }

    def save_json(
        self,
        output_path: Union[str, Path],
        indent     : int = 2,
    ) -> Path:
        """
        Save the entire batch to a single JSON file.

        Args:
            output_path : Destination file path.
            indent      : JSON indentation level.

        Returns:
            Path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
        logger.info(f"Batch saved to JSON: {path} ({self.total} records)")
        return path

    def save_jsonl(self, output_path: Union[str, Path]) -> Path:
        """
        Save the batch to a JSONL file (one JSON record per line).

        This format is preferred for large datasets because it supports
        streaming reads without loading the entire file into memory.

        Args:
            output_path : Destination .jsonl file path.

        Returns:
            Path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(item.to_jsonl_line() + "\n")
        logger.info(f"Batch saved to JSONL: {path} ({self.total} records)")
        return path

    @classmethod
    def load_jsonl(cls, input_path: Union[str, Path]) -> "FashionDatasetBatch":
        """
        Load a batch from a JSONL file (produced by save_jsonl).

        Args:
            input_path : Path to .jsonl file.

        Returns:
            FashionDatasetBatch instance.

        Raises:
            FileNotFoundError   : If input_path does not exist.
            json.JSONDecodeError : If a line is malformed.
            pydantic.ValidationError : If a record violates schema.
        """
        path  = Path(input_path)
        items : List[UnifiedFashionItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(UnifiedFashionItem.from_dict(data))
                except Exception as exc:
                    logger.warning(f"Skipping JSONL line {line_num}: {exc}")
        logger.info(f"Loaded {len(items)} records from {path}")
        return cls(items=items, source=str(path.stem))

    def summary_report(self) -> str:
        """
        Generate a formatted text summary of this batch.

        Returns:
            Multi-line string suitable for printing/logging.
        """
        lines = [
            "=" * 60,
            "FASHION DATASET BATCH SUMMARY",
            "=" * 60,
            f"  Total records      : {self.total:,}",
            f"  Valid records      : {self.valid_count:,} ({self.valid_rate:.1%})",
            f"  Invalid records    : {self.invalid_count:,}",
            f"  Source             : {self.source or 'mixed'}",
            f"  Schema version     : {self.schema_version}",
            f"  Created at         : {self.created_at}",
            "",
            "  Category Distribution:",
        ]
        for cat, count in self.category_distribution().items():
            pct = count / self.total * 100 if self.total else 0
            lines.append(f"    {cat:<20} : {count:>6,}  ({pct:.1f}%)")
        lines.append("")
        lines.append("  Gender Distribution:")
        for gnd, count in self.gender_distribution().items():
            pct = count / self.total * 100 if self.total else 0
            lines.append(f"    {gnd:<20} : {count:>6,}  ({pct:.1f}%)")
        lines.append("")
        lines.append("  Source Distribution:")
        for src, count in self.source_distribution().items():
            lines.append(f"    {src:<20} : {count:>6,}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# ── 5. Private Helper Functions
# =============================================================================

def _safe_category(value: Any) -> CategoryEnum:
    """
    Convert a raw category string to CategoryEnum, falling back to ACCESSORIES.

    Args:
        value : Raw category string from a pipeline record.

    Returns:
        CategoryEnum value.
    """
    if not value:
        return CategoryEnum.ACCESSORIES
    try:
        return CategoryEnum(str(value).strip().lower())
    except ValueError:
        logger.debug(f"Unknown category value: '{value}' → 'accessories'")
        return CategoryEnum.ACCESSORIES


def _safe_gender(value: Any) -> Optional[GenderEnum]:
    """
    Convert a raw gender string to GenderEnum or None.

    Args:
        value : Raw gender string or None.

    Returns:
        GenderEnum value or None.
    """
    if not value:
        return None
    try:
        return GenderEnum(str(value).strip().lower())
    except ValueError:
        logger.debug(f"Unknown gender value: '{value}' → None")
        return None


def _safe_style(value: Any) -> Optional[StyleEnum]:
    """
    Convert a raw style string to StyleEnum or None.

    Args:
        value : Raw style string or None.

    Returns:
        StyleEnum value or None.
    """
    if not value:
        return None
    try:
        return StyleEnum(str(value).strip().lower())
    except ValueError:
        logger.debug(f"Unknown style value: '{value}' → None")
        return None


def _safe_season(value: Any) -> SeasonEnum:
    """
    Convert a raw season string to SeasonEnum, falling back to ALL_SEASON.

    Args:
        value : Raw season string or None.

    Returns:
        SeasonEnum value.
    """
    if not value:
        return SeasonEnum.ALL_SEASON
    try:
        return SeasonEnum(str(value).strip().lower())
    except ValueError:
        logger.debug(f"Unknown season value: '{value}' → 'all_season'")
        return SeasonEnum.ALL_SEASON


def _safe_fit(value: Any) -> Optional[FitEnum]:
    """
    Convert a raw fit string to FitEnum or None.

    Args:
        value : Raw fit string or None.

    Returns:
        FitEnum value or None.
    """
    if not value:
        return None
    try:
        return FitEnum(str(value).strip().lower())
    except ValueError:
        logger.debug(f"Unknown fit value: '{value}' → None")
        return None


def _safe_occasion_list(values: Any) -> List[OccasionEnum]:
    """
    Convert a list of raw occasion strings to validated OccasionEnum values.

    Silently drops unknown occasion strings.

    Args:
        values : List of raw occasion strings or a single string.

    Returns:
        List of OccasionEnum values (may be empty).
    """
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    result: List[OccasionEnum] = []
    seen: set = set()
    for v in values:
        try:
            occ = OccasionEnum(str(v).strip().lower())
            if occ not in seen:
                seen.add(occ)
                result.append(occ)
        except ValueError:
            logger.debug(f"Unknown occasion value: '{v}' — skipped")
    return result


def _parse_landmarks(raw: List[Dict[str, Any]]) -> List[LandmarkPoint]:
    """
    Convert a list of raw landmark dicts to validated LandmarkPoint objects.

    Invalid landmark dicts are skipped with a debug log.

    Args:
        raw : List of dicts with keys: name, x, y, visible.

    Returns:
        List of LandmarkPoint objects (0–6 entries).
    """
    points: List[LandmarkPoint] = []
    for lm in raw:
        if not isinstance(lm, dict):
            continue
        try:
            # Clamp x, y to [0, 1] before construction to avoid ValidationError
            x = max(0.0, min(1.0, float(lm.get("x", 0.0))))
            y = max(0.0, min(1.0, float(lm.get("y", 0.0))))
            points.append(LandmarkPoint(
                name    = str(lm.get("name", "unknown")),
                x       = x,
                y       = y,
                visible = bool(lm.get("visible", False)),
            ))
        except Exception as exc:
            logger.debug(f"Skipping invalid landmark dict {lm}: {exc}")
    return points


def _parse_bounding_box(
    bbox     : List[int],
    bbox_norm: List[float],
) -> Optional[BoundingBox]:
    """
    Convert pixel + normalised bbox lists to a BoundingBox object.

    Args:
        bbox      : [x1, y1, x2, y2] pixel coordinates or [].
        bbox_norm : [nx1, ny1, nx2, ny2] normalised in [0,1] or [].

    Returns:
        BoundingBox object or None if bbox is missing or invalid.
    """
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"Degenerate bbox skipped: {bbox}")
            return None

        nx1 = bbox_norm[0] if len(bbox_norm) == 4 else None
        ny1 = bbox_norm[1] if len(bbox_norm) == 4 else None
        nx2 = bbox_norm[2] if len(bbox_norm) == 4 else None
        ny2 = bbox_norm[3] if len(bbox_norm) == 4 else None

        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            nx1=nx1, ny1=ny1, nx2=nx2, ny2=ny2,
        )
    except Exception as exc:
        logger.debug(f"Failed to parse bounding box {bbox}: {exc}")
        return None


# ── Attribute keyword extractors (DeepFashion) ─────────────────────────────────

_COLOR_KEYWORDS = frozenset({
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "grey", "gray", "beige", "navy", "cream", "ivory",
    "teal", "maroon", "olive", "coral", "turquoise", "gold", "silver",
    "khaki", "lavender", "magenta", "cyan", "indigo", "multicolor", "floral",
})
_FABRIC_KEYWORDS = frozenset({
    "cotton", "polyester", "linen", "silk", "wool", "denim", "leather",
    "nylon", "rayon", "velvet", "satin", "chiffon", "fleece", "knit",
    "mesh", "canvas", "spandex", "lycra", "cashmere", "bamboo", "modal",
    "synthetic", "tweed", "corduroy", "suede",
})
_PATTERN_KEYWORDS = frozenset({
    "solid", "striped", "stripes", "checked", "check", "plaid", "floral",
    "printed", "graphic", "camouflage", "camo", "tie-dye", "paisley",
    "polka dot", "animal print", "abstract", "geometric", "embroidered",
    "lace", "sequin", "ombre", "colorblock",
})
_FIT_KEYWORDS: Dict[str, str] = {
    "slim"    : "slim_fit",
    "slim fit": "slim_fit",
    "regular" : "regular_fit",
    "relaxed" : "relaxed_fit",
    "oversized": "oversized",
    "cropped" : "cropped",
    "skinny"  : "skinny",
    "straight": "straight",
    "athletic": "athletic_fit",
}


def _extract_colors_from_attrs(attrs: List[str]) -> List[str]:
    """Extract color keywords from a DeepFashion attribute list."""
    found: List[str] = []
    seen:  set       = set()
    for a in attrs:
        a_lower = a.lower()
        for kw in _COLOR_KEYWORDS:
            if kw in a_lower and kw not in seen:
                seen.add(kw)
                found.append(a.strip())
                break
    return found


def _extract_fabrics_from_attrs(attrs: List[str]) -> List[str]:
    """Extract fabric keywords from a DeepFashion attribute list."""
    found: List[str] = []
    seen:  set       = set()
    for a in attrs:
        a_lower = a.lower()
        for kw in _FABRIC_KEYWORDS:
            if kw in a_lower and a_lower not in seen:
                seen.add(a_lower)
                found.append(a.strip())
                break
    return found


def _extract_patterns_from_attrs(attrs: List[str]) -> List[str]:
    """Extract pattern keywords from a DeepFashion attribute list."""
    found: List[str] = []
    seen:  set       = set()
    for a in attrs:
        a_lower = a.lower()
        for kw in _PATTERN_KEYWORDS:
            if kw in a_lower and a_lower not in seen:
                seen.add(a_lower)
                found.append(a.strip())
                break
    return found


def _extract_fit_from_attrs(attrs: List[str]) -> Optional[FitEnum]:
    """Extract the first fit keyword from a DeepFashion attribute list."""
    for a in attrs:
        a_lower = a.lower()
        for kw, fit_key in _FIT_KEYWORDS.items():
            if kw in a_lower:
                try:
                    return FitEnum(fit_key)
                except ValueError:
                    pass
    return None


# =============================================================================
# ── 6. Schema Documentation String
# =============================================================================

_SCHEMA_DOCUMENTATION = f"""
# UnifiedFashionItem Schema Documentation
Version: {_SCHEMA_VERSION} | Date: {_SCHEMA_DATE}

## Overview
The UnifiedFashionItem is the canonical, dataset-agnostic schema for all
fashion records in the AI Fashion Design Assistant pipeline.

It unifies records from two source datasets:
  • FashionGen   — HDF5-based, ~293K items, text descriptions, no landmarks
  • DeepFashion  — TXT annotation-based, ~800K items, landmarks, no descriptions

## Field Reference

### Required Fields (must be non-empty)
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| image_id       | str             | Unique ID: "FG_0000042" or "DF_img_..."  |
| image_path     | str             | Relative POSIX path from project root    |
| source_dataset | DatasetSource   | "fashiongen" or "deepfashion"            |
| category       | CategoryEnum    | One of 11 taxonomy keys (see below)      |

### Taxonomy Fields
| Field          | Type            | Values                                   |
|----------------|-----------------|------------------------------------------|
| category       | CategoryEnum    | t_shirts, shirts, hoodies, jackets,      |
|                |                 | pants, jeans, shorts, dresses,           |
|                |                 | ethnic_wear, footwear, accessories       |
| subcategory    | str | None      | Sub-level label, e.g. "graphic_tee"      |
| gender         | GenderEnum|None | men, women, unisex (None if unknown)     |

### Appearance Attribute Fields
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| color          | List[str]       | Normalised color names, e.g. ["Navy"]    |
| fabric         | List[str]       | Normalised fabric names, e.g. ["Cotton"] |
| pattern        | List[str]       | Pattern types, e.g. ["stripes"]          |
| fit            | FitEnum | None  | slim_fit, regular_fit, relaxed_fit,      |
|                |                 | oversized, cropped, skinny, straight,    |
|                |                 | athletic_fit                             |

### Contextual Attribute Fields
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| style          | StyleEnum|None  | streetwear, luxury, formal,              |
|                |                 | business_casual, techwear, minimalist,   |
|                |                 | vintage, athleisure                      |
| season         | SeasonEnum      | spring, summer, autumn, winter,          |
|                |                 | all_season (default)                     |
| occasion       | List[Occasion]  | casual, business_casual, formal, party,  |
|                |                 | sport, outdoor, beach, wedding_festive,  |
|                |                 | lounge                                   |
| description    | str | None      | Human-written text (FashionGen only)     |

### Raw Attribute List
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| attributes     | List[str]       | Free-form attribute strings.             |
|                |                 | FashionGen: from dict flatten.           |
|                |                 | DeepFashion: decoded from 1000-dim vec.  |

### Spatial Data Fields (DeepFashion-specific)
| Field          | Type                 | Description                         |
|----------------|----------------------|-------------------------------------|
| landmarks      | List[LandmarkPoint]  | 0 or 6 landmark points              |
| bounding_box   | BoundingBox | None   | Pixel + normalised bbox             |

LandmarkPoint: name (str), x (float [0,1]), y (float [0,1]), visible (bool)
Landmark names: left_collar, right_collar, left_sleeve, right_sleeve,
                left_hem, right_hem

BoundingBox: x1, y1, x2, y2 (int pixels), nx1, ny1, nx2, ny2 (float [0,1])

### Pipeline Provenance Fields (auto-populated)
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| is_valid       | bool            | True if no hard errors exist             |
| errors         | List[str]       | Hard validation errors                   |
| warnings       | List[str]       | Soft validation warnings                 |
| processed_at   | str             | ISO-8601 UTC timestamp                   |
| schema_version | str             | "{_SCHEMA_VERSION}"                         |

## Dataset Field Coverage Matrix
| Field          | FashionGen | DeepFashion |
|----------------|------------|-------------|
| image_id       | ✅          | ✅           |
| image_path     | ✅          | ✅           |
| category       | ✅          | ✅           |
| subcategory    | ✅          | ⚠️ (raw name)|
| gender         | ✅          | ❌ None      |
| color          | ✅          | ⚠️ extracted |
| fabric         | ✅          | ⚠️ extracted |
| pattern        | ✅          | ⚠️ extracted |
| fit            | ✅          | ⚠️ extracted |
| style          | ✅ inferred | ❌ None      |
| season         | ✅ inferred | ⚠️ all_season|
| occasion       | ✅ inferred | ❌ []        |
| description    | ✅          | ❌ None      |
| attributes     | ⚠️ flattened| ✅           |
| landmarks      | ❌ []       | ✅ 6 points  |
| bounding_box   | ❌ None     | ✅           |

Legend: ✅ = populated  ⚠️ = partial/derived  ❌ = always empty/None

## Validation Rules
1. image_id must be non-empty and contain no whitespace.
2. image_path must use forward slashes only.
3. category must be one of the 11 CategoryEnum values.
4. gender, style, fit, season must be valid Enum values or None.
5. LandmarkPoint x, y must be in [0.0, 1.0].
6. BoundingBox x2 > x1 and y2 > y1 (non-degenerate).
7. Cross-field: dresses + men → warning.
8. Cross-field: accessories/footwear + fit → warning.
9. DeepFashion: landmarks must be 0 or 6 entries.
10. FashionGen: landmarks must be [] (warning if non-empty).
"""


# =============================================================================
# ── 7. Module-Level Convenience Exports
# =============================================================================

__all__ = [
    # ── Primary schema model ──────────────────────────────────────────────────
    "UnifiedFashionItem",
    # ── Batch container ───────────────────────────────────────────────────────
    "FashionDatasetBatch",
    # ── Embedded sub-models ───────────────────────────────────────────────────
    "LandmarkPoint",
    "BoundingBox",
    "SchemaVersion",
    "ValidationReport",
    # ── Enumerations ──────────────────────────────────────────────────────────
    "DatasetSource",
    "GenderEnum",
    "CategoryEnum",
    "StyleEnum",
    "FitEnum",
    "SeasonEnum",
    "OccasionEnum",
    # ── Helper functions ──────────────────────────────────────────────────────
    "safe_category",
    "safe_gender",
    "safe_style",
    "safe_season",
    "safe_fit",
    "safe_occasion_list",
]

# Public aliases for the private helpers (camelCase-free names)
safe_category     = _safe_category
safe_gender       = _safe_gender
safe_style        = _safe_style
safe_season       = _safe_season
safe_fit          = _safe_fit
safe_occasion_list = _safe_occasion_list
