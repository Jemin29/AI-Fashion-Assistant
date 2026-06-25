"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_fashion_schema.py — Unit Tests: Unified Fashion Schema
=============================================================================
Full test suite for data_pipeline/schema/fashion_schema.py.
All tests run without network access or real dataset files.

Test Classes:
  TestEnumerations              — All 7 enums: values, membership, string coercion
  TestLandmarkPoint             — Coords, visibility, name validation
  TestBoundingBox               — Pixel coords, normalised, dimension validation
  TestUnifiedFashionItemBasic   — Construction, required fields, defaults
  TestFieldValidators           — image_id, image_path, list cleaning, description
  TestCrossFieldValidation      — Model-level validators, warnings
  TestSerialisationMethods      — to_dict, to_json, to_jsonl_line, round-trips
  TestDeserialisationMethods    — from_dict, from_json, error handling
  TestFromFashionGen            — Factory from FashionGenRecord mock
  TestFromDeepFashion           — Factory from DeepFashionRecord mock
  TestValidateAndReport         — ValidationReport layers, coverage score
  TestFashionDatasetBatch       — Container operations, filters, stats
  TestBatchSerialisation        — save_json, save_jsonl, load_jsonl
  TestHelperFunctions           — safe_* public helpers
  TestSchemaDocumentation       — get_schema_doc, schema_json

Run:
    pytest tests/test_fashion_schema.py -v
    pytest tests/test_fashion_schema.py -v --cov=data_pipeline.schema.fashion_schema
=============================================================================
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

try:
    from pydantic import ValidationError
except ImportError:
    pytest.skip("Pydantic v2 not installed", allow_module_level=True)

from src.data.schema.fashion_schema import (
    # Primary model
    UnifiedFashionItem,
    FashionDatasetBatch,
    # Sub-models
    LandmarkPoint,
    BoundingBox,
    SchemaVersion,
    ValidationReport,
    # Enums
    DatasetSource,
    GenderEnum,
    CategoryEnum,
    StyleEnum,
    FitEnum,
    SeasonEnum,
    OccasionEnum,
    # Helpers
    safe_category,
    safe_gender,
    safe_style,
    safe_season,
    safe_fit,
    safe_occasion_list,
    # Private helpers accessed via module for thorough testing
    _safe_category,
    _safe_gender,
    _safe_style,
    _safe_season,
    _safe_fit,
    _safe_occasion_list,
    _parse_landmarks,
    _parse_bounding_box,
    _extract_colors_from_attrs,
    _extract_fabrics_from_attrs,
    _extract_patterns_from_attrs,
    _extract_fit_from_attrs,
    _SCHEMA_VERSION,
)


# =============================================================================
# ── Shared Fixtures
# =============================================================================

@pytest.fixture
def minimal_item() -> UnifiedFashionItem:
    """A minimal valid UnifiedFashionItem (only required fields)."""
    return UnifiedFashionItem(
        image_id       = "FG_0000001",
        image_path     = "datasets/fashiongen/images/FG_0000001.jpg",
        source_dataset = DatasetSource.FASHIONGEN,
        category       = CategoryEnum.SHIRTS,
    )


@pytest.fixture
def full_fg_item() -> UnifiedFashionItem:
    """A fully populated FashionGen UnifiedFashionItem."""
    return UnifiedFashionItem(
        image_id       = "FG_0000042",
        image_path     = "datasets/fashiongen/images/FG_0000042.jpg",
        source_dataset = DatasetSource.FASHIONGEN,
        category       = CategoryEnum.SHIRTS,
        subcategory    = "formal_shirt",
        gender         = GenderEnum.MEN,
        color          = ["White", "Blue"],
        fabric         = ["Cotton"],
        pattern        = ["solid"],
        fit            = FitEnum.SLIM_FIT,
        style          = StyleEnum.FORMAL,
        season         = SeasonEnum.ALL_SEASON,
        occasion       = [OccasionEnum.FORMAL, OccasionEnum.BUSINESS_CASUAL],
        description    = "A slim-fit white cotton dress shirt for formal occasions.",
        attributes     = ["formal", "cotton", "slim"],
        landmarks      = [],
        bounding_box   = None,
        is_valid       = True,
        errors         = [],
        warnings       = [],
    )


@pytest.fixture
def full_df_item() -> UnifiedFashionItem:
    """A fully populated DeepFashion UnifiedFashionItem."""
    return UnifiedFashionItem(
        image_id       = "DF_img_Blouse_img_00000001",
        image_path     = "datasets/deepfashion/img/Blouse/img_00000001.jpg",
        source_dataset = DatasetSource.DEEPFASHION,
        category       = CategoryEnum.SHIRTS,
        subcategory    = "blouse",
        gender         = None,
        color          = [],
        fabric         = ["cotton material"],
        pattern        = [],
        fit            = None,
        style          = None,
        season         = SeasonEnum.ALL_SEASON,
        occasion       = [],
        description    = None,
        attributes     = ["floral pattern", "cotton material"],
        landmarks      = [
            LandmarkPoint(name="left_collar",  x=0.32, y=0.05, visible=True),
            LandmarkPoint(name="right_collar", x=0.67, y=0.05, visible=True),
            LandmarkPoint(name="left_sleeve",  x=0.06, y=0.45, visible=True),
            LandmarkPoint(name="right_sleeve", x=0.90, y=0.45, visible=True),
            LandmarkPoint(name="left_hem",     x=0.19, y=0.90, visible=True),
            LandmarkPoint(name="right_hem",    x=0.81, y=0.90, visible=True),
        ],
        bounding_box   = BoundingBox(x1=50, y1=30, x2=206, y2=230,
                                     nx1=0.195, ny1=0.117, nx2=0.805, ny2=0.898),
        is_valid       = True,
        errors         = [],
        warnings       = [],
    )


# ── Mock dataclasses mirroring FashionGenRecord and DeepFashionRecord ─────────

@dataclass
class _MockFashionGenRecord:
    image_id      : str
    image_path    : str
    category      : str = "shirts"
    subcategory   : str = "formal_shirt"
    gender        : str = "men"
    season        : str = "all_season"
    style         : str = "formal"
    description   : str = "A slim-fit white dress shirt."
    attributes    : Dict[str, Any] = field(default_factory=lambda: {
        "colors": ["White"], "fabrics": ["Cotton"],
        "patterns": ["solid"], "fit": "slim_fit",
        "occasions": ["formal"],
    })
    dataset_source: str = "fashiongen"
    is_valid      : bool = True
    errors        : List[str] = field(default_factory=list)
    warnings      : List[str] = field(default_factory=list)
    processed_at  : str = "2026-06-03T10:00:00+00:00"


@dataclass
class _MockDeepFashionRecord:
    image_id        : str = "DF_img_Blouse_img_00000001"
    image_path      : str = "datasets/deepfashion/img/Blouse/img_00000001.jpg"
    category        : str = "shirts"
    category_raw    : str = "Blouse"
    attributes      : List[str] = field(default_factory=lambda: ["floral pattern", "cotton material", "slim fit"])
    landmarks       : List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "left_collar",  "x": 0.32, "y": 0.05, "visible": True},
        {"name": "right_collar", "x": 0.67, "y": 0.05, "visible": True},
        {"name": "left_sleeve",  "x": 0.06, "y": 0.45, "visible": True},
        {"name": "right_sleeve", "x": 0.90, "y": 0.45, "visible": True},
        {"name": "left_hem",     "x": 0.19, "y": 0.90, "visible": True},
        {"name": "right_hem",    "x": 0.81, "y": 0.90, "visible": True},
    ])
    bbox             : List[int]   = field(default_factory=lambda: [50, 30, 206, 230])
    bbox_normalised  : List[float] = field(default_factory=lambda: [0.195, 0.117, 0.805, 0.898])
    dataset_source   : str  = "deepfashion"
    split            : str  = "train"
    is_valid         : bool = True
    errors           : List[str] = field(default_factory=list)
    warnings         : List[str] = field(default_factory=list)


# =============================================================================
# ── 1. TestEnumerations
# =============================================================================

class TestEnumerations:
    """Tests for all 7 enum types in the schema."""

    def test_dataset_source_values(self):
        assert DatasetSource.FASHIONGEN.value  == "fashiongen"
        assert DatasetSource.DEEPFASHION.value == "deepfashion"

    def test_gender_enum_values(self):
        assert GenderEnum.MEN.value    == "men"
        assert GenderEnum.WOMEN.value  == "women"
        assert GenderEnum.UNISEX.value == "unisex"

    def test_category_enum_all_11_values(self):
        expected = {
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
        }
        actual = {e.value for e in CategoryEnum}
        assert actual == expected

    def test_style_enum_all_8_values(self):
        expected = {
            "streetwear", "luxury", "formal", "business_casual",
            "techwear", "minimalist", "vintage", "athleisure"
        }
        actual = {e.value for e in StyleEnum}
        assert actual == expected

    def test_fit_enum_all_8_values(self):
        expected = {
            "slim_fit", "regular_fit", "relaxed_fit", "oversized",
            "cropped", "skinny", "straight", "athletic_fit"
        }
        actual = {e.value for e in FitEnum}
        assert actual == expected

    def test_season_enum_all_5_values(self):
        expected = {"spring", "summer", "autumn", "winter", "all_season"}
        actual   = {e.value for e in SeasonEnum}
        assert actual == expected

    def test_occasion_enum_all_9_values(self):
        expected = {
            "casual", "business_casual", "formal", "party",
            "sport", "outdoor", "beach", "wedding_festive", "lounge"
        }
        actual = {e.value for e in OccasionEnum}
        assert actual == expected

    def test_enum_is_str_subclass(self):
        """All enums inherit from str so they serialise cleanly."""
        assert isinstance(CategoryEnum.SHIRTS, str)
        assert isinstance(GenderEnum.MEN, str)
        assert CategoryEnum.SHIRTS == "shirts"

    def test_category_from_string(self):
        assert CategoryEnum("shirts") == CategoryEnum.SHIRTS
        assert CategoryEnum("t_shirts") == CategoryEnum.T_SHIRTS

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            CategoryEnum("space_suit")


# =============================================================================
# ── 2. TestLandmarkPoint
# =============================================================================

class TestLandmarkPoint:
    """Tests for LandmarkPoint embedded model."""

    def test_valid_landmark(self):
        lm = LandmarkPoint(name="left_collar", x=0.32, y=0.05, visible=True)
        assert lm.name    == "left_collar"
        assert lm.x       == 0.32
        assert lm.visible is True

    def test_coords_must_be_in_unit_interval(self):
        with pytest.raises(ValidationError):
            LandmarkPoint(name="left_collar", x=1.5, y=0.5, visible=True)
        with pytest.raises(ValidationError):
            LandmarkPoint(name="left_collar", x=0.5, y=-0.1, visible=True)

    def test_zero_zero_invisible(self):
        """Invisible landmark with x=0, y=0 is valid."""
        lm = LandmarkPoint(name="right_hem", x=0.0, y=0.0, visible=False)
        assert lm.x == 0.0 and lm.y == 0.0 and lm.visible is False

    def test_frozen_model(self):
        """LandmarkPoint is frozen — assignment must raise."""
        lm = LandmarkPoint(name="left_hem", x=0.5, y=0.5, visible=True)
        with pytest.raises((ValidationError, TypeError)):
            lm.x = 0.9  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            LandmarkPoint(name="left_collar", x=0.5, y=0.5, visible=True, extra="bad")

    def test_non_standard_landmark_name_allowed(self):
        """Non-standard names generate a warning but do not raise."""
        lm = LandmarkPoint(name="custom_point", x=0.5, y=0.5, visible=True)
        assert lm.name == "custom_point"

    def test_boundary_values_accepted(self):
        lm = LandmarkPoint(name="left_collar", x=0.0, y=1.0, visible=True)
        assert lm.x == 0.0 and lm.y == 1.0


# =============================================================================
# ── 3. TestBoundingBox
# =============================================================================

class TestBoundingBox:
    """Tests for BoundingBox embedded model."""

    def test_valid_bbox(self):
        bb = BoundingBox(x1=50, y1=30, x2=206, y2=230)
        assert bb.x1 == 50 and bb.y2 == 230

    def test_width_height_area_aspect(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.width  == 100
        assert bb.height == 200
        assert bb.area   == 20000
        assert bb.aspect_ratio == pytest.approx(0.5)

    def test_degenerate_box_raises(self):
        """x2 <= x1 must raise."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=100, y1=0, x2=50, y2=200)
        with pytest.raises(ValidationError):
            BoundingBox(x1=0, y1=200, x2=100, y2=50)

    def test_negative_coord_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=-1, y1=0, x2=100, y2=200)

    def test_normalised_coords_optional(self):
        bb = BoundingBox(x1=50, y1=30, x2=206, y2=230)
        assert bb.nx1 is None

    def test_normalised_coords_in_range(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=50, y1=30, x2=206, y2=230, nx1=1.5)

    def test_frozen_model(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        with pytest.raises((ValidationError, TypeError)):
            bb.x1 = 10  # type: ignore[misc]


# =============================================================================
# ── 4. TestUnifiedFashionItemBasic
# =============================================================================

class TestUnifiedFashionItemBasic:
    """Basic construction, required fields, and defaults."""

    def test_minimal_construction(self, minimal_item):
        assert minimal_item.image_id == "FG_0000001"
        assert minimal_item.category == CategoryEnum.SHIRTS

    def test_required_image_id(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id       = "",
                image_path     = "p.jpg",
                source_dataset = DatasetSource.FASHIONGEN,
                category       = CategoryEnum.SHIRTS,
            )

    def test_required_image_path(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id       = "FG_001",
                image_path     = "",
                source_dataset = DatasetSource.FASHIONGEN,
                category       = CategoryEnum.SHIRTS,
            )

    def test_defaults_are_safe(self, minimal_item):
        assert minimal_item.gender      is None
        assert minimal_item.style       is None
        assert minimal_item.fit         is None
        assert minimal_item.description is None
        assert minimal_item.color       == []
        assert minimal_item.fabric      == []
        assert minimal_item.pattern     == []
        assert minimal_item.attributes  == []
        assert minimal_item.landmarks   == []
        assert minimal_item.bounding_box is None
        assert minimal_item.is_valid    is False
        assert minimal_item.season      == SeasonEnum.ALL_SEASON
        assert minimal_item.schema_version == _SCHEMA_VERSION

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id       = "FG_001",
                image_path     = "p.jpg",
                source_dataset = DatasetSource.FASHIONGEN,
                category       = CategoryEnum.SHIRTS,
                unknown_field  = "bad",
            )

    def test_invalid_category_type(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id       = "FG_001",
                image_path     = "p.jpg",
                source_dataset = DatasetSource.FASHIONGEN,
                category       = "space_suit",
            )

    def test_source_dataset_string_coercion(self):
        """String 'fashiongen' should coerce to DatasetSource.FASHIONGEN."""
        item = UnifiedFashionItem(
            image_id       = "FG_001",
            image_path     = "p.jpg",
            source_dataset = "fashiongen",  # str, not enum
            category       = "shirts",
        )
        assert item.source_dataset == DatasetSource.FASHIONGEN

    def test_category_string_coercion(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="jeans",
        )
        assert item.category == CategoryEnum.JEANS

    def test_processed_at_is_set(self, minimal_item):
        datetime.fromisoformat(minimal_item.processed_at)  # must not raise


# =============================================================================
# ── 5. TestFieldValidators
# =============================================================================

class TestFieldValidators:
    """Tests for individual field-level validators."""

    def test_image_id_no_whitespace(self):
        with pytest.raises(ValidationError, match="whitespace"):
            UnifiedFashionItem(
                image_id="FG 0001", image_path="p.jpg",
                source_dataset="fashiongen", category="shirts",
            )

    def test_image_id_tab_rejected(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id="FG\t0001", image_path="p.jpg",
                source_dataset="fashiongen", category="shirts",
            )

    def test_image_path_backslash_normalised(self):
        """Backslashes in image_path must be converted to forward slashes."""
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="datasets\\fashiongen\\img.jpg",
            source_dataset="fashiongen", category="shirts",
        )
        assert "\\" not in item.image_path
        assert "/" in item.image_path

    def test_color_deduplication(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            color=["White", "Blue", "White"],
        )
        assert item.color.count("White") == 1

    def test_color_strips_whitespace(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            color=["  White  ", " Blue "],
        )
        assert "White" in item.color
        assert "Blue"  in item.color
        assert all(c == c.strip() for c in item.color)

    def test_color_none_becomes_empty_list(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            color=None,
        )
        assert item.color == []

    def test_attributes_empty_strings_removed(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            attributes=["floral", "", "   ", "cotton"],
        )
        assert "" not in item.attributes
        assert "floral" in item.attributes

    def test_description_stripped(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            description="  A great shirt.  ",
        )
        assert item.description == "A great shirt."

    def test_description_blank_becomes_none(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            description="   ",
        )
        assert item.description is None

    def test_subcategory_normalised(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
            subcategory="Formal Shirt",
        )
        assert item.subcategory == "formal_shirt"

    def test_subcategory_none_stays_none(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
        )
        assert item.subcategory is None

    def test_processed_at_invalid_raises(self):
        with pytest.raises(ValidationError):
            UnifiedFashionItem(
                image_id="FG_001", image_path="p.jpg",
                source_dataset="fashiongen", category="shirts",
                processed_at="not-a-date",
            )


# =============================================================================
# ── 6. TestCrossFieldValidation
# =============================================================================

class TestCrossFieldValidation:
    """Tests for model-level cross-field validators."""

    def test_dresses_with_men_generates_warning(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="dresses",
            gender="men",
        )
        assert any("dresses" in w for w in item.warnings)

    def test_dresses_with_women_no_warning(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="dresses",
            gender="women",
        )
        dress_warnings = [w for w in item.warnings if "dresses" in w]
        assert dress_warnings == []

    def test_accessories_with_fit_generates_warning(self):
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="fashiongen", category="accessories",
            fit="slim_fit",
        )
        assert any("fit" in w.lower() for w in item.warnings)

    def test_deepfashion_id_prefix_warning(self):
        """DF-sourced item with FG_ prefix should generate a warning."""
        item = UnifiedFashionItem(
            image_id="FG_001", image_path="p.jpg",
            source_dataset="deepfashion", category="shirts",
        )
        assert any("DF_" in w for w in item.warnings)

    def test_fashiongen_id_prefix_warning(self):
        """FG-sourced item with DF_ prefix should generate a warning."""
        item = UnifiedFashionItem(
            image_id="DF_001", image_path="p.jpg",
            source_dataset="fashiongen", category="shirts",
        )
        assert any("FG_" in w for w in item.warnings)

    def test_correct_prefix_no_warning(self, full_fg_item):
        prefix_warnings = [w for w in full_fg_item.warnings if "FG_" in w]
        assert prefix_warnings == []


# =============================================================================
# ── 7. TestSerialisationMethods
# =============================================================================

class TestSerialisationMethods:
    """Tests for to_dict, to_json, to_jsonl_line."""

    def test_to_dict_is_dict(self, full_fg_item):
        d = full_fg_item.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_required_fields(self, full_fg_item):
        d = full_fg_item.to_dict()
        for f in ("image_id", "image_path", "source_dataset", "category"):
            assert f in d, f"Missing key: {f}"

    def test_to_dict_enums_as_strings(self, full_fg_item):
        d = full_fg_item.to_dict()
        assert d["category"]       == "shirts"
        assert d["source_dataset"] == "fashiongen"
        assert d["gender"]         == "men"

    def test_to_dict_json_serializable(self, full_fg_item):
        d = full_fg_item.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_landmarks_as_dicts(self, full_df_item):
        d = full_df_item.to_dict()
        assert isinstance(d["landmarks"], list)
        if d["landmarks"]:
            assert isinstance(d["landmarks"][0], dict)
            assert "name" in d["landmarks"][0]

    def test_to_json_returns_string(self, full_fg_item):
        js = full_fg_item.to_json()
        assert isinstance(js, str)
        assert full_fg_item.image_id in js

    def test_to_json_indented(self, full_fg_item):
        js = full_fg_item.to_json(indent=4)
        assert "\n    " in js

    def test_to_jsonl_line_single_line(self, full_fg_item):
        line = full_fg_item.to_jsonl_line()
        assert "\n" not in line
        assert full_fg_item.image_id in line

    def test_to_jsonl_line_is_valid_json(self, full_fg_item):
        line = full_fg_item.to_jsonl_line()
        data = json.loads(line)
        assert data["image_id"] == full_fg_item.image_id


# =============================================================================
# ── 8. TestDeserialisationMethods
# =============================================================================

class TestDeserialisationMethods:
    """Tests for from_dict, from_json round-trips."""

    def test_from_dict_round_trip(self, full_fg_item):
        d     = full_fg_item.to_dict()
        item2 = UnifiedFashionItem.from_dict(d)
        assert item2.image_id == full_fg_item.image_id
        assert item2.category == full_fg_item.category

    def test_from_json_round_trip(self, full_fg_item):
        js    = full_fg_item.to_json()
        item2 = UnifiedFashionItem.from_json(js)
        assert item2.image_id == full_fg_item.image_id

    def test_from_dict_with_df_record(self, full_df_item):
        d     = full_df_item.to_dict()
        item2 = UnifiedFashionItem.from_dict(d)
        assert item2.source_dataset == DatasetSource.DEEPFASHION
        assert len(item2.landmarks)  == len(full_df_item.landmarks)

    def test_from_dict_validates_schema(self):
        """from_dict must run validators — invalid data raises ValidationError."""
        with pytest.raises(ValidationError):
            UnifiedFashionItem.from_dict({
                "image_id": "FG 001",  # whitespace → error
                "image_path": "p.jpg",
                "source_dataset": "fashiongen",
                "category": "shirts",
            })

    def test_from_json_invalid_raises(self):
        with pytest.raises(Exception):
            UnifiedFashionItem.from_json("{not valid json")


# =============================================================================
# ── 9. TestFromFashionGen
# =============================================================================

class TestFromFashionGen:
    """Tests for UnifiedFashionItem.from_fashiongen()."""

    def _make_record(self, **overrides) -> _MockFashionGenRecord:
        rec = _MockFashionGenRecord(image_id="FG_0000042", image_path="p.jpg")
        for k, v in overrides.items():
            object.__setattr__(rec, k, v)
        return rec

    def test_basic_mapping(self):
        rec  = _MockFashionGenRecord(image_id="FG_0000042", image_path="p.jpg")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.image_id        == "FG_0000042"
        assert item.source_dataset  == DatasetSource.FASHIONGEN
        assert item.category        == CategoryEnum.SHIRTS

    def test_gender_mapped(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg", gender="men")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.gender == GenderEnum.MEN

    def test_unknown_gender_becomes_none(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg", gender="robot")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.gender is None

    def test_style_mapped(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg", style="formal")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.style == StyleEnum.FORMAL

    def test_season_mapped(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg", season="summer")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.season == SeasonEnum.SUMMER

    def test_colors_extracted_from_attrs(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     attributes={"colors": ["White", "Navy"]})
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert "White" in item.color
        assert "Navy"  in item.color

    def test_fabrics_extracted_from_attrs(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     attributes={"fabrics": ["Cotton"]})
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert "Cotton" in item.fabric

    def test_fit_extracted_from_attrs(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     attributes={"fit": "slim_fit"})
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.fit == FitEnum.SLIM_FIT

    def test_occasions_extracted_from_attrs(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     attributes={"occasions": ["formal", "casual"]})
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert OccasionEnum.FORMAL in item.occasion
        assert OccasionEnum.CASUAL in item.occasion

    def test_description_passed(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     description="A great shirt.")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.description == "A great shirt."

    def test_fashiongen_has_no_landmarks(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg")
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.landmarks    == []
        assert item.bounding_box is None

    def test_is_valid_carried_over(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg", is_valid=True)
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert item.is_valid is True

    def test_errors_carried_over(self):
        rec  = _MockFashionGenRecord(image_id="FG_001", image_path="p.jpg",
                                     is_valid=False, errors=["some error"])
        item = UnifiedFashionItem.from_fashiongen(rec)
        assert "some error" in item.errors


# =============================================================================
# ── 10. TestFromDeepFashion
# =============================================================================

class TestFromDeepFashion:
    """Tests for UnifiedFashionItem.from_deepfashion()."""

    def test_basic_mapping(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.image_id       == "DF_img_Blouse_img_00000001"
        assert item.source_dataset == DatasetSource.DEEPFASHION
        assert item.category       == CategoryEnum.SHIRTS

    def test_gender_is_none(self):
        """DeepFashion has no gender annotation."""
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.gender is None

    def test_style_is_none(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.style is None

    def test_description_is_none(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.description is None

    def test_season_defaults_to_all_season(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.season == SeasonEnum.ALL_SEASON

    def test_occasions_is_empty(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.occasion == []

    def test_landmarks_parsed(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert len(item.landmarks) == 6
        assert isinstance(item.landmarks[0], LandmarkPoint)

    def test_bounding_box_parsed(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.bounding_box is not None
        assert item.bounding_box.x1 == 50

    def test_attributes_carried_over(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert "floral pattern" in item.attributes

    def test_fabric_extracted_from_attrs(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert any("cotton" in f.lower() for f in item.fabric)

    def test_fit_extracted_from_attrs(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        # "slim fit" is in attributes → should extract slim_fit
        assert item.fit == FitEnum.SLIM_FIT

    def test_category_raw_as_subcategory(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        # category_raw = "Blouse" → subcategory hint
        assert item.subcategory is not None
        assert "blouse" in item.subcategory.lower()

    def test_empty_bbox_gives_none(self):
        rec  = _MockDeepFashionRecord()
        rec.bbox = []
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.bounding_box is None

    def test_is_valid_carried_over(self):
        rec  = _MockDeepFashionRecord()
        item = UnifiedFashionItem.from_deepfashion(rec)
        assert item.is_valid is True


# =============================================================================
# ── 11. TestValidateAndReport
# =============================================================================

class TestValidateAndReport:
    """Tests for validate_and_report() method."""

    def test_valid_item_passes(self, full_fg_item):
        report = full_fg_item.validate_and_report()
        assert report.is_valid
        assert report.errors == []

    def test_report_returns_validation_report_type(self, full_fg_item):
        report = full_fg_item.validate_and_report()
        assert isinstance(report, ValidationReport)

    def test_field_coverage_zero_optional(self, minimal_item):
        report = minimal_item.validate_and_report()
        assert report.field_coverage == 0.0

    def test_field_coverage_increases_with_more_fields(self, full_fg_item):
        report = full_fg_item.validate_and_report()
        assert report.field_coverage > 0.5

    def test_backslash_in_path_error(self):
        item = UnifiedFashionItem(
            image_id="FG_001",
            image_path="p.jpg",  # validator already normalises backslash in __init__
            source_dataset="fashiongen",
            category="shirts",
        )
        # Manually inject a backslash to bypass the field validator
        object.__setattr__(item, "image_path", "datasets\\img\\p.jpg")
        report = item.validate_and_report()
        assert not report.is_valid

    def test_unusual_extension_generates_warning(self, minimal_item):
        object.__setattr__(minimal_item, "image_path", "datasets/img/p.tiff")
        report = minimal_item.validate_and_report()
        assert any("extension" in w.lower() for w in report.warnings)

    def test_no_color_generates_suggestion(self, minimal_item):
        report = minimal_item.validate_and_report()
        assert any("color" in s.lower() for s in report.suggestions)

    def test_df_wrong_landmark_count_is_warning(self):
        item = UnifiedFashionItem(
            image_id="DF_001", image_path="p.jpg",
            source_dataset="deepfashion", category="shirts",
            landmarks=[
                LandmarkPoint(name="left_collar", x=0.5, y=0.5, visible=True),
            ],
        )
        report = item.validate_and_report()
        assert any("landmark" in w.lower() for w in report.warnings)

    def test_fg_non_empty_landmarks_is_warning(self, full_fg_item):
        lm = LandmarkPoint(name="left_collar", x=0.5, y=0.5, visible=True)
        object.__setattr__(full_fg_item, "landmarks", [lm])
        report = full_fg_item.validate_and_report()
        assert any("FashionGen" in w or "landmark" in w.lower() for w in report.warnings)


# =============================================================================
# ── 12. TestFashionDatasetBatch
# =============================================================================

class TestFashionDatasetBatch:
    """Tests for FashionDatasetBatch container."""

    @pytest.fixture
    def mixed_batch(self, full_fg_item, full_df_item, minimal_item) -> FashionDatasetBatch:
        full_df_item_copy = full_df_item.model_copy(update={"is_valid": False})
        return FashionDatasetBatch(
            items  = [full_fg_item, full_df_item_copy, minimal_item],
            source = "mixed",
        )

    def test_len(self, mixed_batch):
        assert len(mixed_batch) == 3

    def test_iter(self, mixed_batch):
        items = list(mixed_batch)
        assert len(items) == 3

    def test_getitem(self, mixed_batch, full_fg_item):
        assert mixed_batch[0].image_id == full_fg_item.image_id

    def test_valid_count(self, mixed_batch):
        # full_fg_item (valid=True) + full_df_item (valid=False) + minimal (valid=False)
        assert mixed_batch.valid_count == 1

    def test_invalid_count(self, mixed_batch):
        assert mixed_batch.invalid_count == 2

    def test_valid_rate(self, mixed_batch):
        assert mixed_batch.valid_rate == pytest.approx(1 / 3, rel=0.01)

    def test_category_distribution(self, mixed_batch):
        dist = mixed_batch.category_distribution()
        assert "shirts" in dist
        assert dist["shirts"] >= 2

    def test_gender_distribution(self, mixed_batch):
        dist = mixed_batch.gender_distribution()
        assert "men"     in dist
        assert "unknown" in dist  # None → "unknown"

    def test_source_distribution(self, mixed_batch):
        dist = mixed_batch.source_distribution()
        assert "fashiongen"  in dist
        assert "deepfashion" in dist

    def test_filter_valid(self, mixed_batch):
        valid_batch = mixed_batch.filter_valid()
        assert len(valid_batch) == 1
        assert all(i.is_valid for i in valid_batch)

    def test_filter_by_category(self, mixed_batch):
        shirts = mixed_batch.filter_by_category("shirts")
        assert all(i.category == CategoryEnum.SHIRTS for i in shirts)

    def test_filter_by_source(self, mixed_batch):
        fg = mixed_batch.filter_by_source("fashiongen")
        assert all(i.source_dataset == DatasetSource.FASHIONGEN for i in fg)

    def test_filter_by_gender(self, mixed_batch):
        men = mixed_batch.filter_by_gender("men")
        assert all(i.gender == GenderEnum.MEN for i in men)

    def test_to_dict_structure(self, mixed_batch):
        d = mixed_batch.to_dict()
        assert "_meta"   in d
        assert "records" in d
        assert d["_meta"]["total"] == 3

    def test_to_dict_category_distribution_in_meta(self, mixed_batch):
        d = mixed_batch.to_dict()
        assert "category_distribution" in d["_meta"]

    def test_summary_report_is_string(self, mixed_batch):
        s = mixed_batch.summary_report()
        assert isinstance(s, str)
        assert "BATCH SUMMARY" in s


# =============================================================================
# ── 13. TestBatchSerialisation
# =============================================================================

class TestBatchSerialisation:
    """Tests for save_json, save_jsonl, load_jsonl."""

    def test_save_json(self, full_fg_item, tmp_path):
        batch = FashionDatasetBatch(items=[full_fg_item])
        path  = batch.save_json(tmp_path / "out.json")
        assert path.exists()
        data  = json.loads(path.read_text("utf-8"))
        assert data["_meta"]["total"] == 1

    def test_save_jsonl(self, full_fg_item, tmp_path):
        batch = FashionDatasetBatch(items=[full_fg_item])
        path  = batch.save_jsonl(tmp_path / "out.jsonl")
        assert path.exists()
        lines = [l for l in path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_load_jsonl_round_trip(self, full_fg_item, full_df_item, tmp_path):
        batch = FashionDatasetBatch(items=[full_fg_item, full_df_item])
        path  = batch.save_jsonl(tmp_path / "out.jsonl")
        loaded = FashionDatasetBatch.load_jsonl(path)
        assert len(loaded) == 2
        ids = {i.image_id for i in loaded}
        assert full_fg_item.image_id in ids
        assert full_df_item.image_id in ids

    def test_load_jsonl_skips_bad_lines(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"image_id": "FG_001", "image_path": "p.jpg", "source_dataset": "fashiongen", "category": "shirts"}\nnot_json_at_all\n', encoding="utf-8")
        batch = FashionDatasetBatch.load_jsonl(p)
        assert len(batch) == 1

    def test_save_json_creates_parent_dir(self, full_fg_item, tmp_path):
        nested = tmp_path / "sub" / "dir" / "out.json"
        batch  = FashionDatasetBatch(items=[full_fg_item])
        batch.save_json(nested)
        assert nested.exists()


# =============================================================================
# ── 14. TestHelperFunctions
# =============================================================================

class TestHelperFunctions:
    """Tests for the safe_* public helper functions."""

    def test_safe_category_valid(self):
        assert safe_category("shirts") == CategoryEnum.SHIRTS

    def test_safe_category_invalid_fallback(self):
        assert safe_category("space_suit") == CategoryEnum.ACCESSORIES

    def test_safe_category_empty_fallback(self):
        assert safe_category("") == CategoryEnum.ACCESSORIES

    def test_safe_category_none_fallback(self):
        assert safe_category(None) == CategoryEnum.ACCESSORIES

    def test_safe_gender_valid(self):
        assert safe_gender("men")   == GenderEnum.MEN
        assert safe_gender("women") == GenderEnum.WOMEN

    def test_safe_gender_invalid_none(self):
        assert safe_gender("alien") is None

    def test_safe_gender_empty_none(self):
        assert safe_gender("") is None

    def test_safe_style_valid(self):
        assert safe_style("formal") == StyleEnum.FORMAL

    def test_safe_style_invalid_none(self):
        assert safe_style("cyberpunk") is None

    def test_safe_season_valid(self):
        assert safe_season("summer") == SeasonEnum.SUMMER

    def test_safe_season_invalid_fallback(self):
        assert safe_season("monsoon") == SeasonEnum.ALL_SEASON

    def test_safe_fit_valid(self):
        assert safe_fit("slim_fit") == FitEnum.SLIM_FIT

    def test_safe_fit_invalid_none(self):
        assert safe_fit("perfect_fit") is None

    def test_safe_occasion_list_valid(self):
        result = safe_occasion_list(["formal", "casual"])
        assert OccasionEnum.FORMAL in result
        assert OccasionEnum.CASUAL in result

    def test_safe_occasion_list_deduplicates(self):
        result = safe_occasion_list(["formal", "formal"])
        assert result.count(OccasionEnum.FORMAL) == 1

    def test_safe_occasion_list_drops_unknown(self):
        result = safe_occasion_list(["formal", "intergalactic"])
        assert len(result) == 1

    def test_parse_landmarks_valid(self):
        raw = [
            {"name": "left_collar", "x": 0.32, "y": 0.05, "visible": True}
        ]
        points = _parse_landmarks(raw)
        assert len(points) == 1
        assert isinstance(points[0], LandmarkPoint)

    def test_parse_landmarks_clamps_out_of_range(self):
        raw = [{"name": "left_collar", "x": 1.5, "y": 0.5, "visible": True}]
        points = _parse_landmarks(raw)
        assert len(points) == 1
        assert points[0].x <= 1.0

    def test_parse_landmarks_skips_invalid_type(self):
        raw = ["not_a_dict", {"name": "left_collar", "x": 0.5, "y": 0.5, "visible": True}]
        points = _parse_landmarks(raw)
        assert len(points) == 1

    def test_parse_bounding_box_valid(self):
        bb = _parse_bounding_box([50, 30, 206, 230], [0.195, 0.117, 0.805, 0.898])
        assert bb is not None
        assert bb.x1 == 50

    def test_parse_bounding_box_empty_returns_none(self):
        assert _parse_bounding_box([], []) is None

    def test_parse_bounding_box_degenerate_returns_none(self):
        assert _parse_bounding_box([100, 0, 50, 200], []) is None

    def test_extract_colors(self):
        result = _extract_colors_from_attrs(["floral pattern", "black color", "cotton"])
        assert any("black" in r.lower() or "floral" in r.lower() for r in result)

    def test_extract_fabrics(self):
        result = _extract_fabrics_from_attrs(["cotton material", "red color"])
        assert any("cotton" in r.lower() for r in result)

    def test_extract_patterns(self):
        result = _extract_patterns_from_attrs(["striped design", "blue color"])
        assert any("strip" in r.lower() for r in result)

    def test_extract_fit(self):
        result = _extract_fit_from_attrs(["slim fit", "cotton"])
        assert result == FitEnum.SLIM_FIT

    def test_extract_fit_none_when_no_match(self):
        result = _extract_fit_from_attrs(["floral", "cotton"])
        assert result is None


# =============================================================================
# ── 15. TestSchemaDocumentation
# =============================================================================

class TestSchemaDocumentation:
    """Tests for schema documentation methods."""

    def test_get_schema_doc_returns_string(self):
        doc = UnifiedFashionItem.get_schema_doc()
        assert isinstance(doc, str)
        assert len(doc) > 100

    def test_schema_doc_contains_all_categories(self):
        doc = UnifiedFashionItem.get_schema_doc()
        for cat in ["t_shirts", "shirts", "hoodies", "jackets", "pants",
                    "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"]:
            assert cat in doc, f"Category '{cat}' missing from schema doc"

    def test_schema_doc_contains_all_styles(self):
        doc = UnifiedFashionItem.get_schema_doc()
        for style in ["streetwear", "luxury", "formal", "business_casual",
                      "techwear", "minimalist", "vintage", "athleisure"]:
            assert style in doc

    def test_schema_doc_mentions_both_datasets(self):
        doc = UnifiedFashionItem.get_schema_doc()
        assert "FashionGen"  in doc
        assert "DeepFashion" in doc

    def test_schema_json_returns_string(self):
        js = UnifiedFashionItem.schema_json()
        assert isinstance(js, str)

    def test_schema_json_is_valid_json(self):
        js = UnifiedFashionItem.schema_json()
        data = json.loads(js)
        assert "properties" in data or "title" in data

    def test_schema_version_constant(self):
        assert _SCHEMA_VERSION == "1.0.0"

    def test_model_repr(self, minimal_item):
        r = repr(minimal_item)
        assert "UnifiedFashionItem" in r
        assert minimal_item.image_id in r
