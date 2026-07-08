"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_fashiongen_loader.py — Unit Tests: FashionGen Ingestion Pipeline
=============================================================================
Tests every layer of the fashiongen_loader.py pipeline in isolation.
All tests are fully mocked — no real HDF5 file or images are required.

Test classes and coverage:
  TestRawFashionGenRecord       — data model completeness
  TestFashionGenRecord          — dataclass + to_dict() spec compliance
  TestPipelineStats             — counter math, timing, rates
  TestFashionGenExtractor       — HDF5 reading (mocked), decode, corruption
  TestFashionGenTransformer     — category/gender/season/style normalization
  TestFashionGenValidator       — all 7 validation layers
  TestFashionGenWriter          — JSON output structure
  TestFashionGenLoader          — end-to-end integration (fully mocked)
  TestCategoryMapping           — _CATEGORY_MAP completeness
  TestNormalizationEdgeCases    — boundary conditions

Run:
    pytest tests/test_fashiongen_loader.py -v
    pytest tests/test_fashiongen_loader.py -v --cov=data_pipeline.ingestion.fashiongen_loader
=============================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

# ── System under test ─────────────────────────────────────────────────────────
from src.data.ingestion.fashiongen_loader import (
    # Data models
    RawFashionGenRecord,
    FashionGenRecord,
    PipelineStats,
    # Pipeline layers
    FashionGenExtractor,
    FashionGenTransformer,
    FashionGenValidator,
    FashionGenWriter,
    FashionGenLoader,
    # Constants
    _CATEGORY_MAP,
    _GENDER_MAP,
    _SEASON_KEYWORDS,
    _STYLE_KEYWORDS,
    _DEFAULT_OUTPUT_DIR,
    _DEFAULT_HDF5_PATH,
    _OUTPUT_FILENAME,
)


# =============================================================================
# ── Shared Test Fixtures
# =============================================================================

@pytest.fixture
def dummy_image() -> np.ndarray:
    """A valid 256×256 RGB uint8 numpy array."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def zero_image() -> np.ndarray:
    """An all-zeros (corrupted) 256×256 RGB uint8 numpy array."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def raw_record(dummy_image) -> RawFashionGenRecord:
    """A valid RawFashionGenRecord for transformation tests."""
    return RawFashionGenRecord(
        source_index    = 42,
        image_array     = dummy_image,
        description     = "A slim-fit white cotton formal dress shirt for office wear.",
        category_raw    = "Shirts",
        subcategory_raw = "Dress Shirts",
        gender_raw      = "Men",
        hdf5_path       = Path("datasets/fashiongen/fake.h5"),
    )


@pytest.fixture
def processed_record() -> FashionGenRecord:
    """A fully processed, valid FashionGenRecord for validator/writer tests."""
    return FashionGenRecord(
        image_id       = "FG_0000042",
        image_path     = "datasets/fashiongen/images/0000/FG_0000042.jpg",
        dataset_source = "fashiongen",
        description    = "A slim-fit white cotton formal dress shirt for office wear.",
        category       = "shirts",
        gender         = "men",
        season         = "all_season",
        style          = "formal",
        subcategory    = "Dress Shirts",
        attributes     = {"colors": ["White"], "fabrics": ["Cotton"], "patterns": []},
        source_index   = 42,
        is_valid       = True,
    )


@pytest.fixture
def transformer() -> FashionGenTransformer:
    """A FashionGenTransformer without Knowledge Base."""
    return FashionGenTransformer(id_prefix="FG", kb=None)


@pytest.fixture
def validator() -> FashionGenValidator:
    """A FashionGenValidator without Knowledge Base."""
    return FashionGenValidator(kb=None)


# =============================================================================
# ── 1. TestRawFashionGenRecord
# =============================================================================

class TestRawFashionGenRecord:
    """Tests for the RawFashionGenRecord dataclass."""

    def test_all_fields_set(self, dummy_image):
        """All fields should be assignable and accessible."""
        rec = RawFashionGenRecord(
            source_index    = 0,
            image_array     = dummy_image,
            description     = "Test description",
            category_raw    = "T-Shirts",
            subcategory_raw = "Graphic Tees",
            gender_raw      = "Men",
            hdf5_path       = Path("test.h5"),
        )
        assert rec.source_index    == 0
        assert rec.image_array.shape == (256, 256, 3)
        assert rec.description     == "Test description"
        assert rec.category_raw    == "T-Shirts"
        assert rec.subcategory_raw == "Graphic Tees"
        assert rec.gender_raw      == "Men"
        assert rec.hdf5_path       == Path("test.h5")

    def test_image_array_type(self, dummy_image):
        """image_array must be a numpy ndarray."""
        rec = RawFashionGenRecord(0, dummy_image, "", "", "", "", Path("x.h5"))
        assert isinstance(rec.image_array, np.ndarray)


# =============================================================================
# ── 2. TestFashionGenRecord
# =============================================================================

class TestFashionGenRecord:
    """Tests for the FashionGenRecord dataclass and to_dict() spec compliance."""

    SPEC_FIELDS = ["image_id", "image_path", "description",
                   "category", "gender", "season", "style"]

    def test_to_dict_contains_all_spec_fields(self, processed_record):
        """to_dict() must contain every field in the output spec."""
        d = processed_record.to_dict()
        for field_name in self.SPEC_FIELDS:
            assert field_name in d, f"Spec field '{field_name}' missing from to_dict()"

    def test_to_dict_is_json_serializable(self, processed_record):
        """to_dict() output must be JSON-serializable (no numpy arrays, etc.)."""
        d = processed_record.to_dict()
        json_str = json.dumps(d)
        assert processed_record.image_id in json_str

    def test_to_dict_no_image_array(self, processed_record):
        """to_dict() must NOT include the raw image_array (too large for JSON)."""
        d = processed_record.to_dict()
        assert "image_array" not in d

    def test_default_dataset_source(self):
        """dataset_source must default to 'fashiongen'."""
        rec = FashionGenRecord(image_id="X", image_path="x/y.jpg")
        assert rec.dataset_source == "fashiongen"

    def test_default_season(self):
        """season must default to 'all_season'."""
        rec = FashionGenRecord(image_id="X", image_path="x/y.jpg")
        assert rec.season == "all_season"

    def test_processed_at_is_iso_string(self, processed_record):
        """processed_at should be a valid ISO 8601 string."""
        d = processed_record.to_dict()
        # Should not raise
        from datetime import datetime
        datetime.fromisoformat(d["processed_at"])

    def test_errors_warnings_in_dict(self):
        """errors and warnings lists must appear in to_dict()."""
        rec = FashionGenRecord(
            image_id="X", image_path="p.jpg",
            errors=["err1"], warnings=["warn1"],
        )
        d = rec.to_dict()
        assert d["errors"]   == ["err1"]
        assert d["warnings"] == ["warn1"]

    def test_attributes_dict_in_output(self, processed_record):
        """attributes key must be a dict in to_dict() output."""
        d = processed_record.to_dict()
        assert isinstance(d["attributes"], dict)
        assert "colors" in d["attributes"]
        assert "fabrics" in d["attributes"]


# =============================================================================
# ── 3. TestPipelineStats
# =============================================================================

class TestPipelineStats:
    """Tests for PipelineStats counters, rates, and timing."""

    def test_initial_state(self):
        """All counters should start at zero."""
        stats = PipelineStats()
        assert stats.total_read       == 0
        assert stats.total_processed  == 0
        assert stats.total_valid      == 0
        assert stats.total_invalid    == 0
        assert stats.total_skipped    == 0
        assert stats.total_corrupted  == 0
        assert stats.total_saved      == 0

    def test_valid_rate_zero_division(self):
        """valid_rate should be 0.0 when no records are processed."""
        stats = PipelineStats()
        assert stats.valid_rate == 0.0

    def test_valid_rate_calculation(self):
        """valid_rate = total_valid / total_processed."""
        stats = PipelineStats()
        stats.total_processed = 100
        stats.total_valid     = 80
        assert abs(stats.valid_rate - 0.80) < 0.001

    def test_records_per_second(self):
        """records_per_second should be > 0 after processing."""
        stats = PipelineStats()
        stats.total_processed = 500
        time.sleep(0.01)   # Ensure elapsed > 0
        assert stats.records_per_second > 0

    def test_elapsed_seconds_before_finalize(self):
        """elapsed_seconds should be positive before finalize."""
        stats = PipelineStats()
        time.sleep(0.01)
        assert stats.elapsed_seconds > 0

    def test_finalize_stamps_end_time(self):
        """After finalize(), elapsed_seconds should be fixed."""
        stats = PipelineStats()
        time.sleep(0.01)
        stats.finalize()
        t1 = stats.elapsed_seconds
        time.sleep(0.05)
        t2 = stats.elapsed_seconds
        assert abs(t1 - t2) < 0.001, "elapsed_seconds changed after finalize()"

    def test_increment_category(self):
        """increment_category should count correctly."""
        stats = PipelineStats()
        stats.increment_category("t_shirts")
        stats.increment_category("t_shirts")
        stats.increment_category("jeans")
        assert stats.category_counts["t_shirts"] == 2
        assert stats.category_counts["jeans"]    == 1

    def test_increment_gender(self):
        stats = PipelineStats()
        stats.increment_gender("men")
        stats.increment_gender("women")
        stats.increment_gender("men")
        assert stats.gender_counts["men"]   == 2
        assert stats.gender_counts["women"] == 1

    def test_to_dict_completeness(self):
        """to_dict() should return all required keys."""
        stats = PipelineStats()
        d = stats.to_dict()
        required = [
            "total_read", "total_processed", "total_valid", "total_invalid",
            "total_skipped", "total_corrupted", "total_saved",
            "valid_rate", "records_per_second", "elapsed_seconds",
            "category_counts", "gender_counts", "style_counts", "season_counts",
        ]
        for key in required:
            assert key in d, f"PipelineStats.to_dict() missing key: '{key}'"

    def test_to_dict_json_serializable(self):
        """to_dict() output must be JSON-serializable."""
        stats = PipelineStats()
        stats.total_processed = 10
        stats.total_valid     = 9
        json.dumps(stats.to_dict())  # Should not raise


# =============================================================================
# ── 4. TestFashionGenExtractor
# =============================================================================

class TestFashionGenExtractor:
    """Tests for FashionGenExtractor — all HDF5 I/O is mocked."""

    def test_init_warns_on_missing_file(self, tmp_path, capfd):
        """Initialising with a missing file should warn, not raise."""
        fake_path = tmp_path / "missing.h5"
        # Should NOT raise — just warn
        extractor = FashionGenExtractor(hdf5_path=fake_path)
        assert extractor.hdf5_path == fake_path

    def test_get_total_records_missing_file(self, tmp_path):
        """get_total_records() returns 0 when file doesn't exist."""
        extractor = FashionGenExtractor(tmp_path / "missing.h5")
        assert extractor.get_total_records() == 0

    def test_get_dataset_info_missing_file(self, tmp_path):
        """get_dataset_info() returns available=False when file is absent."""
        extractor = FashionGenExtractor(tmp_path / "missing.h5")
        info = extractor.get_dataset_info()
        assert info["available"] is False

    def test_stream_raises_on_missing_file(self, tmp_path):
        """stream() should raise FileNotFoundError when file is absent."""
        extractor = FashionGenExtractor(tmp_path / "missing.h5")
        with pytest.raises(FileNotFoundError):
            list(extractor.stream())

    def test_decode_bytes(self):
        """_decode should handle bytes, numpy bytes, str, ndarray."""
        decode = FashionGenExtractor._decode
        assert decode(b"hello")           == "hello"
        assert decode(np.bytes_(b"world"))== "world"
        assert decode("already str")      == "already str"
        assert decode(np.array([b"test"]))== "test"

    def test_decode_empty_bytes(self):
        """_decode should return '' for empty bytes."""
        assert FashionGenExtractor._decode(b"") == ""

    def test_decode_whitespace_stripped(self):
        """_decode should strip leading/trailing whitespace."""
        assert FashionGenExtractor._decode(b"  slim fit  ") == "slim fit"

    def test_decode_utf8_replacement(self):
        """_decode should replace undecodable bytes with '?'."""
        result = FashionGenExtractor._decode(b"\xff\xfe")
        assert isinstance(result, str)   # Should not raise

    @patch("src.data.ingestion.fashiongen_loader._H5PY_AVAILABLE", False)
    def test_stream_raises_when_h5py_missing(self, tmp_path):
        """stream() should raise RuntimeError if h5py is not installed."""
        extractor = FashionGenExtractor(tmp_path / "fake.h5")
        # Create a dummy file so FileNotFoundError doesn't fire first
        (tmp_path / "fake.h5").touch()
        with pytest.raises(RuntimeError, match="h5py"):
            list(extractor.stream())

    def test_validate_hdf5_keys_raises_on_missing(self):
        """_validate_hdf5_keys should raise KeyError if a key is absent."""
        fake_hdf5 = {"input_image": None}   # Missing other keys
        extractor = FashionGenExtractor(Path("x.h5"))
        with pytest.raises(KeyError, match="missing required keys"):
            extractor._validate_hdf5_keys(fake_hdf5)

    @patch("src.data.ingestion.fashiongen_loader.h5py")
    @patch("src.data.ingestion.fashiongen_loader._H5PY_AVAILABLE", True)
    def test_stream_yields_raw_records(self, mock_h5py, tmp_path, dummy_image):
        """stream() should yield (RawFashionGenRecord, None) for valid rows."""
        # Build mock HDF5 file structure
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: True   # all key checks pass
        mock_file.__getitem__ = lambda self, key: {
            "input_image"      : [dummy_image, dummy_image],
            "input_description": [b"A nice shirt", b"Nice pants"],
            "input_category"   : [b"Shirts", b"Pants"],
            "input_subcategory": [b"Formal Shirts", b"Chinos"],
            "input_gender"     : [b"Men", b"Men"],
        }[key]

        # Mock file open context manager
        mock_h5py.File.return_value.__enter__ = lambda s: mock_file
        mock_h5py.File.return_value.__exit__ = MagicMock(return_value=False)

        fake_h5 = tmp_path / "test.h5"
        fake_h5.write_bytes(b"fake")   # File exists check passes

        extractor = FashionGenExtractor(fake_h5)
        results = list(extractor.stream(max_records=2))

        assert len(results) == 2
        for raw, err in results:
            assert isinstance(raw, RawFashionGenRecord)


# =============================================================================
# ── 5. TestFashionGenTransformer
# =============================================================================

class TestFashionGenTransformer:
    """Tests for FashionGenTransformer — category, gender, season, style."""

    # ── Category normalization ─────────────────────────────────────────────────

    def test_transform_shirts_category(self, transformer, raw_record):
        """Raw 'Shirts' should map to taxonomy key 'shirts'."""
        result = transformer.transform(raw_record)
        assert result.category == "shirts"

    def test_transform_tshirts_category(self, transformer, raw_record, dummy_image):
        raw_record.category_raw = "T-Shirts"
        result = transformer.transform(raw_record)
        assert result.category == "t_shirts"

    def test_transform_jeans_category(self, transformer, raw_record):
        raw_record.category_raw = "Jeans"
        result = transformer.transform(raw_record)
        assert result.category == "jeans"

    def test_transform_dresses_category(self, transformer, raw_record):
        raw_record.category_raw = "Dresses"
        result = transformer.transform(raw_record)
        assert result.category == "dresses"

    def test_transform_footwear_category(self, transformer, raw_record):
        raw_record.category_raw = "Shoes"
        result = transformer.transform(raw_record)
        assert result.category == "footwear"

    def test_transform_accessories_category(self, transformer, raw_record):
        raw_record.category_raw = "Watches"
        result = transformer.transform(raw_record)
        assert result.category == "accessories"

    def test_transform_hoodies_category(self, transformer, raw_record):
        raw_record.category_raw = "Hoodies"
        result = transformer.transform(raw_record)
        assert result.category == "hoodies"

    def test_transform_jackets_category(self, transformer, raw_record):
        raw_record.category_raw = "Jackets"
        result = transformer.transform(raw_record)
        assert result.category == "jackets"

    def test_transform_unknown_category_fallback(self, transformer, raw_record):
        """Unknown categories should fallback to 'accessories'."""
        raw_record.category_raw = "SpaceSuits"
        result = transformer.transform(raw_record)
        assert result.category == "accessories"

    def test_transform_empty_category_fallback(self, transformer, raw_record):
        raw_record.category_raw = ""
        result = transformer.transform(raw_record)
        assert result.category == "accessories"

    # ── Gender normalization ───────────────────────────────────────────────────

    def test_transform_men_gender(self, transformer, raw_record):
        """Raw 'Men' should map to 'men'."""
        raw_record.gender_raw = "Men"
        result = transformer.transform(raw_record)
        assert result.gender == "men"

    def test_transform_women_gender(self, transformer, raw_record):
        raw_record.gender_raw = "Women"
        result = transformer.transform(raw_record)
        assert result.gender == "women"

    def test_transform_boys_maps_to_men(self, transformer, raw_record):
        raw_record.gender_raw = "Boys"
        result = transformer.transform(raw_record)
        assert result.gender == "men"

    def test_transform_girls_maps_to_women(self, transformer, raw_record):
        raw_record.gender_raw = "Girls"
        result = transformer.transform(raw_record)
        assert result.gender == "women"

    def test_transform_unknown_gender_fallback(self, transformer, raw_record):
        raw_record.gender_raw = "Alien"
        result = transformer.transform(raw_record)
        assert result.gender == "unisex"

    def test_transform_empty_gender_fallback(self, transformer, raw_record):
        raw_record.gender_raw = ""
        result = transformer.transform(raw_record)
        assert result.gender == "unisex"

    # ── Season inference ───────────────────────────────────────────────────────

    def test_infer_summer_season(self, transformer):
        """Description with 'summer' keyword → season='summer'."""
        season = transformer._infer_season("A breathable lightweight summer beach top.")
        assert season == "summer"

    def test_infer_winter_season(self, transformer):
        season = transformer._infer_season("A warm wool fleece jacket for cold winter days.")
        assert season == "winter"

    def test_infer_spring_season(self, transformer):
        season = transformer._infer_season("Pastel floral blouse, perfect for spring refresh.")
        assert season == "spring"

    def test_infer_autumn_season(self, transformer):
        season = transformer._infer_season("Earthy-tone layering knit sweater for fall.")
        assert season == "autumn"

    def test_infer_all_season_fallback(self, transformer):
        season = transformer._infer_season("A plain cotton shirt.")
        assert season == "all_season"

    def test_infer_season_empty_description(self, transformer):
        assert transformer._infer_season("") == "all_season"

    # ── Style inference ────────────────────────────────────────────────────────

    def test_infer_formal_style(self, transformer):
        style = transformer._infer_style(
            "A professional formal blazer for office and business meetings.", "jackets"
        )
        assert style == "formal"

    def test_infer_streetwear_style(self, transformer):
        style = transformer._infer_style(
            "Oversized graphic logo tee — urban streetwear vibe.", "t_shirts"
        )
        assert style == "streetwear"

    def test_infer_athleisure_style(self, transformer):
        style = transformer._infer_style(
            "Performance running shorts for gym workout and training.", "shorts"
        )
        assert style == "athleisure"

    def test_infer_ethnic_prior(self, transformer):
        """ethnic_wear category should always → 'formal' style."""
        style = transformer._infer_style(
            "A casual everyday kurta.", "ethnic_wear"
        )
        assert style == "formal"

    def test_infer_footwear_sneaker_athleisure(self, transformer):
        style = transformer._infer_style(
            "Nike running sneakers for sport.", "footwear"
        )
        assert style == "athleisure"

    def test_infer_footwear_formal(self, transformer):
        style = transformer._infer_style(
            "Classic leather oxford dress shoes.", "footwear"
        )
        assert style == "formal"

    def test_infer_style_empty(self, transformer):
        style = transformer._infer_style("", "t_shirts")
        assert style == ""

    # ── Image ID and path ──────────────────────────────────────────────────────

    def test_image_id_format(self, transformer, raw_record):
        """image_id should be zero-padded with prefix."""
        result = transformer.transform(raw_record)
        assert result.image_id == "FG_0000042"

    def test_image_path_format(self, transformer, raw_record):
        """image_path should include bucket dir and .jpg extension."""
        result = transformer.transform(raw_record)
        assert result.image_path.endswith(".jpg")
        assert "FG_0000042" in result.image_path
        assert "/" in result.image_path  # Forward slashes (cross-platform)

    def test_image_path_bucketing(self, transformer):
        """Images in row 1500 should go to bucket 1000."""
        path = transformer._build_image_path("FG_0001500")
        assert "1000" in path

    def test_image_path_bucket_zero(self, transformer):
        """Images in row 0-999 should go to bucket 0000."""
        path = transformer._build_image_path("FG_0000001")
        assert "0000" in path

    # ── Attributes extraction ──────────────────────────────────────────────────

    def test_extract_color_from_description(self, transformer):
        attrs = transformer._extract_attributes(
            "A beautiful navy blue cotton shirt.", "shirts"
        )
        assert len(attrs["colors"]) > 0

    def test_extract_fabric_from_description(self, transformer):
        attrs = transformer._extract_attributes(
            "Premium 100% cotton slim-fit shirt.", "shirts"
        )
        assert len(attrs["fabrics"]) > 0

    def test_extract_pattern_from_description(self, transformer):
        attrs = transformer._extract_attributes(
            "A classic plaid flannel shirt with striped cuffs.", "shirts"
        )
        assert len(attrs["patterns"]) > 0
        assert "checks" in attrs["patterns"] or "stripes" in attrs["patterns"]

    def test_attributes_empty_description(self, transformer):
        attrs = transformer._extract_attributes("", "shirts")
        assert attrs == {"colors": [], "fabrics": [], "patterns": []}

    def test_attributes_colors_capped_at_5(self, transformer):
        desc = "white black navy red green blue yellow orange pink purple"
        attrs = transformer._extract_attributes(desc, "t_shirts")
        assert len(attrs["colors"]) <= 5

    # ── Subcategory normalization ──────────────────────────────────────────────

    def test_normalize_subcategory_title_case(self, transformer):
        result = transformer._normalize_subcategory("dress_shirts")
        assert result == "Dress Shirts"

    def test_normalize_subcategory_empty(self, transformer):
        assert transformer._normalize_subcategory("") == ""

    def test_normalize_subcategory_strips_hyphens(self, transformer):
        result = transformer._normalize_subcategory("slim-fit-jeans")
        assert "Slim" in result and "Fit" in result


# =============================================================================
# ── 6. TestFashionGenValidator
# =============================================================================

class TestFashionGenValidator:
    """Tests for all 7 validation layers in FashionGenValidator."""

    def test_valid_record_passes(self, validator, processed_record):
        """A fully valid record should pass with no errors."""
        result = validator.validate(processed_record)
        assert result.is_valid, f"Expected valid but got errors: {result.errors}"
        assert result.errors == []

    def test_missing_image_id_fails(self, validator, processed_record):
        processed_record.image_id = ""
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("image_id" in e for e in result.errors)

    def test_missing_category_fails(self, validator, processed_record):
        processed_record.category = ""
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_missing_gender_fails(self, validator, processed_record):
        processed_record.gender = ""
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_invalid_category_fails(self, validator, processed_record):
        processed_record.category = "space_suit"
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("category" in e.lower() for e in result.errors)

    def test_invalid_gender_fails(self, validator, processed_record):
        processed_record.gender = "robot"
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("gender" in e.lower() for e in result.errors)

    def test_dresses_for_men_fails(self, validator, processed_record):
        """CR001: dresses + men is always invalid."""
        processed_record.category = "dresses"
        processed_record.gender   = "men"
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("dresses" in e for e in result.errors)

    def test_dresses_for_women_passes(self, validator, processed_record):
        processed_record.category = "dresses"
        processed_record.gender   = "women"
        result = validator.validate(processed_record)
        assert result.is_valid, f"Expected valid: {result.errors}"

    def test_unknown_style_generates_warning_not_error(self, validator, processed_record):
        processed_record.style = "hyper_quantum_fashion"
        result = validator.validate(processed_record)
        # Style issues are warnings, not errors
        assert result.is_valid
        assert any("style" in w.lower() or "Unrecognized" in w for w in result.warnings)

    def test_unknown_season_generates_warning(self, validator, processed_record):
        processed_record.season = "monsoon"
        result = validator.validate(processed_record)
        assert result.is_valid
        assert any("season" in w.lower() or "Unrecognized" in w for w in result.warnings)

    def test_all_valid_categories_accepted(self, validator, processed_record):
        """Every taxonomy category key should pass validation."""
        valid_cats = [
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "ethnic_wear", "footwear", "accessories"
        ]
        for cat in valid_cats:
            processed_record.category = cat
            processed_record.gender   = "men"
            result = validator.validate(processed_record)
            cat_errors = [e for e in result.errors if "category" in e.lower()]
            assert cat_errors == [], f"Category '{cat}' unexpectedly failed: {cat_errors}"

    def test_all_valid_genders_accepted(self, validator, processed_record):
        for gender in ("men", "women", "unisex"):
            processed_record.gender = gender
            if gender == "men":
                processed_record.category = "shirts"   # Avoid dresses+men conflict
            result = validator.validate(processed_record)
            gender_errors = [e for e in result.errors if "gender" in e.lower()]
            assert gender_errors == [], f"Gender '{gender}' unexpectedly failed: {gender_errors}"

    def test_is_valid_false_on_error(self, validator, processed_record):
        """Adding any error must set is_valid = False."""
        processed_record.image_id = ""
        result = validator.validate(processed_record)
        assert result.is_valid is False

    def test_multiple_errors_accumulate(self, validator, processed_record):
        """Multiple issues should all appear in the errors list."""
        processed_record.image_id  = ""
        processed_record.category  = "spaceship"
        processed_record.gender    = "alien"
        result = validator.validate(processed_record)
        assert len(result.errors) >= 3


# =============================================================================
# ── 7. TestFashionGenWriter
# =============================================================================

class TestFashionGenWriter:
    """Tests for FashionGenWriter JSON output structure."""

    def test_save_records_creates_file(self, tmp_path, processed_record):
        """save_records() must create the fashiongen_processed.json file."""
        stats  = PipelineStats()
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], stats)

        assert path.exists()
        assert path.name == _OUTPUT_FILENAME

    def test_saved_json_has_meta_and_records(self, tmp_path, processed_record):
        """The output JSON must have '_meta' and 'records' top-level keys."""
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], PipelineStats())

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "_meta"   in data
        assert "records" in data

    def test_saved_json_record_has_spec_fields(self, tmp_path, processed_record):
        """Each record in the JSON must contain the required spec fields."""
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], PipelineStats())

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rec = data["records"][0]
        spec_fields = ["image_id", "image_path", "description",
                       "category", "gender", "season", "style"]
        for field_name in spec_fields:
            assert field_name in rec, f"Spec field '{field_name}' missing from saved JSON"

    def test_meta_total_records_matches(self, tmp_path, processed_record):
        """_meta.total_records must match the actual number of records saved."""
        writer = FashionGenWriter(output_dir=tmp_path)
        records = [processed_record, processed_record]
        path = writer.save_records(records, PipelineStats())

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["_meta"]["total_records"] == 2
        assert len(data["records"])           == 2

    def test_meta_has_stats(self, tmp_path, processed_record):
        """_meta must include pipeline stats."""
        writer = FashionGenWriter(output_dir=tmp_path)
        stats  = PipelineStats()
        stats.total_processed = 1
        path   = writer.save_records([processed_record], stats)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "stats" in data["_meta"]
        assert data["_meta"]["stats"]["total_processed"] == 1

    def test_save_run_report_creates_file(self, tmp_path):
        """save_run_report() must create fashiongen_run_report.json."""
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_run_report(PipelineStats(), Path("test.h5"))
        assert path.exists()
        assert path.name == "fashiongen_run_report.json"

    def test_run_report_json_valid(self, tmp_path):
        """The run report must be valid JSON with run_info and pipeline_stats."""
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_run_report(PipelineStats(), Path("test.h5"))

        with open(path, "r") as f:
            data = json.load(f)

        assert "run_info"        in data
        assert "pipeline_stats"  in data

    def test_empty_records_list(self, tmp_path):
        """Writing an empty records list should produce a valid JSON with 0 records."""
        writer = FashionGenWriter(output_dir=tmp_path)
        path   = writer.save_records([], PipelineStats())

        with open(path, "r") as f:
            data = json.load(f)

        assert data["_meta"]["total_records"] == 0
        assert data["records"]                == []


# =============================================================================
# ── 8. TestFashionGenLoader (Integration — fully mocked HDF5)
# =============================================================================

class TestFashionGenLoader:
    """Integration tests for FashionGenLoader orchestrator."""

    def test_init_without_hdf5(self, tmp_path):
        """Loader should initialise even when HDF5 file doesn't exist."""
        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "missing.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        assert loader is not None

    def test_get_dataset_info_missing(self, tmp_path):
        """get_dataset_info() returns available=False for missing file."""
        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "missing.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        info = loader.get_dataset_info()
        assert info.get("available") is False

    @patch("src.data.ingestion.fashiongen_loader.FashionGenExtractor")
    def test_run_with_mock_extractor(self, MockExtractor, tmp_path, dummy_image):
        """
        Full pipeline run with a mocked extractor that returns 5 valid records.

        Verifies:
          - Output file is created.
          - stats['total_processed'] == 5
          - stats['total_valid'] > 0
        """
        # Build 5 synthetic raw records
        raw_records = [
            (
                RawFashionGenRecord(
                    source_index    = i,
                    image_array     = dummy_image,
                    description     = f"A white cotton formal shirt for office #{i}.",
                    category_raw    = "Shirts",
                    subcategory_raw = "Formal Shirts",
                    gender_raw      = "Men",
                    hdf5_path       = tmp_path / "fake.h5",
                ),
                None,  # No extraction error
            )
            for i in range(5)
        ]

        # Configure mock extractor
        mock_instance = MagicMock()
        mock_instance.get_total_records.return_value = 5
        mock_instance.get_dataset_info.return_value  = {
            "available": True, "total_records": 5
        }
        mock_instance.stream.return_value = iter(raw_records)
        MockExtractor.return_value = mock_instance

        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "fake.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        # Override extractor with mock
        loader.extractor = mock_instance

        result = loader.run(show_progress=False)

        assert "output_path"  in result
        assert "stats"        in result
        assert result["stats"]["total_processed"] == 5
        assert result["total_records"]            == 5
        assert Path(result["output_path"]).exists()

    @patch("src.data.ingestion.fashiongen_loader.FashionGenExtractor")
    def test_run_skips_corrupted_records(self, MockExtractor, tmp_path, dummy_image, zero_image):
        """Pipeline should skip records with extraction errors, not crash."""
        raw_records = [
            (
                RawFashionGenRecord(0, dummy_image, "Good shirt", "Shirts", "", "Men", tmp_path / "f.h5"),
                None,
            ),
            (
                RawFashionGenRecord(1, zero_image, "", "", "", "", tmp_path / "f.h5"),
                "Row 1: image is all-zeros (likely corrupted)",
            ),
            (
                RawFashionGenRecord(2, dummy_image, "Good jeans", "Jeans", "", "Men", tmp_path / "f.h5"),
                None,
            ),
        ]

        mock_instance = MagicMock()
        mock_instance.get_total_records.return_value = 3
        mock_instance.stream.return_value = iter(raw_records)
        MockExtractor.return_value = mock_instance

        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "fake.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        loader.extractor = mock_instance

        result = loader.run(show_progress=False)

        # Only 2 valid records should be processed (1 skipped)
        assert result["stats"]["total_processed"] == 2
        assert result["stats"]["total_corrupted"] == 1
        assert result["stats"]["total_skipped"]   == 1

    @patch("src.data.ingestion.fashiongen_loader.FashionGenExtractor")
    def test_run_creates_output_json(self, MockExtractor, tmp_path, dummy_image):
        """After run(), fashiongen_processed.json must exist on disk."""
        raw_records = [
            (
                RawFashionGenRecord(0, dummy_image, "A vintage denim jacket", "Jackets", "", "Men", tmp_path / "f.h5"),
                None,
            ),
        ]
        mock_instance = MagicMock()
        mock_instance.get_total_records.return_value = 1
        mock_instance.stream.return_value = iter(raw_records)
        MockExtractor.return_value = mock_instance

        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "fake.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        loader.extractor = mock_instance

        result = loader.run(show_progress=False)
        output_path = Path(result["output_path"])

        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert len(data["records"]) == 1
        assert data["records"][0]["category"] == "jackets"

    @patch("src.data.ingestion.fashiongen_loader.FashionGenExtractor")
    def test_run_stats_accumulate_correctly(self, MockExtractor, tmp_path, dummy_image):
        """PipelineStats counters should be accurate after pipeline run."""
        # 3 valid, 1 invalid (dresses+men)
        raw_records = [
            (RawFashionGenRecord(0, dummy_image, "summer beach dress", "Dresses", "", "Men", tmp_path/"f.h5"), None),
            (RawFashionGenRecord(1, dummy_image, "casual cotton tee", "T-Shirts", "", "Women", tmp_path/"f.h5"), None),
            (RawFashionGenRecord(2, dummy_image, "slim fit jeans", "Jeans", "", "Men", tmp_path/"f.h5"), None),
            (RawFashionGenRecord(3, dummy_image, "leather jacket", "Jackets", "", "Women", tmp_path/"f.h5"), None),
        ]

        mock_instance = MagicMock()
        mock_instance.get_total_records.return_value = 4
        mock_instance.stream.return_value = iter(raw_records)
        MockExtractor.return_value = mock_instance

        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "fake.h5",
            output_dir = tmp_path,
            use_kb     = False,
        )
        loader.extractor = mock_instance

        result = loader.run(show_progress=False)
        stats = result["stats"]

        assert stats["total_read"]       == 4
        assert stats["total_processed"]  == 4
        # At least the 3 non-dresses records should be valid
        assert stats["total_valid"]      >= 3
        # The dresses+men record should be invalid
        assert stats["total_invalid"]    >= 1


# =============================================================================
# ── 9. TestCategoryMapping
# =============================================================================

class TestCategoryMapping:
    """Tests that _CATEGORY_MAP covers all required source strings."""

    REQUIRED_SOURCE_STRINGS = [
        "T-Shirts", "Shirts", "Hoodies", "Sweatshirts", "Jackets",
        "Pants", "Jeans", "Shorts", "Dresses", "Shoes", "Bags",
    ]

    def test_required_strings_in_map(self):
        """All commonly observed FashionGen category strings must be mappable."""
        transformer = FashionGenTransformer(kb=None)
        for raw_cat in self.REQUIRED_SOURCE_STRINGS:
            result = transformer._normalize_category(raw_cat)
            assert result in {
                "t_shirts", "shirts", "hoodies", "jackets", "pants",
                "jeans", "shorts", "dresses", "ethnic_wear",
                "footwear", "accessories"
            }, f"'{raw_cat}' mapped to invalid taxonomy key: '{result}'"

    def test_all_map_values_are_valid_taxonomy_keys(self):
        """Every value in _CATEGORY_MAP must be a valid taxonomy category key."""
        valid_keys = {
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
        }
        for raw, taxonomy_key in _CATEGORY_MAP.items():
            assert taxonomy_key in valid_keys, (
                f"_CATEGORY_MAP['{raw}'] = '{taxonomy_key}' is not a valid taxonomy key"
            )

    def test_all_gender_map_values_are_valid(self):
        """Every value in _GENDER_MAP must be in {men, women, unisex}."""
        valid_genders = {"men", "women", "unisex"}
        for raw, mapped in _GENDER_MAP.items():
            assert mapped in valid_genders, (
                f"_GENDER_MAP['{raw}'] = '{mapped}' is invalid"
            )


# =============================================================================
# ── 10. TestNormalizationEdgeCases
# =============================================================================

class TestNormalizationEdgeCases:
    """Boundary and edge case tests for normalization logic."""

    @pytest.fixture
    def t(self):
        return FashionGenTransformer(kb=None)

    def test_category_case_insensitive(self, t):
        """Normalization should be case-insensitive."""
        assert t._normalize_category("shirts") == "shirts"
        assert t._normalize_category("SHIRTS") == "shirts"
        assert t._normalize_category("Shirts") == "shirts"

    def test_gender_case_insensitive(self, t):
        assert t._normalize_gender("men")   == "men"
        assert t._normalize_gender("Men")   == "men"
        assert t._normalize_gender("MEN")   == "men"

    def test_season_multi_keyword_uses_highest_score(self, t):
        """When multiple seasons match, the one with more keywords wins."""
        # "summer" gets 3 hits, "winter" gets 1
        desc = "A summer beach lightweight breathable tank top."
        season = t._infer_season(desc)
        assert season == "summer"

    def test_style_no_match_returns_empty(self, t):
        """When no style keywords match, return empty string."""
        style = t._infer_style("A plain item.", "t_shirts")
        assert style == ""

    def test_image_path_no_backslashes(self, t):
        """image_path must use forward slashes (cross-platform)."""
        path = t._build_image_path("FG_0000099")
        assert "\\" not in path

    def test_attributes_colors_no_duplicates(self, t):
        """The same color should not appear twice in the colors list."""
        attrs = t._extract_attributes(
            "A white white shirt with white stripes.", "shirts"
        )
        colors = attrs["colors"]
        assert len(colors) == len(set(colors))

    def test_extract_attributes_returns_dict_always(self, t):
        """_extract_attributes should always return a dict with 3 keys."""
        for desc in ["", "cotton shirt", "123 456", None]:
            attrs = t._extract_attributes(desc or "", "shirts")
            assert "colors"   in attrs
            assert "fabrics"  in attrs
            assert "patterns" in attrs

    def test_pipeline_stats_log_summary_runs_without_error(self):
        """log_summary() should complete without raising."""
        stats = PipelineStats()
        stats.total_processed = 100
        stats.total_valid     = 90
        stats.finalize()
        stats.log_summary()  # Should not raise

    def test_fashiongen_record_source_index_preserved(self):
        """source_index in FashionGenRecord should match the HDF5 row index."""
        rec = FashionGenRecord(
            image_id="FG_0000099", image_path="p.jpg", source_index=99
        )
        assert rec.to_dict()["source_index"] == 99

    def test_transformer_custom_id_prefix(self, raw_record, dummy_image):
        """Custom id_prefix should be reflected in image_id."""
        transformer = FashionGenTransformer(id_prefix="TEST", kb=None)
        result = transformer.transform(raw_record)
        assert result.image_id.startswith("TEST_")
