"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_validation.py — Legacy Validation Tests (updated for new API)
=============================================================================
These tests were originally written for the stub DataValidator. They are
kept for backward-compatibility verification and have been updated to use
the new ValidationConfig-based API while preserving all assertions.

For the comprehensive new test suite see: tests/test_data_validator.py

Run:
    pytest tests/test_validation.py -v
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.validation import FashionDataValidator, ValidationConfig


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_minimal_record(
    image_id       : str   = "TEST_0000001",
    dataset_source : str   = "fashiongen",
    category       : str   = "dresses",
    description    : str   = "A vibrant floral summer dress with spaghetti straps.",
    **overrides,
) -> Dict[str, Any]:
    """
    Build a minimal valid record compatible with FashionDataValidator.
    No real files needed — verify_image_exists is False in all fixtures.
    """
    base = {
        "image_id"      : image_id,
        "source_dataset": dataset_source,
        "image_path"    : f"datasets/fashiongen/images/{image_id}.jpg",
        "category"      : category,
        "description"   : description,
        "gender"        : "women",
        "color"         : ["Floral"],
        "season"        : "summer",
        "occasion"      : ["casual"],
        "attributes"    : [],
        "is_valid"      : True,
        "processed_at"  : "2026-06-03T12:00:00+00:00",
    }
    base.update(overrides)
    return base


# ─── Shared validator fixture ──────────────────────────────────────────────────

@pytest.fixture
def validator():
    """DataValidator alias — file existence checking disabled for unit tests."""
    cfg = ValidationConfig(verify_image_exists=False, verify_image_readable=False)
    return FashionDataValidator(config=cfg)


# =============================================================================
# TestDataValidatorSchema  (maps to old required-field + categorical checks)
# =============================================================================

class TestDataValidatorSchema:
    """Tests for required-field validation and categorical field checks."""

    def test_valid_record_passes(self, validator):
        record = _make_minimal_record()
        result = validator.validate_record(record)
        assert result.is_valid, f"Unexpected errors: {result.error_messages}"

    def test_missing_required_field(self, validator):
        record = _make_minimal_record()
        del record["image_id"]
        result = validator.validate_record(record)
        assert not result.is_valid
        assert any("image_id" in e.field for e in result.errors)

    def test_none_required_field(self, validator):
        record = _make_minimal_record(category=None)
        result = validator.validate_record(record)
        assert not result.is_valid
        assert any("category" in e.field for e in result.errors)

    def test_invalid_split(self, validator):
        """'split' is not a field in the new schema — test invalid source instead."""
        record = _make_minimal_record(source_dataset="holdout")
        result = validator.validate_record(record)
        assert not result.is_valid

    def test_invalid_dataset_source(self, validator):
        record = _make_minimal_record(source_dataset="imagenet")
        result = validator.validate_record(record)
        assert not result.is_valid

    def test_invalid_color_mode(self, validator):
        """
        color_mode was part of the old schema. In the new schema, 'color'
        is a list of color names. Test that invalid gender is rejected.
        """
        record = _make_minimal_record(gender="alien")
        result = validator.validate_record(record)
        assert not result.is_valid


# =============================================================================
# TestDataValidatorDimensions  (maps to old image-dimension checks)
# =============================================================================

class TestDataValidatorDimensions:
    """
    The new FashionDataValidator checks image dimensions only when
    verify_image_readable=True and Pillow is installed. With exists=False
    these are skipped. We instead test bounding box geometry checks.
    """

    @pytest.fixture
    def validator(self):
        cfg = ValidationConfig(
            verify_image_exists  = False,
            verify_image_readable= False,
            min_image_width_px   = 64,
            min_image_height_px  = 64,
        )
        return FashionDataValidator(config=cfg)

    def test_too_narrow_image(self, validator):
        """Invalid bounding box (x2 <= x1) maps to the dimension-failure intent."""
        record = _make_minimal_record(
            bounding_box={"x1": 100, "y1": 10, "x2": 50, "y2": 200}
        )
        result = validator.validate_record(record)
        assert not result.is_valid
        assert any("bounding_box" in e.field for e in result.errors)

    def test_too_short_image(self, validator):
        """Invalid bounding box (y2 <= y1)."""
        record = _make_minimal_record(
            bounding_box={"x1": 10, "y1": 100, "x2": 200, "y2": 50}
        )
        result = validator.validate_record(record)
        assert not result.is_valid
        assert any("bounding_box" in e.field for e in result.errors)

    def test_exact_minimum_passes(self, validator):
        """Valid bounding box: x2 > x1 and y2 > y1."""
        record = _make_minimal_record(
            bounding_box={"x1": 0, "y1": 0, "x2": 64, "y2": 64}
        )
        result = validator.validate_record(record)
        bbox_errors = [e for e in result.errors if "bounding_box" in e.field]
        assert not bbox_errors


# =============================================================================
# TestDataValidatorDescription
# =============================================================================

class TestDataValidatorDescription:
    """Tests for text description checks."""

    @pytest.fixture
    def validator(self):
        cfg = ValidationConfig(
            verify_image_exists  = False,
            min_description_chars= 10,
        )
        return FashionDataValidator(config=cfg)

    def test_short_description_fails(self, validator):
        record = _make_minimal_record(description="Shirt")  # Only 5 chars
        result = validator.validate_record(record)
        assert not result.is_valid
        assert any("short" in e.message.lower() for e in result.errors)

    def test_adequate_description_passes(self, validator):
        record = _make_minimal_record(
            description="A long enough description of the fashion item."
        )
        result = validator.validate_record(record)
        desc_errors = [e for e in result.errors if "description" in e.field.lower()]
        assert not desc_errors

    def test_numeric_description_warns(self, validator):
        record = _make_minimal_record(description="12345678901234567890")  # All digits
        result = validator.validate_record(record)
        digit_warns = [
            w for w in result.warnings
            if "digit" in w.message.lower() or "number" in w.message.lower()
        ]
        assert digit_warns


# =============================================================================
# TestBatchValidation
# =============================================================================

class TestBatchValidation:
    """Tests for batch validation and report aggregation."""

    def test_batch_counts(self, validator):
        records = [_make_minimal_record(image_id=f"R{i:04d}") for i in range(5)]
        # Corrupt one record with an invalid source
        records[2]["source_dataset"] = "invalid_source"

        batch = validator.validate_batch(records)
        assert batch.total_records   == 5
        assert batch.valid_records   == 4
        assert batch.failed_records  == 1

    def test_batch_summary_string(self, validator):
        records = [_make_minimal_record(image_id=f"S{i:04d}") for i in range(3)]
        batch   = validator.validate_batch(records)
        summary = batch.summary()
        # Summary should mention total records
        assert "3" in summary

    def test_invalid_records_filter(self, validator):
        records = [_make_minimal_record(image_id=f"V{i:04d}") for i in range(4)]
        records[1]["description"] = "X"  # Too short
        batch    = validator.validate_batch(records)
        invalids = batch.invalid_results()
        assert len(invalids) == 1
