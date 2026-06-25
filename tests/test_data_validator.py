"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_data_validator.py — Unit Tests: Fashion Dataset Validation Framework
=============================================================================
Full test suite for:
  data_pipeline/validation/data_validator.py

Test Classes:
  TestSeverity                   — Severity constant values
  TestValidationIssue            — Dataclass, is_error, to_dict
  TestRecordValidationResult     — add_issue, compute_score, to_dict, errors
  TestBatchValidationResult      — aggregate stats, summary, to_dict
  TestValidationConfig           — Defaults, to_dict
  TestLayer1RequiredFields       — Missing/empty image_id, image_path, category
  TestLayer2MissingImages        — Extension, existence, size (no-disk mode)
  TestLayer3CategoryValidation   — All 11 categories, invalid values
  TestLayer4MissingMetadata      — Gender, color, season, description warnings
  TestLayer5CorruptedRecords     — Type errors, bbox, landmarks, timestamps
  TestLayer6EmptyDescriptions    — Blank, short, digit ratio, control chars
  TestLayer7InvalidAttributes    — Bad style/fit/gender/season/occasion
  TestFashionDataValidatorBatch  — Batch mode, aggregation, breakdown
  TestSaveReport                 — JSON report structure, file writing
  TestValidDatasets              — Complete valid records for FashionGen & DF
  TestEdgeCases                  — Empty batch, single record, all-error record

Total: ~175 tests

Run:
    pytest tests/test_data_validator.py -v
    pytest tests/test_data_validator.py -v --cov=data_pipeline.validation
=============================================================================
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.validation.data_validator import (
    FashionDataValidator,
    DataValidator,
    ValidationIssue,
    RecordValidationResult,
    BatchValidationResult,
    ValidationConfig,
    Severity,
)


# =============================================================================
# ── Shared Fixtures & Helpers
# =============================================================================

def _valid_fashiongen_record(**overrides) -> Dict[str, Any]:
    """Build a minimal valid FashionGen-style record dict."""
    base = {
        "image_id"      : "FG_0001",
        "image_path"    : "datasets/fashiongen/images/FG_0001.jpg",
        "category"      : "shirts",
        "source_dataset": "fashiongen",
        "gender"        : "men",
        "color"         : ["White"],
        "fabric"        : ["Cotton"],
        "pattern"       : ["solid"],
        "fit"           : "slim_fit",
        "style"         : "formal",
        "season"        : "all_season",
        "occasion"      : ["formal"],
        "description"   : "A classic slim fit white cotton dress shirt for formal occasions.",
        "attributes"    : ["collar", "button-down"],
        "landmarks"     : [],
        "bounding_box"  : None,
        "is_valid"      : True,
        "processed_at"  : "2026-06-03T12:00:00+00:00",
    }
    base.update(overrides)
    return base


def _valid_deepfashion_record(**overrides) -> Dict[str, Any]:
    """Build a minimal valid DeepFashion-style record dict."""
    base = {
        "image_id"      : "DF_001",
        "image_path"    : "datasets/deepfashion/img/Blouse/img_00000001.jpg",
        "category"      : "shirts",
        "source_dataset": "deepfashion",
        "gender"        : None,
        "color"         : [],
        "attributes"    : ["stripe", "v-neck"],
        "landmarks"     : [
            {"name": "left_collar",  "x": 0.3, "y": 0.1, "visible": True},
            {"name": "right_collar", "x": 0.7, "y": 0.1, "visible": True},
        ],
        "bounding_box"  : {"x1": 10, "y1": 20, "x2": 200, "y2": 240},
        "description"   : None,
        "season"        : "all_season",
        "is_valid"      : True,
        "processed_at"  : "2026-06-03T12:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def validator() -> FashionDataValidator:
    """Validator with image existence check disabled (no real disk files needed)."""
    cfg = ValidationConfig(
        verify_image_exists  = False,
        verify_image_readable= False,
    )
    return FashionDataValidator(config=cfg)


@pytest.fixture(scope="module")
def strict_validator() -> FashionDataValidator:
    """Validator with stricter thresholds for edge-case tests."""
    cfg = ValidationConfig(
        verify_image_exists     = False,
        min_description_chars   = 20,
        max_color_count         = 3,
    )
    return FashionDataValidator(config=cfg)


# =============================================================================
# ── 1. TestSeverity
# =============================================================================

class TestSeverity:

    def test_error_constant(self):
        assert Severity.ERROR == "ERROR"

    def test_warning_constant(self):
        assert Severity.WARNING == "WARNING"

    def test_hint_constant(self):
        assert Severity.HINT == "HINT"


# =============================================================================
# ── 2. TestValidationIssue
# =============================================================================

class TestValidationIssue:

    def test_is_error_true(self):
        issue = ValidationIssue(Severity.ERROR, "image_file", "image_path", "File missing")
        assert issue.is_error() is True

    def test_is_error_false_for_warning(self):
        issue = ValidationIssue(Severity.WARNING, "metadata", "gender", "No gender")
        assert issue.is_error() is False

    def test_is_warning_true(self):
        issue = ValidationIssue(Severity.WARNING, "metadata", "color", "No color")
        assert issue.is_warning() is True

    def test_to_dict_has_all_keys(self):
        issue = ValidationIssue(Severity.ERROR, "category", "category", "Bad value", "foo")
        d = issue.to_dict()
        assert set(d.keys()) == {"severity", "layer", "field", "message", "value"}

    def test_to_dict_value_truncated(self):
        long_val = "x" * 300
        issue = ValidationIssue(Severity.WARNING, "desc", "description", "Long", long_val)
        d = issue.to_dict()
        assert len(d["value"]) <= 200

    def test_to_dict_none_value(self):
        issue = ValidationIssue(Severity.ERROR, "required", "image_id", "Missing")
        d = issue.to_dict()
        assert d["value"] is None


# =============================================================================
# ── 3. TestRecordValidationResult
# =============================================================================

class TestRecordValidationResult:

    def test_starts_valid(self):
        r = RecordValidationResult(image_id="test_001")
        assert r.is_valid is True

    def test_add_error_marks_invalid(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_error("layer", "field", "An error")
        assert r.is_valid is False

    def test_add_warning_keeps_valid(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_warning("layer", "field", "A warning")
        assert r.is_valid is True

    def test_add_hint_keeps_valid(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_hint("layer", "field", "A hint")
        assert r.is_valid is True

    def test_errors_property(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_error("layer", "field", "Error 1")
        r.add_warning("layer", "field", "Warning 1")
        assert len(r.errors) == 1
        assert len(r.warnings) == 1

    def test_error_messages_property(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_error("layer", "field", "Error A")
        assert "Error A" in r.error_messages

    def test_compute_score_perfect(self):
        r = RecordValidationResult(image_id="test_001")
        r.checks_total  = 10
        r.checks_passed = 10
        r.compute_score()
        assert r.quality_score == 1.0

    def test_compute_score_half(self):
        r = RecordValidationResult(image_id="test_001")
        r.checks_total  = 10
        r.checks_passed = 5
        r.compute_score()
        assert r.quality_score == 0.5

    def test_compute_score_zero_total(self):
        r = RecordValidationResult(image_id="test_001")
        r.checks_total  = 0
        r.checks_passed = 0
        r.compute_score()
        assert r.quality_score == 0.0

    def test_to_dict_has_required_keys(self):
        r = RecordValidationResult(image_id="test_001")
        d = r.to_dict()
        for k in ("image_id", "is_valid", "quality_score", "error_count",
                  "warning_count", "issues", "checks_total", "checks_passed"):
            assert k in d

    def test_to_dict_issues_list(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_error("layer", "field", "err")
        d = r.to_dict()
        assert len(d["issues"]) == 1

    def test_to_dict_hints_excluded_by_default(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_hint("layer", "field", "hint")
        d = r.to_dict(include_hints=False)
        assert all(i["severity"] != Severity.HINT for i in d["issues"])

    def test_repr_valid(self):
        r = RecordValidationResult(image_id="test_001")
        assert "VALID" in repr(r)

    def test_repr_invalid(self):
        r = RecordValidationResult(image_id="test_001")
        r.add_error("layer", "field", "error")
        assert "INVALID" in repr(r)


# =============================================================================
# ── 4. TestBatchValidationResult
# =============================================================================

class TestBatchValidationResult:

    def _make_batch(self) -> BatchValidationResult:
        batch = BatchValidationResult(total_records=3)
        r1 = RecordValidationResult(image_id="A", is_valid=True,  quality_score=1.0)
        r2 = RecordValidationResult(image_id="B", is_valid=False, quality_score=0.5)
        r3 = RecordValidationResult(image_id="C", is_valid=True,  quality_score=0.8)
        r3.add_warning("L4", "gender", "Missing gender")
        batch._record_results = [r1, r2, r3]
        batch.valid_records   = 2
        batch.failed_records  = 1
        batch.warning_records = 1
        batch.success_rate    = 2 / 3
        batch.quality_score   = round((1.0 + 0.5 + 0.8) / 3, 4)
        return batch

    def test_invalid_results(self):
        batch = self._make_batch()
        assert len(batch.invalid_results()) == 1
        assert batch.invalid_results()[0].image_id == "B"

    def test_valid_results(self):
        batch = self._make_batch()
        assert len(batch.valid_results()) == 2

    def test_results_with_warnings(self):
        batch = self._make_batch()
        with_warn = batch.results_with_warnings()
        assert any(r.image_id == "C" for r in with_warn)

    def test_summary_contains_key_info(self):
        batch = self._make_batch()
        s = batch.summary()
        assert "total" in s.lower() or "TOTAL" in s
        assert "valid" in s.lower() or "VALID" in s

    def test_to_dict_summary_keys(self):
        batch = self._make_batch()
        d = batch.to_dict()
        summary = d["summary"]
        for k in ("total_records", "valid_records", "failed_records",
                  "warning_records", "success_rate", "quality_score"):
            assert k in summary

    def test_to_dict_records_list(self):
        batch = self._make_batch()
        d = batch.to_dict()
        # Invalid records should appear; valid-without-warnings should not
        ids_in_report = {r["image_id"] for r in d["records"]}
        assert "B" in ids_in_report  # failed
        assert "C" in ids_in_report  # valid but has warning

    def test_to_dict_generated_at(self):
        batch = self._make_batch()
        d = batch.to_dict()
        assert "generated_at" in d
        assert "schema_version" in d

    def test_all_errors_flat_list(self):
        batch = BatchValidationResult(total_records=1)
        r = RecordValidationResult(image_id="X")
        r.add_error("L1", "image_id", "Missing")
        r.add_error("L2", "image_path", "Not found")
        batch._record_results = [r]
        assert len(batch.all_errors) == 2

    def test_all_warnings_flat_list(self):
        batch = BatchValidationResult(total_records=1)
        r = RecordValidationResult(image_id="X")
        r.add_warning("L4", "gender", "No gender")
        batch._record_results = [r]
        assert len(batch.all_warnings) == 1


# =============================================================================
# ── 5. TestValidationConfig
# =============================================================================

class TestValidationConfig:

    def test_default_thresholds(self):
        cfg = ValidationConfig()
        assert cfg.verify_image_exists      is True
        assert cfg.verify_image_readable    is False
        assert cfg.min_image_width_px       == 32
        assert cfg.min_description_chars    == 10
        assert cfg.warn_missing_gender      is True

    def test_custom_thresholds(self):
        cfg = ValidationConfig(min_description_chars=50, max_color_count=5)
        assert cfg.min_description_chars == 50
        assert cfg.max_color_count       == 5

    def test_to_dict_has_keys(self):
        cfg = ValidationConfig()
        d   = cfg.to_dict()
        assert "min_description_chars" in d
        assert "verify_image_exists"   in d
        assert "valid_extensions"      in d

    def test_valid_extensions_include_jpg(self):
        cfg = ValidationConfig()
        assert ".jpg" in cfg.valid_extensions
        assert ".png" in cfg.valid_extensions


# =============================================================================
# ── 6. TestLayer1RequiredFields
# =============================================================================

class TestLayer1RequiredFields:

    def test_valid_record_passes(self, validator):
        rec    = _valid_fashiongen_record()
        result = validator.validate_record(rec)
        assert result.is_valid is True
        assert not result.errors

    def test_missing_image_id(self, validator):
        rec = _valid_fashiongen_record()
        del rec["image_id"]
        result = validator.validate_record(rec)
        assert any("image_id" in e.field for e in result.errors)

    def test_empty_image_id(self, validator):
        rec    = _valid_fashiongen_record(image_id="   ")
        result = validator.validate_record(rec)
        assert any("image_id" in e.field for e in result.errors)

    def test_missing_image_path(self, validator):
        rec = _valid_fashiongen_record()
        del rec["image_path"]
        result = validator.validate_record(rec)
        assert any("image_path" in e.field for e in result.errors)

    def test_empty_image_path(self, validator):
        rec    = _valid_fashiongen_record(image_path="")
        result = validator.validate_record(rec)
        assert any("image_path" in e.field for e in result.errors)

    def test_missing_category(self, validator):
        rec = _valid_fashiongen_record()
        del rec["category"]
        result = validator.validate_record(rec)
        assert any("category" in e.field for e in result.errors)

    def test_none_category(self, validator):
        rec    = _valid_fashiongen_record(category=None)
        result = validator.validate_record(rec)
        assert any("category" in e.field for e in result.errors)

    def test_missing_source_dataset(self, validator):
        rec = _valid_fashiongen_record()
        del rec["source_dataset"]
        result = validator.validate_record(rec)
        # Should still be caught even without the key
        assert result.checks_total > 0


# =============================================================================
# ── 7. TestLayer2MissingImages
# =============================================================================

class TestLayer2MissingImages:

    def test_valid_jpg_extension(self, validator):
        rec    = _valid_fashiongen_record(image_path="datasets/test.jpg")
        result = validator.validate_record(rec)
        image_errs = [e for e in result.errors if e.layer == "image_file" and "extension" in e.message.lower()]
        assert len(image_errs) == 0

    def test_invalid_extension_gif(self, validator):
        rec    = _valid_fashiongen_record(image_path="datasets/test.gif")
        result = validator.validate_record(rec)
        image_errs = [e for e in result.errors if e.layer == "image_file" and "extension" in e.message.lower()]
        assert len(image_errs) == 1

    def test_valid_png_extension(self, validator):
        rec    = _valid_fashiongen_record(image_path="datasets/img/item.png")
        result = validator.validate_record(rec)
        ext_errs = [e for e in result.errors if e.layer == "image_file" and "extension" in e.message.lower()]
        assert len(ext_errs) == 0

    def test_valid_webp_extension(self, validator):
        rec    = _valid_fashiongen_record(image_path="datasets/img/item.webp")
        result = validator.validate_record(rec)
        ext_errs = [e for e in result.errors if e.layer == "image_file" and "extension" in e.message.lower()]
        assert len(ext_errs) == 0

    def test_backslash_path_generates_warning(self, validator):
        rec     = _valid_fashiongen_record(image_path="datasets\\test.jpg")
        result  = validator.validate_record(rec)
        bslash  = [w for w in result.warnings if "backslash" in w.message.lower()]
        assert len(bslash) == 1

    def test_no_file_check_without_existence_flag(self, validator):
        # validator has verify_image_exists=False → no existence error
        rec    = _valid_fashiongen_record(image_path="does/not/exist/at/all.jpg")
        result = validator.validate_record(rec)
        exist_errs = [e for e in result.errors if "does not exist" in e.message]
        assert len(exist_errs) == 0

    def test_file_existence_error(self, tmp_path):
        cfg = ValidationConfig(
            verify_image_exists  = True,
            verify_image_readable= False,
        )
        v   = FashionDataValidator(config=cfg, project_root=str(tmp_path))
        rec = _valid_fashiongen_record(image_path="nonexistent/path.jpg")
        result = v.validate_record(rec)
        exist_errs = [e for e in result.errors if "does not exist" in e.message]
        assert len(exist_errs) == 1


# =============================================================================
# ── 8. TestLayer3CategoryValidation
# =============================================================================

class TestLayer3CategoryValidation:

    @pytest.mark.parametrize("category", [
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    ])
    def test_valid_category(self, validator, category):
        rec    = _valid_fashiongen_record(category=category)
        result = validator.validate_record(rec)
        cat_errs = [e for e in result.errors if e.layer == "category"]
        assert len(cat_errs) == 0, f"Expected no category error for '{category}', got: {cat_errs}"

    def test_invalid_category(self, validator):
        rec    = _valid_fashiongen_record(category="underwear")
        result = validator.validate_record(rec)
        cat_errs = [e for e in result.errors if e.layer == "category"]
        assert len(cat_errs) >= 1

    def test_category_case_sensitive(self, validator):
        # "Shirts" (uppercase S) should fail — taxonomy is lowercase
        rec    = _valid_fashiongen_record(category="Shirts")
        result = validator.validate_record(rec)
        cat_errs = [e for e in result.errors if e.layer == "category"]
        assert len(cat_errs) == 1

    def test_invalid_source_dataset(self, validator):
        rec    = _valid_fashiongen_record(source_dataset="someotherdataset")
        result = validator.validate_record(rec)
        src_errs = [e for e in result.errors if "source" in e.field]
        assert len(src_errs) == 1

    def test_valid_deepfashion_source(self, validator):
        rec    = _valid_deepfashion_record()
        result = validator.validate_record(rec)
        src_errs = [e for e in result.errors if "source" in e.field]
        assert len(src_errs) == 0


# =============================================================================
# ── 9. TestLayer4MissingMetadata
# =============================================================================

class TestLayer4MissingMetadata:

    def test_missing_gender_generates_warning(self, validator):
        rec    = _valid_fashiongen_record(gender=None)
        result = validator.validate_record(rec)
        gender_warns = [w for w in result.warnings if "gender" in w.field.lower()]
        assert len(gender_warns) == 1
        assert result.is_valid is True  # Warning, not error

    def test_missing_color_generates_warning(self, validator):
        rec    = _valid_fashiongen_record(color=[])
        result = validator.validate_record(rec)
        color_warns = [w for w in result.warnings if "color" in w.field.lower()]
        assert len(color_warns) == 1
        assert result.is_valid is True

    def test_missing_season_generates_warning(self, validator):
        rec    = _valid_fashiongen_record(season=None)
        result = validator.validate_record(rec)
        season_warns = [w for w in result.warnings if "season" in w.field.lower()]
        assert len(season_warns) >= 1

    def test_missing_description_error_for_fashiongen(self, validator):
        rec    = _valid_fashiongen_record(description=None)
        result = validator.validate_record(rec)
        desc_issues = [
            i for i in result.issues
            if "description" in i.field.lower()
            and i.layer == "metadata_completeness"
        ]
        assert len(desc_issues) >= 1

    def test_missing_description_deepfashion_is_warning(self, validator):
        rec    = _valid_deepfashion_record(description=None)
        result = validator.validate_record(rec)
        # DeepFashion has no description → should be warning not an error
        desc_errs = [
            e for e in result.errors
            if "description" in e.field.lower() and e.layer == "metadata_completeness"
        ]
        assert len(desc_errs) == 0

    def test_complete_metadata_no_warnings(self, validator):
        rec    = _valid_fashiongen_record()
        result = validator.validate_record(rec)
        meta_warns = [w for w in result.warnings if w.layer == "metadata_completeness"]
        assert len(meta_warns) == 0


# =============================================================================
# ── 10. TestLayer5CorruptedRecords
# =============================================================================

class TestLayer5CorruptedRecords:

    def test_image_id_non_string(self, validator):
        rec    = _valid_fashiongen_record(image_id=12345)
        result = validator.validate_record(rec)
        type_errs = [e for e in result.errors if "image_id" in e.field and "string" in e.message.lower()]
        assert len(type_errs) == 1

    def test_category_non_string(self, validator):
        rec    = _valid_fashiongen_record(category=["shirts"])
        result = validator.validate_record(rec)
        # category is a list not a str → corruption error
        assert not result.is_valid

    def test_color_non_list(self, validator):
        rec    = _valid_fashiongen_record(color="White")
        result = validator.validate_record(rec)
        col_errs = [e for e in result.errors if "color" in e.field and "list" in e.message.lower()]
        assert len(col_errs) == 1

    def test_invalid_bbox_x2_less_than_x1(self, validator):
        rec    = _valid_fashiongen_record(
            bounding_box={"x1": 100, "y1": 10, "x2": 50, "y2": 200}
        )
        result = validator.validate_record(rec)
        bbox_errs = [e for e in result.errors if "bounding_box" in e.field.lower()]
        assert len(bbox_errs) == 1

    def test_valid_bbox_passes(self, validator):
        rec    = _valid_fashiongen_record(
            bounding_box={"x1": 10, "y1": 20, "x2": 200, "y2": 240}
        )
        result = validator.validate_record(rec)
        bbox_errs = [e for e in result.errors if "bounding_box" in e.field.lower()]
        assert len(bbox_errs) == 0

    def test_invalid_landmark_missing_keys(self, validator):
        rec = _valid_fashiongen_record(landmarks=[{"name": "left_collar", "x": 0.3}])
        result = validator.validate_record(rec)
        lm_errs = [e for e in result.errors if "landmark" in e.field.lower()]
        assert len(lm_errs) >= 1

    def test_invalid_landmark_out_of_range(self, validator):
        rec = _valid_fashiongen_record(
            landmarks=[{"name": "left_collar", "x": 1.5, "y": 0.3, "visible": True}]
        )
        result = validator.validate_record(rec)
        lm_errs = [e for e in result.errors if "landmark" in e.field.lower()]
        assert len(lm_errs) >= 1

    def test_valid_landmarks_pass(self, validator):
        rec = _valid_fashiongen_record(
            landmarks=[
                {"name": "left_collar",  "x": 0.3, "y": 0.1, "visible": True},
                {"name": "right_collar", "x": 0.7, "y": 0.1, "visible": True},
            ]
        )
        result = validator.validate_record(rec)
        lm_errs = [e for e in result.errors if "landmark" in e.field.lower()]
        assert len(lm_errs) == 0

    def test_invalid_timestamp(self, validator):
        rec    = _valid_fashiongen_record(processed_at="not-a-date")
        result = validator.validate_record(rec)
        ts_errs = [e for e in result.errors if "processed_at" in e.field]
        assert len(ts_errs) == 1

    def test_valid_timestamp(self, validator):
        rec    = _valid_fashiongen_record(processed_at="2026-06-03T12:00:00+00:00")
        result = validator.validate_record(rec)
        ts_errs = [e for e in result.errors if "processed_at" in e.field]
        assert len(ts_errs) == 0

    def test_is_valid_non_bool(self, validator):
        rec    = _valid_fashiongen_record(is_valid="True")
        result = validator.validate_record(rec)
        iv_errs = [e for e in result.errors if "is_valid" in e.field]
        assert len(iv_errs) == 1

    def test_attributes_non_list(self, validator):
        rec    = _valid_fashiongen_record(attributes="collar button")
        result = validator.validate_record(rec)
        attr_errs = [e for e in result.errors if "attributes" in e.field and "list" in e.message.lower()]
        assert len(attr_errs) == 1


# =============================================================================
# ── 11. TestLayer6EmptyDescriptions
# =============================================================================

class TestLayer6EmptyDescriptions:

    def test_none_description_skipped(self, validator):
        # None description is OK for DeepFashion (Layer 4 handles it)
        rec    = _valid_deepfashion_record(description=None)
        result = validator.validate_record(rec)
        desc_l6 = [e for e in result.errors if e.layer == "description"]
        assert len(desc_l6) == 0

    def test_blank_description_error(self, validator):
        rec    = _valid_fashiongen_record(description="   ")
        result = validator.validate_record(rec)
        blank  = [e for e in result.errors if "blank" in e.message.lower()]
        assert len(blank) >= 1

    def test_too_short_description(self, validator):
        rec    = _valid_fashiongen_record(description="Hi")
        result = validator.validate_record(rec)
        short  = [e for e in result.errors if "too short" in e.message.lower()]
        assert len(short) >= 1

    def test_valid_description_passes(self, validator):
        rec    = _valid_fashiongen_record(description="A slim fit white cotton shirt.")
        result = validator.validate_record(rec)
        desc_errs = [e for e in result.errors if e.layer == "description"]
        assert len(desc_errs) == 0

    def test_digit_heavy_description_warning(self, validator):
        rec    = _valid_fashiongen_record(description="1234567890 123456789012345")
        result = validator.validate_record(rec)
        digit_warns = [w for w in result.warnings if "digit" in w.message.lower()]
        assert len(digit_warns) >= 1

    def test_control_char_in_description(self, validator):
        rec    = _valid_fashiongen_record(description="A shirt\x01with control chars")
        result = validator.validate_record(rec)
        ctrl_errs = [e for e in result.errors if "control" in e.message.lower()]
        assert len(ctrl_errs) >= 1

    def test_non_string_description(self, validator):
        rec    = _valid_fashiongen_record(description=123)
        result = validator.validate_record(rec)
        type_errs = [e for e in result.errors if e.layer == "description"]
        assert len(type_errs) >= 1

    def test_very_long_description_warning_only(self, validator):
        rec    = _valid_fashiongen_record(description="word " * 900)
        result = validator.validate_record(rec)
        # Should not be a hard error for long descriptions
        long_errs = [e for e in result.errors if e.layer == "description" and "long" in e.message.lower()]
        assert len(long_errs) == 0


    def test_empty_string_description(self, validator):
        rec    = _valid_fashiongen_record(description="")
        result = validator.validate_record(rec)
        blank  = [e for e in result.errors if e.layer == "description"]
        assert len(blank) >= 1


# =============================================================================
# ── 12. TestLayer7InvalidAttributes
# =============================================================================

class TestLayer7InvalidAttributes:

    @pytest.mark.parametrize("gender", ["men", "women", "unisex"])
    def test_valid_gender_values(self, validator, gender):
        rec    = _valid_fashiongen_record(gender=gender)
        result = validator.validate_record(rec)
        gender_errs = [e for e in result.errors if e.field == "gender" and e.layer == "attributes"]
        assert len(gender_errs) == 0

    def test_invalid_gender(self, validator):
        rec    = _valid_fashiongen_record(gender="nonbinary")
        result = validator.validate_record(rec)
        gender_errs = [e for e in result.errors if e.field == "gender" and e.layer == "attributes"]
        assert len(gender_errs) == 1

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "formal", "business_casual",
        "techwear", "minimalist", "vintage", "athleisure"
    ])
    def test_valid_style_values(self, validator, style):
        rec    = _valid_fashiongen_record(style=style)
        result = validator.validate_record(rec)
        style_errs = [e for e in result.errors if e.field == "style"]
        assert len(style_errs) == 0

    def test_invalid_style(self, validator):
        rec    = _valid_fashiongen_record(style="boho")
        result = validator.validate_record(rec)
        style_errs = [e for e in result.errors if e.field == "style"]
        assert len(style_errs) == 1

    @pytest.mark.parametrize("fit", [
        "slim_fit", "regular_fit", "relaxed_fit", "oversized",
        "cropped", "skinny", "straight", "athletic_fit"
    ])
    def test_valid_fit_values(self, validator, fit):
        rec    = _valid_fashiongen_record(fit=fit)
        result = validator.validate_record(rec)
        fit_errs = [e for e in result.errors if e.field == "fit"]
        assert len(fit_errs) == 0

    def test_invalid_fit(self, validator):
        rec    = _valid_fashiongen_record(fit="baggy_fit")
        result = validator.validate_record(rec)
        fit_errs = [e for e in result.errors if e.field == "fit"]
        assert len(fit_errs) == 1

    @pytest.mark.parametrize("season", [
        "spring", "summer", "autumn", "winter", "all_season"
    ])
    def test_valid_season_values(self, validator, season):
        rec    = _valid_fashiongen_record(season=season)
        result = validator.validate_record(rec)
        season_errs = [e for e in result.errors if e.field == "season" and e.layer == "attributes"]
        assert len(season_errs) == 0

    def test_invalid_season(self, validator):
        rec    = _valid_fashiongen_record(season="monsoon")
        result = validator.validate_record(rec)
        season_errs = [e for e in result.errors if e.field == "season" and e.layer == "attributes"]
        assert len(season_errs) == 1

    def test_invalid_occasion_value(self, validator):
        rec    = _valid_fashiongen_record(occasion=["casual", "nightclub"])
        result = validator.validate_record(rec)
        occ_errs = [e for e in result.errors if e.field == "occasion"]
        assert len(occ_errs) == 1

    def test_valid_occasion_list(self, validator):
        rec    = _valid_fashiongen_record(occasion=["casual", "formal"])
        result = validator.validate_record(rec)
        occ_errs = [e for e in result.errors if e.field == "occasion"]
        assert len(occ_errs) == 0

    def test_color_list_too_long(self, strict_validator):
        rec    = _valid_fashiongen_record(color=["Red", "Blue", "Green", "Black"])
        result = strict_validator.validate_record(rec)
        col_warns = [w for w in result.warnings if "color" in w.field and "entries" in w.message.lower()]
        assert len(col_warns) >= 1

    def test_control_char_in_attributes(self, validator):
        rec    = _valid_fashiongen_record(attributes=["collar", "button\x07down"])
        result = validator.validate_record(rec)
        ctrl_errs = [e for e in result.errors if "attributes" in e.field and "control" in e.message.lower()]
        assert len(ctrl_errs) == 1

    def test_too_many_landmarks(self, validator):
        landmarks = [
            {"name": "left_collar", "x": 0.1 * i, "y": 0.1, "visible": True}
            for i in range(10)
        ]
        rec    = _valid_fashiongen_record(landmarks=landmarks)
        result = validator.validate_record(rec)
        lm_warns = [w for w in result.warnings if "landmark" in w.field and "too many" in w.message.lower()]
        assert len(lm_warns) >= 1


# =============================================================================
# ── 13. TestFashionDataValidatorBatch
# =============================================================================

class TestFashionDataValidatorBatch:

    def test_batch_returns_correct_counts(self, validator):
        records = [
            _valid_fashiongen_record(image_id="A"),
            _valid_fashiongen_record(image_id="B", category="INVALID_CAT"),
            _valid_fashiongen_record(image_id="C"),
        ]
        batch = validator.validate_batch(records)
        assert batch.total_records   == 3
        assert batch.valid_records   == 2
        assert batch.failed_records  == 1

    def test_success_rate_computation(self, validator):
        records = [_valid_fashiongen_record(image_id=f"R{i}") for i in range(10)]
        batch   = validator.validate_batch(records)
        assert batch.success_rate == pytest.approx(1.0)

    def test_batch_quality_score_in_range(self, validator):
        records = [
            _valid_fashiongen_record(image_id="X"),
            _valid_fashiongen_record(image_id="Y", category="BAD"),
        ]
        batch = validator.validate_batch(records)
        assert 0.0 <= batch.quality_score <= 1.0

    def test_batch_error_breakdown(self, validator):
        records = [
            _valid_fashiongen_record(image_id=f"E{i}", category="not_a_real_cat")
            for i in range(5)
        ]
        batch = validator.validate_batch(records)
        assert len(batch.error_breakdown) > 0

    def test_batch_warning_records_counted(self, validator):
        records = [
            _valid_fashiongen_record(image_id=f"W{i}", gender=None)
            for i in range(3)
        ]
        batch = validator.validate_batch(records)
        assert batch.warning_records == 3

    def test_batch_preserves_record_results(self, validator):
        records = [_valid_fashiongen_record(image_id=f"P{i}") for i in range(5)]
        batch   = validator.validate_batch(records)
        assert len(batch._record_results) == 5

    def test_batch_empty_list(self, validator):
        batch = validator.validate_batch([])
        assert batch.total_records  == 0
        assert batch.valid_records  == 0
        assert batch.failed_records == 0
        assert batch.success_rate   == 0.0

    def test_batch_single_record(self, validator):
        batch = validator.validate_batch([_valid_fashiongen_record()])
        assert batch.total_records == 1

    def test_batch_processing_time_positive(self, validator):
        records = [_valid_fashiongen_record(image_id=f"T{i}") for i in range(3)]
        batch   = validator.validate_batch(records)
        assert batch.processing_time_s >= 0


# =============================================================================
# ── 14. TestSaveReport
# =============================================================================

class TestSaveReport:

    def test_save_creates_file(self, validator, tmp_path):
        records = [_valid_fashiongen_record(image_id=f"R{i}") for i in range(3)]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "validation_report.json"
        path    = validator.save_report(batch, out)
        assert path.exists()

    def test_save_creates_parent_dirs(self, validator, tmp_path):
        records = [_valid_fashiongen_record()]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "subdir" / "deep" / "validation_report.json"
        validator.save_report(batch, out)
        assert out.exists()

    def test_report_json_structure(self, validator, tmp_path):
        records = [
            _valid_fashiongen_record(image_id="A"),
            _valid_fashiongen_record(image_id="B", category="bad_cat"),
        ]
        batch = validator.validate_batch(records)
        out   = tmp_path / "report.json"
        validator.save_report(batch, out)
        data  = json.loads(out.read_text(encoding="utf-8"))

        # Top-level keys
        assert "generated_at"   in data
        assert "schema_version" in data
        assert "summary"        in data
        assert "records"        in data
        assert "config"         in data

    def test_report_summary_fields(self, validator, tmp_path):
        records = [_valid_fashiongen_record(), _valid_fashiongen_record(image_id="B", category="bad")]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "r.json"
        validator.save_report(batch, out)
        data    = json.loads(out.read_text(encoding="utf-8"))
        summary = data["summary"]

        assert "total_records"   in summary
        assert "failed_records"  in summary
        assert "warning_records" in summary
        assert "success_rate"    in summary

    def test_report_summary_values_correct(self, validator, tmp_path):
        records = [
            _valid_fashiongen_record(image_id="X"),
            _valid_fashiongen_record(image_id="Y", category="BAD"),
        ]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "r.json"
        validator.save_report(batch, out)
        data    = json.loads(out.read_text(encoding="utf-8"))
        summary = data["summary"]

        assert summary["total_records"]  == 2
        assert summary["failed_records"] == 1
        assert summary["success_rate"]   == pytest.approx(0.5)

    def test_report_is_valid_json(self, validator, tmp_path):
        records = [_valid_fashiongen_record()]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "r.json"
        validator.save_report(batch, out)
        # Should not raise
        json.loads(out.read_text(encoding="utf-8"))

    def test_report_config_section(self, validator, tmp_path):
        records = [_valid_fashiongen_record()]
        batch   = validator.validate_batch(records)
        out     = tmp_path / "r.json"
        validator.save_report(batch, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "min_description_chars" in data["config"]


# =============================================================================
# ── 15. TestValidDatasets
# =============================================================================

class TestValidDatasets:

    def test_complete_fashiongen_record_is_valid(self, validator):
        result = validator.validate_record(_valid_fashiongen_record())
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_complete_deepfashion_record_is_valid(self, validator):
        result = validator.validate_record(_valid_deepfashion_record())
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_quality_score_perfect_for_complete_record(self, validator):
        result = validator.validate_record(_valid_fashiongen_record())
        result.compute_score()
        # Perfect record should score above 0.85
        assert result.quality_score >= 0.85

    def test_multiple_valid_records_all_pass(self, validator):
        records = [
            _valid_fashiongen_record(image_id=f"FG_{i:04d}", category=cat)
            for i, cat in enumerate([
                "t_shirts", "shirts", "hoodies", "jackets", "pants",
                "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
            ])
        ]
        batch = validator.validate_batch(records)
        assert batch.failed_records == 0

    def test_backward_compat_alias(self):
        # DataValidator should be the same class as FashionDataValidator
        assert DataValidator is FashionDataValidator


# =============================================================================
# ── 16. TestEdgeCases
# =============================================================================

class TestEdgeCases:

    def test_empty_record_dict(self, validator):
        result = validator.validate_record({})
        assert result.is_valid is False

    def test_record_with_only_image_id(self, validator):
        result = validator.validate_record({"image_id": "X"})
        assert result.is_valid is False

    def test_all_fields_none(self, validator):
        rec = {
            "image_id": None, "image_path": None,
            "category": None, "source_dataset": None,
        }
        result = validator.validate_record(rec)
        assert result.is_valid is False
        assert len(result.errors) >= 3

    def test_quality_score_is_float(self, validator):
        result = validator.validate_record(_valid_fashiongen_record())
        assert isinstance(result.quality_score, float)

    def test_checks_total_positive(self, validator):
        result = validator.validate_record(_valid_fashiongen_record())
        assert result.checks_total > 0

    def test_checks_passed_lte_total(self, validator):
        result = validator.validate_record(_valid_fashiongen_record())
        assert result.checks_passed <= result.checks_total

    def test_validated_at_is_iso(self, validator):
        from datetime import datetime
        result = validator.validate_record(_valid_fashiongen_record())
        datetime.fromisoformat(result.validated_at)  # must not raise

    def test_validator_accepts_unified_schema_dict(self, validator):
        """Test validator works with UnifiedFashionItem-style dicts."""
        unified = _valid_fashiongen_record()
        unified["subcategory"]    = "formal_shirt"
        unified["schema_version"] = "1.0.0"
        result = validator.validate_record(unified)
        assert result.is_valid is True

    def test_dataset_source_key_alternate(self, validator):
        """Accept 'dataset_source' (loader style) as well as 'source_dataset'."""
        rec = _valid_fashiongen_record()
        # Rename to loader-style key
        rec["dataset_source"] = rec.pop("source_dataset")
        result = validator.validate_record(rec)
        assert result.is_valid is True
