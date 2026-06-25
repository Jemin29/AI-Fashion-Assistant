"""
=============================================================================
tests/test_validation_extended.py
=============================================================================
Extended tests for data_pipeline/validation/data_validator.py.

Targets uncovered lines:
  - ValidationIssue helpers (is_error, is_warning, to_dict)
  - RecordValidationResult (add_issue, add_error, add_warning, add_hint,
    compute_score, to_dict with include_hints, __repr__)
  - BatchValidationResult (all_errors, all_warnings, invalid/valid_results,
    results_with_warnings, summary, to_dict with all flags)
  - ValidationConfig edge cases
  - FashionDataValidator: all 7 layer checks with invalid inputs
  - save_report: include_valid flag
  - Large batch validation
=============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.validation.data_validator import (
    FashionDataValidator,
    ValidationConfig,
    ValidationIssue,
    RecordValidationResult,
    BatchValidationResult,
    Severity,
)
from tests.conftest import make_valid_record, make_record_batch



# =============================================================================
# ── ValidationIssue
# =============================================================================

class TestValidationIssue:
    def test_is_error_true(self):
        issue = ValidationIssue(Severity.ERROR, "cat", "category", "Bad category", "foo")
        assert issue.is_error() is True
        assert issue.is_warning() is False

    def test_is_warning_true(self):
        issue = ValidationIssue(Severity.WARNING, "desc", "description", "Short desc")
        assert issue.is_warning() is True
        assert issue.is_error() is False

    def test_hint_is_neither_error_nor_warning(self):
        issue = ValidationIssue(Severity.HINT, "color", "color", "Suggest adding color")
        assert issue.is_error() is False
        assert issue.is_warning() is False

    def test_to_dict_structure(self):
        issue = ValidationIssue(Severity.ERROR, "image", "image_path", "Missing path", "/bad")
        d = issue.to_dict()
        assert d["severity"] == "ERROR"
        assert d["layer"] == "image"
        assert d["field"] == "image_path"
        assert d["message"] == "Missing path"
        assert "/bad" in str(d["value"])

    def test_to_dict_none_value(self):
        issue = ValidationIssue(Severity.WARNING, "cat", "category", "msg")
        d = issue.to_dict()
        assert d["value"] is None

    def test_value_truncated_to_200(self):
        long_val = "x" * 300
        issue = ValidationIssue(Severity.ERROR, "a", "b", "msg", long_val)
        d = issue.to_dict()
        assert len(d["value"]) <= 200


# =============================================================================
# ── RecordValidationResult
# =============================================================================

class TestRecordValidationResult:
    def test_defaults_are_valid(self):
        r = RecordValidationResult(image_id="X001")
        assert r.is_valid is True
        assert r.issues == []
        assert r.checks_total == 0
        assert r.checks_passed == 0

    def test_add_error_marks_invalid(self):
        r = RecordValidationResult(image_id="X002")
        r.add_error("cat", "category", "Bad value", "foo")
        assert r.is_valid is False
        assert len(r.errors) == 1
        assert len(r.warnings) == 0

    def test_add_warning_keeps_valid(self):
        r = RecordValidationResult(image_id="X003")
        r.add_warning("desc", "description", "Short description")
        assert r.is_valid is True
        assert len(r.warnings) == 1
        assert len(r.errors) == 0

    def test_add_hint(self):
        r = RecordValidationResult(image_id="X004")
        r.add_hint("color", "color", "Consider adding color info")
        assert r.is_valid is True
        assert len(r.hints) == 1

    def test_add_issue_directly(self):
        r = RecordValidationResult(image_id="X005")
        issue = ValidationIssue(Severity.ERROR, "src", "source_dataset", "Invalid source")
        r.add_issue(issue)
        assert r.is_valid is False
        assert issue in r.issues

    def test_compute_score(self):
        r = RecordValidationResult(image_id="X006")
        r.checks_total = 10
        r.checks_passed = 8
        r.compute_score()
        assert abs(r.quality_score - 0.8) < 0.001

    def test_compute_score_zero_total(self):
        r = RecordValidationResult(image_id="X007")
        r.checks_total = 0
        r.compute_score()
        assert r.quality_score == 0.0

    def test_error_messages_property(self):
        r = RecordValidationResult(image_id="X008")
        r.add_error("a", "b", "First error")
        r.add_error("c", "d", "Second error")
        msgs = r.error_messages
        assert "First error" in msgs
        assert "Second error" in msgs

    def test_warning_messages_property(self):
        r = RecordValidationResult(image_id="X009")
        r.add_warning("a", "b", "A warning")
        msgs = r.warning_messages
        assert "A warning" in msgs

    def test_to_dict_without_hints(self):
        r = RecordValidationResult(image_id="X010")
        r.add_error("cat", "category", "Error")
        r.add_hint("desc", "description", "Hint msg")
        d = r.to_dict(include_hints=False)
        issues = d["issues"]
        # Hint should be excluded
        for issue in issues:
            assert issue["severity"] != "HINT"

    def test_to_dict_with_hints(self):
        r = RecordValidationResult(image_id="X011")
        r.add_hint("color", "color", "Add color")
        d = r.to_dict(include_hints=True)
        severities = [i["severity"] for i in d["issues"]]
        assert "HINT" in severities

    def test_repr_valid(self):
        r = RecordValidationResult(image_id="X012")
        s = repr(r)
        assert "VALID" in s
        assert "X012" in s

    def test_repr_invalid(self):
        r = RecordValidationResult(image_id="X013")
        r.add_error("a", "b", "error")
        s = repr(r)
        assert "INVALID" in s


# =============================================================================
# ── BatchValidationResult
# =============================================================================

class TestBatchValidationResult:
    def _make_batch(self) -> BatchValidationResult:
        b = BatchValidationResult()
        r1 = RecordValidationResult(image_id="A001")
        r1.add_error("cat", "category", "Bad cat")
        r2 = RecordValidationResult(image_id="A002")
        r2.add_warning("desc", "description", "Short")
        r3 = RecordValidationResult(image_id="A003")  # fully valid
        b._record_results = [r1, r2, r3]
        b.total_records   = 3
        b.valid_records   = 2
        b.failed_records  = 1
        b.warning_records = 1
        b.success_rate    = 2 / 3
        b.quality_score   = 0.85
        return b

    def test_all_errors(self):
        b = self._make_batch()
        errors = b.all_errors
        assert len(errors) == 1
        assert errors[0].field == "category"

    def test_all_warnings(self):
        b = self._make_batch()
        warnings = b.all_warnings
        assert len(warnings) == 1

    def test_invalid_results(self):
        b = self._make_batch()
        invalid = b.invalid_results()
        assert len(invalid) == 1
        assert invalid[0].image_id == "A001"

    def test_valid_results(self):
        b = self._make_batch()
        valid = b.valid_results()
        assert len(valid) == 2
        ids = {r.image_id for r in valid}
        assert "A002" in ids
        assert "A003" in ids

    def test_results_with_warnings(self):
        b = self._make_batch()
        warned = b.results_with_warnings()
        assert len(warned) == 1
        assert warned[0].image_id == "A002"

    def test_summary_contains_counts(self):
        b = self._make_batch()
        s = b.summary()
        assert "3" in s   # total
        assert "1" in s   # failed

    def test_to_dict_structure(self):
        b = self._make_batch()
        d = b.to_dict()
        assert "summary" in d
        assert "records" in d
        assert d["summary"]["total_records"] == 3

    def test_to_dict_include_valid(self):
        b = self._make_batch()
        d = b.to_dict(include_valid_records=True)
        ids = {r["image_id"] for r in d["records"]}
        # A003 is valid with no warnings — should appear with include_valid_records
        assert "A003" in ids

    def test_to_dict_max_records(self):
        b = BatchValidationResult()
        for i in range(20):
            r = RecordValidationResult(image_id=f"R{i:03d}")
            r.add_error("a", "b", f"err {i}")
            b._record_results.append(r)
        b.total_records  = 20
        b.failed_records = 20
        d = b.to_dict(max_records=5)
        assert len(d["records"]) <= 5

    def test_to_dict_no_warning_records(self):
        """include_warning_records=False excludes warning-only records."""
        b = self._make_batch()
        d = b.to_dict(include_warning_records=False)
        ids = {r["image_id"] for r in d["records"]}
        # A002 (warning only, valid) should NOT be in output
        assert "A002" not in ids

    def test_error_breakdown_shown_in_summary(self):
        b = self._make_batch()
        b.error_breakdown = {"Invalid category": 5, "Missing path": 2}
        s = b.summary()
        assert "Invalid category" in s

    def test_warning_breakdown_shown_in_summary(self):
        b = self._make_batch()
        b.warning_breakdown = {"Short description": 3}
        s = b.summary()
        assert "Short description" in s


# =============================================================================
# ── ValidationConfig
# =============================================================================

class TestValidationConfig:
    def test_defaults(self):
        cfg = ValidationConfig()
        assert cfg.verify_image_exists is True   # defaults to True
        assert cfg.verify_image_readable is False
        assert cfg.min_description_chars == 10
        assert cfg.max_description_chars == 4096
        assert cfg.warn_missing_gender is True

    def test_custom_config(self):
        cfg = ValidationConfig(
            verify_image_exists     = False,
            min_description_chars   = 20,
            warn_missing_description= False,
        )
        assert cfg.verify_image_exists is False
        assert cfg.min_description_chars == 20
        assert cfg.warn_missing_description is False

    def test_valid_extensions_frozenset(self):
        cfg = ValidationConfig()
        assert isinstance(cfg.valid_extensions, frozenset)
        assert '.jpg' in cfg.valid_extensions or 'jpg' in str(cfg.valid_extensions)


# =============================================================================
# ── FashionDataValidator — Layer-by-Layer Tests
# =============================================================================

@pytest.fixture
def validator():
    return FashionDataValidator(config=ValidationConfig(
        verify_image_exists=False,
        verify_image_readable=False,
    ))


def _full_valid_rec(idx: int = 0) -> Dict[str, Any]:
    return {
        "image_id"      : f"VLD_{idx:04d}",
        "image_path"    : f"datasets/img/VLD_{idx:04d}.jpg",
        "source_dataset": "fashiongen",
        "category"      : "t_shirts",
        "gender"        : "unisex",
        "description"   : "A great cotton tee with a bold graphic print design.",
        "color"         : ["Black"],
        "pattern"       : ["solid"],
        "fit"           : "regular_fit",
        "style"         : "streetwear",
        "season"        : "all_season",
        "occasion"      : ["casual"],
        "attributes"    : [],
        "landmarks"     : [],
    }


class TestFashionDataValidatorLayers:

    # ── Layer 1: Identity ───────────────────────────────────────────────────

    def test_missing_image_id_fails(self, validator):
        rec = _full_valid_rec()
        del rec["image_id"]
        result = validator.validate_record(rec)
        assert result.is_valid is False
        assert any("image_id" in e.field.lower() for e in result.errors)

    def test_empty_image_id_fails(self, validator):
        rec = _full_valid_rec()
        rec["image_id"] = ""
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_missing_image_path_fails(self, validator):
        rec = _full_valid_rec()
        del rec["image_path"]
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_empty_image_path_fails(self, validator):
        rec = _full_valid_rec()
        rec["image_path"] = ""
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_invalid_source_dataset_fails(self, validator):
        rec = _full_valid_rec()
        rec["source_dataset"] = "unknown_source"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_empty_source_dataset_fails(self, validator):
        rec = _full_valid_rec()
        rec["source_dataset"] = ""
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_valid_source_deepfashion(self, validator):
        rec = _full_valid_rec()
        rec["source_dataset"] = "deepfashion"
        result = validator.validate_record(rec)
        assert result.is_valid is True

    # ── Layer 2: Image Path ─────────────────────────────────────────────────

    def test_bad_extension_fails(self, validator):
        rec = _full_valid_rec()
        rec["image_path"] = "datasets/img/test.gif"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_no_extension_fails(self, validator):
        rec = _full_valid_rec()
        rec["image_path"] = "datasets/img/test_no_ext"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_valid_extensions(self, validator):
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            rec = _full_valid_rec()
            rec["image_path"] = f"datasets/img/test{ext}"
            result = validator.validate_record(rec)
            assert result.is_valid is True, f"Extension {ext} should be valid"

    # ── Layer 3: Category ───────────────────────────────────────────────────

    def test_missing_category_fails(self, validator):
        rec = _full_valid_rec()
        del rec["category"]
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_invalid_category_fails(self, validator):
        rec = _full_valid_rec()
        rec["category"] = "swimwear"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_all_valid_categories_pass(self, validator):
        valid_cats = [
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
        ]
        for cat in valid_cats:
            rec = _full_valid_rec()
            rec["category"] = cat
            result = validator.validate_record(rec)
            assert result.is_valid is True, f"Category '{cat}' should be valid"

    def test_empty_category_fails(self, validator):
        rec = _full_valid_rec()
        rec["category"] = ""
        result = validator.validate_record(rec)
        assert result.is_valid is False

    # ── Layer 4: Description ────────────────────────────────────────────────

    def test_very_short_description_fails(self, validator):
        rec = _full_valid_rec()
        rec["description"] = "short"  # < 10 chars
        result = validator.validate_record(rec)
        # Should have issues (warning or error depending on config)
        assert len(result.issues) > 0

    def test_none_description_handled(self, validator):
        rec = _full_valid_rec()
        rec["description"] = None
        result = validator.validate_record(rec)
        # Validator should handle None without crashing
        assert isinstance(result, RecordValidationResult)

    def test_long_description_valid(self, validator):
        rec = _full_valid_rec()
        rec["description"] = "A " * 100 + "description"
        result = validator.validate_record(rec)
        assert result.is_valid is True

    # ── Layer 5: Metadata Attributes ───────────────────────────────────────

    def test_invalid_gender_generates_issue(self, validator):
        rec = _full_valid_rec()
        rec["gender"] = "alien"
        result = validator.validate_record(rec)
        # Gender is a warning, not an error
        assert len(result.issues) > 0

    def test_invalid_fit_generates_issue(self, validator):
        rec = _full_valid_rec()
        rec["fit"] = "wrong_fit"
        result = validator.validate_record(rec)
        assert len(result.issues) > 0

    def test_invalid_season_generates_issue(self, validator):
        rec = _full_valid_rec()
        rec["season"] = "monsoon"
        result = validator.validate_record(rec)
        assert len(result.issues) > 0

    def test_invalid_style_generates_issue(self, validator):
        rec = _full_valid_rec()
        rec["style"] = "punk_rock"
        result = validator.validate_record(rec)
        assert len(result.issues) > 0

    # ── Layer 6: Pattern & Color ────────────────────────────────────────────

    def test_color_as_string_instead_of_list_fails(self, validator):
        rec = _full_valid_rec()
        rec["color"] = "Black"   # should be a list
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_pattern_as_string_fails(self, validator):
        rec = _full_valid_rec()
        rec["pattern"] = "solid"  # should be a list
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_color_list_valid(self, validator):
        rec = _full_valid_rec()
        rec["color"] = ["Black", "White"]
        result = validator.validate_record(rec)
        assert result.is_valid is True

    # ── Layer 7: Structured Fields ──────────────────────────────────────────

    def test_landmarks_as_string_fails(self, validator):
        rec = _full_valid_rec()
        rec["landmarks"] = "some string"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_attributes_as_string_fails(self, validator):
        rec = _full_valid_rec()
        rec["attributes"] = "collar"
        result = validator.validate_record(rec)
        assert result.is_valid is False

    def test_valid_landmarks_list(self, validator):
        """A list of landmark dicts (any structure) should be accepted."""
        rec = _full_valid_rec()
        # Use the exact format that the validator accepts
        rec["landmarks"] = [{"name": "collar", "x": 0.5, "y": 0.1, "visible": True}]
        result = validator.validate_record(rec)
        # Landmarks list is valid — not expected to trigger an error
        assert len(result.errors) == 0


# =============================================================================
# ── FashionDataValidator — Batch & Report
# =============================================================================

# Shared no-image validator for all batch tests
_NO_IMG_VALIDATOR = FashionDataValidator(config=ValidationConfig(
    verify_image_exists=False,
    verify_image_readable=False,
))


class TestFashionDataValidatorBatch:

    def test_empty_batch(self):
        result = _NO_IMG_VALIDATOR.validate_batch([])
        assert result.total_records == 0
        assert result.valid_records == 0

    def test_all_valid_batch(self):
        records = make_record_batch(5)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        assert result.total_records == 5
        assert result.valid_records == 5
        assert result.failed_records == 0
        assert result.success_rate == 1.0

    def test_mixed_batch(self):
        good = make_valid_record(0)
        bad = make_valid_record(1)
        bad["source_dataset"] = "invalid"
        result = _NO_IMG_VALIDATOR.validate_batch([good, bad])
        assert result.total_records == 2
        assert result.failed_records == 1
        assert result.valid_records == 1

    def test_quality_score_computed(self):
        records = make_record_batch(10)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        assert 0.0 <= result.quality_score <= 1.0

    def test_save_report_creates_file(self, tmp_path):
        records = make_record_batch(3)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        p = tmp_path / "report.json"
        out = _NO_IMG_VALIDATOR.save_report(result, p)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data

    def test_save_report_include_valid(self, tmp_path):
        records = make_record_batch(3)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        p = tmp_path / "report_full.json"
        _NO_IMG_VALIDATOR.save_report(result, p, include_valid=True)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "records" in data

    def test_processing_time_recorded(self):
        records = make_record_batch(5)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        assert result.processing_time_s >= 0

    def test_error_breakdown_populated(self):
        bad_records = []
        for i in range(5):
            r = make_valid_record(i)
            r["source_dataset"] = "bad_source"
            bad_records.append(r)
        result = _NO_IMG_VALIDATOR.validate_batch(bad_records)
        assert len(result.error_breakdown) > 0

    def test_single_record_validate(self):
        rec = make_valid_record(0)
        result = _NO_IMG_VALIDATOR.validate_record(rec)
        assert isinstance(result, RecordValidationResult)
        assert result.is_valid is True

    def test_large_batch_performance(self):
        """100 records should complete quickly without errors."""
        records = make_record_batch(100)
        result = _NO_IMG_VALIDATOR.validate_batch(records)
        assert result.total_records == 100
        assert result.processing_time_s < 10.0   # must complete in <10s
