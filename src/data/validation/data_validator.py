"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/validation/data_validator.py
=============================================================================
MODULE  : Fashion Dataset Validation Framework
WEEK    : 1 — Fashion Domain Research & Dataset Curation
AUTHOR  : Fashion AI Team

PURPOSE
-------
Production-grade validation framework that performs 7 layers of checks on
every fashion record and emits a structured JSON validation report.

VALIDATION LAYERS
-----------------
  Layer 1 — Required Fields        : image_id, image_path, category, source
  Layer 2 — Missing Images         : file existence, extension, zero-byte guard
  Layer 3 — Category Validation    : taxonomy lock-in against VALID_CATEGORIES
  Layer 4 — Missing Metadata       : gender, color, season, description
  Layer 5 — Corrupted Records      : malformed types, JSON decode errors,
                                     negative dimensions, broken landmarks
  Layer 6 — Empty Descriptions     : blank/None, too short, all-numeric
  Layer 7 — Invalid Attributes     : unknown styles/fits, out-of-range coords,
                                     invalid season/occasion, emoji/control chars

SEVERITY LEVELS
---------------
  ERROR   : Hard failure — record is INVALID, excluded from training
  WARNING : Soft flag — record is kept, issue logged for review
  HINT    : Improvement suggestion — no impact on is_valid status

CONFIDENCE SCORING
------------------
  record_score = (checks_passed / total_checks) × 100
  dataset_quality_score = weighted mean of per-record scores

OUTPUT
------
  validate_record(record)  → RecordValidationResult
  validate_batch(records)  → BatchValidationResult
  save_report(path)        → validation_report.json  (spec requirement)

VALIDATION REPORT JSON STRUCTURE
---------------------------------
  {
    "generated_at"     : "2026-06-03T17:30:00Z",
    "schema_version"   : "1.0.0",
    "config"           : { ... validation thresholds ... },
    "summary": {
      "total_records"  : 10000,
      "valid_records"  : 9210,
      "failed_records" : 790,
      "warning_records": 1420,
      "success_rate"   : 0.921,
      "quality_score"  : 0.874,
      "error_breakdown": { ... },
      "warning_breakdown": { ... }
    },
    "records": [ ... per-record detail (only failures + warnings) ... ]
  }

USAGE
-----
  from src.data.validation import FashionDataValidator

  validator = FashionDataValidator(project_root=".")
  result    = validator.validate_record(record_dict)
  batch_res = validator.validate_batch(list_of_records)
  validator.save_report(batch_res, "datasets/metadata/validation_report.json")

ARCHITECTURE
------------
  ValidationIssue         — Dataclass: one error/warning/hint
  RecordValidationResult  — Dataclass: all issues for one record + score
  BatchValidationResult   — Dataclass: aggregate stats + per-record details
  ValidationConfig        — Dataclass: all thresholds (overridable)
  FashionDataValidator    — Orchestrator: runs 7 layers, saves report
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Third-party ──────────────────────────────────────────────────────────────
from loguru import logger

# ─── Optional: PIL for deep image validation ──────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning(
        "Pillow not installed — deep image corruption checks disabled. "
        "Install: pip install Pillow"
    )

# ─── Resolve project root ──────────────────────────────────────────────────────
_FILE_DIR    = Path(__file__).resolve().parent         # data_pipeline/validation/
_PROJECT_ROOT = _FILE_DIR.parent.parent.parent                # fashion-ai-assistant/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Knowledge-base constants (graceful inline fallback) ──────────────────────
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
    VALID_STYLES = frozenset({
        "streetwear", "luxury", "formal", "business_casual",
        "techwear", "minimalist", "vintage", "athleisure"
    })
    VALID_FITS = frozenset({
        "slim_fit", "regular_fit", "relaxed_fit", "oversized",
        "cropped", "skinny", "straight", "athletic_fit"
    })
    VALID_SEASONS   = frozenset({"spring", "summer", "autumn", "winter", "all_season"})
    VALID_OCCASIONS = frozenset({
        "casual", "business_casual", "formal", "party",
        "sport", "outdoor", "beach", "wedding_festive", "lounge"
    })
    VALID_GENDERS = frozenset({"men", "women", "unisex"})
    logger.warning("KB import failed — using inline taxonomy constants.")

# ─── Schema version constant ───────────────────────────────────────────────────
_VALIDATOR_VERSION = "1.0.0"

# ─── Valid dataset sources ─────────────────────────────────────────────────────
_VALID_SOURCES = frozenset({"fashiongen", "deepfashion"})

# ─── Valid image extensions ────────────────────────────────────────────────────
_VALID_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# ─── Regex: control characters (non-printable, non-space) ─────────────────────
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


# =============================================================================
# ── 1. Data Models
# =============================================================================

class Severity:
    """Severity constants for ValidationIssue."""
    ERROR   = "ERROR"     # Hard failure — record marked is_valid=False
    WARNING = "WARNING"   # Soft flag — record kept but flagged
    HINT    = "HINT"      # Improvement suggestion — no validity impact


@dataclass
class ValidationIssue:
    """
    A single validation finding (error, warning, or hint) for one field.

    Attributes:
        severity : ERROR | WARNING | HINT
        layer    : Which layer raised this issue (e.g., "image", "category")
        field    : The specific field name involved (e.g., "image_path")
        message  : Human-readable description of the problem
        value    : The actual field value that triggered the issue (optional)
    """
    severity : str
    layer    : str
    field    : str
    message  : str
    value    : Any = None

    def is_error(self) -> bool:
        return self.severity == Severity.ERROR

    def is_warning(self) -> bool:
        return self.severity == Severity.WARNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "layer"   : self.layer,
            "field"   : self.field,
            "message" : self.message,
            "value"   : str(self.value)[:200] if self.value is not None else None,
        }


@dataclass
class RecordValidationResult:
    """
    Complete validation result for a single fashion record.

    Attributes:
        image_id       : Record identifier.
        is_valid       : True when zero ERROR-severity issues found.
        issues         : All ValidationIssue objects (errors + warnings + hints).
        checks_total   : Total validation checks performed.
        checks_passed  : Number of checks that passed cleanly.
        quality_score  : checks_passed / checks_total  (0.0 – 1.0).
        validated_at   : ISO-8601 UTC timestamp.
    """
    image_id      : str
    is_valid      : bool       = True
    issues        : List[ValidationIssue] = field(default_factory=list)
    checks_total  : int        = 0
    checks_passed : int        = 0
    quality_score : float      = 1.0
    validated_at  : str        = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def errors(self) -> List[ValidationIssue]:
        """All ERROR-severity issues."""
        return [i for i in self.issues if i.is_error()]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """All WARNING-severity issues."""
        return [i for i in self.issues if i.is_warning()]

    @property
    def hints(self) -> List[ValidationIssue]:
        """All HINT-severity issues."""
        return [i for i in self.issues if i.severity == Severity.HINT]

    @property
    def error_messages(self) -> List[str]:
        return [i.message for i in self.errors]

    @property
    def warning_messages(self) -> List[str]:
        return [i.message for i in self.warnings]

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue and update is_valid if it's an error."""
        self.issues.append(issue)
        if issue.is_error():
            self.is_valid = False

    def add_error(self, layer: str, field: str, message: str, value: Any = None) -> None:
        """Convenience: add an ERROR-severity issue."""
        self.add_issue(ValidationIssue(Severity.ERROR, layer, field, message, value))

    def add_warning(self, layer: str, field: str, message: str, value: Any = None) -> None:
        """Convenience: add a WARNING-severity issue."""
        self.add_issue(ValidationIssue(Severity.WARNING, layer, field, message, value))

    def add_hint(self, layer: str, field: str, message: str, value: Any = None) -> None:
        """Convenience: add a HINT-severity issue."""
        self.add_issue(ValidationIssue(Severity.HINT, layer, field, message, value))

    def compute_score(self) -> None:
        """Recompute quality_score from checks_passed / checks_total."""
        if self.checks_total > 0:
            self.quality_score = round(self.checks_passed / self.checks_total, 4)
        else:
            self.quality_score = 0.0

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self, include_hints: bool = False) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        issue_list = [
            i.to_dict() for i in self.issues
            if include_hints or i.severity != Severity.HINT
        ]
        return {
            "image_id"     : self.image_id,
            "is_valid"     : self.is_valid,
            "quality_score": round(self.quality_score, 4),
            "checks_total" : self.checks_total,
            "checks_passed": self.checks_passed,
            "error_count"  : len(self.errors),
            "warning_count": len(self.warnings),
            "issues"       : issue_list,
            "validated_at" : self.validated_at,
        }

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"RecordValidationResult({status} | id={self.image_id} | "
            f"score={self.quality_score:.0%} | "
            f"err={len(self.errors)} | warn={len(self.warnings)})"
        )


@dataclass
class BatchValidationResult:
    """
    Aggregate validation result for a batch of records.

    Spec requirement fields:
        total_records   : int
        failed_records  : int
        warning_records : int
        success_rate    : float

    Extended:
        valid_records   : int
        quality_score   : float (weighted mean of per-record scores)
        error_breakdown : {error_message_prefix: count}
        warning_breakdown : {warning_message_prefix: count}
        records         : per-record results (failures + warnings only by default)
    """
    total_records    : int   = 0
    valid_records    : int   = 0
    failed_records   : int   = 0
    warning_records  : int   = 0
    success_rate     : float = 0.0
    quality_score    : float = 0.0
    error_breakdown  : Dict[str, int] = field(default_factory=dict)
    warning_breakdown: Dict[str, int] = field(default_factory=dict)
    generated_at     : str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    processing_time_s: float = 0.0
    # Full list of individual record results (in-memory)
    _record_results  : List[RecordValidationResult] = field(default_factory=list)

    # ── Computed helpers ───────────────────────────────────────────────────────

    @property
    def all_errors(self) -> List[ValidationIssue]:
        """Flat list of all ERROR issues across the batch."""
        return [issue for r in self._record_results for issue in r.errors]

    @property
    def all_warnings(self) -> List[ValidationIssue]:
        """Flat list of all WARNING issues across the batch."""
        return [issue for r in self._record_results for issue in r.warnings]

    def invalid_results(self) -> List[RecordValidationResult]:
        """Return only failed record results."""
        return [r for r in self._record_results if not r.is_valid]

    def valid_results(self) -> List[RecordValidationResult]:
        """Return only passing record results."""
        return [r for r in self._record_results if r.is_valid]

    def results_with_warnings(self) -> List[RecordValidationResult]:
        """Return records that are valid but have warnings."""
        return [r for r in self._record_results if r.is_valid and r.warnings]

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable multi-line summary report."""
        lines = [
            "=" * 72,
            "FASHION DATASET VALIDATION REPORT — SUMMARY",
            "=" * 72,
            f"  Total records    : {self.total_records:>8,}",
            f"  Valid records    : {self.valid_records:>8,}  ({self.success_rate:.1%})",
            f"  Failed records   : {self.failed_records:>8,}",
            f"  Warning records  : {self.warning_records:>8,}",
            f"  Quality score    : {self.quality_score:>8.1%}",
            f"  Processing time  : {self.processing_time_s:.2f}s",
            f"  Generated at     : {self.generated_at}",
            "-" * 72,
        ]
        if self.error_breakdown:
            lines.append("  TOP ERRORS:")
            for msg, count in list(self.error_breakdown.items())[:8]:
                lines.append(f"    [{count:>5}x] {msg}")
        if self.warning_breakdown:
            lines.append("  TOP WARNINGS:")
            for msg, count in list(self.warning_breakdown.items())[:5]:
                lines.append(f"    [{count:>5}x] {msg}")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(
        self,
        include_valid_records  : bool = False,
        include_warning_records: bool = True,
        max_records            : int  = 5000,
    ) -> Dict[str, Any]:
        """
        Serialise to a JSON-safe dict.

        Args:
            include_valid_records  : If True, also include passing records (large).
            include_warning_records: If True, include valid records with warnings.
            max_records            : Cap on number of record dicts emitted.

        Returns:
            JSON-safe dict matching the spec's validation_report.json schema.
        """
        # Select which records to include in the report
        records_to_emit: List[RecordValidationResult] = []
        records_to_emit.extend(self.invalid_results())
        if include_warning_records:
            records_to_emit.extend(self.results_with_warnings())
        if include_valid_records:
            records_to_emit.extend(self.valid_results())

        # Deduplicate and cap
        seen = set()
        unique_records = []
        for r in records_to_emit:
            if r.image_id not in seen:
                seen.add(r.image_id)
                unique_records.append(r)
        unique_records = unique_records[:max_records]

        return {
            "generated_at"     : self.generated_at,
            "schema_version"   : _VALIDATOR_VERSION,
            "summary": {
                "total_records"   : self.total_records,
                "valid_records"   : self.valid_records,
                "failed_records"  : self.failed_records,
                "warning_records" : self.warning_records,
                "success_rate"    : round(self.success_rate, 4),
                "quality_score"   : round(self.quality_score, 4),
                "processing_time_s": round(self.processing_time_s, 3),
                "error_breakdown" : self.error_breakdown,
                "warning_breakdown": self.warning_breakdown,
            },
            "records": [
                r.to_dict() for r in unique_records
            ],
        }


# =============================================================================
# ── 2. Validation Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """
    All configurable thresholds for the FashionDataValidator.

    Every threshold can be overridden at construction time so the same
    validator class works for strict training-data gates and looser
    exploratory-analysis modes.
    """
    # ── Image checks ──────────────────────────────────────────────────────────
    verify_image_exists     : bool  = True   # Check file existence on disk
    verify_image_readable   : bool  = False  # Open with Pillow (slower)
    min_image_width_px      : int   = 32     # Minimum pixel width
    min_image_height_px     : int   = 32     # Minimum pixel height
    max_image_size_mb       : float = 50.0   # Maximum file size
    valid_extensions        : frozenset = field(
        default_factory=lambda: _VALID_EXTENSIONS
    )

    # ── Description checks ────────────────────────────────────────────────────
    min_description_chars   : int   = 10    # Minimum characters
    max_description_chars   : int   = 4096  # Maximum characters
    max_digit_ratio         : float = 0.5   # Max fraction of digits in description
    require_description_for : frozenset = field(
        default_factory=lambda: frozenset({"fashiongen"})
    )  # Sources where description is mandatory

    # ── Attribute checks ──────────────────────────────────────────────────────
    max_color_count         : int   = 20    # Maximum colors in list
    max_attribute_count     : int   = 100   # Maximum attributes
    max_landmark_count      : int   = 6     # DeepFashion has exactly 6
    min_landmark_count_df   : int   = 1     # DeepFashion minimum expected

    # ── Metadata completeness checks ──────────────────────────────────────────
    warn_missing_gender     : bool  = True
    warn_missing_color      : bool  = True
    warn_missing_style      : bool  = False  # Style is often optional
    warn_missing_description: bool  = True

    # ── Project root (for resolving relative image paths) ─────────────────────
    project_root            : Optional[Path] = None

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = _PROJECT_ROOT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verify_image_exists"    : self.verify_image_exists,
            "verify_image_readable"  : self.verify_image_readable,
            "min_image_width_px"     : self.min_image_width_px,
            "min_image_height_px"    : self.min_image_height_px,
            "max_image_size_mb"      : self.max_image_size_mb,
            "valid_extensions"       : sorted(self.valid_extensions),
            "min_description_chars"  : self.min_description_chars,
            "max_description_chars"  : self.max_description_chars,
            "max_digit_ratio"        : self.max_digit_ratio,
            "max_color_count"        : self.max_color_count,
            "max_attribute_count"    : self.max_attribute_count,
            "max_landmark_count"     : self.max_landmark_count,
            "warn_missing_gender"    : self.warn_missing_gender,
            "warn_missing_color"     : self.warn_missing_color,
            "warn_missing_description": self.warn_missing_description,
        }


# =============================================================================
# ── 3. FashionDataValidator — Main Orchestrator
# =============================================================================

class FashionDataValidator:
    """
    Production-grade fashion dataset validation framework.

    Performs 7 sequential validation layers on every record and aggregates
    results into a structured JSON report.

    Validation Layers:
        1. Required fields          — critical identity + taxonomy fields
        2. Missing images           — file presence, size, extension, readability
        3. Category validation      — taxonomy lock-in (11 allowed categories)
        4. Missing metadata         — gender, color, season, description
        5. Corrupted records        — type errors, malformed data, broken fields
        6. Empty descriptions       — blank, too-short, all-numeric descriptions
        7. Invalid attributes       — bad styles/fits/occasions/patterns,
                                      out-of-range coordinates, control chars

    Args:
        config      : ValidationConfig instance (optional — defaults provided).
        project_root: Path to fashion-ai-assistant/ root for image path resolution.
    """

    # ── Layer names (used in ValidationIssue.layer) ───────────────────────────
    _L1 = "required_fields"
    _L2 = "image_file"
    _L3 = "category"
    _L4 = "metadata_completeness"
    _L5 = "corruption"
    _L6 = "description"
    _L7 = "attributes"

    def __init__(
        self,
        config      : Optional[ValidationConfig] = None,
        project_root: Optional[str | Path] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        if project_root:
            self.config.project_root = Path(project_root)

        logger.info(
            f"FashionDataValidator v{_VALIDATOR_VERSION} initialised | "
            f"image_check={self.config.verify_image_exists} | "
            f"deep_image={self.config.verify_image_readable} | "
            f"min_desc={self.config.min_description_chars}chars"
        )

    # =========================================================================
    # ── Public API
    # =========================================================================

    def validate_record(self, record: Dict[str, Any]) -> RecordValidationResult:
        """
        Run all 7 validation layers on a single fashion record dict.

        Compatible with dicts from:
          - FashionGenRecord.to_dict()
          - DeepFashionRecord dict
          - UnifiedFashionItem.to_dict()
          - any dict with image_id, image_path, category, source_dataset keys

        Args:
            record : Fashion item dict.

        Returns:
            RecordValidationResult with is_valid, issues, and quality_score.
        """
        image_id = str(record.get("image_id") or record.get("image_id", "unknown"))
        result   = RecordValidationResult(image_id=image_id)

        # ── Run each layer; track check counts ────────────────────────────────
        layers = [
            self._check_required_fields,
            self._check_image_file,
            self._check_category,
            self._check_metadata_completeness,
            self._check_corruption,
            self._check_description,
            self._check_attributes,
        ]

        for layer_fn in layers:
            passed, total = layer_fn(record, result)
            result.checks_passed += passed
            result.checks_total  += total

        result.compute_score()

        log_level = "DEBUG" if result.is_valid else "WARNING"
        logger.log(
            log_level,
            f"Validated {image_id} | "
            f"{'PASS' if result.is_valid else 'FAIL'} | "
            f"score={result.quality_score:.0%} | "
            f"err={len(result.errors)} | warn={len(result.warnings)}"
        )
        return result

    def validate_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> BatchValidationResult:
        """
        Validate a list of records and return an aggregate report.

        Args:
            records : List of fashion item dicts.

        Returns:
            BatchValidationResult with summary statistics and per-record details.
        """
        t_start = time.perf_counter()
        logger.info(f"Starting batch validation: {len(records):,} records")

        batch = BatchValidationResult(total_records=len(records))

        for i, record in enumerate(records):
            try:
                res = self.validate_record(record)
                batch._record_results.append(res)

                if res.is_valid:
                    batch.valid_records += 1
                else:
                    batch.failed_records += 1

                if res.warnings:
                    batch.warning_records += 1

            except Exception as exc:
                # Unhandled exception → treat record as corrupted
                img_id = str(record.get("image_id", f"index_{i}"))
                logger.error(f"Validation crashed on record {img_id}: {exc}")
                failed_res = RecordValidationResult(image_id=img_id)
                failed_res.add_error(self._L5, "record", f"Validation exception: {exc}")
                failed_res.compute_score()
                batch._record_results.append(failed_res)
                batch.failed_records += 1

            if (i + 1) % 500 == 0:
                logger.info(
                    f"  Progress: {i+1:,}/{len(records):,} "
                    f"({(i+1)/len(records):.0%}) | "
                    f"valid={batch.valid_records} fail={batch.failed_records}"
                )

        # ── Compute aggregate statistics ──────────────────────────────────────
        batch.success_rate = (
            batch.valid_records / batch.total_records
            if batch.total_records > 0 else 0.0
        )
        scores = [r.quality_score for r in batch._record_results]
        batch.quality_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        batch.error_breakdown   = self._build_breakdown(batch.all_errors)
        batch.warning_breakdown = self._build_breakdown(batch.all_warnings)
        batch.processing_time_s = round(time.perf_counter() - t_start, 3)

        logger.success(
            f"Batch validation complete | "
            f"total={batch.total_records:,} | "
            f"valid={batch.valid_records:,} ({batch.success_rate:.1%}) | "
            f"failed={batch.failed_records:,} | "
            f"score={batch.quality_score:.1%} | "
            f"time={batch.processing_time_s:.2f}s"
        )
        logger.info(batch.summary())
        return batch

    def save_report(
        self,
        batch_result : BatchValidationResult,
        output_path  : str | Path,
        include_valid: bool = False,
        indent       : int  = 2,
    ) -> Path:
        """
        Serialise the batch result to a validation_report.json file.

        The JSON structure satisfies the spec:
          {
            "summary": {
              "total_records"  : int,
              "failed_records" : int,
              "warning_records": int,
              "success_rate"   : float,
              ... more fields ...
            },
            "records": [ ... per-record details (failures + warnings) ... ]
          }

        Args:
            batch_result : BatchValidationResult from validate_batch().
            output_path  : Path to write the JSON file to.
            include_valid: If True, also write passing records (large output).
            indent       : JSON indentation level.

        Returns:
            Resolved absolute path to the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = batch_result.to_dict(
            include_valid_records   = include_valid,
            include_warning_records = True,
        )
        # Inject config so the report is self-documenting
        report_dict["config"] = self.config.to_dict()

        path.write_text(
            json.dumps(report_dict, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )

        size_kb = path.stat().st_size / 1024
        logger.success(
            f"Validation report saved | {path} | {size_kb:.1f} KB | "
            f"{len(report_dict['records'])} record entries"
        )
        return path.resolve()

    # =========================================================================
    # ── Layer 1: Required Fields
    # =========================================================================

    def _check_required_fields(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Verify that the four critical identity fields are present and non-empty.

        Required: image_id, image_path, category, source_dataset (or dataset_source).

        Returns:
            (checks_passed, checks_total) tuple.
        """
        REQUIRED = [
            ("image_id",       "Unique record identifier"),
            ("image_path",     "Path to the fashion image file"),
            ("category",       "Product category (taxonomy key)"),
        ]
        # Accept both 'source_dataset' (schema) and 'dataset_source' (loader dicts)
        source_key = "source_dataset" if "source_dataset" in record else "dataset_source"
        REQUIRED.append((source_key, "Source dataset name (fashiongen/deepfashion)"))

        passed = 0
        total  = len(REQUIRED)

        for field_name, description in REQUIRED:
            value = record.get(field_name)
            if value is None:
                result.add_error(
                    self._L1, field_name,
                    f"Required field '{field_name}' is missing: {description}",
                    value,
                )
            elif isinstance(value, str) and not value.strip():
                result.add_error(
                    self._L1, field_name,
                    f"Required field '{field_name}' is empty (blank string)",
                    value,
                )
            elif isinstance(value, (list, dict)) and len(value) == 0:
                # Lists like color/attributes can be empty; only flag if it's
                # a required identity field with no content
                result.add_error(
                    self._L1, field_name,
                    f"Required field '{field_name}' is empty (empty container)",
                    value,
                )
            else:
                passed += 1

        return passed, total

    # =========================================================================
    # ── Layer 2: Missing Images
    # =========================================================================

    def _check_image_file(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Validate image file existence, extension, size, and optionally readability.

        Checks:
          1. image_path field has a valid extension
          2. image_path contains no backslashes (POSIX convention)
          3. File exists on disk (if config.verify_image_exists)
          4. File is not zero-byte
          5. File size does not exceed maximum
          6. File is readable by Pillow (if config.verify_image_readable)
        """
        passed = 0
        total  = 0
        path_raw: str = record.get("image_path") or ""

        if not path_raw:
            # Already caught by Layer 1 — skip to avoid duplicate errors
            return 0, 0

        # ── Check 1: Extension ──────────────────────────────────────────────
        total += 1
        suffix = Path(path_raw).suffix.lower()
        if suffix not in self.config.valid_extensions:
            result.add_error(
                self._L2, "image_path",
                f"Unsupported image extension '{suffix}'. "
                f"Expected one of: {sorted(self.config.valid_extensions)}",
                path_raw,
            )
        else:
            passed += 1

        # ── Check 2: Forward slashes ────────────────────────────────────────
        total += 1
        if "\\" in path_raw:
            result.add_warning(
                self._L2, "image_path",
                "image_path uses backslashes — should be forward slashes",
                path_raw,
            )
            # Still count as passed (warning only)
            passed += 1
        else:
            passed += 1

        # ── Resolve absolute path for existence checks ──────────────────────
        if not self.config.verify_image_exists:
            return passed, total

        abs_path: Optional[Path] = None
        path_obj = Path(path_raw)
        if path_obj.is_absolute():
            abs_path = path_obj
        else:
            abs_path = self.config.project_root / path_obj

        # ── Check 3: Existence ───────────────────────────────────────────────
        total += 1
        if not abs_path.exists():
            result.add_error(
                self._L2, "image_path",
                f"Image file does not exist: {abs_path}",
                str(abs_path),
            )
            return passed, total  # Can't check size/readability if file missing
        else:
            passed += 1

        # ── Check 4: Non-zero size ───────────────────────────────────────────
        total += 1
        try:
            file_size = abs_path.stat().st_size
        except OSError:
            file_size = -1

        if file_size == 0:
            result.add_error(
                self._L2, "image_path",
                f"Image file is empty (0 bytes): {abs_path}",
                str(abs_path),
            )
        else:
            passed += 1

        # ── Check 5: Max size ────────────────────────────────────────────────
        total += 1
        max_bytes = int(self.config.max_image_size_mb * 1_000_000)
        if 0 < file_size > max_bytes:
            result.add_error(
                self._L2, "image_path",
                f"Image file too large: {file_size/1e6:.1f}MB "
                f"(max {self.config.max_image_size_mb}MB)",
                str(abs_path),
            )
        else:
            passed += 1

        # ── Check 6: Readability (PIL, optional) ─────────────────────────────
        if self.config.verify_image_readable and _PIL_AVAILABLE:
            total += 1
            try:
                with PILImage.open(abs_path) as img:
                    img.verify()

                    # Width/Height check while we have the image open
                    w, h = img.size
                    if w < self.config.min_image_width_px:
                        result.add_error(
                            self._L2, "image_path",
                            f"Image width {w}px < minimum {self.config.min_image_width_px}px",
                            str(abs_path),
                        )
                    elif h < self.config.min_image_height_px:
                        result.add_error(
                            self._L2, "image_path",
                            f"Image height {h}px < minimum {self.config.min_image_height_px}px",
                            str(abs_path),
                        )
                    else:
                        passed += 1

            except Exception as exc:
                result.add_error(
                    self._L2, "image_path",
                    f"Corrupted image — cannot open with Pillow: {exc}",
                    str(abs_path),
                )

        return passed, total

    # =========================================================================
    # ── Layer 3: Category Validation
    # =========================================================================

    def _check_category(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Validate the category field against the 11-key taxonomy.

        Checks:
          1. category is a non-empty string
          2. category value is in VALID_CATEGORIES (taxonomy lock-in)
          3. source_dataset value is a known dataset source
        """
        passed = 0
        total  = 0

        # ── Check 1 + 2: Category taxonomy ──────────────────────────────────
        total += 1
        category = record.get("category")
        if not category or not isinstance(category, str):
            result.add_error(
                self._L3, "category",
                "category field is missing or not a string",
                category,
            )
        elif category.strip() not in VALID_CATEGORIES:
            result.add_error(
                self._L3, "category",
                f"Invalid category '{category}'. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}",
                category,
            )
        else:
            passed += 1

        # ── Check 3: Source dataset ──────────────────────────────────────────
        total += 1
        source = (
            record.get("source_dataset")
            or record.get("dataset_source")
            or ""
        )
        if source and str(source) not in _VALID_SOURCES:
            result.add_error(
                self._L3, "source_dataset",
                f"Invalid source_dataset '{source}'. "
                f"Must be one of: {sorted(_VALID_SOURCES)}",
                source,
            )
        else:
            passed += 1

        return passed, total

    # =========================================================================
    # ── Layer 4: Missing Metadata
    # =========================================================================

    def _check_metadata_completeness(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Check for absent optional-but-expected metadata fields.

        Issues warnings (not errors) for missing:
          - gender
          - color (list)
          - season
          - description (for FashionGen records)
          - style
        """
        passed = 0
        total  = 0
        source = str(
            record.get("source_dataset") or record.get("dataset_source") or ""
        )

        # ── Gender ────────────────────────────────────────────────────────────
        if self.config.warn_missing_gender:
            total += 1
            gender = record.get("gender")
            if not gender or gender == "null":
                result.add_warning(
                    self._L4, "gender",
                    "Gender is not set — consider annotating for downstream filtering",
                )
            else:
                passed += 1

        # ── Color ─────────────────────────────────────────────────────────────
        if self.config.warn_missing_color:
            total += 1
            color = record.get("color") or []
            if not color:
                result.add_warning(
                    self._L4, "color",
                    "No color annotation — auto-generation recommended",
                )
            else:
                passed += 1

        # ── Season ────────────────────────────────────────────────────────────
        total += 1
        season = record.get("season")
        if not season:
            result.add_warning(
                self._L4, "season",
                "Season is not set — defaulting to all_season is acceptable",
            )
        else:
            passed += 1

        # ── Description (mandatory for FashionGen) ────────────────────────────
        if self.config.warn_missing_description:
            total += 1
            desc = record.get("description")
            if not desc or (isinstance(desc, str) and not desc.strip()):
                if source in self.config.require_description_for:
                    result.add_error(
                        self._L4, "description",
                        f"FashionGen record is missing a description — "
                        f"descriptions are required for {source} records",
                    )
                else:
                    result.add_warning(
                        self._L4, "description",
                        "Description is absent — auto-generation recommended",
                    )
            else:
                passed += 1

        return passed, total

    # =========================================================================
    # ── Layer 5: Corrupted Records
    # =========================================================================

    def _check_corruption(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Detect malformed, type-inconsistent, or structurally broken records.

        Checks:
          1. image_id type is string
          2. category type is string
          3. color field type is list (not str, dict, etc.)
          4. fabric/attributes/landmarks are lists
          5. Landmark entries have required x, y, visible keys
          6. Bounding box (if present) has x1 < x2 and y1 < y2
          7. Processed_at / created_at is a parseable ISO-8601 date
          8. Errors/warnings fields (if present) are lists
          9. is_valid field (if present) is a bool
        """
        passed = 0
        total  = 0

        # ── Check 1: image_id type ────────────────────────────────────────────
        total += 1
        image_id = record.get("image_id")
        if image_id is not None and not isinstance(image_id, str):
            result.add_error(
                self._L5, "image_id",
                f"image_id must be a string, got {type(image_id).__name__}",
                image_id,
            )
        else:
            passed += 1

        # ── Check 2: category type ────────────────────────────────────────────
        total += 1
        category = record.get("category")
        if category is not None and not isinstance(category, str):
            result.add_error(
                self._L5, "category",
                f"category must be a string, got {type(category).__name__}",
                category,
            )
        else:
            passed += 1

        # ── Check 3: color must be list ───────────────────────────────────────
        total += 1
        color = record.get("color")
        if color is not None and not isinstance(color, list):
            result.add_error(
                self._L5, "color",
                f"color must be a list, got {type(color).__name__}",
                color,
            )
        else:
            passed += 1

        # ── Check 4: list-typed fields ────────────────────────────────────────
        list_fields = ("fabric", "attributes", "landmarks", "occasion", "pattern")
        for fname in list_fields:
            val = record.get(fname)
            if val is not None:
                total += 1
                if not isinstance(val, list):
                    result.add_error(
                        self._L5, fname,
                        f"'{fname}' must be a list, got {type(val).__name__}",
                        val,
                    )
                else:
                    passed += 1

        # ── Check 5: landmark structure ───────────────────────────────────────
        landmarks = record.get("landmarks") or []
        if isinstance(landmarks, list) and landmarks:
            total += 1
            bad_landmarks = []
            for i, lm in enumerate(landmarks):
                if not isinstance(lm, dict):
                    bad_landmarks.append(f"index {i}: not a dict")
                elif not all(k in lm for k in ("x", "y", "visible")):
                    missing = [k for k in ("x", "y", "visible") if k not in lm]
                    bad_landmarks.append(f"index {i}: missing keys {missing}")
                else:
                    x, y = lm.get("x"), lm.get("y")
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        bad_landmarks.append(f"index {i}: x/y must be numeric")
                    elif not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                        bad_landmarks.append(
                            f"index {i}: x={x}, y={y} outside [0,1]"
                        )
            if bad_landmarks:
                result.add_error(
                    self._L5, "landmarks",
                    f"Malformed landmarks: {'; '.join(bad_landmarks[:5])}",
                    landmarks,
                )
            else:
                passed += 1

        # ── Check 6: bounding box geometry ───────────────────────────────────
        bbox = record.get("bounding_box") or record.get("bbox")
        if isinstance(bbox, dict) and bbox:
            total += 1
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)
            if x2 <= x1 or y2 <= y1:
                result.add_error(
                    self._L5, "bounding_box",
                    f"Invalid bounding box: (x1={x1}, y1={y1}) must be < (x2={x2}, y2={y2})",
                    bbox,
                )
            else:
                passed += 1
        elif isinstance(bbox, list) and len(bbox) == 4:
            total += 1
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                result.add_error(
                    self._L5, "bounding_box",
                    f"Invalid bounding box list: [{x1},{y1},{x2},{y2}] — x2>x1 and y2>y1 required",
                    bbox,
                )
            else:
                passed += 1

        # ── Check 7: Timestamp format ─────────────────────────────────────────
        ts = record.get("processed_at") or record.get("created_at")
        if ts and isinstance(ts, str):
            total += 1
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                passed += 1
            except ValueError:
                result.add_error(
                    self._L5, "processed_at",
                    f"Invalid ISO-8601 timestamp: '{ts}'",
                    ts,
                )

        # ── Check 8: is_valid type ────────────────────────────────────────────
        is_valid_raw = record.get("is_valid")
        if is_valid_raw is not None:
            total += 1
            if not isinstance(is_valid_raw, bool):
                result.add_error(
                    self._L5, "is_valid",
                    f"is_valid must be bool, got {type(is_valid_raw).__name__}",
                    is_valid_raw,
                )
            else:
                passed += 1

        return passed, total

    # =========================================================================
    # ── Layer 6: Empty / Invalid Descriptions
    # =========================================================================

    def _check_description(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Validate the quality of the description field.

        Checks:
          1. description is a string (or None — which is OK for DeepFashion)
          2. description is not blank / whitespace-only
          3. description meets minimum character count
          4. description does not exceed maximum character count
          5. description digit ratio does not exceed threshold
          6. description contains no control characters
          7. Emoji-only descriptions generate a hint
        """
        passed = 0
        total  = 0

        desc = record.get("description")

        # None description is acceptable (DeepFashion has none)
        if desc is None:
            return 0, 0

        # ── Check 1: type ─────────────────────────────────────────────────────
        total += 1
        if not isinstance(desc, str):
            result.add_error(
                self._L6, "description",
                f"description must be a string or None, got {type(desc).__name__}",
                desc,
            )
            return passed, total
        passed += 1

        desc_stripped = desc.strip()

        # ── Check 2: non-blank ────────────────────────────────────────────────
        total += 1
        if not desc_stripped:
            result.add_error(
                self._L6, "description",
                "description is blank / whitespace-only",
                repr(desc),
            )
            return passed, total
        passed += 1

        # ── Check 3: minimum length ───────────────────────────────────────────
        total += 1
        if len(desc_stripped) < self.config.min_description_chars:
            result.add_error(
                self._L6, "description",
                f"description too short: {len(desc_stripped)} chars "
                f"(minimum {self.config.min_description_chars})",
                desc_stripped[:80],
            )
        else:
            passed += 1

        # ── Check 4: maximum length ───────────────────────────────────────────
        total += 1
        if len(desc_stripped) > self.config.max_description_chars:
            result.add_warning(
                self._L6, "description",
                f"description very long: {len(desc_stripped)} chars "
                f"(max recommended {self.config.max_description_chars})",
            )
            passed += 1  # Warning only — not an error
        else:
            passed += 1

        # ── Check 5: digit ratio ──────────────────────────────────────────────
        total += 1
        digit_count = sum(c.isdigit() for c in desc_stripped)
        digit_ratio = digit_count / max(len(desc_stripped), 1)
        if digit_ratio > self.config.max_digit_ratio:
            result.add_warning(
                self._L6, "description",
                f"description appears to be mostly numbers "
                f"(digit_ratio={digit_ratio:.2f}): '{desc_stripped[:60]}'",
                digit_ratio,
            )
        passed += 1

        # ── Check 6: control characters ───────────────────────────────────────
        total += 1
        ctrl_matches = _CONTROL_CHAR_RE.findall(desc_stripped)
        if ctrl_matches:
            result.add_error(
                self._L6, "description",
                f"description contains {len(ctrl_matches)} control character(s)",
                repr(desc_stripped[:60]),
            )
        else:
            passed += 1

        # ── Check 7: emoji-only hint ──────────────────────────────────────────
        printable_non_emoji = re.sub(r"[^\x20-\x7e]", "", desc_stripped)
        if len(printable_non_emoji) < 3 and len(desc_stripped) > 0:
            result.add_hint(
                self._L6, "description",
                "description appears to contain mostly emoji/non-ASCII — "
                "consider adding textual content",
                desc_stripped[:40],
            )

        return passed, total

    # =========================================================================
    # ── Layer 7: Invalid Attributes
    # =========================================================================

    def _check_attributes(
        self,
        record: Dict[str, Any],
        result: RecordValidationResult,
    ) -> Tuple[int, int]:
        """
        Validate all taxonomy-bound attribute fields.

        Checks:
          1. gender value in VALID_GENDERS (if set)
          2. style value in VALID_STYLES (if set)
          3. fit value in VALID_FITS (if set)
          4. season value in VALID_SEASONS (if set)
          5. each occasion value in VALID_OCCASIONS (if set)
          6. color list not too long
          7. attributes list not too long
          8. landmark count does not exceed max
          9. attribute strings have no control chars
        """
        passed = 0
        total  = 0

        # ── Check 1: gender ───────────────────────────────────────────────────
        gender = record.get("gender")
        if gender and isinstance(gender, str):
            total += 1
            if gender not in VALID_GENDERS:
                result.add_error(
                    self._L7, "gender",
                    f"Invalid gender '{gender}'. Must be one of: {sorted(VALID_GENDERS)}",
                    gender,
                )
            else:
                passed += 1

        # ── Check 2: style ────────────────────────────────────────────────────
        style = record.get("style")
        if style and isinstance(style, str):
            total += 1
            if style not in VALID_STYLES:
                result.add_error(
                    self._L7, "style",
                    f"Invalid style '{style}'. Must be one of: {sorted(VALID_STYLES)}",
                    style,
                )
            else:
                passed += 1

        # ── Check 3: fit ──────────────────────────────────────────────────────
        fit = record.get("fit")
        if fit and isinstance(fit, str):
            total += 1
            if fit not in VALID_FITS:
                result.add_error(
                    self._L7, "fit",
                    f"Invalid fit '{fit}'. Must be one of: {sorted(VALID_FITS)}",
                    fit,
                )
            else:
                passed += 1

        # ── Check 4: season ───────────────────────────────────────────────────
        season = record.get("season")
        if season and isinstance(season, str):
            total += 1
            if season not in VALID_SEASONS:
                result.add_error(
                    self._L7, "season",
                    f"Invalid season '{season}'. Must be one of: {sorted(VALID_SEASONS)}",
                    season,
                )
            else:
                passed += 1

        # ── Check 5: occasion list ────────────────────────────────────────────
        occasions = record.get("occasion") or []
        if isinstance(occasions, list) and occasions:
            total += 1
            bad_occasions = [
                occ for occ in occasions
                if isinstance(occ, str) and occ not in VALID_OCCASIONS
            ]
            if bad_occasions:
                result.add_error(
                    self._L7, "occasion",
                    f"Invalid occasion values: {bad_occasions}. "
                    f"Must be from: {sorted(VALID_OCCASIONS)}",
                    bad_occasions,
                )
            else:
                passed += 1

        # ── Check 6: color list size ──────────────────────────────────────────
        colors = record.get("color") or []
        if isinstance(colors, list):
            total += 1
            if len(colors) > self.config.max_color_count:
                result.add_warning(
                    self._L7, "color",
                    f"color list has {len(colors)} entries "
                    f"(max recommended {self.config.max_color_count})",
                    len(colors),
                )
            passed += 1  # Warning only

        # ── Check 7: attributes list size ────────────────────────────────────
        attributes = record.get("attributes") or []
        if isinstance(attributes, list):
            total += 1
            if len(attributes) > self.config.max_attribute_count:
                result.add_warning(
                    self._L7, "attributes",
                    f"attributes list has {len(attributes)} entries "
                    f"(max recommended {self.config.max_attribute_count})",
                    len(attributes),
                )
            passed += 1  # Warning only

        # ── Check 8: landmark count ───────────────────────────────────────────
        landmarks = record.get("landmarks") or []
        if isinstance(landmarks, list) and landmarks:
            total += 1
            source = str(
                record.get("source_dataset") or record.get("dataset_source") or ""
            )
            if len(landmarks) > self.config.max_landmark_count:
                result.add_warning(
                    self._L7, "landmarks",
                    f"Too many landmarks: {len(landmarks)} "
                    f"(max {self.config.max_landmark_count} for DeepFashion)",
                    len(landmarks),
                )
            if source == "deepfashion" and len(landmarks) < self.config.min_landmark_count_df:
                result.add_warning(
                    self._L7, "landmarks",
                    f"DeepFashion record has only {len(landmarks)} landmark(s) — "
                    f"expected at least {self.config.min_landmark_count_df}",
                    len(landmarks),
                )
            passed += 1

        # ── Check 9: attribute strings — no control chars ─────────────────────
        if isinstance(attributes, list) and attributes:
            total += 1
            corrupted_attrs = [
                a for a in attributes
                if isinstance(a, str) and _CONTROL_CHAR_RE.search(a)
            ]
            if corrupted_attrs:
                result.add_error(
                    self._L7, "attributes",
                    f"{len(corrupted_attrs)} attribute string(s) contain control characters",
                    corrupted_attrs[:3],
                )
            else:
                passed += 1

        return passed, total

    # =========================================================================
    # ── Private Helpers
    # =========================================================================

    @staticmethod
    def _build_breakdown(issues: List[ValidationIssue]) -> Dict[str, int]:
        """
        Build a frequency map of issue messages (first 80 chars as key).

        Sorted descending by count.

        Args:
            issues : List of ValidationIssue objects.

        Returns:
            Dict mapping message_prefix → occurrence_count.
        """
        counts: Dict[str, int] = {}
        for issue in issues:
            # Strip variable parts (paths, values) by truncating to 80 chars
            key = re.sub(r"'[^']{20,}'", "'...'", issue.message)[:80]
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# =============================================================================
# ── 4. Backward-Compat Alias (preserves existing DataValidator import)
# =============================================================================

# The original stub was imported as DataValidator; keep the alias
# so existing code in tests and notebooks continues to work.
DataValidator = FashionDataValidator
