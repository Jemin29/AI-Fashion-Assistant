"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/validation/__init__.py — Validation Sub-Package
=============================================================================
Public API for the Fashion Dataset Validation Framework.

Primary exports:

    from src.data.validation import FashionDataValidator
    from src.data.validation import DataValidator  # alias

    # Quick usage:
    validator = FashionDataValidator()
    result    = validator.validate_record(record_dict)
    batch_res = validator.validate_batch(list_of_records)
    validator.save_report(batch_res, "datasets/metadata/validation_report.json")

7 Validation Layers:
  1. Required Fields          — image_id, image_path, category, source
  2. Missing Images           — existence, extension, size, readability
  3. Category Validation      — taxonomy lock-in (11 allowed values)
  4. Missing Metadata         — gender, color, season, description
  5. Corrupted Records        — type errors, malformed data
  6. Empty Descriptions       — blank, too short, all-numeric
  7. Invalid Attributes       — bad styles/fits/occasions, coordinate ranges

Data Models:
    ValidationIssue         — Single finding (ERROR | WARNING | HINT)
    RecordValidationResult  — All issues for one record + quality score
    BatchValidationResult   — Aggregate stats + per-record list
    ValidationConfig        — All configurable thresholds
=============================================================================
"""

from src.data.validation.data_validator import (
    # ── Main validator class ──────────────────────────────────────────────────
    FashionDataValidator,
    # ── Backward-compat alias ─────────────────────────────────────────────────
    DataValidator,
    # ── Data models ───────────────────────────────────────────────────────────
    ValidationIssue,
    RecordValidationResult,
    BatchValidationResult,
    ValidationConfig,
    # ── Severity constants ─────────────────────────────────────────────────────
    Severity,
)

__all__ = [
    "FashionDataValidator",
    "DataValidator",
    "ValidationIssue",
    "RecordValidationResult",
    "BatchValidationResult",
    "ValidationConfig",
    "Severity",
]
