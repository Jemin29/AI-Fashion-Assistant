"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/schema/__init__.py — Schema Sub-Package
=============================================================================
Public API for the Unified Fashion Dataset Schema.

Usage:
    from src.data.schema import UnifiedFashionItem, FashionDatasetBatch
    from src.data.schema import CategoryEnum, GenderEnum, StyleEnum
    from src.data.schema import LandmarkPoint, BoundingBox
=============================================================================
"""

from src.data.schema.fashion_schema import (
    # ── Primary schema model ──────────────────────────────────────────────────
    UnifiedFashionItem,
    # ── Batch container ───────────────────────────────────────────────────────
    FashionDatasetBatch,
    # ── Embedded sub-models ───────────────────────────────────────────────────
    LandmarkPoint,
    BoundingBox,
    SchemaVersion,
    ValidationReport,
    # ── Enumerations ──────────────────────────────────────────────────────────
    DatasetSource,
    GenderEnum,
    CategoryEnum,
    StyleEnum,
    FitEnum,
    SeasonEnum,
    OccasionEnum,
    # ── Helper functions ──────────────────────────────────────────────────────
    safe_category,
    safe_gender,
    safe_style,
    safe_season,
    safe_fit,
    safe_occasion_list,
)

__all__ = [
    "UnifiedFashionItem",
    "FashionDatasetBatch",
    "LandmarkPoint",
    "BoundingBox",
    "SchemaVersion",
    "ValidationReport",
    "DatasetSource",
    "GenderEnum",
    "CategoryEnum",
    "StyleEnum",
    "FitEnum",
    "SeasonEnum",
    "OccasionEnum",
    "safe_category",
    "safe_gender",
    "safe_style",
    "safe_season",
    "safe_fit",
    "safe_occasion_list",
]
