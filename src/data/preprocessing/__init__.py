"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/preprocessing/__init__.py — Preprocessing Sub-Package
=============================================================================
Public API for the Fashion Preprocessing Pipeline.

Primary exports:

    from src.data.preprocessing import PreprocessingPipeline
    from src.data.preprocessing import PipelineConfig
    from src.data.preprocessing import FashionPreprocessor   # image only

    # Full 7-stage pipeline:
    pipeline = PreprocessingPipeline()
    result   = pipeline.run(records)
    pipeline.save(result, "datasets/processed/clean_dataset.json")

    # Convenience: load from JSON files
    result = pipeline.run_from_json(
        "datasets/processed/fashiongen_processed.json",
        "datasets/processed/deepfashion_processed.json",
    )

Pipeline Stages:
  Stage 1 — Image Resizing            resize + aspect ratio metadata
  Stage 2 — Image Normalization       pixel stats, ImageNet reference values
  Stage 3 — Duplicate Detection       pHash / path-hash dedup
  Stage 4 — Description Cleaning      HTML strip, whitespace, control chars
  Stage 5 — Attribute Normalization   color/style/fit/season/occasion aliases
  Stage 6 — Category Normalization    11-key taxonomy mapping
  Stage 7 — Balancing Statistics      per-field distributions + recommendations

Standalone stage functions (also exported for unit testing):
    stage1_image_resize, stage2_image_normalize,
    stage3_dedup, stage4_clean_description,
    stage5_normalize_attributes, stage6_normalize_category,
    compute_balance_stats, build_dedup_hash,
    clean_description, normalize_category, normalize_value,
    normalize_list_field
=============================================================================
"""

# ── Image preprocessor (original, image-only) ──────────────────────────────
from src.data.preprocessing.image_preprocessor import (
    FashionPreprocessor,
    PreprocessorConfig,
    ProcessingResult,
)

# ── Full 7-stage preprocessing pipeline ────────────────────────────────────
from src.data.preprocessing.preprocessing_pipeline import (
    # Main classes
    PreprocessingPipeline,
    PipelineConfig,
    PipelineRunResult,
    StageResult,

    # Standalone stage functions
    stage1_image_resize,
    stage2_image_normalize,
    stage3_dedup,
    stage4_clean_description,
    stage5_normalize_attributes,
    stage6_normalize_category,
    compute_balance_stats,

    # Helper functions
    build_dedup_hash,
    clean_description,
    normalize_category,
    normalize_value,
    normalize_list_field,

    # Taxonomy alias tables (useful for tests / extensions)
    _CATEGORY_ALIASES,
    _GENDER_ALIASES,
    _COLOR_ALIASES,
    _STYLE_ALIASES,
    _FIT_ALIASES,
    _SEASON_ALIASES,
    _OCCASION_ALIASES,
)

__all__ = [
    # Image preprocessor
    "FashionPreprocessor",
    "PreprocessorConfig",
    "ProcessingResult",

    # Pipeline
    "PreprocessingPipeline",
    "PipelineConfig",
    "PipelineRunResult",
    "StageResult",

    # Stage functions
    "stage1_image_resize",
    "stage2_image_normalize",
    "stage3_dedup",
    "stage4_clean_description",
    "stage5_normalize_attributes",
    "stage6_normalize_category",
    "compute_balance_stats",

    # Helpers
    "build_dedup_hash",
    "clean_description",
    "normalize_category",
    "normalize_value",
    "normalize_list_field",

    # Alias tables
    "_CATEGORY_ALIASES",
    "_GENDER_ALIASES",
    "_COLOR_ALIASES",
    "_STYLE_ALIASES",
    "_FIT_ALIASES",
    "_SEASON_ALIASES",
    "_OCCASION_ALIASES",
]
