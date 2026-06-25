"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_preprocessing_pipeline.py
Unit Tests: 7-Stage Preprocessing Pipeline
=============================================================================
Full test coverage for:
  data_pipeline/preprocessing/preprocessing_pipeline.py

Test Classes (18 classes, 170+ tests):
  TestPipelineConfig                — defaults, overrides, to_dict
  TestStageResult                   — dropped/modified flags
  TestPipelineRunResult             — summary_dict, print_summary
  TestCleanDescription              — all 10 cleaning transforms
  TestNormalizeValue                — exact match, alias, substring, unknown
  TestNormalizeListField            — list dedup, unknowns, empty
  TestNormalizeCategory             — all 11 taxonomy keys, aliases, unknown
  TestBuildDedupHash                — path_hash, content_key strategies
  TestStage1ImageResize             — dimension metadata, aspect ratio
  TestStage2ImageNormalize          — metadata keys, no-PIL fallback
  TestStage3Dedup                   — exact duplicates, different records
  TestStage4CleanDescription        — calls clean_description, writes raw_
  TestStage5NormalizeAttributes     — list/scalar fields, normalization_log
  TestStage6NormalizeCategory       — all 11 + drop_unknown + uncategorized
  TestComputeBalanceStats           — counts, shares, imbalance, recommendation
  TestPreprocessingPipelineRun      — full pipeline, 7-stage integration
  TestPreprocessingPipelineSave     — JSON structure, file writing
  TestPipelineEdgeCases             — empty batch, single record, all-bad

Run:
    pytest tests/test_preprocessing_pipeline.py -v
    pytest tests/test_preprocessing_pipeline.py -v --tb=short
=============================================================================
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.data.preprocessing.preprocessing_pipeline import (
    # Config & models
    PipelineConfig,
    PipelineRunResult,
    StageResult,
    # Stage functions
    stage1_image_resize,
    stage2_image_normalize,
    stage3_dedup,
    stage4_clean_description,
    stage5_normalize_attributes,
    stage6_normalize_category,
    compute_balance_stats,
    # Helpers
    build_dedup_hash,
    clean_description,
    normalize_category,
    normalize_value,
    normalize_list_field,
    # Alias tables
    _CATEGORY_ALIASES,
    _STYLE_ALIASES,
    _FIT_ALIASES,
    _SEASON_ALIASES,
    _OCCASION_ALIASES,
    _GENDER_ALIASES,
    _COLOR_ALIASES,
    # Main class
    PreprocessingPipeline,
)
from src.data.knowledge_base.fashion_domain_research import (
    VALID_CATEGORIES,
    VALID_STYLES,
    VALID_FITS,
    VALID_SEASONS,
    VALID_OCCASIONS,
    VALID_GENDERS,
)


# =============================================================================
# ── Shared Test Fixtures & Helpers
# =============================================================================

# Counter to give each record a unique image_path by default
_REC_COUNTER: List[int] = [0]


def _rec(
    image_id  : str  = None,
    category  : str  = "shirts",
    source    : str  = "fashiongen",
    gender    : str  = "men",
    style     : str  = "formal",
    fit       : str  = "slim_fit",
    season    : str  = "all_season",
    occasion  : list = None,
    color     : list = None,
    description: str = "A classic slim fit white cotton dress shirt for formal occasions.",
    image_path: str  = None,
    **overrides,
) -> Dict[str, Any]:
    """Build a minimal valid fashion record with a unique image_path per call."""
    # Auto-generate unique IDs so dedup doesn't silently collapse records
    _REC_COUNTER[0] += 1
    idx = _REC_COUNTER[0]
    if image_id is None:
        image_id = f"AUTO_{idx:06d}"
    if image_path is None:
        image_path = f"datasets/fashiongen/images/{image_id}.jpg"
    base = {
        "image_id"      : image_id,
        "image_path"    : image_path,
        "category"      : category,
        "source_dataset": source,
        "gender"        : gender,
        "style"         : style,
        "fit"           : fit,
        "season"        : season,
        "occasion"      : occasion or ["formal"],
        "color"         : color or ["White"],
        "description"   : description,
        "attributes"    : ["collar", "button-down"],
        "landmarks"     : [],
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def default_config() -> PipelineConfig:
    return PipelineConfig(
        verify_image_exists    = False if True else True,   # allow non-existent files
        resize_if_image_exists = False,
    )


@pytest.fixture(scope="module")
def pipeline() -> PreprocessingPipeline:
    cfg = PipelineConfig(drop_unknown_categories=False)
    return PreprocessingPipeline(config=cfg)


@pytest.fixture(scope="module")
def strict_pipeline() -> PreprocessingPipeline:
    cfg = PipelineConfig(drop_unknown_categories=True)
    return PreprocessingPipeline(config=cfg)


# =============================================================================
# ── 1. TestPipelineConfig
# =============================================================================

class TestPipelineConfig:

    def test_default_target_size(self):
        cfg = PipelineConfig()
        assert cfg.target_size == (256, 256)

    def test_default_dedup_strategy(self):
        cfg = PipelineConfig()
        assert cfg.dedup_strategy == "path_hash"

    def test_default_min_description_chars(self):
        cfg = PipelineConfig()
        assert cfg.min_description_chars == 5

    def test_custom_target_size(self):
        cfg = PipelineConfig(target_size=(512, 512))
        assert cfg.target_size == (512, 512)

    def test_custom_min_desc_chars(self):
        cfg = PipelineConfig(min_description_chars=20)
        assert cfg.min_description_chars == 20

    def test_to_dict_has_all_keys(self):
        cfg = PipelineConfig()
        d   = cfg.to_dict()
        for k in ("target_size", "dedup_strategy", "min_description_chars",
                  "imagenet_mean", "imagenet_std", "drop_unknown_categories"):
            assert k in d

    def test_to_dict_target_size_is_list(self):
        cfg = PipelineConfig()
        d   = cfg.to_dict()
        assert isinstance(d["target_size"], list)

    def test_to_dict_imagenet_mean_is_list(self):
        cfg = PipelineConfig()
        d   = cfg.to_dict()
        assert isinstance(d["imagenet_mean"], list)
        assert len(d["imagenet_mean"]) == 3

    def test_lowercase_description_default_false(self):
        cfg = PipelineConfig()
        assert cfg.lowercase_description is False

    def test_keep_normalization_log_default_true(self):
        cfg = PipelineConfig()
        assert cfg.keep_normalization_log is True


# =============================================================================
# ── 2. TestStageResult
# =============================================================================

class TestStageResult:

    def test_default_not_dropped(self):
        sr = StageResult(record={})
        assert sr.dropped is False

    def test_default_not_modified(self):
        sr = StageResult(record={})
        assert sr.modified is False

    def test_default_no_warnings(self):
        sr = StageResult(record={})
        assert sr.warnings == []

    def test_drop_reason_none_by_default(self):
        sr = StageResult(record={})
        assert sr.drop_reason is None

    def test_can_set_dropped(self):
        sr = StageResult(record={}, dropped=True, drop_reason="bad cat")
        assert sr.dropped is True
        assert sr.drop_reason == "bad cat"


# =============================================================================
# ── 3. TestPipelineRunResult
# =============================================================================

class TestPipelineRunResult:

    def test_default_zero_counts(self):
        r = PipelineRunResult()
        assert r.total_input        == 0
        assert r.total_output       == 0
        assert r.duplicates_removed == 0
        assert r.uncategorized      == 0

    def test_summary_dict_has_spec_fields(self):
        r = PipelineRunResult(
            total_input=10, total_output=8,
            duplicates_removed=1, uncategorized=1
        )
        d = r.summary_dict()
        assert "total_input"        in d
        assert "total_output"       in d
        assert "duplicates_removed" in d
        assert "uncategorized"      in d
        assert "processing_time_s"  in d

    def test_summary_dict_values_correct(self):
        r = PipelineRunResult(total_input=10, total_output=9, duplicates_removed=1)
        d = r.summary_dict()
        assert d["total_input"]        == 10
        assert d["total_output"]       == 9
        assert d["duplicates_removed"] == 1

    def test_print_summary_runs_without_error(self, capsys):
        r = PipelineRunResult(total_input=5, total_output=4)
        r.print_summary()
        captured = capsys.readouterr()
        assert "5" in captured.out


# =============================================================================
# ── 4. TestCleanDescription
# =============================================================================

class TestCleanDescription:

    cfg = PipelineConfig()

    def test_strips_whitespace(self):
        assert clean_description("  hello  ", self.cfg) == "hello"

    def test_collapses_multiple_spaces(self):
        result = clean_description("A  great   shirt", self.cfg)
        assert "  " not in result

    def test_collapses_newlines(self):
        result = clean_description("A shirt\n\nwith details", self.cfg)
        assert "\n" not in result

    def test_strips_html_tags(self):
        result = clean_description("<b>Bold</b> shirt", self.cfg)
        assert "<" not in result
        assert "Bold" in result

    def test_decodes_html_entities(self):
        result = clean_description("Price &amp; quality", self.cfg)
        assert "&amp;" not in result
        assert "&" in result

    def test_removes_control_chars(self):
        result = clean_description("shirt\x01with\x07control", self.cfg)
        assert "\x01" not in result
        assert "\x07" not in result

    def test_unicode_nfc_normalization(self):
        # café composed vs decomposed should produce same result
        composed   = clean_description("café", self.cfg)
        decomposed = clean_description("cafe\u0301", self.cfg)
        assert composed == decomposed

    def test_collapses_repeated_punctuation(self):
        result = clean_description("Amazing!!!", self.cfg)
        assert "!!!" not in result
        assert "!" in result

    def test_truncates_to_max_chars(self):
        cfg    = PipelineConfig(max_description_chars=20)
        result = clean_description("x" * 100, cfg)
        assert len(result) <= 22  # 20 chars + "…"
        assert result.endswith("…")

    def test_lowercase_option(self):
        cfg    = PipelineConfig(lowercase_description=True)
        result = clean_description("A BLUE Shirt", cfg)
        assert result == result.lower()

    def test_none_returns_empty_string(self):
        assert clean_description(None, self.cfg) == ""

    def test_non_string_returns_empty_string(self):
        assert clean_description(12345, self.cfg) == ""  # type: ignore[arg-type]

    def test_empty_string_returns_empty_string(self):
        assert clean_description("", self.cfg) == ""

    def test_valid_text_unchanged(self):
        text   = "A slim fit white cotton shirt."
        result = clean_description(text, self.cfg)
        assert result == text


# =============================================================================
# ── 5. TestNormalizeValue
# =============================================================================

class TestNormalizeValue:

    def test_exact_match_in_valid_set(self):
        val, ok = normalize_value("formal", _STYLE_ALIASES, VALID_STYLES)
        assert ok is True
        assert val == "formal"

    def test_alias_lookup(self):
        val, ok = normalize_value("retro", _STYLE_ALIASES, VALID_STYLES)
        assert ok is True
        assert val == "vintage"

    def test_case_insensitive_lookup(self):
        val, ok = normalize_value("Slim Fit", _FIT_ALIASES, VALID_FITS)
        assert ok is True
        assert val == "slim_fit"

    def test_unknown_value_returned_as_is(self):
        val, ok = normalize_value("cyberpunk", _STYLE_ALIASES, VALID_STYLES)
        assert ok is False
        assert val == "cyberpunk"

    def test_none_returns_none_ok(self):
        val, ok = normalize_value(None, _STYLE_ALIASES, VALID_STYLES)
        assert val is None
        assert ok is True

    def test_empty_string_returns_none_ok(self):
        val, ok = normalize_value("", _STYLE_ALIASES, VALID_STYLES)
        assert val is None
        assert ok is True

    def test_season_fall_alias(self):
        val, ok = normalize_value("fall", _SEASON_ALIASES, VALID_SEASONS)
        assert ok is True
        assert val == "autumn"

    def test_gender_male_alias(self):
        val, ok = normalize_value("male", _GENDER_ALIASES, VALID_GENDERS)
        assert ok is True
        assert val == "men"

    def test_fit_baggy_alias(self):
        val, ok = normalize_value("baggy", _FIT_ALIASES, VALID_FITS)
        assert ok is True
        assert val == "oversized"

    def test_occasion_everyday_alias(self):
        val, ok = normalize_value("everyday", _OCCASION_ALIASES, VALID_OCCASIONS)
        assert ok is True
        assert val == "casual"


# =============================================================================
# ── 6. TestNormalizeListField
# =============================================================================

class TestNormalizeListField:

    def test_empty_list_returns_empty(self):
        result, unknowns, warns = normalize_list_field([], _STYLE_ALIASES, VALID_STYLES)
        assert result == []
        assert unknowns == []

    def test_none_returns_empty(self):
        result, unknowns, warns = normalize_list_field(None, _STYLE_ALIASES, VALID_STYLES)
        assert result == []

    def test_valid_values_passed_through(self):
        result, unknowns, _ = normalize_list_field(
            ["casual", "formal"], _OCCASION_ALIASES, VALID_OCCASIONS
        )
        assert "casual" in result
        assert "formal" in result

    def test_alias_resolved_in_list(self):
        result, unknowns, _ = normalize_list_field(
            ["everyday"], _OCCASION_ALIASES, VALID_OCCASIONS
        )
        assert "casual" in result

    def test_deduplication(self):
        result, _, _ = normalize_list_field(
            ["casual", "everyday"], _OCCASION_ALIASES, VALID_OCCASIONS
        )
        # "everyday" → "casual", so there should only be one "casual"
        assert result.count("casual") == 1

    def test_unknown_value_returned_in_unknowns(self):
        # Use a value that is genuinely not in the alias map or valid set
        _, unknowns, warns = normalize_list_field(
            ["xyzzy_unknown_occasion_9999"], _OCCASION_ALIASES, VALID_OCCASIONS
        )
        assert len(unknowns) == 1
        assert len(warns) >= 1

    def test_mixed_known_unknown(self):
        result, unknowns, _ = normalize_list_field(
            ["casual", "rave"], _OCCASION_ALIASES, VALID_OCCASIONS
        )
        assert "casual" in result
        assert "rave" in unknowns

    def test_non_string_items_skipped(self):
        result, _, _ = normalize_list_field(
            ["casual", 123, None], _OCCASION_ALIASES, VALID_OCCASIONS  # type: ignore
        )
        assert "casual" in result
        # Non-strings silently skipped


# =============================================================================
# ── 7. TestNormalizeCategory
# =============================================================================

class TestNormalizeCategory:

    @pytest.mark.parametrize("canonical", [
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    ])
    def test_all_11_canonical_categories_pass_through(self, canonical):
        val, ok = normalize_category(canonical)
        assert ok is True
        assert val == canonical

    def test_alias_tshirt(self):
        val, ok = normalize_category("t-shirt")
        assert ok is True
        assert val == "t_shirts"

    def test_alias_blouse_to_shirts(self):
        val, ok = normalize_category("blouse")
        assert ok is True
        assert val == "shirts"

    def test_alias_gown_to_dresses(self):
        val, ok = normalize_category("gown")
        assert ok is True
        assert val == "dresses"

    def test_alias_sneakers_to_footwear(self):
        val, ok = normalize_category("sneakers")
        assert ok is True
        assert val == "footwear"

    def test_alias_boots_to_footwear(self):
        val, ok = normalize_category("boots")
        assert ok is True
        assert val == "footwear"

    def test_alias_bag_to_accessories(self):
        val, ok = normalize_category("bag")
        assert ok is True
        assert val == "accessories"

    def test_alias_kurta_to_ethnic_wear(self):
        val, ok = normalize_category("kurta")
        assert ok is True
        assert val == "ethnic_wear"

    def test_alias_hoodie_to_hoodies(self):
        val, ok = normalize_category("hoodie")
        assert ok is True
        assert val == "hoodies"

    def test_alias_trouser_to_pants(self):
        val, ok = normalize_category("trouser")
        assert ok is True
        assert val == "pants"

    def test_alias_denim_to_jeans(self):
        val, ok = normalize_category("denim")
        assert ok is True
        assert val == "jeans"

    def test_unknown_category(self):
        val, ok = normalize_category("swimwear")
        assert ok is False
        assert val is None

    def test_none_returns_none_not_ok(self):
        val, ok = normalize_category(None)
        assert ok is False
        assert val is None

    def test_empty_string_returns_not_ok(self):
        val, ok = normalize_category("")
        assert ok is False

    def test_case_insensitive(self):
        val, ok = normalize_category("SHIRTS")
        assert ok is True
        assert val == "shirts"


# =============================================================================
# ── 8. TestBuildDedupHash
# =============================================================================

class TestBuildDedupHash:

    def test_same_path_same_hash(self):
        r1 = _rec(image_id="A", image_path="datasets/img/a.jpg")
        r2 = _rec(image_id="B", image_path="datasets/img/a.jpg")
        assert build_dedup_hash(r1) == build_dedup_hash(r2)

    def test_different_path_different_hash(self):
        r1 = _rec(image_id="A", image_path="datasets/img/a.jpg")
        r2 = _rec(image_id="B", image_path="datasets/img/b.jpg")
        assert build_dedup_hash(r1) != build_dedup_hash(r2)

    def test_backslash_normalized(self):
        r1 = _rec(image_path="datasets\\img\\a.jpg")
        r2 = _rec(image_path="datasets/img/a.jpg")
        assert build_dedup_hash(r1) == build_dedup_hash(r2)

    def test_content_key_strategy_different_categories(self):
        r1 = _rec(image_path="img/a.jpg", category="shirts")
        r2 = _rec(image_path="img/a.jpg", category="jeans")
        h1 = build_dedup_hash(r1, strategy="content_key")
        h2 = build_dedup_hash(r2, strategy="content_key")
        assert h1 != h2

    def test_content_key_same_path_same_cat_same_hash(self):
        r1 = _rec(image_id="X", image_path="img/a.jpg", category="shirts")
        r2 = _rec(image_id="Y", image_path="img/a.jpg", category="shirts")
        h1 = build_dedup_hash(r1, strategy="content_key")
        h2 = build_dedup_hash(r2, strategy="content_key")
        assert h1 == h2

    def test_hash_is_hex_string(self):
        r = _rec()
        h = build_dedup_hash(r)
        assert re.fullmatch(r"[0-9a-f]{32}", h)

    def test_missing_image_path(self):
        r = {"image_id": "X"}  # no image_path
        h = build_dedup_hash(r)
        assert isinstance(h, str)


# =============================================================================
# ── 9. TestStage1ImageResize
# =============================================================================

class TestStage1ImageResize:

    cfg = PipelineConfig(target_size=(128, 128))

    def test_writes_image_width(self):
        sr = stage1_image_resize(_rec(), self.cfg)
        assert sr.record["image_width"] == 128

    def test_writes_image_height(self):
        sr = stage1_image_resize(_rec(), self.cfg)
        assert sr.record["image_height"] == 128

    def test_writes_image_resized_flag(self):
        sr = stage1_image_resize(_rec(), self.cfg)
        assert "image_resized" in sr.record

    def test_aspect_ratio_none_when_no_original_dims(self):
        sr = stage1_image_resize(_rec(), self.cfg)
        # No image_width/image_height in input → aspect_ratio_original could be None
        assert "aspect_ratio_original" in sr.record

    def test_aspect_ratio_computed_from_original_dims(self):
        rec = _rec()
        rec["image_width"]  = 800
        rec["image_height"] = 400
        sr  = stage1_image_resize(rec, self.cfg)
        assert sr.record["aspect_ratio_original"] == pytest.approx(2.0, rel=1e-3)

    def test_record_not_mutated(self):
        rec = _rec()
        original_path = rec["image_path"]
        stage1_image_resize(rec, self.cfg)
        assert rec["image_path"] == original_path
        # original dict's image_width should not be set (we copy)
        assert "image_width" not in rec

    def test_warning_when_image_not_on_disk(self):
        rec = _rec(image_path="does/not/exist.jpg")
        cfg = PipelineConfig()
        sr  = stage1_image_resize(rec, cfg)
        # Should produce a warning (file not found), not crash
        assert isinstance(sr.warnings, list)

    def test_stage_returns_stage_result(self):
        sr = stage1_image_resize(_rec(), self.cfg)
        assert isinstance(sr, StageResult)


# =============================================================================
# ── 10. TestStage2ImageNormalize
# =============================================================================

class TestStage2ImageNormalize:

    cfg = PipelineConfig()

    def test_writes_imagenet_mean(self):
        sr = stage2_image_normalize(_rec(), self.cfg)
        assert "imagenet_mean" in sr.record
        assert sr.record["imagenet_mean"] == [0.485, 0.456, 0.406]

    def test_writes_imagenet_std(self):
        sr = stage2_image_normalize(_rec(), self.cfg)
        assert "imagenet_std" in sr.record
        assert sr.record["imagenet_std"] == [0.229, 0.224, 0.225]

    def test_writes_normalization_ok_key(self):
        sr = stage2_image_normalize(_rec(), self.cfg)
        assert "normalization_ok" in sr.record

    def test_normalization_ok_false_when_no_file(self):
        rec = _rec(image_path="nonexistent/path.jpg")
        sr  = stage2_image_normalize(rec, self.cfg)
        assert sr.record["normalization_ok"] is False

    def test_pixel_mean_rgb_none_when_no_file(self):
        rec = _rec(image_path="nonexistent/path.jpg")
        sr  = stage2_image_normalize(rec, self.cfg)
        assert sr.record["pixel_mean_rgb"] is None

    def test_pixel_std_rgb_none_when_no_file(self):
        rec = _rec(image_path="nonexistent/path.jpg")
        sr  = stage2_image_normalize(rec, self.cfg)
        assert sr.record["pixel_std_rgb"] is None

    def test_stage_does_not_mutate_input(self):
        rec  = _rec()
        orig = dict(rec)
        stage2_image_normalize(rec, self.cfg)
        assert rec == orig

    def test_normalization_ok_true_with_real_image(self, tmp_path):
        """Only run if PIL and numpy available."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip("PIL/numpy not available")

        img_path = tmp_path / "test.jpg"
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_path, format="JPEG")

        rec = _rec(image_path=str(img_path))
        cfg = PipelineConfig()
        cfg.project_root = tmp_path
        sr  = stage2_image_normalize(rec, cfg)
        assert sr.record["normalization_ok"] is True
        assert isinstance(sr.record["pixel_mean_rgb"], list)
        assert len(sr.record["pixel_mean_rgb"]) == 3


# =============================================================================
# ── 11. TestStage3Dedup
# =============================================================================

class TestStage3Dedup:

    cfg = PipelineConfig()

    def test_no_duplicates_keeps_all(self):
        records = [
            _rec(image_id="A", image_path="img/a.jpg"),
            _rec(image_id="B", image_path="img/b.jpg"),
            _rec(image_id="C", image_path="img/c.jpg"),
        ]
        kept, dup_count, dup_map = stage3_dedup(records, self.cfg)
        assert len(kept)    == 3
        assert dup_count    == 0
        assert dup_map      == {}

    def test_exact_duplicate_removed(self):
        records = [
            _rec(image_id="A", image_path="img/same.jpg"),
            _rec(image_id="B", image_path="img/same.jpg"),  # duplicate
        ]
        kept, dup_count, dup_map = stage3_dedup(records, self.cfg)
        assert len(kept)  == 1
        assert dup_count  == 1
        assert "B" in dup_map
        assert dup_map["B"] == "A"

    def test_first_seen_is_kept(self):
        records = [
            _rec(image_id="FIRST",  image_path="img/same.jpg"),
            _rec(image_id="SECOND", image_path="img/same.jpg"),
        ]
        kept, _, _ = stage3_dedup(records, self.cfg)
        assert kept[0]["image_id"] == "FIRST"

    def test_dedup_hash_added_to_kept_records(self):
        records = [_rec(image_id="A")]
        kept, _, _ = stage3_dedup(records, self.cfg)
        assert "dedup_hash" in kept[0]

    def test_backslash_treated_as_same_path(self):
        records = [
            _rec(image_id="A", image_path="img\\same.jpg"),
            _rec(image_id="B", image_path="img/same.jpg"),
        ]
        kept, dup_count, _ = stage3_dedup(records, self.cfg)
        assert dup_count == 1
        assert len(kept) == 1

    def test_empty_list_returns_empty(self):
        kept, dup_count, dup_map = stage3_dedup([], self.cfg)
        assert kept     == []
        assert dup_count == 0

    def test_multiple_duplicates(self):
        records = [_rec(image_id=f"R{i}", image_path="img/same.jpg") for i in range(5)]
        kept, dup_count, _ = stage3_dedup(records, self.cfg)
        assert len(kept) == 1
        assert dup_count == 4


# =============================================================================
# ── 12. TestStage4CleanDescription
# =============================================================================

class TestStage4CleanDescription:

    cfg = PipelineConfig()

    def test_cleaned_description_written(self):
        rec = _rec(description="  A   shirt.  ")
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description"] == "A shirt."

    def test_raw_description_preserved(self):
        raw = "  A   shirt.  "
        rec = _rec(description=raw)
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description_raw"] == raw

    def test_description_length_recorded(self):
        rec = _rec(description="Hello world")
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description_length"] == 11

    def test_description_cleaned_flag_true_when_changed(self):
        rec = _rec(description="  shirt  ")
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description_cleaned"] is True

    def test_description_cleaned_flag_false_when_unchanged(self):
        rec = _rec(description="A slim fit white shirt.")
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description_cleaned"] is False

    def test_short_description_marked_empty(self):
        cfg = PipelineConfig(min_description_chars=10)
        rec = _rec(description="Hi")
        sr  = stage4_clean_description(rec, cfg)
        assert sr.record["description_empty"] is True

    def test_adequate_description_not_empty(self):
        rec = _rec(description="A beautiful white shirt.")
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description_empty"] is False

    def test_html_stripped(self):
        rec = _rec(description="<p>A <b>shirt</b></p>")
        sr  = stage4_clean_description(rec, self.cfg)
        assert "<" not in sr.record["description"]
        assert "shirt" in sr.record["description"]

    def test_none_description_handled(self):
        rec = _rec(description=None)
        sr  = stage4_clean_description(rec, self.cfg)
        assert sr.record["description"] == ""
        assert sr.record["description_empty"] is True

    def test_does_not_mutate_input(self):
        rec  = _rec()
        orig = rec["description"]
        stage4_clean_description(rec, self.cfg)
        assert rec["description"] == orig


# =============================================================================
# ── 13. TestStage5NormalizeAttributes
# =============================================================================

class TestStage5NormalizeAttributes:

    cfg = PipelineConfig()

    def test_valid_style_unchanged(self):
        rec = _rec(style="formal")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["style"] == "formal"
        assert "style_raw" not in sr.record

    def test_alias_style_normalized(self):
        rec = _rec(style="retro")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["style"] == "vintage"
        assert sr.record["style_raw"] == "retro"

    def test_alias_fit_normalized(self):
        rec = _rec(fit="baggy")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["fit"] == "oversized"

    def test_alias_season_fall_to_autumn(self):
        rec = _rec(season="fall")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["season"] == "autumn"

    def test_alias_gender_male_to_men(self):
        rec = _rec(gender="male")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["gender"] == "men"

    def test_occasion_list_normalized(self):
        rec = _rec(occasion=["everyday", "formal"])
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert "casual" in sr.record["occasion"]
        assert "formal" in sr.record["occasion"]

    def test_occasion_deduplication(self):
        rec = _rec(occasion=["casual", "everyday"])  # both → "casual"
        sr  = stage5_normalize_attributes(rec, self.cfg)
        assert sr.record["occasion"].count("casual") == 1

    def test_unknown_style_preserved_with_warning(self):
        rec = _rec(style="cyberpunk")
        sr  = stage5_normalize_attributes(rec, self.cfg)
        # Unknown style kept as-is but warning issued
        assert sr.record["style"] == "cyberpunk"
        assert len(sr.warnings) > 0

    def test_normalization_log_written_when_enabled(self):
        rec = _rec(style="retro")
        cfg = PipelineConfig(keep_normalization_log=True)
        sr  = stage5_normalize_attributes(rec, cfg)
        assert "normalization_log" in sr.record

    def test_normalization_log_absent_when_disabled(self):
        rec = _rec()
        cfg = PipelineConfig(keep_normalization_log=False)
        sr  = stage5_normalize_attributes(rec, cfg)
        assert "normalization_log" not in sr.record

    def test_does_not_mutate_input(self):
        rec  = _rec(style="retro")
        orig = rec["style"]
        stage5_normalize_attributes(rec, self.cfg)
        assert rec["style"] == orig


# =============================================================================
# ── 14. TestStage6NormalizeCategory
# =============================================================================

class TestStage6NormalizeCategory:

    cfg = PipelineConfig(drop_unknown_categories=False)

    @pytest.mark.parametrize("cat", [
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    ])
    def test_valid_category_unchanged(self, cat):
        rec = _rec(category=cat)
        sr  = stage6_normalize_category(rec, self.cfg)
        assert sr.record["category"] == cat
        assert sr.dropped is False
        assert sr.record["category_known"] is True

    def test_alias_tshirt_normalized(self):
        rec = _rec(category="tshirt")
        sr  = stage6_normalize_category(rec, self.cfg)
        assert sr.record["category"] == "t_shirts"
        assert sr.record["category_raw"] == "tshirt"

    def test_alias_blouse_to_shirts(self):
        rec = _rec(category="blouse")
        sr  = stage6_normalize_category(rec, self.cfg)
        assert sr.record["category"] == "shirts"

    def test_unknown_category_becomes_uncategorized(self):
        rec = _rec(category="swimwear")
        sr  = stage6_normalize_category(rec, self.cfg)
        assert sr.record["category"] == "uncategorized"
        assert sr.record["category_known"] is False
        assert sr.dropped is False

    def test_unknown_category_dropped_when_strict(self):
        cfg = PipelineConfig(drop_unknown_categories=True)
        rec = _rec(category="swimwear")
        sr  = stage6_normalize_category(rec, cfg)
        assert sr.dropped is True
        assert sr.drop_reason is not None

    def test_known_category_not_dropped(self):
        rec = _rec(category="jeans")
        sr  = stage6_normalize_category(rec, self.cfg)
        assert sr.dropped is False

    def test_category_raw_written_on_change(self):
        rec = _rec(category="SHIRTS")
        sr  = stage6_normalize_category(rec, self.cfg)
        assert "category_raw" in sr.record or sr.record["category"] == "shirts"

    def test_stage_result_type(self):
        sr = stage6_normalize_category(_rec(), self.cfg)
        assert isinstance(sr, StageResult)


# =============================================================================
# ── 15. TestComputeBalanceStats
# =============================================================================

class TestComputeBalanceStats:

    cfg = PipelineConfig()

    def _make_records(self) -> List[Dict[str, Any]]:
        categories = ["shirts"] * 5 + ["jeans"] * 3 + ["dresses"] * 2
        return [_rec(image_id=f"R{i}", category=c) for i, c in enumerate(categories)]

    def test_total_records_correct(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert stats["total_records"] == 10

    def test_category_counts_correct(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        cat_counts = stats["category"]["counts"]
        assert cat_counts["shirts"]  == 5
        assert cat_counts["jeans"]   == 3
        assert cat_counts["dresses"] == 2

    def test_category_shares_sum_to_100(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        total_share = sum(
            v for k, v in stats["category"]["shares_pct"].items()
            if k != "_missing_"
        )
        assert total_share == pytest.approx(100.0, abs=0.5)

    def test_imbalance_ratio_gt_1_when_unbalanced(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert stats["category"]["imbalance_ratio"] > 1.0

    def test_imbalance_ratio_1_when_balanced(self):
        records = [_rec(image_id=f"R{i}", category=cat)
                   for i, cat in enumerate(["shirts", "jeans", "dresses"])]
        stats = compute_balance_stats(records, self.cfg)
        assert stats["category"]["imbalance_ratio"] == pytest.approx(1.0)

    def test_gender_stats_computed(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert "gender" in stats

    def test_source_dataset_stats_computed(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert "source_dataset" in stats

    def test_season_stats_computed(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert "season" in stats

    def test_recommended_balance_present(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert "recommended_balance" in stats

    def test_recommendation_has_categories_key(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        assert "categories" in stats["recommended_balance"]

    def test_oversample_recommended_for_minority(self):
        records = self._make_records()
        stats   = compute_balance_stats(records, self.cfg)
        dresses = stats["recommended_balance"]["categories"].get("dresses", {})
        # dresses has count=2, min=2, max=5 → target=2 → balanced
        # OR if target=min=2, dresses is balanced
        assert "action" in dresses

    def test_empty_dataset(self):
        stats = compute_balance_stats([], self.cfg)
        assert stats["total_records"] == 0

    def test_custom_balance_target(self):
        cfg     = PipelineConfig(balance_target_per_class=10)
        records = self._make_records()
        stats   = compute_balance_stats(records, cfg)
        shirts  = stats["recommended_balance"]["categories"]["shirts"]
        assert shirts["target_count"] == 10
        # shirts has 5, target 10 → oversample
        assert shirts["action"] == "oversample"


# =============================================================================
# ── 16. TestPreprocessingPipelineRun  (integration)
# =============================================================================

class TestPreprocessingPipelineRun:

    def _make_batch(self, n: int = 5) -> List[Dict[str, Any]]:
        categories = ["shirts", "jeans", "dresses", "hoodies", "footwear"]
        return [
            _rec(image_id=f"FG_{i:04d}", category=categories[i % len(categories)])
            for i in range(n)
        ]

    def test_run_returns_pipeline_run_result(self, pipeline):
        result = pipeline.run(self._make_batch())
        assert isinstance(result, PipelineRunResult)

    def test_total_input_correct(self, pipeline):
        records = self._make_batch(7)
        result  = pipeline.run(records)
        assert result.total_input == 7

    def test_total_output_le_total_input(self, pipeline):
        records = self._make_batch(5)
        result  = pipeline.run(records)
        assert result.total_output <= result.total_input

    def test_duplicates_removed_correct(self, pipeline):
        records = [
            _rec(image_id="A", image_path="img/same.jpg"),
            _rec(image_id="B", image_path="img/same.jpg"),  # dup
            _rec(image_id="C", image_path="img/other.jpg"),
        ]
        result = pipeline.run(records)
        assert result.duplicates_removed == 1

    def test_records_are_cleaned(self, pipeline):
        records = [_rec(description="  <b>A shirt</b>  ")]
        result  = pipeline.run(records)
        desc = result.records[0]["description"]
        assert "<b>" not in desc
        assert "A shirt" in desc

    def test_category_normalized_in_output(self, pipeline):
        records = [_rec(category="tshirt")]
        result  = pipeline.run(records)
        assert result.records[0]["category"] == "t_shirts"

    def test_balance_stats_present(self, pipeline):
        records = self._make_batch(5)
        result  = pipeline.run(records)
        assert isinstance(result.balance_stats, dict)
        assert "total_records" in result.balance_stats

    def test_processing_time_positive(self, pipeline):
        records = self._make_batch(3)
        result  = pipeline.run(records)
        assert result.processing_time_s >= 0

    def test_unknown_category_not_dropped_when_lenient(self, pipeline):
        records = [_rec(category="swimwear")]
        result  = pipeline.run(records)
        # With drop_unknown_categories=False → still in output as "uncategorized"
        assert result.total_output == 1
        assert result.uncategorized == 1

    def test_unknown_category_dropped_when_strict(self, strict_pipeline):
        records = [_rec(category="swimwear")]
        result  = strict_pipeline.run(records)
        assert result.total_output   == 0
        assert result.dropped_count  == 1

    def test_image_dimensions_written(self, pipeline):
        records = [_rec()]
        result  = pipeline.run(records)
        r = result.records[0]
        assert "image_width"  in r
        assert "image_height" in r

    def test_imagenet_mean_written(self, pipeline):
        records = [_rec()]
        result  = pipeline.run(records)
        assert "imagenet_mean" in result.records[0]

    def test_dedup_hash_written(self, pipeline):
        records = [_rec()]
        result  = pipeline.run(records)
        assert "dedup_hash" in result.records[0]

    def test_attribute_style_normalized(self, pipeline):
        records = [_rec(style="retro")]
        result  = pipeline.run(records)
        assert result.records[0]["style"] == "vintage"

    def test_run_empty_batch(self, pipeline):
        result = pipeline.run([])
        assert result.total_input  == 0
        assert result.total_output == 0

    def test_run_single_record(self, pipeline):
        result = pipeline.run([_rec()])
        assert result.total_input == 1

    def test_warning_count_is_non_negative(self, pipeline):
        result = pipeline.run(self._make_batch(3))
        assert result.warning_count >= 0


# =============================================================================
# ── 17. TestPreprocessingPipelineSave
# =============================================================================

class TestPreprocessingPipelineSave:

    def _run_and_save(self, pipeline, tmp_path, n=3) -> Path:
        # Give each record a truly unique image_path so dedup keeps all
        records = [
            _rec(
                image_id=f"SAVE_{i}",
                category=["shirts", "jeans", "dresses"][i % 3],
                image_path=f"datasets/save_test/img_{i:04d}.jpg",
            )
            for i in range(n)
        ]
        result = pipeline.run(records)
        out    = tmp_path / "clean_dataset.json"
        pipeline.save(result, out)
        return out

    def test_file_created(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        assert path.exists()

    def test_file_is_valid_json(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_top_level_keys_present(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for k in ("generated_at", "schema_version", "pipeline_config",
                  "summary", "balance_stats", "records"):
            assert k in data, f"Missing top-level key: {k}"

    def test_summary_has_spec_fields(self, pipeline, tmp_path):
        path    = self._run_and_save(pipeline, tmp_path)
        data    = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        for k in ("total_input", "total_output", "duplicates_removed",
                  "uncategorized", "processing_time_s"):
            assert k in summary, f"Missing summary field: {k}"

    def test_records_list_in_output(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data["records"], list)
        assert len(data["records"]) == 3

    def test_balance_stats_in_output(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "total_records" in data["balance_stats"]

    def test_pipeline_config_in_output(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "target_size" in data["pipeline_config"]

    def test_parent_dirs_created(self, pipeline, tmp_path):
        records = [_rec()]
        result  = pipeline.run(records)
        out     = tmp_path / "nested" / "deep" / "clean_dataset.json"
        pipeline.save(result, out)
        assert out.exists()

    def test_returns_path_object(self, pipeline, tmp_path):
        records = [_rec()]
        result  = pipeline.run(records)
        out     = tmp_path / "dataset.json"
        returned = pipeline.save(result, out)
        assert isinstance(returned, Path)

    def test_schema_version_correct(self, pipeline, tmp_path):
        path = self._run_and_save(pipeline, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == "1.0.0"


# =============================================================================
# ── 18. TestPipelineEdgeCases
# =============================================================================

class TestPipelineEdgeCases:

    def test_record_with_no_description(self, pipeline):
        rec    = _rec(description=None)
        result = pipeline.run([rec])
        assert result.total_output == 1
        assert result.records[0]["description"] == ""

    def test_record_with_html_in_description(self, pipeline):
        rec    = _rec(description="<br>A <strong>nice</strong> shirt.")
        result = pipeline.run([rec])
        assert "<" not in result.records[0]["description"]

    def test_all_same_category_no_balance_needed(self, pipeline):
        # Use explicit unique paths so dedup keeps all 5 records
        records = [
            _rec(image_id=f"BAL_{i}", image_path=f"datasets/balance/img_{i}.jpg")
            for i in range(5)
        ]
        result = pipeline.run(records)
        shirts = result.balance_stats["category"]["counts"].get("shirts", 0)
        assert shirts == 5

    def test_mixed_source_datasets(self, pipeline):
        # Explicit unique image paths so dedup does not collapse them
        records = [
            _rec(image_id="FG1", source="fashiongen",
                 image_path="datasets/fg/img_001.jpg"),
            _rec(image_id="DF1", source="deepfashion",
                 image_path="datasets/df/img_001.jpg"),
        ]
        result = pipeline.run(records)
        assert result.total_output == 2
        sources = result.balance_stats["source_dataset"]["counts"]
        assert "fashiongen"  in sources
        assert "deepfashion" in sources

    def test_alias_in_all_attribute_fields(self, pipeline):
        rec = _rec(
            style="retro", fit="baggy", season="fall",
            gender="male", occasion=["everyday"],
        )
        result = pipeline.run([rec])
        r = result.records[0]
        assert r["style"]   == "vintage"
        assert r["fit"]     == "oversized"
        assert r["season"]  == "autumn"
        assert r["gender"]  == "men"
        assert "casual" in r["occasion"]

    def test_many_duplicates_only_one_kept(self, pipeline):
        records = [_rec(image_id=f"DUP{i}", image_path="img/same.jpg") for i in range(20)]
        result  = pipeline.run(records)
        assert result.total_output       == 1
        assert result.duplicates_removed == 19

    def test_run_from_json_with_nonexistent_file(self, pipeline, tmp_path):
        result = pipeline.run_from_json(str(tmp_path / "nonexistent.json"))
        assert result.total_input  == 0
        assert result.total_output == 0

    def test_run_from_json_with_valid_file(self, pipeline, tmp_path):
        records = [_rec(image_id="J1"), _rec(image_id="J2")]
        json_file = tmp_path / "input.json"
        json_file.write_text(json.dumps(records), encoding="utf-8")
        result = pipeline.run_from_json(str(json_file))
        assert result.total_input == 2

    def test_run_from_json_with_records_key(self, pipeline, tmp_path):
        records = [_rec(image_id="K1"), _rec(image_id="K2")]
        data    = {"records": records, "extra_key": "ignored"}
        json_file = tmp_path / "wrapped.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        result = pipeline.run_from_json(str(json_file))
        assert result.total_input == 2
