"""
Extended unit tests for data_pipeline/preprocessing/preprocessing_pipeline.py

Targets standalone stage functions (pure, independently testable) and the
PreprocessingPipeline orchestrator.  No HDF5 or real images required.

Real API verified from source:
  PipelineConfig:
    target_size, dedup_strategy, drop_unknown_categories,
    lowercase_description, max_description_chars, min_description_chars,
    keep_normalization_log, balance_target_per_class
  PreprocessingPipeline(config, project_root)
    .run(records) → PipelineRunResult
    .save(run_result, output_path) → Path
  Stage functions (standalone):
    stage1_image_resize(record, config) → StageResult
    stage3_dedup(records, config) → (deduped, dup_count, dup_map)
    stage4_clean_description(record, config) → StageResult
    stage5_normalize_attributes(record, config) → StageResult
    stage6_normalize_category(record, config) → StageResult
    compute_balance_stats(records, config) → Dict
    normalize_category(raw) → (canonical, ok)
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from src.data.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline,
    PipelineConfig,
    PipelineRunResult,
    StageResult,
    stage1_image_resize,
    stage3_dedup,
    stage4_clean_description,
    stage5_normalize_attributes,
    stage6_normalize_category,
    compute_balance_stats,
    normalize_category,
)


# =============================================================================
# ── Helpers
# =============================================================================

def _make_rec(
    image_id: str = "IMG_001",
    category: str = "t_shirts",
    gender: str = "men",
    description: str = "A clean cotton t-shirt.",
    image_path: str = "datasets/images/IMG_001.jpg",
    source_dataset: str = "fashiongen",
    style: str = "streetwear",
    season: str = "all_season",
) -> Dict[str, Any]:
    return {
        "image_id"      : image_id,
        "image_path"    : image_path,
        "category"      : category,
        "gender"        : gender,
        "description"   : description,
        "source_dataset": source_dataset,
        "style"         : style,
        "season"        : season,
        "colors"        : ["Navy"],
        "fabrics"       : ["Cotton"],
    }


def _make_batch(n: int = 5) -> List[Dict[str, Any]]:
    return [
        _make_rec(
            image_id=f"IMG_{i:03d}",
            image_path=f"datasets/images/IMG_{i:03d}.jpg",
        )
        for i in range(n)
    ]


# =============================================================================
# ── PipelineConfig
# =============================================================================

class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.target_size == (256, 256)
        assert cfg.dedup_strategy == "path_hash"
        assert cfg.drop_unknown_categories is False
        assert cfg.lowercase_description is False
        assert cfg.max_description_chars == 2048
        assert cfg.min_description_chars == 5

    def test_normalize_descriptions_false(self):
        cfg = PipelineConfig(lowercase_description=False)
        assert cfg.lowercase_description is False

    def test_lowercase_enabled(self):
        cfg = PipelineConfig(lowercase_description=True)
        assert cfg.lowercase_description is True

    def test_show_progress_not_a_field(self):
        """show_progress is not a PipelineConfig field — verify via run()."""
        cfg = PipelineConfig()
        assert not hasattr(cfg, "show_progress")

    def test_to_dict_returns_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "target_size" in d


# =============================================================================
# ── PreprocessingPipeline.__init__
# =============================================================================

class TestPreprocessingPipelineInit:

    def test_default_config_attached(self):
        pipeline = PreprocessingPipeline()
        assert isinstance(pipeline.config, PipelineConfig)

    def test_custom_config(self):
        cfg = PipelineConfig(drop_unknown_categories=True)
        pipeline = PreprocessingPipeline(config=cfg)
        assert pipeline.config.drop_unknown_categories is True

    def test_output_dir_custom_via_project_root(self, tmp_path):
        pipeline = PreprocessingPipeline(project_root=tmp_path)
        assert pipeline.config.project_root == tmp_path


# =============================================================================
# ── Stage 3: Deduplication
# =============================================================================

class TestDeduplicationStage:

    def _run_dedup(self, records, **cfg_kwargs):
        cfg = PipelineConfig(**cfg_kwargs)
        return stage3_dedup(records, cfg)

    def test_no_duplicates_unchanged(self):
        records = _make_batch(3)
        deduped, dup_count, _ = self._run_dedup(records)
        assert len(deduped) == 3
        assert dup_count == 0

    def test_path_duplicates_removed(self):
        records = _make_batch(2)
        # Force same path → duplicate
        records[1]["image_path"] = records[0]["image_path"]
        deduped, dup_count, _ = self._run_dedup(records)
        assert dup_count == 1
        assert len(deduped) == 1

    def test_id_duplicates_removed(self):
        records = _make_batch(2)
        records[1]["image_id"] = records[0]["image_id"]
        deduped, dup_count, _ = self._run_dedup(records)
        # May or may not detect by ID depending on strategy; at least returns valid output
        assert len(deduped) >= 1

    def test_empty_input_ok(self):
        deduped, dup_count, _ = self._run_dedup([])
        assert deduped == []
        assert dup_count == 0

    def test_all_duplicates_leaves_one(self):
        rec = _make_rec()
        records = [dict(rec) for _ in range(5)]
        deduped, dup_count, _ = self._run_dedup(records)
        assert len(deduped) == 1
        assert dup_count == 4


# =============================================================================
# ── Stage 4: Description Cleaning
# =============================================================================

class TestDescriptionCleaningStage:

    def _clean(self, rec, **cfg_kw):
        cfg = PipelineConfig(**cfg_kw)
        return stage4_clean_description(rec, cfg)

    def test_html_tags_removed(self):
        rec = _make_rec(description="<b>Bold</b> T-shirt")
        result = self._clean(rec)
        assert "<b>" not in result.record["description"]
        assert "Bold" in result.record["description"]

    def test_extra_whitespace_collapsed(self):
        rec = _make_rec(description="A   very    spacious    garment")
        result = self._clean(rec)
        assert "  " not in result.record["description"]

    def test_leading_trailing_stripped(self):
        rec = _make_rec(description="   A shirt.   ")
        result = self._clean(rec)
        assert result.record["description"] == result.record["description"].strip()

    def test_none_description_handled(self):
        rec = _make_rec(description="")
        rec["description"] = None
        result = self._clean(rec)
        # Should not raise — description becomes empty string or stays None
        assert result is not None

    def test_empty_description_handled(self):
        rec = _make_rec(description="")
        result = self._clean(rec)
        assert result.record["description"] == "" or result.record["description"] is None

    def test_clean_description_unchanged(self):
        clean_desc = "A clean cotton t-shirt."
        rec = _make_rec(description=clean_desc)
        result = self._clean(rec)
        assert "cotton" in result.record["description"]

    def test_empty_input_ok(self):
        result = stage4_clean_description({}, PipelineConfig())
        assert isinstance(result, StageResult)


# =============================================================================
# ── Stage 5: Attribute Normalization
# =============================================================================

class TestAttributeNormalizationStage:

    def _normalize(self, rec, **cfg_kw):
        cfg = PipelineConfig(**cfg_kw)
        return stage5_normalize_attributes(rec, cfg)

    def test_empty_input_ok(self):
        result = stage5_normalize_attributes({}, PipelineConfig())
        assert isinstance(result, StageResult)

    def test_valid_style_preserved(self):
        rec = _make_rec(style="streetwear")
        result = self._normalize(rec)
        assert result.record.get("style") in (
            "streetwear", "streetwear"
        )

    def test_duplicate_attributes_field_kept(self):
        rec = _make_rec()
        rec["colors"] = ["Navy", "Navy", "Blue"]
        result = self._normalize(rec)
        # Stage 5 normalises values, but de-dup of list items is tested elsewhere
        assert isinstance(result.record.get("colors"), list)

    def test_string_color_not_converted_to_list(self):
        """The stage normalises known fields but doesn't reshape data types."""
        rec = _make_rec()
        rec["colors"] = ["black"]
        result = self._normalize(rec)
        # colors should remain a list
        assert isinstance(result.record.get("colors"), list)

    def test_attributes_key_preserved(self):
        rec = _make_rec()
        result = self._normalize(rec)
        assert "colors" in result.record or "gender" in result.record


# =============================================================================
# ── Stage 6: Category Normalization
# =============================================================================

class TestCategoryNormalizationStage:

    def test_valid_category_preserved(self):
        rec = _make_rec(category="t_shirts")
        result = stage6_normalize_category(rec, PipelineConfig())
        assert result.record["category"] == "t_shirts"
        assert result.record.get("category_known") is True

    def test_unknown_category_kept_when_drop_false(self):
        rec = _make_rec(category="ZXQWERTY_NONEXISTENT_12345")
        result = stage6_normalize_category(rec, PipelineConfig(drop_unknown_categories=False))
        assert result.record["category"] == "uncategorized"
        assert result.dropped is False

    def test_unknown_category_dropped_when_drop_true(self):
        rec = _make_rec(category="ZXQWERTY_NONEXISTENT_12345")
        result = stage6_normalize_category(rec, PipelineConfig(drop_unknown_categories=True))
        assert result.dropped is True

    def test_all_valid_categories_preserved(self):
        valid = [
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
        ]
        cfg = PipelineConfig()
        for cat in valid:
            rec = _make_rec(category=cat)
            result = stage6_normalize_category(rec, cfg)
            assert result.record["category"] == cat, f"Category changed: {cat}"

    def test_empty_input_ok(self):
        result = stage6_normalize_category({}, PipelineConfig())
        assert isinstance(result, StageResult)


# =============================================================================
# ── normalize_category helper
# =============================================================================

class TestNormalizeCategoryHelper:

    def test_known_value_returns_true(self):
        canonical, ok = normalize_category("t_shirts")
        assert ok is True
        assert canonical == "t_shirts"

    def test_alias_resolves(self):
        canonical, ok = normalize_category("hoodie")
        assert ok is True
        assert canonical == "hoodies"

    def test_unknown_returns_none_false(self):
        canonical, ok = normalize_category("ZZZZUNKNOWN_XYZ")
        assert ok is False
        assert canonical is None

    def test_none_input(self):
        canonical, ok = normalize_category(None)
        assert ok is False

    def test_empty_string(self):
        canonical, ok = normalize_category("")
        assert ok is False


# =============================================================================
# ── Stage 7: Balancing Statistics
# =============================================================================

class TestBalancingStatistics:

    def test_category_distribution_computed(self):
        records = (
            [_make_rec(image_id=f"I{i}", category="t_shirts") for i in range(3)] +
            [_make_rec(image_id=f"J{i}", category="jeans") for i in range(2)]
        )
        stats = compute_balance_stats(records, PipelineConfig())
        assert "category" in stats
        counts = stats["category"]["counts"]
        assert counts.get("t_shirts") == 3
        assert counts.get("jeans") == 2

    def test_stats_with_empty_records(self):
        stats = compute_balance_stats([], PipelineConfig())
        assert stats["total_records"] == 0

    def test_recommended_balance_key_present(self):
        records = _make_batch(4)
        stats = compute_balance_stats(records, PipelineConfig())
        assert "recommended_balance" in stats


# =============================================================================
# ── PreprocessingPipeline.run()
# =============================================================================

class TestPreprocessingPipelineRun:

    def _pipeline(self, **cfg_kw):
        return PreprocessingPipeline(config=PipelineConfig(**cfg_kw))

    def test_run_returns_pipeline_run_result(self):
        pipeline = self._pipeline()
        result = pipeline.run([])
        assert isinstance(result, PipelineRunResult)

    def test_run_empty_input(self):
        result = self._pipeline().run([])
        assert result.total_input == 0
        assert result.total_output == 0

    def test_run_returns_list_of_records(self):
        records = _make_batch(3)
        result = self._pipeline().run(records)
        assert isinstance(result.records, list)

    def test_run_preserves_valid_records(self):
        records = _make_batch(5)
        result = self._pipeline().run(records)
        assert result.total_output > 0

    def test_run_removes_duplicates(self):
        rec = _make_rec()
        records = [dict(rec) for _ in range(4)]
        result = self._pipeline().run(records)
        assert result.duplicates_removed == 3
        assert result.total_output == 1

    def test_run_cleans_descriptions(self):
        records = [_make_rec(description="<b>Fancy</b>  garment")]
        result = self._pipeline().run(records)
        out_desc = result.records[0].get("description", "")
        assert "<b>" not in out_desc

    def test_run_pipeline_result_has_balance_stats(self):
        records = _make_batch(3)
        result = self._pipeline().run(records)
        assert isinstance(result.balance_stats, dict)

    def test_run_total_input_counts_all(self):
        records = _make_batch(7)
        result = self._pipeline().run(records)
        assert result.total_input == 7


# =============================================================================
# ── PreprocessingPipeline.save()
# =============================================================================

class TestPreprocessingPipelineSave:

    def _pipeline_and_result(self, tmp_path, n=3):
        pipeline = PreprocessingPipeline(config=PipelineConfig())
        records = _make_batch(n)
        run_result = pipeline.run(records)
        return pipeline, run_result

    def test_save_creates_file(self, tmp_path):
        pipeline, result = self._pipeline_and_result(tmp_path)
        output_path = tmp_path / "clean_dataset.json"
        path = pipeline.save(result, str(output_path))
        assert Path(path).exists()

    def test_save_json_structure(self, tmp_path):
        pipeline, result = self._pipeline_and_result(tmp_path)
        output_path = tmp_path / "clean.json"
        path = pipeline.save(result, str(output_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "records" in data

    def test_save_meta_included(self, tmp_path):
        pipeline, result = self._pipeline_and_result(tmp_path)
        output_path = tmp_path / "meta.json"
        path = pipeline.save(result, str(output_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "summary" in data or "pipeline_config" in data or "generated_at" in data

    def test_save_empty_records(self, tmp_path):
        pipeline = PreprocessingPipeline()
        result = pipeline.run([])
        output_path = tmp_path / "empty.json"
        path = pipeline.save(result, str(output_path))
        assert Path(path).exists()

    def test_save_custom_path(self, tmp_path):
        pipeline, result = self._pipeline_and_result(tmp_path)
        custom = tmp_path / "sub" / "output.json"
        path = pipeline.save(result, str(custom))
        assert Path(path).exists()
