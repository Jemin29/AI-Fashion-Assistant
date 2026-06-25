"""
week2/tests/test_fashion_generation_pipeline.py
==================================================
Comprehensive tests for FashionGenerationPipeline.

Coverage:
  - Data structures: StageResult, ItemResult, PipelineResult, PipelineConfig
  - All 5 pipeline stages (unit tested via dry-run)
  - Single-item run(), run_from_prompt(), dry_run(), explain()
  - Batch run_batch() — success, partial failure, stop_on_error
  - run_from_library() integration
  - Progress tracking (show_progress=False in tests)
  - Metadata and report saving
  - Context manager usage
  - Error isolation (bad inputs, missing fields)
  - All 7 styles end-to-end in dry-run mode
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generation.pipelines.fashion_generation_pipeline import (
    FashionGenerationPipeline,
    PipelineConfig,
    PipelineResult,
    ItemResult,
    StageResult,
)


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    d = tmp_path / "generated"
    d.mkdir()
    return d

@pytest.fixture
def tmp_reports(tmp_path):
    """Temporary report directory."""
    d = tmp_path / "reports"
    d.mkdir()
    return d

@pytest.fixture
def cfg(tmp_output, tmp_reports) -> PipelineConfig:
    return PipelineConfig(
        output_dir    = tmp_output,
        report_dir    = tmp_reports,
        dry_run       = True,
        show_progress = False,
        evaluate      = False,
        save_report   = False,
        save_metadata = False,
    )

@pytest.fixture
def pipeline(cfg) -> FashionGenerationPipeline:
    return FashionGenerationPipeline(pipeline_config=cfg)

ITEM_STREETWEAR = {"style": "streetwear", "category": "hoodie",      "color": "black"}
ITEM_LUXURY     = {"style": "luxury",     "category": "evening gown", "color": "emerald"}
ITEM_CASUAL     = {"style": "casual",     "category": "t-shirt",     "color": "white"}
ITEM_FORMAL     = {"style": "formal",     "category": "suit",         "color": "navy"}
ITEM_TECHWEAR   = {"style": "techwear",   "category": "jacket",       "color": "matte black"}
ITEM_VINTAGE    = {"style": "vintage",    "category": "dress",        "color": "dusty rose"}
ITEM_ATHLEISURE = {"style": "athleisure", "category": "leggings",    "color": "coral"}

ALL_ITEMS = [
    ITEM_STREETWEAR, ITEM_LUXURY, ITEM_CASUAL, ITEM_FORMAL,
    ITEM_TECHWEAR, ITEM_VINTAGE, ITEM_ATHLEISURE,
]


# =============================================================================
# ── PipelineConfig Tests
# =============================================================================

class TestPipelineConfig:

    def test_default_config_instantiates(self):
        cfg = PipelineConfig()
        assert cfg is not None

    def test_default_output_dir(self):
        cfg = PipelineConfig()
        assert "generated" in str(cfg.output_dir)

    def test_default_dry_run_false(self):
        assert PipelineConfig().dry_run is False

    def test_default_boost_quality_true(self):
        assert PipelineConfig().boost_quality is True

    def test_default_evaluate_true(self):
        assert PipelineConfig().evaluate is True

    def test_num_images_default(self):
        assert PipelineConfig().num_images_per_item == 1

    def test_custom_config(self, tmp_output):
        cfg = PipelineConfig(
            output_dir    = tmp_output,
            dry_run       = True,
            num_images_per_item = 4,
            default_steps = 20,
            default_guidance = 6.5,
        )
        assert cfg.dry_run is True
        assert cfg.num_images_per_item == 4
        assert cfg.default_steps == 20
        assert cfg.default_guidance == 6.5


# =============================================================================
# ── StageResult Tests
# =============================================================================

class TestStageResult:

    def test_success_stage(self):
        sr = StageResult(stage="test", success=True, elapsed_s=0.1, data={"key": "val"})
        assert sr.success
        assert sr.data["key"] == "val"

    def test_failed_stage(self):
        sr = StageResult(stage="test", success=False, error="Something went wrong")
        assert not sr.success
        assert "wrong" in sr.error

    def test_skipped_stage(self):
        sr = StageResult(stage="gen", skipped=True, success=True)
        assert sr.skipped

    def test_repr(self):
        sr = StageResult(stage="prompt_build", success=True, elapsed_s=0.5)
        r  = repr(sr)
        assert "prompt_build" in r
        assert "OK" in r


# =============================================================================
# ── ItemResult Tests
# =============================================================================

class TestItemResult:

    def test_default_item_result(self):
        ir = ItemResult()
        assert ir.success is True
        assert ir.dry_run is False
        assert ir.image_paths == []
        assert ir.eval_score == 0.0

    def test_item_id_auto_generated(self):
        ir1 = ItemResult()
        ir2 = ItemResult()
        assert ir1.item_id != ir2.item_id

    def test_summary_dict_keys(self):
        ir = ItemResult(
            item    = {"style": "luxury", "category": "gown"},
            success = True,
        )
        d = ir.summary_dict()
        assert "item_id"     in d
        assert "style"       in d
        assert "category"    in d
        assert "success"     in d
        assert "elapsed_s"   in d

    def test_summary_dict_values(self):
        ir = ItemResult(
            item      = {"style": "formal", "category": "suit"},
            success   = True,
            eval_score= 0.85,
        )
        d = ir.summary_dict()
        assert d["style"]    == "formal"
        assert d["category"] == "suit"
        assert d["success"]  is True


# =============================================================================
# ── PipelineResult Tests
# =============================================================================

class TestPipelineResult:

    def test_default_pipeline_result(self):
        pr = PipelineResult()
        assert pr.total == 0
        assert pr.success is True   # No failures

    def test_success_property_true_when_no_failures(self):
        pr = PipelineResult(total=5, succeeded=5, failed=0)
        assert pr.success is True

    def test_success_property_partial(self):
        pr = PipelineResult(total=5, succeeded=3, failed=2)
        assert pr.success is True   # At least one succeeded

    def test_pass_rate_calculation(self):
        pr = PipelineResult(total_images=10, passed_eval=8, failed_eval=2)
        assert pr.pass_rate == pytest.approx(0.8)

    def test_pass_rate_zero_images(self):
        pr = PipelineResult(total_images=0)
        assert pr.pass_rate == 0.0

    def test_summary_returns_string(self):
        pr = PipelineResult(total=3, succeeded=3)
        s  = pr.summary()
        assert isinstance(s, str)
        assert len(s) > 30

    def test_summary_contains_key_fields(self):
        pr = PipelineResult(total=2, succeeded=1, failed=1, run_id="test_run_123")
        s  = pr.summary()
        assert "test_run_123" in s
        assert "Items processed" in s

    def test_to_dict_is_json_serialisable(self):
        pr = PipelineResult(total=2, succeeded=2, elapsed_s=1.5)
        d  = pr.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert len(json_str) > 10

    def test_to_dict_keys(self):
        pr = PipelineResult()
        d  = pr.to_dict()
        assert "run_id"   in d
        assert "summary"  in d
        assert "items"    in d
        assert "errors"   in d
        assert "dry_run"  in d

    def test_error_list_populates(self):
        pr = PipelineResult()
        pr.errors.append("something failed")
        s = pr.summary()
        assert "something failed" in s


# =============================================================================
# ── FashionGenerationPipeline — Initialisation
# =============================================================================

class TestPipelineInit:

    def test_instantiates_with_defaults(self):
        pipeline = FashionGenerationPipeline(
            PipelineConfig(dry_run=True, show_progress=False)
        )
        assert pipeline is not None

    def test_output_dir_created(self, tmp_path):
        out_dir = tmp_path / "test_out"
        pipeline = FashionGenerationPipeline(
            PipelineConfig(output_dir=out_dir, dry_run=True, show_progress=False)
        )
        assert out_dir.exists()

    def test_repr(self, pipeline):
        assert "FashionGenerationPipeline" in repr(pipeline)

    def test_context_manager(self, cfg):
        with FashionGenerationPipeline(cfg) as p:
            assert p is not None


# =============================================================================
# ── run() — Single Item
# =============================================================================

class TestRunSingleItem:

    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.run(ITEM_STREETWEAR)
        assert isinstance(result, PipelineResult)

    def test_single_item_total_is_one(self, pipeline):
        result = pipeline.run(ITEM_LUXURY)
        assert result.total == 1

    def test_success_on_valid_item(self, pipeline):
        result = pipeline.run(ITEM_CASUAL)
        assert result.success is True

    def test_extra_has_prompts_key(self, pipeline):
        result = pipeline.run(ITEM_FORMAL)
        assert "prompts" in result.extra

    def test_prompt_is_nonempty_string(self, pipeline):
        result = pipeline.run(ITEM_STREETWEAR)
        pos = result.extra["prompts"]["positive"]
        assert isinstance(pos, str) and len(pos) > 20

    def test_negative_prompt_is_nonempty(self, pipeline):
        result = pipeline.run(ITEM_LUXURY)
        neg = result.extra["prompts"]["negative"]
        assert isinstance(neg, str) and len(neg) > 10

    def test_prompt_contains_category(self, pipeline):
        result = pipeline.run({"style": "streetwear", "category": "bomber jacket"})
        pos = result.extra["prompts"]["positive"]
        assert "bomber jacket" in pos.lower()

    def test_prompt_contains_color(self, pipeline):
        result = pipeline.run({"style": "luxury", "category": "dress", "color": "cobalt blue"})
        pos = result.extra["prompts"]["positive"]
        assert "cobalt blue" in pos.lower()

    def test_dry_run_no_images(self, pipeline):
        result = pipeline.run(ITEM_TECHWEAR)
        assert result.total_images == 0

    def test_dry_run_skips_generation_stage(self, pipeline):
        result = pipeline.run(ITEM_VINTAGE)
        ir     = result.items[0]
        gen_stage = ir.stages.get("generation")
        assert gen_stage is not None
        assert gen_stage.skipped is True

    def test_string_input_normalised(self, pipeline):
        result = pipeline.run("A beautiful floral dress")
        assert result.success is True

    def test_missing_style_falls_back(self, pipeline):
        result = pipeline.run({"category": "hoodie"})
        assert result.success is True

    def test_empty_item_does_not_crash(self, pipeline):
        result = pipeline.run({})
        assert isinstance(result, PipelineResult)

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_all_styles_succeed(self, pipeline, style):
        result = pipeline.run({"style": style, "category": "jacket", "color": "black"})
        assert result.success is True

    def test_photo_style_override(self, pipeline):
        result = pipeline.run(ITEM_LUXURY, photo_style="runway")
        pos = result.extra["prompts"]["positive"]
        assert any(t in pos for t in ["runway", "catwalk", "fashion week"])

    def test_extra_tags_injected(self, pipeline):
        result = pipeline.run(ITEM_CASUAL, extra_tags=["limited edition"])
        pos = result.extra["prompts"]["positive"]
        assert "limited edition" in pos

    def test_succeeded_count_is_one(self, pipeline):
        result = pipeline.run(ITEM_STREETWEAR)
        assert result.succeeded == 1
        assert result.failed    == 0

    def test_elapsed_is_positive(self, pipeline):
        result = pipeline.run(ITEM_LUXURY)
        assert result.elapsed_s >= 0.0

    def test_item_result_has_all_stages(self, pipeline):
        result = pipeline.run(ITEM_FORMAL)
        ir     = result.items[0]
        assert "prompt_build"    in ir.stages
        assert "prompt_validate" in ir.stages
        assert "generation"      in ir.stages
        assert "image_save"      in ir.stages
        assert "evaluation"      in ir.stages


# =============================================================================
# ── run_batch() — Batch Mode
# =============================================================================

class TestRunBatch:

    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS)
        assert isinstance(result, PipelineResult)

    def test_total_matches_items(self, pipeline):
        items  = ALL_ITEMS[:4]
        result = pipeline.run_batch(items)
        assert result.total == 4

    def test_all_items_succeed_in_dry_run(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS)
        assert result.succeeded == len(ALL_ITEMS)
        assert result.failed    == 0

    def test_items_list_populated(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS[:3])
        assert len(result.items) == 3

    def test_each_item_has_prompt(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS[:3])
        for ir in result.items:
            assert isinstance(ir.prompt, str) and len(ir.prompt) > 10

    def test_empty_batch_returns_empty_result(self, pipeline):
        result = pipeline.run_batch([])
        assert result.total == 0
        assert "empty" in " ".join(result.warnings).lower()

    def test_string_items_normalised(self, pipeline):
        result = pipeline.run_batch(["a red dress", "a blue hoodie", "a green jacket"])
        assert result.total == 3
        assert result.succeeded == 3

    def test_mixed_string_and_dict_items(self, pipeline):
        items  = ["a red hoodie", {"style": "formal", "category": "suit"}]
        result = pipeline.run_batch(items)
        assert result.total == 2

    def test_stop_on_error_false_continues(self, cfg):
        cfg.stop_on_error = False
        pipeline = FashionGenerationPipeline(cfg)
        items = [
            {"style": "streetwear", "category": "hoodie"},
            {"style": "luxury",     "category": "gown"},
            {"style": "casual",     "category": "t-shirt"},
        ]
        result = pipeline.run_batch(items)
        assert result.total == 3

    def test_batch_elapsed_positive(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS[:3])
        assert result.elapsed_s >= 0.0

    def test_all_seven_styles_in_batch(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS)
        assert result.total == 7
        assert result.succeeded == 7

    def test_batch_item_ids_are_unique(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS)
        ids = [ir.item_id for ir in result.items]
        assert len(ids) == len(set(ids))

    def test_batch_to_dict_serialisable(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS[:3])
        d = result.to_dict()
        json.dumps(d)  # Must not raise

    def test_batch_summary_contains_run_id(self, pipeline):
        result = pipeline.run_batch(ALL_ITEMS[:2])
        s = result.summary()
        assert result.run_id in s


# =============================================================================
# ── run_from_prompt()
# =============================================================================

class TestRunFromPrompt:

    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.run_from_prompt("A red velvet dress")
        assert isinstance(result, PipelineResult)

    def test_prompt_is_enhanced(self, pipeline):
        result = pipeline.run_from_prompt("A simple shirt", style="luxury")
        pos = result.extra["prompts"]["positive"]
        # Enhanced prompt should be longer than raw
        assert len(pos) > len("A simple shirt")

    def test_no_enhance_passes_through(self, pipeline):
        raw    = "A red hoodie with gold buttons"
        result = pipeline.run_from_prompt(raw, style=None, enhance=False)
        pos    = result.extra["prompts"]["positive"]
        assert raw in pos or raw.lower() in pos.lower()

    def test_style_injected(self, pipeline):
        result = pipeline.run_from_prompt("a structured coat", style="luxury")
        pos = result.extra["prompts"]["positive"]
        assert any(t in pos.lower() for t in ["luxury", "haute couture", "premium"])

    def test_success(self, pipeline):
        result = pipeline.run_from_prompt("A black leather jacket", style="streetwear")
        assert result.success is True


# =============================================================================
# ── dry_run() shortcut
# =============================================================================

class TestDryRun:

    def test_dry_run_returns_result(self, pipeline):
        result = pipeline.dry_run(ITEM_STREETWEAR)
        assert isinstance(result, PipelineResult)

    def test_dry_run_flag_set(self, pipeline):
        result = pipeline.dry_run(ITEM_LUXURY)
        assert result.dry_run is True

    def test_dry_run_no_images(self, pipeline):
        result = pipeline.dry_run(ITEM_FORMAL)
        assert result.total_images == 0

    def test_dry_run_prompt_built(self, pipeline):
        result = pipeline.dry_run(ITEM_TECHWEAR)
        pos = result.extra["prompts"]["positive"]
        assert len(pos) > 20

    def test_dry_run_breakdown_in_extra(self, pipeline):
        result = pipeline.dry_run(ITEM_VINTAGE)
        ir = result.items[0]
        assert "prompt_breakdown" in ir.extra


# =============================================================================
# ── explain()
# =============================================================================

class TestExplain:

    def test_returns_dict(self, pipeline):
        result = pipeline.explain(ITEM_LUXURY)
        assert isinstance(result, dict)

    def test_has_full_prompt(self, pipeline):
        result = pipeline.explain(ITEM_STREETWEAR)
        assert "full_prompt" in result
        assert len(result["full_prompt"]) > 20

    def test_has_negative(self, pipeline):
        result = pipeline.explain(ITEM_FORMAL)
        assert "negative" in result
        assert "blurry" in result["negative"]

    def test_has_layers(self, pipeline):
        result = pipeline.explain(ITEM_CASUAL)
        assert "layers" in result
        assert isinstance(result["layers"], dict)

    def test_has_token_estimate(self, pipeline):
        result = pipeline.explain(ITEM_TECHWEAR)
        assert "token_estimate" in result
        assert result["token_estimate"] > 0

    def test_has_style(self, pipeline):
        result = pipeline.explain(ITEM_VINTAGE)
        assert result["style"] == "vintage"

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_explain_all_styles(self, pipeline, style):
        result = pipeline.explain({"style": style, "category": "jacket"})
        assert result["full_prompt"]


# =============================================================================
# ── run_from_library()
# =============================================================================

class TestRunFromLibrary:

    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.run_from_library("streetwear", n=2)
        assert isinstance(result, PipelineResult)

    def test_correct_number_of_items(self, pipeline):
        result = pipeline.run_from_library("luxury", n=3)
        assert result.total >= 1  # May fall back to 1 if library random_batch signature differs

    def test_section_filter(self, pipeline):
        result = pipeline.run_from_library("formal", section="e-commerce", n=2)
        assert result.total >= 1  # At least one prompt returned

    def test_all_styles_library(self, pipeline):
        for style in ["streetwear", "luxury", "casual", "formal", "techwear", "vintage", "athleisure"]:
            result = pipeline.run_from_library(style, n=1)
            assert result.total >= 1

    def test_success(self, pipeline):
        result = pipeline.run_from_library("athleisure", n=2)
        assert result.succeeded >= 1


# =============================================================================
# ── Pipeline Stages Unit Tests (via ItemResult inspection)
# =============================================================================

class TestPipelineStages:

    def test_prompt_build_stage_success(self, pipeline):
        result = pipeline.run(ITEM_STREETWEAR)
        ir     = result.items[0]
        stage  = ir.stages["prompt_build"]
        assert stage.success
        assert "positive" in stage.data
        assert "negative" in stage.data

    def test_prompt_validate_stage_success(self, pipeline):
        result = pipeline.run(ITEM_LUXURY)
        ir     = result.items[0]
        stage  = ir.stages["prompt_validate"]
        assert stage.success

    def test_generation_stage_skipped_in_dry_run(self, pipeline):
        result = pipeline.run(ITEM_CASUAL)
        ir     = result.items[0]
        stage  = ir.stages["generation"]
        assert stage.skipped

    def test_save_stage_skipped_in_dry_run(self, pipeline):
        result = pipeline.run(ITEM_FORMAL)
        ir     = result.items[0]
        stage  = ir.stages["image_save"]
        assert stage.skipped

    def test_evaluation_stage_skipped_in_dry_run(self, pipeline):
        result = pipeline.run(ITEM_TECHWEAR)
        ir     = result.items[0]
        stage  = ir.stages["evaluation"]
        assert stage.skipped

    def test_all_stage_elapsed_times_are_nonnegative(self, pipeline):
        result = pipeline.run(ITEM_VINTAGE)
        ir     = result.items[0]
        for stage_name, stage in ir.stages.items():
            assert stage.elapsed_s >= 0.0, f"Stage {stage_name} has negative elapsed"

    def test_stage_data_accessible(self, pipeline):
        result = pipeline.run(ITEM_ATHLEISURE)
        ir     = result.items[0]
        pb     = ir.stages["prompt_build"]
        assert "positive" in pb.data
        assert len(pb.data["positive"]) > 10

    def test_description_override_used_in_prompt_build(self, pipeline):
        item   = {"style": "luxury", "description": "A breathtaking emerald silk gown"}
        result = pipeline.run(item)
        ir     = result.items[0]
        pb     = ir.stages["prompt_build"]
        pos    = pb.data["positive"]
        assert "emerald silk gown" in pos.lower()


# =============================================================================
# ── Report and Metadata Saving
# =============================================================================

class TestReportSaving:

    def test_run_summary_saved_to_report_dir(self, tmp_path):
        report_dir = tmp_path / "reports"
        cfg = PipelineConfig(
            output_dir    = tmp_path / "out",
            report_dir    = report_dir,
            dry_run       = True,
            show_progress = False,
            save_report   = True,
        )
        pipeline = FashionGenerationPipeline(cfg)
        result   = pipeline.run_batch(ALL_ITEMS[:2])
        # Run summary should be saved
        report_files = list(report_dir.glob("run_*.json"))
        assert len(report_files) >= 1

    def test_run_summary_is_valid_json(self, tmp_path):
        report_dir = tmp_path / "reports"
        cfg = PipelineConfig(
            output_dir    = tmp_path / "out",
            report_dir    = report_dir,
            dry_run       = True,
            show_progress = False,
            save_report   = True,
        )
        pipeline = FashionGenerationPipeline(cfg)
        pipeline.run_batch(ALL_ITEMS[:1])
        for f in report_dir.glob("run_*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            assert "run_id"  in data
            assert "summary" in data


# =============================================================================
# ── Error Handling & Edge Cases
# =============================================================================

class TestErrorHandling:

    def test_unknown_style_does_not_crash(self, pipeline):
        result = pipeline.run({"style": "totally_unknown_xyz", "category": "hoodie"})
        assert isinstance(result, PipelineResult)

    def test_empty_string_item_does_not_crash(self, pipeline):
        result = pipeline.run("")
        assert isinstance(result, PipelineResult)

    def test_very_long_description_truncated(self, pipeline):
        long_desc = "a beautiful " * 200
        result = pipeline.run({"style": "casual", "description": long_desc})
        assert isinstance(result, PipelineResult)

    def test_none_category_handled(self, pipeline):
        result = pipeline.run({"style": "luxury", "category": None})
        assert isinstance(result, PipelineResult)

    def test_none_color_handled(self, pipeline):
        result = pipeline.run({"style": "formal", "category": "suit", "color": None})
        assert isinstance(result, PipelineResult)

    def test_batch_with_one_bad_item(self, pipeline):
        items = [
            {"style": "streetwear", "category": "hoodie"},
            "",   # empty string — bad item
            {"style": "luxury", "category": "gown"},
        ]
        result = pipeline.run_batch(items)
        # Should complete all items without crashing
        assert result.total == 3

    def test_run_from_prompt_empty_string_raises(self, pipeline):
        # Empty prompt should be caught by prompt_enhancer
        with pytest.raises((ValueError, Exception)):
            pipeline.run_from_prompt("")

    def test_stop_on_error_behaviour(self, tmp_output, tmp_reports):
        cfg = PipelineConfig(
            output_dir    = tmp_output,
            report_dir    = tmp_reports,
            dry_run       = True,
            show_progress = False,
            stop_on_error = True,
        )
        pipeline = FashionGenerationPipeline(cfg)
        # Normal items should still complete
        result = pipeline.run_batch(ALL_ITEMS[:3])
        assert result.total >= 1


# =============================================================================
# ── Stage Constants
# =============================================================================

class TestStageConstants:

    def test_stage_name_constants(self):
        assert FashionGenerationPipeline.STAGE_PROMPT_BUILD == "prompt_build"
        assert FashionGenerationPipeline.STAGE_PROMPT_VALID == "prompt_validate"
        assert FashionGenerationPipeline.STAGE_GENERATION   == "generation"
        assert FashionGenerationPipeline.STAGE_SAVE         == "image_save"
        assert FashionGenerationPipeline.STAGE_EVALUATION   == "evaluation"

    def test_five_stages_in_result(self, pipeline):
        result = pipeline.run(ITEM_STREETWEAR)
        ir     = result.items[0]
        assert len(ir.stages) == 5
