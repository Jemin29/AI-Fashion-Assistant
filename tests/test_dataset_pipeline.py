"""
=============================================================================
tests/test_dataset_pipeline.py
=============================================================================
Unit test suite for dataset_pipeline.py — the Master Pipeline Controller.

Coverage:
  - MasterPipelineConfig   : defaults, custom values, output_dir creation
  - StageTimer             : timing, error capture, to_dict()
  - ExecutionSummary       : finalize(), to_dict(), print_summary()
  - _build_demo_records    : record schema, count, field types
  - DemoPipelineController : end-to-end with all 6 stages (no real datasets)
  - MasterPipelineController: stage skip flags, partial-run modes
  - _load_records_from_json: valid / invalid / missing file cases
  - CLI argument parser     : all flags parsed correctly
  - Output files            : JSON files written, structure verified

Run with:
  pytest tests/test_dataset_pipeline.py -v
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ─── Ensure project root is on sys.path ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Module under test ───────────────────────────────────────────────────────
from dataset_pipeline import (
    MasterPipelineConfig,
    StageTimer,
    ExecutionSummary,
    MasterPipelineController,
    DemoPipelineController,
    _build_demo_records,
    _build_cli_parser,
)

# ─── Valid taxonomy values (must match what the validator accepts) ────────────
_VALID_CATEGORIES = {
    "t_shirts", "shirts", "hoodies", "jackets", "pants",
    "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories",
}
_VALID_SOURCES = {"fashiongen", "deepfashion"}


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Return a fresh temp directory to use as output_dir."""
    out = tmp_path / "processed"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def demo_cfg(tmp_output: Path) -> MasterPipelineConfig:
    """A MasterPipelineConfig configured for fast unit-test runs."""
    return MasterPipelineConfig(
        enable_fashiongen       = False,
        enable_deepfashion      = False,
        enable_metadata_gen     = True,
        metadata_nlp_enabled    = False,   # skip spaCy for speed
        enable_preprocessing    = True,
        enable_validation       = True,
        show_progress           = False,
        output_dir              = tmp_output,
        use_kb                  = False,
    )


@pytest.fixture
def demo_records() -> List[Dict[str, Any]]:
    """20 synthetic demo records via the module helper."""
    return _build_demo_records(20)


# =============================================================================
# ── 1. MasterPipelineConfig
# =============================================================================

class TestMasterPipelineConfig:
    """Tests for MasterPipelineConfig dataclass."""

    def test_defaults(self):
        cfg = MasterPipelineConfig()
        assert cfg.enable_fashiongen is True
        assert cfg.enable_deepfashion is True
        assert cfg.enable_metadata_gen is True
        assert cfg.enable_preprocessing is True
        assert cfg.enable_validation is True
        assert cfg.fashiongen_max_records is None
        assert cfg.deepfashion_max_records is None
        assert cfg.deepfashion_split == "train"
        assert cfg.preproc_target_size == (256, 256)
        assert cfg.preproc_dedup_strategy == "path_hash"
        assert cfg.export_include_invalid is False
        assert cfg.export_pretty_json is True
        assert cfg.show_progress is True
        assert cfg.use_kb is True

    def test_output_dir_created(self, tmp_path: Path):
        """__post_init__ must create the output directory if absent."""
        out = tmp_path / "brand_new" / "subdir"
        assert not out.exists()
        cfg = MasterPipelineConfig(output_dir=out)
        assert out.exists(), "output_dir was not created by __post_init__"

    def test_custom_values(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen       = False,
            fashiongen_max_records  = 500,
            deepfashion_split       = "val",
            deepfashion_max_records = 200,
            preproc_target_size     = (128, 128),
            validation_image_check  = True,
            output_dir              = tmp_output,
        )
        assert cfg.enable_fashiongen is False
        assert cfg.fashiongen_max_records == 500
        assert cfg.deepfashion_split == "val"
        assert cfg.deepfashion_max_records == 200
        assert cfg.preproc_target_size == (128, 128)
        assert cfg.validation_image_check is True

    def test_output_dir_stored_as_path(self, tmp_output: Path):
        cfg = MasterPipelineConfig(output_dir=str(tmp_output))
        assert isinstance(cfg.output_dir, Path)


# =============================================================================
# ── 2. StageTimer
# =============================================================================

class TestStageTimer:
    """Tests for StageTimer context manager."""

    def test_basic_timing(self):
        with StageTimer("test stage") as t:
            time.sleep(0.01)
        assert t.elapsed_s >= 0.009
        assert t.error is None

    def test_to_dict_on_success(self):
        with StageTimer("success stage") as t:
            pass
        d = t.to_dict()
        assert d["stage"] == "success stage"
        assert d["status"] == "ok"
        assert d["error"] is None
        assert isinstance(d["elapsed_s"], float)
        assert d["elapsed_s"] >= 0

    def test_elapsed_before_exit(self):
        t = StageTimer("mid-run")
        t.start_time = time.perf_counter() - 1.0  # simulate 1s elapsed
        assert t.elapsed_s >= 0.9

    def test_elapsed_after_exit_is_fixed(self):
        with StageTimer("fixed") as t:
            pass
        e1 = t.elapsed_s
        time.sleep(0.05)
        e2 = t.elapsed_s
        assert abs(e1 - e2) < 0.01, "elapsed_s should be fixed after context exit"

    def test_error_captured(self):
        t = StageTimer("failing stage")
        try:
            with t:
                raise ValueError("test error")
        except ValueError:
            pass
        assert t.error is not None
        assert "ValueError" in t.error
        assert "test error" in t.error
        d = t.to_dict()
        assert d["status"] == "error"
        assert d["error"] is not None


# =============================================================================
# ── 3. ExecutionSummary
# =============================================================================

class TestExecutionSummary:
    """Tests for ExecutionSummary dataclass."""

    def test_default_status_is_running(self):
        s = ExecutionSummary()
        assert s.status == "running"

    def test_finalize_success(self):
        s = ExecutionSummary()
        s.final_records = 100
        s.finalize()
        assert s.status == "success"
        assert s.completed_at != ""
        assert s.total_elapsed_s >= 0

    def test_finalize_partial(self):
        s = ExecutionSummary()
        s.errors.append("some error")
        s.final_records = 50
        s.finalize()
        assert s.status == "partial"

    def test_finalize_failed(self):
        s = ExecutionSummary()
        s.errors.append("fatal error")
        s.final_records = 0
        s.finalize()
        assert s.status == "failed"

    def test_to_dict_structure(self):
        s = ExecutionSummary()
        s.fashiongen_records  = 100
        s.deepfashion_records = 200
        s.total_raw_records   = 300
        s.validation_valid    = 280
        s.validation_failed   = 20
        s.final_records       = 280
        s.finalize()
        d = s.to_dict()

        assert "pipeline_version" in d
        assert "run_id" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "status" in d
        assert "record_counts" in d
        assert "quality_metrics" in d
        assert "stage_timings" in d
        assert "output_paths" in d
        assert "errors" in d
        assert "warnings" in d

        rc = d["record_counts"]
        assert rc["fashiongen_raw"] == 100
        assert rc["deepfashion_raw"] == 200
        assert rc["total_raw"] == 300
        assert rc["validation_valid"] == 280
        assert rc["validation_failed"] == 20
        assert rc["final_exported"] == 280

    def test_to_dict_timers_included(self):
        s = ExecutionSummary()
        t = StageTimer("dummy stage")
        t.start_time = t.end_time = time.perf_counter()
        s.stage_timers.append(t)
        d = s.to_dict()
        assert len(d["stage_timings"]) == 1
        assert d["stage_timings"][0]["stage"] == "dummy stage"

    def test_print_summary_no_crash(self, capsys):
        """print_summary must not raise even with all-zero counts."""
        s = ExecutionSummary()
        s.finalize()
        s.print_summary()   # should not raise
        captured = capsys.readouterr()
        assert "MASTER PIPELINE" in captured.out
        assert "EXECUTION SUMMARY" in captured.out

    def test_print_summary_with_errors(self, capsys):
        s = ExecutionSummary()
        s.errors = ["error one", "error two"]
        s.final_records = 0
        s.finalize()
        s.print_summary()
        captured = capsys.readouterr()
        assert "error one" in captured.out

    def test_print_summary_outputs_counts(self, capsys):
        s = ExecutionSummary()
        s.fashiongen_records  = 1234
        s.deepfashion_records = 5678
        s.final_records = 6000
        s.finalize()
        s.print_summary()
        captured = capsys.readouterr()
        assert "1,234" in captured.out
        assert "5,678" in captured.out


# =============================================================================
# ── 4. _build_demo_records
# =============================================================================

class TestBuildDemoRecords:
    """Tests for the synthetic demo record factory."""

    def test_returns_correct_count(self):
        for n in (1, 5, 11, 20, 33):
            records = _build_demo_records(n)
            assert len(records) == n, f"Expected {n} records, got {len(records)}"

    def test_required_fields_present(self):
        required = {
            "image_id", "image_path", "source_dataset",
            "category", "gender", "description",
            "color", "season", "fit", "occasion",
            "pattern", "attributes", "landmarks",
        }
        for rec in _build_demo_records(11):
            missing = required - rec.keys()
            assert not missing, f"Record missing fields: {missing}"

    def test_source_dataset_valid(self):
        """validator only accepts 'fashiongen' | 'deepfashion'."""
        for rec in _build_demo_records(11):
            assert rec["source_dataset"] in _VALID_SOURCES, (
                f"Invalid source_dataset: {rec['source_dataset']}"
            )

    def test_category_valid(self):
        for rec in _build_demo_records(11):
            assert rec["category"] in _VALID_CATEGORIES, (
                f"Invalid category: {rec['category']}"
            )

    def test_color_is_list(self):
        for rec in _build_demo_records(11):
            assert isinstance(rec["color"], list), (
                f"color must be list, got {type(rec['color'])}"
            )

    def test_pattern_is_list(self):
        for rec in _build_demo_records(11):
            assert isinstance(rec["pattern"], list), (
                f"pattern must be list, got {type(rec['pattern'])}"
            )

    def test_occasion_is_list(self):
        for rec in _build_demo_records(11):
            assert isinstance(rec["occasion"], list)

    def test_attributes_is_list(self):
        for rec in _build_demo_records(11):
            assert isinstance(rec["attributes"], list)

    def test_landmarks_is_list(self):
        for rec in _build_demo_records(11):
            assert isinstance(rec["landmarks"], list)

    def test_image_id_unique(self):
        records = _build_demo_records(30)
        ids = [r["image_id"] for r in records]
        assert len(ids) == len(set(ids)), "image_id values must be unique"

    def test_description_non_empty(self):
        for rec in _build_demo_records(11):
            assert rec.get("description"), "description must be non-empty"

    def test_cycles_through_all_categories(self):
        """With 11 records we should see all 11 taxonomy categories."""
        records = _build_demo_records(11)
        cats = {r["category"] for r in records}
        assert cats == _VALID_CATEGORIES


# =============================================================================
# ── 5. _load_records_from_json (static helper)
# =============================================================================

class TestLoadRecordsFromJson:
    """Tests for the static JSON record loader."""

    def test_loads_records_key(self, tmp_path: Path):
        p = tmp_path / "test.json"
        data = {"records": [{"id": 1}, {"id": 2}]}
        p.write_text(json.dumps(data), encoding="utf-8")
        records = MasterPipelineController._load_records_from_json(p)
        assert len(records) == 2
        assert records[0]["id"] == 1

    def test_loads_meta_and_records_key(self, tmp_path: Path):
        p = tmp_path / "loader.json"
        data = {"_meta": {"total": 1}, "records": [{"cat": "jeans"}]}
        p.write_text(json.dumps(data), encoding="utf-8")
        records = MasterPipelineController._load_records_from_json(p)
        assert len(records) == 1
        assert records[0]["cat"] == "jeans"

    def test_returns_empty_on_missing_file(self, tmp_path: Path):
        p = tmp_path / "nonexistent.json"
        records = MasterPipelineController._load_records_from_json(p)
        assert records == []

    def test_returns_empty_on_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{{", encoding="utf-8")
        records = MasterPipelineController._load_records_from_json(p)
        assert records == []

    def test_returns_empty_when_records_is_not_list(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"records": {"a": 1}}), encoding="utf-8")
        records = MasterPipelineController._load_records_from_json(p)
        assert records == []

    def test_empty_records_list(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"records": []}), encoding="utf-8")
        records = MasterPipelineController._load_records_from_json(p)
        assert records == []


# =============================================================================
# ── 6. DemoPipelineController — end-to-end
# =============================================================================

class TestDemoPipelineController:
    """End-to-end tests using DemoPipelineController (no real datasets needed)."""

    def test_run_returns_execution_summary(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert isinstance(summary, ExecutionSummary)

    def test_status_is_success(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.status == "success", (
            f"Expected status='success', got '{summary.status}'. "
            f"Errors: {summary.errors}"
        )

    def test_total_raw_records_count(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.total_raw_records == 11

    def test_final_records_exported(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.final_records > 0, "At least some records should be exported"

    def test_validation_success_rate_high(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.validation_success_rate >= 0.9, (
            f"Expected >=90% success rate, got {summary.validation_success_rate:.1%}"
        )

    def test_six_stage_timers_recorded(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert len(summary.stage_timers) == 6, (
            f"Expected 6 stage timers, got {len(summary.stage_timers)}"
        )

    def test_output_files_created(self, demo_cfg: MasterPipelineConfig, tmp_output: Path):
        from dataset_pipeline import _FINAL_JSON, _SUMMARY_JSON
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        ctrl.run()
        # Files that respect config.output_dir (tmp_output):
        assert (tmp_output / "clean_dataset.json").exists(), "clean_dataset.json missing"
        assert (tmp_output / "validation_report.json").exists(), "validation_report.json missing"
        # Files written to module-level global paths:
        assert _FINAL_JSON.exists(),  "final_fashion_dataset.json missing"
        assert _SUMMARY_JSON.exists(), "execution_summary.json missing"

    def test_final_dataset_json_structure(self, demo_cfg: MasterPipelineConfig, tmp_output: Path):
        """final_fashion_dataset.json (_FINAL_JSON) must have _meta, summary, records keys."""
        from dataset_pipeline import _FINAL_JSON
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        ctrl.run()
        assert _FINAL_JSON.exists()
        data = json.loads(_FINAL_JSON.read_text(encoding="utf-8"))
        assert "_meta"   in data
        assert "summary" in data
        assert "records" in data
        assert data["_meta"]["total_records"] >= 1

    def test_execution_summary_json_content(self, demo_cfg: MasterPipelineConfig, tmp_output: Path):
        """execution_summary.json is written to the module-level _SUMMARY_JSON path."""
        from dataset_pipeline import _SUMMARY_JSON
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        ctrl.run()
        assert _SUMMARY_JSON.exists()
        data = json.loads(_SUMMARY_JSON.read_text(encoding="utf-8"))
        assert data["status"] == "success"
        assert data["record_counts"]["total_raw"] == 11

    def test_no_errors_in_summary(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.errors == [], f"Unexpected errors: {summary.errors}"

    def test_metadata_enriched_equals_total_raw(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.metadata_enriched == 11

    def test_validation_failed_is_zero_for_valid_demo(self, demo_cfg: MasterPipelineConfig):
        ctrl = DemoPipelineController(config=demo_cfg, n_records=11)
        summary = ctrl.run()
        assert summary.validation_failed == 0, (
            f"Expected 0 validation failures, got {summary.validation_failed}"
        )

    def test_different_n_records(self, tmp_output: Path):
        """Pipeline should handle different record counts."""
        for n in (1, 5, 22):
            cfg = MasterPipelineConfig(
                enable_fashiongen    = False,
                enable_deepfashion  = False,
                metadata_nlp_enabled= False,
                enable_validation   = True,
                show_progress       = False,
                output_dir          = tmp_output,
                use_kb              = False,
            )
            ctrl = DemoPipelineController(config=cfg, n_records=n)
            summary = ctrl.run()
            assert summary.total_raw_records == n
            assert summary.status in ("success", "partial")

    def test_include_invalid_flag(self, tmp_output: Path):
        """export_include_invalid=True should include all records in final export."""
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            metadata_nlp_enabled= False,
            enable_validation   = True,
            export_include_invalid = True,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = DemoPipelineController(config=cfg, n_records=11)
        summary = ctrl.run()
        # All records (valid + invalid) should be exported
        assert summary.final_records == summary.validation_valid + summary.validation_failed


# =============================================================================
# ── 7. MasterPipelineController — Selective Stage Skipping
# =============================================================================

class TestMasterPipelineStageSkipping:
    """Tests that individual stages can be disabled without crashing."""

    def _make_ctrl(self, tmp_output, **overrides) -> MasterPipelineController:
        base = dict(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_preprocessing= False,
            enable_validation   = False,
            show_progress       = False,
            output_dir          = tmp_output,
            use_kb              = False,
        )
        base.update(overrides)  # overrides win — avoids duplicate-kwarg TypeError
        cfg = MasterPipelineConfig(**base)
        return MasterPipelineController(config=cfg)

    def test_all_stages_disabled_no_crash(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output)
        summary = ctrl.run()
        # No records → "failed" status but no exception
        assert isinstance(summary, ExecutionSummary)

    def test_fashiongen_disabled_records_empty(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output)
        records = ctrl._run_fashiongen_stage()
        assert records == []

    def test_deepfashion_disabled_records_empty(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output)
        records = ctrl._run_deepfashion_stage()
        assert records == []

    def test_metadata_stage_disabled_returns_original(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output, enable_metadata_gen=False)
        input_records = [{"image_id": "X001", "description": "test"}]
        result = ctrl._run_metadata_stage(input_records)
        assert result is input_records  # same object returned unchanged

    def test_preprocessing_stage_disabled_returns_original(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output, enable_preprocessing=False)
        input_records = [{"image_id": "X001"}]
        result = ctrl._run_preprocessing_stage(input_records)
        assert result is input_records

    def test_validation_stage_disabled_all_valid(self, tmp_output: Path):
        ctrl = self._make_ctrl(tmp_output, enable_validation=False)
        input_records = [{"image_id": "X001"}, {"image_id": "X002"}]
        valid, invalid = ctrl._run_validation_stage(input_records)
        assert len(valid) == 2
        assert invalid == []


# =============================================================================
# ── 8. Metadata Stage
# =============================================================================

class TestMetadataStage:
    """Tests for the metadata enrichment stage in isolation."""

    def test_enrichment_fills_missing_fields(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            metadata_nlp_enabled= False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{
            "image_id"      : "T001",
            "image_path"    : "test/image.jpg",
            "source_dataset": "fashiongen",
            "category"      : "hoodies",
            "description"   : "A black oversized hoodie with graphic print",
        }]
        enriched = ctrl._run_metadata_stage(records)
        assert len(enriched) == 1
        rec = enriched[0]
        # After enrichment, metadata fields should be populated
        assert "_auto_generated" in rec
        assert rec["_auto_generated"] is True

    def test_enrichment_preserves_existing_fields(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            metadata_nlp_enabled= False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{
            "image_id"      : "T002",
            "image_path"    : "test/image.jpg",
            "source_dataset": "fashiongen",
            "category"      : "jackets",
            "style"         : "luxury",      # pre-set — should NOT be overwritten
            "description"   : "A streetwear bomber jacket",
        }]
        enriched = ctrl._run_metadata_stage(records)
        # "style" was already set — must remain "luxury"
        assert enriched[0]["style"] == "luxury"

    def test_empty_description_handled(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            metadata_nlp_enabled= False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{
            "image_id"      : "T003",
            "image_path"    : "test/image.jpg",
            "source_dataset": "fashiongen",
            "category"      : "shirts",
            "description"   : "",     # empty
        }]
        enriched = ctrl._run_metadata_stage(records)
        assert len(enriched) == 1  # record should still be returned


# =============================================================================
# ── 9. CLI Argument Parser
# =============================================================================

class TestCLIParser:
    """Tests for the _build_cli_parser argument parser."""

    def test_defaults(self):
        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.max_fg is None
        assert args.max_df is None
        assert args.df_split == "train"
        assert args.no_fashiongen is False
        assert args.no_deepfashion is False
        assert args.no_metadata is False
        assert args.no_preproc is False
        assert args.no_validation is False
        assert args.include_invalid is False
        assert args.demo is False
        assert args.demo_n == 20
        assert args.no_progress is False

    def test_demo_flag(self):
        args = _build_cli_parser().parse_args(["--demo"])
        assert args.demo is True

    def test_demo_n_custom(self):
        args = _build_cli_parser().parse_args(["--demo", "--demo-n", "50"])
        assert args.demo_n == 50

    def test_max_fg_and_df(self):
        args = _build_cli_parser().parse_args(["--max-fg", "100", "--max-df", "200"])
        assert args.max_fg == 100
        assert args.max_df == 200

    def test_disable_fashiongen(self):
        args = _build_cli_parser().parse_args(["--no-fashiongen"])
        assert args.no_fashiongen is True

    def test_disable_deepfashion(self):
        args = _build_cli_parser().parse_args(["--no-deepfashion"])
        assert args.no_deepfashion is True

    def test_disable_metadata(self):
        args = _build_cli_parser().parse_args(["--no-metadata"])
        assert args.no_metadata is True

    def test_disable_preproc(self):
        args = _build_cli_parser().parse_args(["--no-preproc"])
        assert args.no_preproc is True

    def test_disable_validation(self):
        args = _build_cli_parser().parse_args(["--no-validation"])
        assert args.no_validation is True

    def test_include_invalid(self):
        args = _build_cli_parser().parse_args(["--include-invalid"])
        assert args.include_invalid is True

    def test_no_progress(self):
        args = _build_cli_parser().parse_args(["--no-progress"])
        assert args.no_progress is True

    def test_df_split_choices(self):
        for split in ("train", "val", "test", "all"):
            args = _build_cli_parser().parse_args(["--df-split", split])
            assert args.df_split == split

    def test_invalid_df_split_raises(self):
        with pytest.raises(SystemExit):
            _build_cli_parser().parse_args(["--df-split", "invalid"])

    def test_output_dir(self, tmp_path: Path):
        args = _build_cli_parser().parse_args(["--output-dir", str(tmp_path)])
        assert args.output_dir == str(tmp_path)


# =============================================================================
# ── 10. Preprocessing Stage Integration
# =============================================================================

class TestPreprocessingStage:
    """Tests for Stage 4 preprocessing integration."""

    def test_preprocessing_cleans_descriptions(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_validation   = False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{
            "image_id"      : "P001",
            "image_path"    : "test/img.jpg",
            "source_dataset": "fashiongen",
            "category"      : "t_shirts",
            "description"   : "  <b>Great tee</b>   with  lots   of  spaces!!!  ",
        }]
        cleaned = ctrl._run_preprocessing_stage(records)
        assert len(cleaned) == 1
        desc = cleaned[0].get("description", "")
        assert "<b>" not in desc
        assert "  " not in desc   # no multiple spaces

    def test_preprocessing_deduplication(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_validation   = False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        # Two identical image paths — second should be deduplicated
        records = [
            {"image_id": "P001", "image_path": "test/same.jpg",
             "source_dataset": "fashiongen", "category": "shirts",
             "description": "A shirt"},
            {"image_id": "P002", "image_path": "test/same.jpg",
             "source_dataset": "fashiongen", "category": "shirts",
             "description": "A shirt"},
        ]
        cleaned = ctrl._run_preprocessing_stage(records)
        # One of the two duplicates should be removed
        assert len(cleaned) == 1
        assert ctrl.summary.duplicates_removed == 1

    def test_preprocessing_writes_clean_json(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_validation   = False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{
            "image_id"      : "P010",
            "image_path"    : "test/img.jpg",
            "source_dataset": "fashiongen",
            "category"      : "jeans",
            "description"   : "Blue slim-fit jeans.",
        }]
        ctrl._run_preprocessing_stage(records)
        clean_json = tmp_output / "clean_dataset.json"
        assert clean_json.exists(), "clean_dataset.json must be written by preprocessing"


# =============================================================================
# ── 11. Validation Stage Integration
# =============================================================================

class TestValidationStage:
    """Tests for Stage 5 validation integration."""

    def test_valid_records_separated(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_preprocessing= False,
            enable_validation   = True,
            validation_image_check= False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [
            {
                "image_id"      : "V001",
                "image_path"    : "datasets/test/img1.jpg",
                "source_dataset": "fashiongen",
                "category"      : "t_shirts",
                "description"   : "A great cotton tee with a cool print on the front.",
                "color"         : ["Black"],
                "gender"        : "unisex",
                "style"         : "streetwear",
                "season"        : "summer",
                "fit"           : "regular_fit",
                "occasion"      : ["casual"],
                "pattern"       : ["solid"],
                "attributes"    : [],
                "landmarks"     : [],
            },
            {
                # Missing required source_dataset → should fail validation
                "image_id"      : "V002",
                "image_path"    : "datasets/test/img2.jpg",
                "source_dataset": "",   # empty → invalid
                "category"      : "jeans",
                "description"   : "Some jeans",
            },
        ]
        valid, invalid = ctrl._run_validation_stage(records)
        # V001 should be valid, V002 should be invalid
        valid_ids   = {r["image_id"] for r in valid}
        invalid_ids = {r["image_id"] for r in invalid}
        assert "V001" in valid_ids
        assert "V002" in invalid_ids

    def test_validation_report_written(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_preprocessing= False,
            enable_validation   = True,
            validation_image_check= False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        ctrl._run_validation_stage([])
        report = tmp_output / "validation_report.json"
        assert report.exists()

    def test_validation_skipped_returns_all_as_valid(self, tmp_output: Path):
        cfg = MasterPipelineConfig(
            enable_fashiongen    = False,
            enable_deepfashion  = False,
            enable_metadata_gen = False,
            enable_preprocessing= False,
            enable_validation   = False,
            show_progress        = False,
            output_dir           = tmp_output,
            use_kb               = False,
        )
        ctrl = MasterPipelineController(config=cfg)
        records = [{"image_id": "A"}, {"image_id": "B"}]
        valid, invalid = ctrl._run_validation_stage(records)
        assert len(valid) == 2
        assert invalid == []


# =============================================================================
# ── 12. ExecutionSummary — JSON round-trip
# =============================================================================

class TestExecutionSummaryJsonRoundtrip:
    """Verify ExecutionSummary can be serialised and de-serialised via JSON."""

    def test_round_trip(self):
        s = ExecutionSummary()
        s.fashiongen_records  = 100
        s.deepfashion_records = 200
        s.total_raw_records   = 300
        s.metadata_enriched   = 300
        s.after_preprocessing = 295
        s.duplicates_removed  = 5
        s.validation_valid    = 290
        s.validation_failed   = 5
        s.final_records       = 290
        s.validation_success_rate  = 290 / 295
        s.validation_quality_score = 0.97
        s.output_paths = {"final_dataset": "/some/path/final.json"}
        t = StageTimer("stage1")
        t.start_time = t.end_time = time.perf_counter()
        s.stage_timers.append(t)
        s.finalize()

        json_str = json.dumps(s.to_dict(), ensure_ascii=False)
        loaded   = json.loads(json_str)

        assert loaded["status"] == "success"
        assert loaded["record_counts"]["total_raw"] == 300
        assert loaded["record_counts"]["final_exported"] == 290
        assert len(loaded["stage_timings"]) == 1
        assert loaded["output_paths"]["final_dataset"] == "/some/path/final.json"
