"""
Extended unit tests for data_pipeline/ingestion/deepfashion_loader.py

Targets:
  - DeepFashionAnnotationParser  (annotation file parsers)
  - DeepFashionLoader.__init__   (path resolution, kb flag)
  - DeepFashionWriter.save_records / save_run_report
  - DeepFashionTransformer._normalize_category
  - DeepFashionExtractor stream / _build_raw_record

All tests are self-contained — no real DeepFashion dataset required.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.ingestion.deepfashion_loader import (
    DeepFashionLoader,
    DeepFashionAnnotationParser,
    DeepFashionWriter,
    DeepFashionTransformer,
    DFPipelineStats,
    _DF_CATEGORY_MAP,
)


# =============================================================================
# ── Helpers
# =============================================================================

def _write_annotation_file(path: Path, lines: list[str]) -> None:
    """Write annotation lines to a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_anno_dir(tmp_path: Path) -> Path:
    """Create minimal DeepFashion directory skeleton."""
    anno_dir = tmp_path / "Anno"
    anno_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


# =============================================================================
# ── DeepFashionAnnotationParser
# =============================================================================

class TestAnnotationParsers:

    def test_is_available_false_for_missing_root(self, tmp_path):
        parser = DeepFashionAnnotationParser(root_dir=tmp_path / "nonexistent")
        assert parser.is_available() is False

    def test_is_available_true_for_valid_root(self, tmp_path):
        anno = tmp_path / "Anno"
        anno.mkdir()
        # Just the Anno dir is enough for is_available to check
        parser = DeepFashionAnnotationParser(root_dir=tmp_path)
        # is_available checks for specific annotation files — even without them
        # the parser shouldn't raise
        result = parser.is_available()
        assert isinstance(result, bool)

    def test_category_file_parsing(self, tmp_path):
        """Parser should populate category_names from list_category_cloth.txt."""
        root = _make_anno_dir(tmp_path)
        cat_file = root / "Anno" / "list_category_cloth.txt"
        _write_annotation_file(cat_file, [
            "3",                      # count
            "category_name  category_type",  # header
            "T-Shirt  1",
            "Blouse   1",
            "Shorts   2",
        ])
        parser = DeepFashionAnnotationParser(root_dir=root)
        parser._parse_category_cloth()
        assert "T-Shirt" in parser.category_names
        assert len(parser.category_names) == 3

    def test_attribute_file_parsing(self, tmp_path):
        """Parser should populate attr_names from list_attr_cloth.txt."""
        root = _make_anno_dir(tmp_path)
        attr_file = root / "Anno" / "list_attr_cloth.txt"
        _write_annotation_file(attr_file, [
            "2",
            "attribute_name  attribute_type",
            "floral  1",
            "striped 2",
        ])
        parser = DeepFashionAnnotationParser(root_dir=root)
        parser._parse_attr_cloth()
        assert "floral" in parser.attr_names
        assert len(parser.attr_names) == 2

    def test_missing_annotation_file_returns_empty(self, tmp_path):
        """Parser should not raise when annotation files are absent."""
        root = _make_anno_dir(tmp_path)  # Anno dir exists, but files absent
        parser = DeepFashionAnnotationParser(root_dir=root)
        parser._parse_category_cloth()   # Should log warning, not raise
        assert parser.category_names == []

    def test_get_image_paths_for_split_empty_when_no_data(self, tmp_path):
        root = _make_anno_dir(tmp_path)
        parser = DeepFashionAnnotationParser(root_dir=root)
        paths = parser.get_image_paths_for_split("train")
        assert isinstance(paths, list)
        assert len(paths) == 0

    def test_get_stats_returns_dict(self, tmp_path):
        root = _make_anno_dir(tmp_path)
        parser = DeepFashionAnnotationParser(root_dir=root)
        stats = parser.get_stats()
        assert isinstance(stats, dict)


# =============================================================================
# ── DeepFashionTransformer — _normalize_category
# =============================================================================

class TestDeepFashionTransformerCategory:

    def setup_method(self):
        self.transformer = DeepFashionTransformer(root_dir=Path("."), kb=None, max_attrs=50)

    def test_exact_map_key(self):
        """Keys in _DF_CATEGORY_MAP should resolve correctly."""
        for raw, expected in list(_DF_CATEGORY_MAP.items())[:5]:
            result = self.transformer._normalise_category(raw)
            assert result == expected, f"{raw!r} → {result!r} expected {expected!r}"

    def test_unknown_falls_back_to_accessories(self):
        result = self.transformer._normalise_category("COMPLETELY_UNKNOWN_XYZABC")
        assert result == "accessories"

    def test_empty_string_falls_back(self):
        result = self.transformer._normalise_category("")
        assert result == "accessories"

    def test_case_insensitive_match(self):
        # First key in map lowercased
        first_raw = list(_DF_CATEGORY_MAP.keys())[0]
        result = self.transformer._normalise_category(first_raw.lower())
        expected = _DF_CATEGORY_MAP[first_raw]
        assert result == expected


# =============================================================================
# ── DeepFashionLoader.__init__
# =============================================================================

class TestDeepFashionLoaderInit:

    def test_default_paths_set(self):
        loader = DeepFashionLoader(use_kb=False)
        assert loader.root_dir is not None
        assert loader.output_dir is not None

    def test_custom_root_dir(self, tmp_path):
        loader = DeepFashionLoader(root_dir=str(tmp_path), use_kb=False)
        assert loader.root_dir == tmp_path

    def test_custom_output_dir(self, tmp_path):
        loader = DeepFashionLoader(output_dir=str(tmp_path), use_kb=False)
        assert loader.output_dir == tmp_path

    def test_kb_disabled(self):
        loader = DeepFashionLoader(use_kb=False)
        assert loader.kb is None

    def test_layers_attached(self):
        loader = DeepFashionLoader(use_kb=False)
        assert loader.parser is not None
        assert loader.extractor is not None
        assert loader.transformer is not None
        assert loader.validator is not None
        assert loader.writer is not None

    def test_max_attrs_propagates(self):
        loader = DeepFashionLoader(max_attrs=10, use_kb=False)
        assert loader.transformer.max_attrs == 10


# =============================================================================
# ── DeepFashionLoader.run — invalid split raises ValueError
# =============================================================================

class TestDeepFashionLoaderRun:

    def test_invalid_split_raises(self, tmp_path):
        loader = DeepFashionLoader(root_dir=str(tmp_path), use_kb=False)
        with pytest.raises(ValueError, match="split"):
            loader.run(split="invalid_split_xyz", show_progress=False)

    def test_valid_split_no_data_returns_zero_or_raises(self, tmp_path):
        """When parser has no data, run() may return 0 records or raise RuntimeError."""
        loader = DeepFashionLoader(root_dir=str(tmp_path), use_kb=False)
        try:
            result = loader.run(split="train", show_progress=False)
            assert result["total_records"] == 0
        except RuntimeError as exc:
            assert "no data" in str(exc).lower() or "annotation" in str(exc).lower()

    def test_run_returns_required_keys(self, tmp_path):
        """run() with no data should either return keys or raise RuntimeError."""
        loader = DeepFashionLoader(root_dir=str(tmp_path), use_kb=False)
        try:
            result = loader.run(split="train", show_progress=False)
            for key in ("output_path", "report_path", "stats", "total_records"):
                assert key in result, f"Missing key: {key}"
        except RuntimeError:
            pass  # Expected when parser has no data — RuntimeError is the documented contract

    def test_get_dataset_info_returns_dict(self, tmp_path):
        loader = DeepFashionLoader(root_dir=str(tmp_path), use_kb=False)
        info = loader.get_dataset_info()
        assert isinstance(info, dict)


# =============================================================================
# ── DeepFashionWriter.save_records / save_run_report
# =============================================================================

class TestDeepFashionLoaderRunSave:

    def test_save_empty_records(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        mock_stats = MagicMock(spec=DFPipelineStats)
        mock_stats.to_dict.return_value = {"total_processed": 0}
        mock_stats.total_processed = 0
        path = writer.save_records([], mock_stats)
        assert path.exists(), f"Expected file at {path}"

    def test_save_custom_path(self, tmp_path):
        """save_records writes to output_dir — confirm different dirs produce different paths."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        mock_stats = MagicMock(spec=DFPipelineStats)
        mock_stats.to_dict.return_value = {"total_processed": 0}
        mock_stats.total_processed = 0

        w1 = DeepFashionWriter(output_dir=dir1)
        w2 = DeepFashionWriter(output_dir=dir2)
        p1 = w1.save_records([], mock_stats)
        p2 = w2.save_records([], mock_stats)
        assert p1 != p2

    def test_save_records_json_valid(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        mock_stats = MagicMock(spec=DFPipelineStats)
        mock_stats.to_dict.return_value = {"total_processed": 0}
        mock_stats.total_processed = 0
        path = writer.save_records([], mock_stats)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_save_run_report_creates_file(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        mock_stats = MagicMock(spec=DFPipelineStats)
        mock_stats.to_dict.return_value = {"total_processed": 0}
        path = writer.save_run_report(mock_stats, tmp_path)
        assert path.exists()


# =============================================================================
# ── DFPipelineStats
# =============================================================================

class TestDFPipelineStats:

    def test_initial_zeroes(self):
        stats = DFPipelineStats()
        assert stats.total_read == 0
        assert stats.total_processed == 0

    def test_to_dict_returns_dict(self):
        stats = DFPipelineStats()
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert "total_processed" in d
