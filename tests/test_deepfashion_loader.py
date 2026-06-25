"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_deepfashion_loader.py — Unit Tests: DeepFashion Ingestion Pipeline
=============================================================================
Full test suite for deepfashion_loader.py.
All tests are fully mocked — no real annotation files or images required.

Test classes and coverage:
  TestDeepFashionRecord              — dataclass + to_dict() spec compliance
  TestDFPipelineStats                — counters, rates, timing
  TestDeepFashionAnnotationParser    — TXT parsing logic (in-memory stubs)
  TestDeepFashionExtractor           — record assembly, missing-value handling
  TestDeepFashionTransformer         — category, attributes, landmarks, bbox
  TestDeepFashionValidator           — all 7 validation layers
  TestDeepFashionWriter              — JSON output structure + spec fields
  TestDeepFashionLoader              — end-to-end orchestration (fully mocked)
  TestCategoryMapping                — DF category map completeness
  TestNormalisationEdgeCases         — boundary conditions

Run:
    pytest tests/test_deepfashion_loader.py -v
    pytest tests/test_deepfashion_loader.py -v --cov=data_pipeline.ingestion.deepfashion_loader
=============================================================================
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── System under test ─────────────────────────────────────────────────────────
from src.data.ingestion.deepfashion_loader import (
    # Data models
    RawDeepFashionRecord,
    DeepFashionRecord,
    DFPipelineStats,
    # Pipeline layers
    DeepFashionAnnotationParser,
    DeepFashionExtractor,
    DeepFashionTransformer,
    DeepFashionValidator,
    DeepFashionWriter,
    DeepFashionLoader,
    # Constants
    _DF_CATEGORY_MAP,
    _LANDMARK_NAMES,
    _VALID_CATEGORIES,
    _OUTPUT_FILENAME,
    _DEFAULT_ROOT_DIR,
    _DEFAULT_OUTPUT_DIR,
)


# =============================================================================
# ── Shared Fixtures
# =============================================================================

@pytest.fixture
def raw_record() -> RawDeepFashionRecord:
    """A complete, valid RawDeepFashionRecord."""
    return RawDeepFashionRecord(
        image_rel_path  = "img/Blouse/img_00000001.jpg",
        category_raw    = "Blouse",
        category_id     = 1,
        attr_vector     = [1, -1, 1, -1, 0] + [-1] * 995,
        attr_names      = ["floral pattern", "sleeveless", "cotton material",
                           "solid color", "loose fit"] + ["attr_" + str(i) for i in range(995)],
        landmarks_raw   = [
            {"name": "left_collar",  "x": 100, "y": 40,  "visible": True},
            {"name": "right_collar", "x": 156, "y": 40,  "visible": True},
            {"name": "left_sleeve",  "x": 60,  "y": 120, "visible": True},
            {"name": "right_sleeve", "x": 196, "y": 120, "visible": True},
            {"name": "left_hem",     "x": 80,  "y": 210, "visible": True},
            {"name": "right_hem",    "x": 176, "y": 210, "visible": True},
        ],
        bbox            = [50, 30, 206, 230],
        split           = "train",
        image_width     = 256,
        image_height    = 256,
    )


@pytest.fixture
def processed_record() -> DeepFashionRecord:
    """A fully processed, valid DeepFashionRecord."""
    return DeepFashionRecord(
        image_id         = "DF_img_Blouse_img_00000001",
        category         = "shirts",
        attributes       = ["floral pattern", "cotton material"],
        landmarks        = [
            {"name": "left_collar",  "x": 0.3125, "y": 0.05, "visible": True},
            {"name": "right_collar", "x": 0.6719, "y": 0.05, "visible": True},
            {"name": "left_sleeve",  "x": 0.0625, "y": 0.45, "visible": True},
            {"name": "right_sleeve", "x": 0.9063, "y": 0.45, "visible": True},
            {"name": "left_hem",     "x": 0.1875, "y": 0.90, "visible": True},
            {"name": "right_hem",    "x": 0.8125, "y": 0.90, "visible": True},
        ],
        image_path       = "datasets/deepfashion/img/Blouse/img_00000001.jpg",
        dataset_source   = "deepfashion",
        category_raw     = "Blouse",
        category_id      = 1,
        split            = "train",
        bbox             = [50, 30, 206, 230],
        bbox_normalised  = [0.1953, 0.1172, 0.8047, 0.8984],
        attr_count       = 2,
        is_valid         = True,
    )


@pytest.fixture
def transformer() -> DeepFashionTransformer:
    """DeepFashionTransformer without KB."""
    return DeepFashionTransformer(kb=None)


@pytest.fixture
def validator() -> DeepFashionValidator:
    return DeepFashionValidator(kb=None)


# ── Parser stub helpers ───────────────────────────────────────────────────────

def _make_parser_stub(tmp_path: Path) -> DeepFashionAnnotationParser:
    """
    Return a DeepFashionAnnotationParser with its internals populated
    from in-memory data (no real files required).
    """
    parser = DeepFashionAnnotationParser.__new__(DeepFashionAnnotationParser)
    parser.root_dir      = tmp_path
    parser.category_names = [
        "Blouse", "Button-Down", "Dress Shirt", "Jeans", "Shorts",
        "Jacket", "Dress", "Shawl", "Hoodie", "Tee",
    ]
    parser.attr_names = ["floral pattern", "sleeveless", "cotton material",
                         "solid color", "loose fit"]
    parser.split_map = {
        "img/Blouse/img_00000001.jpg"     : "train",
        "img/Blouse/img_00000002.jpg"     : "val",
        "img/Jeans/img_00000001.jpg"      : "train",
        "img/Dress/img_00000001.jpg"      : "test",
        "img/Jacket/img_00000001.jpg"     : "train",
    }
    parser.category_map = {
        "img/Blouse/img_00000001.jpg"  : ("Blouse",  1),
        "img/Blouse/img_00000002.jpg"  : ("Blouse",  1),
        "img/Jeans/img_00000001.jpg"   : ("Jeans",   4),
        "img/Dress/img_00000001.jpg"   : ("Dress",   7),
        "img/Jacket/img_00000001.jpg"  : ("Jacket",  6),
    }
    parser.attr_map = {
        "img/Blouse/img_00000001.jpg" : [1, -1, 1, -1, 0],
        "img/Jeans/img_00000001.jpg"  : [-1, 1, -1, 1, -1],
        "img/Dress/img_00000001.jpg"  : [1, 1, -1, -1, 1],
        "img/Jacket/img_00000001.jpg" : [-1, -1, 1, -1, -1],
    }
    parser.bbox_map = {
        "img/Blouse/img_00000001.jpg" : [50, 30, 206, 230],
        "img/Jeans/img_00000001.jpg"  : [40, 80, 220, 256],
    }
    parser.landmark_map = {
        "img/Blouse/img_00000001.jpg": [
            {"name": n, "x": 100 + i * 10, "y": 40 + i * 30, "visible": True}
            for i, n in enumerate(_LANDMARK_NAMES)
        ],
    }
    return parser


# =============================================================================
# ── 1. TestDeepFashionRecord
# =============================================================================

class TestDeepFashionRecord:
    """Tests for DeepFashionRecord dataclass and to_dict() spec compliance."""

    SPEC_FIELDS = ["image_id", "category", "attributes", "landmarks", "image_path"]

    def test_to_dict_contains_all_spec_fields(self, processed_record):
        """to_dict() must contain every field in the output spec."""
        d = processed_record.to_dict()
        for f in self.SPEC_FIELDS:
            assert f in d, f"Spec field '{f}' missing from to_dict()"

    def test_to_dict_is_json_serializable(self, processed_record):
        """to_dict() output must be JSON-serializable."""
        d = processed_record.to_dict()
        s = json.dumps(d)
        assert processed_record.image_id in s

    def test_to_dict_attributes_is_list(self, processed_record):
        d = processed_record.to_dict()
        assert isinstance(d["attributes"], list)

    def test_to_dict_landmarks_is_list(self, processed_record):
        d = processed_record.to_dict()
        assert isinstance(d["landmarks"], list)
        assert len(d["landmarks"]) == 6

    def test_to_dict_landmark_has_all_keys(self, processed_record):
        d = processed_record.to_dict()
        for lm in d["landmarks"]:
            assert "name"    in lm
            assert "x"       in lm
            assert "y"       in lm
            assert "visible" in lm

    def test_default_dataset_source(self):
        rec = DeepFashionRecord(
            image_id="DF_x", category="shirts", attributes=[],
            landmarks=[], image_path="p.jpg"
        )
        assert rec.dataset_source == "deepfashion"

    def test_processed_at_is_iso_string(self, processed_record):
        from datetime import datetime
        d = processed_record.to_dict()
        datetime.fromisoformat(d["processed_at"])  # must not raise

    def test_errors_warnings_in_dict(self):
        rec = DeepFashionRecord(
            image_id="X", category="shirts", attributes=[],
            landmarks=[], image_path="p.jpg",
            errors=["e1"], warnings=["w1"]
        )
        d = rec.to_dict()
        assert d["errors"]   == ["e1"]
        assert d["warnings"] == ["w1"]

    def test_image_path_in_output(self, processed_record):
        d = processed_record.to_dict()
        assert "img" in d["image_path"] or "deepfashion" in d["image_path"]


# =============================================================================
# ── 2. TestDFPipelineStats
# =============================================================================

class TestDFPipelineStats:
    """Tests for DFPipelineStats counters, rates, and timing."""

    def test_initial_state(self):
        stats = DFPipelineStats()
        assert stats.total_read       == 0
        assert stats.total_processed  == 0
        assert stats.total_valid      == 0
        assert stats.total_invalid    == 0
        assert stats.total_skipped    == 0
        assert stats.total_missing_img == 0
        assert stats.total_saved      == 0

    def test_valid_rate_zero_division(self):
        assert DFPipelineStats().valid_rate == 0.0

    def test_valid_rate_calculation(self):
        s = DFPipelineStats()
        s.total_processed = 200
        s.total_valid     = 190
        assert abs(s.valid_rate - 0.95) < 0.001

    def test_records_per_second_positive(self):
        s = DFPipelineStats()
        s.total_processed = 100
        time.sleep(0.01)
        assert s.records_per_second > 0

    def test_finalize_stamps_time(self):
        s = DFPipelineStats()
        time.sleep(0.01)
        s.finalize()
        t1 = s.elapsed_seconds
        time.sleep(0.05)
        t2 = s.elapsed_seconds
        assert abs(t1 - t2) < 0.002, "elapsed_seconds changed after finalize()"

    def test_increment_category(self):
        s = DFPipelineStats()
        s.increment_category("shirts")
        s.increment_category("shirts")
        s.increment_category("jeans")
        assert s.category_counts["shirts"] == 2
        assert s.category_counts["jeans"]  == 1

    def test_increment_split(self):
        s = DFPipelineStats()
        s.increment_split("train")
        s.increment_split("val")
        s.increment_split("train")
        assert s.split_counts["train"] == 2
        assert s.split_counts["val"]   == 1

    def test_to_dict_completeness(self):
        s = DFPipelineStats()
        d = s.to_dict()
        required = [
            "total_read", "total_processed", "total_valid", "total_invalid",
            "total_skipped", "total_missing_img", "total_saved",
            "valid_rate", "records_per_second", "elapsed_seconds",
            "category_counts", "split_counts", "landmark_coverage",
        ]
        for key in required:
            assert key in d, f"DFPipelineStats.to_dict() missing: '{key}'"

    def test_to_dict_json_serializable(self):
        s = DFPipelineStats()
        s.total_processed = 10; s.total_valid = 8
        json.dumps(s.to_dict())  # must not raise

    def test_log_summary_no_crash(self):
        s = DFPipelineStats()
        s.total_processed = 100; s.total_valid = 90
        s.finalize()
        s.log_summary()  # must not raise


# =============================================================================
# ── 3. TestDeepFashionAnnotationParser
# =============================================================================

class TestDeepFashionAnnotationParser:
    """Tests for TXT annotation file parsing using in-memory temp files."""

    def _write_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _make_root(self, tmp_path: Path) -> Path:
        """Create minimal annotation files in a temp root directory."""
        root = tmp_path / "deepfashion"

        # list_category_cloth.txt
        self._write_file(root / "Anno" / "list_category_cloth.txt",
            "3\n"
            "category_name category_type\n"
            "Blouse 1\n"
            "Jeans 2\n"
            "Jacket 3\n"
        )

        # list_attr_cloth.txt
        self._write_file(root / "Anno" / "list_attr_cloth.txt",
            "3\n"
            "attribute_name attribute_type\n"
            "floral_pattern 1\n"
            "sleeveless 2\n"
            "cotton 3\n"
        )

        # list_eval_partition.txt
        self._write_file(root / "Eval" / "list_eval_partition.txt",
            "4\n"
            "image_name evaluation_status\n"
            "img/Blouse/img_00000001.jpg train\n"
            "img/Blouse/img_00000002.jpg val\n"
            "img/Jeans/img_00000001.jpg  train\n"
            "img/Jacket/img_00000001.jpg test\n"
        )

        # list_category_img.txt
        self._write_file(root / "Anno" / "list_category_img.txt",
            "4\n"
            "image_name category_label\n"
            "img/Blouse/img_00000001.jpg 1\n"
            "img/Blouse/img_00000002.jpg 1\n"
            "img/Jeans/img_00000001.jpg  2\n"
            "img/Jacket/img_00000001.jpg 3\n"
        )

        # list_attr_img.txt
        self._write_file(root / "Anno" / "list_attr_img.txt",
            "4\n"
            "3\n"
            "image_name attribute_labels\n"
            "img/Blouse/img_00000001.jpg 1 -1 1\n"
            "img/Blouse/img_00000002.jpg -1 1 -1\n"
            "img/Jeans/img_00000001.jpg  1 1 -1\n"
            "img/Jacket/img_00000001.jpg -1 -1 1\n"
        )

        # list_bbox.txt
        self._write_file(root / "Anno" / "list_bbox.txt",
            "4\n"
            "image_name x_1 y_1 x_2 y_2\n"
            "img/Blouse/img_00000001.jpg 50 30 200 220\n"
            "img/Jeans/img_00000001.jpg  40 80 220 256\n"
        )

        # list_landmarks.txt (18 cols per image: 6 landmarks × 3 values)
        self._write_file(root / "Anno" / "list_landmarks.txt",
            "2\n"
            "image_name lm_x1 lm_y1 lm_vis1 lm_x2 lm_y2 lm_vis2 "
            "lm_x3 lm_y3 lm_vis3 lm_x4 lm_y4 lm_vis4 "
            "lm_x5 lm_y5 lm_vis5 lm_x6 lm_y6 lm_vis6\n"
            "img/Blouse/img_00000001.jpg 100 40 1 156 40 1 60 120 1 196 120 1 80 210 1 176 210 1\n"
            "img/Jeans/img_00000001.jpg  0 0 0 0 0 0 50 180 1 200 180 1 60 240 1 190 240 1\n"
        )

        return root

    def test_parse_category_cloth(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        assert len(parser.category_names) == 3
        assert parser.category_names[0] == "Blouse"
        assert parser.category_names[1] == "Jeans"

    def test_parse_attr_cloth(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        assert len(parser.attr_names) == 3
        assert "floral" in parser.attr_names[0].lower()
        assert "sleeveless" in parser.attr_names[1].lower()

    def test_parse_eval_partition(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        assert len(parser.split_map) == 4
        assert parser.split_map["img/Blouse/img_00000001.jpg"] == "train"
        assert parser.split_map["img/Blouse/img_00000002.jpg"] == "val"
        assert parser.split_map["img/Jacket/img_00000001.jpg"] == "test"

    def test_parse_category_img(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        cat_name, cat_id = parser.category_map["img/Blouse/img_00000001.jpg"]
        assert cat_name == "Blouse"
        assert cat_id   == 1

    def test_parse_attr_img_active_attrs(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        vec = parser.attr_map["img/Blouse/img_00000001.jpg"]
        assert vec[0] == 1    # floral_pattern present
        assert vec[1] == -1   # sleeveless absent
        assert vec[2] == 1    # cotton present

    def test_parse_bbox_coords(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        bbox = parser.bbox_map["img/Blouse/img_00000001.jpg"]
        assert bbox == [50, 30, 200, 220]

    def test_parse_landmarks_count(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        lms = parser.landmark_map["img/Blouse/img_00000001.jpg"]
        assert len(lms) == 6

    def test_parse_landmarks_visibility(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        lms = parser.landmark_map["img/Blouse/img_00000001.jpg"]
        assert all(lm["visible"] for lm in lms)
        jeans_lms = parser.landmark_map["img/Jeans/img_00000001.jpg"]
        # First two are invisible (vis=0)
        assert not jeans_lms[0]["visible"]
        assert not jeans_lms[1]["visible"]

    def test_get_image_paths_for_split_train(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        paths = parser.get_image_paths_for_split("train")
        assert len(paths) == 2
        assert "img/Blouse/img_00000001.jpg" in paths

    def test_get_image_paths_for_split_all(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        all_paths = parser.get_image_paths_for_split("all")
        assert len(all_paths) == 4

    def test_is_available_true_when_parsed(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        assert parser.is_available() is True

    def test_is_available_false_when_no_root(self, tmp_path):
        parser = DeepFashionAnnotationParser(tmp_path / "nonexistent")
        assert parser.is_available() is False

    def test_get_stats_keys(self, tmp_path):
        root = self._make_root(tmp_path)
        parser = DeepFashionAnnotationParser(root)
        stats = parser.get_stats()
        for key in ["total_images", "total_category_names", "total_attr_names",
                    "total_landmark_map", "split_distribution"]:
            assert key in stats

    def test_missing_file_graceful(self, tmp_path):
        """Parser should warn and continue if a file is missing."""
        root = tmp_path / "partial_df"
        root.mkdir()
        # Only eval_partition — no other files
        (root / "Eval").mkdir()
        (root / "Eval" / "list_eval_partition.txt").write_text(
            "1\nimage_name evaluation_status\nimg/x.jpg train\n"
        )
        (root / "Anno").mkdir()
        parser = DeepFashionAnnotationParser(root)
        # Should not raise — partial parse is fine
        assert parser.is_available() is True
        assert parser.category_names == []

    def test_negative_bbox_clamped(self, tmp_path):
        """Negative bbox coordinates should be clamped to 0."""
        root = tmp_path / "df_neg"
        root.mkdir()
        (root / "Anno").mkdir()
        (root / "Anno" / "list_bbox.txt").write_text(
            "1\nimage_name x_1 y_1 x_2 y_2\n"
            "img/x.jpg -10 -5 200 250\n"
        )
        # Create other required minimal stubs
        for k in ["category_cloth", "attr_cloth", "category_img", "attr_img", "landmarks"]:
            (root / _ANNO_FILES_PATHS[k]).parent.mkdir(parents=True, exist_ok=True)
            (root / _ANNO_FILES_PATHS[k]).write_text("0\nheader\n")
        (root / "Eval").mkdir(exist_ok=True)
        (root / "Eval" / "list_eval_partition.txt").write_text(
            "0\nimage_name split\n"
        )
        parser = DeepFashionAnnotationParser(root)
        bbox = parser.bbox_map.get("img/x.jpg", [])
        if bbox:
            assert bbox[0] >= 0
            assert bbox[1] >= 0


# Helper for test_negative_bbox_clamped
_ANNO_FILES_PATHS = {
    "category_cloth" : "Anno/list_category_cloth.txt",
    "attr_cloth"     : "Anno/list_attr_cloth.txt",
    "category_img"   : "Anno/list_category_img.txt",
    "attr_img"       : "Anno/list_attr_img.txt",
    "landmarks"      : "Anno/list_landmarks.txt",
    "bbox"           : "Anno/list_bbox.txt",
}


# =============================================================================
# ── 4. TestDeepFashionExtractor
# =============================================================================

class TestDeepFashionExtractor:
    """Tests for DeepFashionExtractor record assembly."""

    def test_stream_yields_raw_records(self, tmp_path):
        """stream() should yield RawDeepFashionRecord objects."""
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)

        results = list(extractor.stream(split="train", max_records=2))
        assert len(results) == 2
        for raw, err in results:
            assert isinstance(raw, RawDeepFashionRecord)

    def test_stream_train_split_filtered(self, tmp_path):
        """Only train-split images should be yielded for split='train'."""
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)

        results = list(extractor.stream(split="train"))
        paths = [r.image_rel_path for r, _ in results]
        assert all(
            parser.split_map[p] == "train" for p in paths
        )

    def test_stream_val_split_filtered(self, tmp_path):
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)

        results = list(extractor.stream(split="val"))
        assert len(results) == 1
        assert results[0][0].split == "val"

    def test_stream_all_split(self, tmp_path):
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)
        results   = list(extractor.stream(split="all"))
        assert len(results) == 5  # all 5 entries in stub

    def test_stream_max_records_respected(self, tmp_path):
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)
        results   = list(extractor.stream(split="all", max_records=2))
        assert len(results) == 2

    def test_stream_raises_when_no_data(self, tmp_path):
        """stream() should raise RuntimeError if parser has no data."""
        empty_parser = DeepFashionAnnotationParser.__new__(DeepFashionAnnotationParser)
        empty_parser.root_dir      = tmp_path
        empty_parser.category_names = []
        empty_parser.attr_names     = []
        empty_parser.split_map      = {}   # empty → is_available() = False
        empty_parser.category_map   = {}
        empty_parser.attr_map       = {}
        empty_parser.bbox_map       = {}
        empty_parser.landmark_map   = {}

        extractor = DeepFashionExtractor(parser=empty_parser, root_dir=tmp_path)
        with pytest.raises(RuntimeError, match="no data"):
            list(extractor.stream())

    def test_missing_category_annotation_default(self, tmp_path):
        """Record with no category annotation should get category_raw='unknown'."""
        parser = _make_parser_stub(tmp_path)
        parser.split_map["img/Unknown/img_x.jpg"] = "train"
        # No category_map entry for this path

        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)
        results   = [r for r, _ in extractor.stream(split="train")]
        unknown_records = [r for r in results if r.image_rel_path == "img/Unknown/img_x.jpg"]
        assert len(unknown_records) == 1
        assert unknown_records[0].category_raw == "unknown"

    def test_missing_bbox_returns_empty_list(self, tmp_path):
        """Records without bbox annotation should have bbox=[]."""
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)

        # Blouse/img_00000002.jpg has no bbox in stub
        results = list(extractor.stream(split="val"))
        rec     = results[0][0]
        assert rec.bbox == []

    def test_missing_landmarks_returns_defaults(self, tmp_path):
        """Records without landmark annotation get all-invisible defaults."""
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)

        results = list(extractor.stream(split="test"))
        rec     = results[0][0]  # Dress — no landmarks in stub
        assert len(rec.landmarks_raw) == 6
        assert all(not lm["visible"] for lm in rec.landmarks_raw)

    def test_get_dataset_info_keys(self, tmp_path):
        """get_dataset_info() should return expected keys."""
        parser    = _make_parser_stub(tmp_path)
        extractor = DeepFashionExtractor(parser=parser, root_dir=tmp_path)
        info      = extractor.get_dataset_info()

        assert "total_images"    in info
        assert "available"       in info
        assert "img_dir_exists"  in info


# =============================================================================
# ── 5. TestDeepFashionTransformer
# =============================================================================

class TestDeepFashionTransformer:
    """Tests for DeepFashionTransformer — category, attributes, landmarks, bbox."""

    # ── Category normalisation ─────────────────────────────────────────────────

    def test_blouse_maps_to_shirts(self, transformer):
        assert transformer._normalise_category("Blouse") == "shirts"

    def test_jeans_maps_to_jeans(self, transformer):
        assert transformer._normalise_category("Jeans") == "jeans"

    def test_jacket_maps_to_jackets(self, transformer):
        assert transformer._normalise_category("Jacket") == "jackets"

    def test_dress_maps_to_dresses(self, transformer):
        assert transformer._normalise_category("Dress") == "dresses"

    def test_shorts_maps_to_shorts(self, transformer):
        assert transformer._normalise_category("Shorts") == "shorts"

    def test_hoodie_maps_to_hoodies(self, transformer):
        assert transformer._normalise_category("Hoodie") == "hoodies"

    def test_tee_maps_to_t_shirts(self, transformer):
        assert transformer._normalise_category("Tee") == "t_shirts"

    def test_tank_maps_to_t_shirts(self, transformer):
        assert transformer._normalise_category("Tank") == "t_shirts"

    def test_shawl_maps_to_accessories(self, transformer):
        assert transformer._normalise_category("Shawl") == "accessories"

    def test_unknown_maps_to_accessories(self, transformer):
        assert transformer._normalise_category("unknown") == "accessories"

    def test_empty_maps_to_accessories(self, transformer):
        assert transformer._normalise_category("") == "accessories"

    def test_case_insensitive_map(self, transformer):
        """Category lookup should be case-insensitive."""
        assert transformer._normalise_category("jeans") == "jeans"
        assert transformer._normalise_category("JEANS") == "jeans"

    def test_all_map_keys_produce_valid_category(self, transformer):
        """Every key in _DF_CATEGORY_MAP must produce a valid taxonomy key."""
        for raw_cat in _DF_CATEGORY_MAP:
            result = transformer._normalise_category(raw_cat)
            assert result in _VALID_CATEGORIES, \
                f"'{raw_cat}' mapped to invalid key: '{result}'"

    # ── Attribute decoding ────────────────────────────────────────────────────

    def test_decode_attrs_only_positive_one(self, transformer):
        """Only +1 values should appear in output."""
        names  = ["floral", "sleeveless", "cotton"]
        vector = [1, -1, 1]
        result = transformer._decode_attributes(vector, names)
        assert "floral"    in result
        assert "sleeveless" not in result
        assert "cotton"    in result

    def test_decode_attrs_zero_excluded(self, transformer):
        """Zero (missing) values should not appear in output."""
        names  = ["attr_a", "attr_b"]
        vector = [0, 0]
        result = transformer._decode_attributes(vector, names)
        assert result == []

    def test_decode_attrs_empty_vector(self, transformer):
        result = transformer._decode_attributes([], ["a", "b"])
        assert result == []

    def test_decode_attrs_no_duplicates(self, transformer):
        names  = ["x", "x", "y"]   # same name twice
        vector = [1, 1, 1]
        result = transformer._decode_attributes(vector, names)
        assert result.count("x") == 1  # deduplication

    def test_decode_attrs_capped_at_max(self):
        t   = DeepFashionTransformer(max_attrs=3, kb=None)
        n   = ["a", "b", "c", "d", "e"]
        vec = [1, 1, 1, 1, 1]
        result = t._decode_attributes(vec, n)
        assert len(result) <= 3

    def test_clean_attr_name_lowercases(self, transformer):
        assert transformer._clean_attr_name("Floral_Pattern") == "floral pattern"

    def test_clean_attr_name_underscores_to_spaces(self, transformer):
        assert transformer._clean_attr_name("floral_pattern") == "floral pattern"

    def test_clean_attr_name_empty(self, transformer):
        assert transformer._clean_attr_name("") == ""

    def test_clean_attr_name_collapses_spaces(self, transformer):
        assert transformer._clean_attr_name("a  b") == "a b"

    # ── Landmark normalisation ─────────────────────────────────────────────────

    def test_normalise_landmarks_with_bbox(self, transformer):
        """Landmarks should be normalised relative to bbox."""
        bbox = [50, 30, 206, 230]   # w=156, h=200
        lms  = [{"name": "left_collar", "x": 50, "y": 30, "visible": True}]
        result = transformer._normalise_landmarks(lms, bbox)
        assert result[0]["x"] == pytest.approx(0.0, abs=0.001)
        assert result[0]["y"] == pytest.approx(0.0, abs=0.001)

    def test_normalise_landmarks_top_right_corner(self, transformer):
        bbox = [0, 0, 256, 256]
        lms  = [{"name": "right_hem", "x": 256, "y": 256, "visible": True}]
        result = transformer._normalise_landmarks(lms, bbox)
        # Clamped to 1.0
        assert result[0]["x"] <= 1.0
        assert result[0]["y"] <= 1.0

    def test_normalise_landmarks_clamped_above_bbox(self, transformer):
        """Coordinates outside bbox should be clamped to [0,1]."""
        bbox = [50, 50, 150, 150]
        lms  = [{"name": "left_hem", "x": 300, "y": 300, "visible": True}]
        result = transformer._normalise_landmarks(lms, bbox)
        assert result[0]["x"] == pytest.approx(1.0)
        assert result[0]["y"] == pytest.approx(1.0)

    def test_normalise_landmarks_invisible_gives_zero(self, transformer):
        """Invisible landmarks must get x=0.0, y=0.0."""
        lms = [{"name": "left_collar", "x": 100, "y": 100, "visible": False}]
        result = transformer._normalise_landmarks(lms, [])
        assert result[0]["x"]   == 0.0
        assert result[0]["y"]   == 0.0
        assert result[0]["visible"] is False

    def test_normalise_landmarks_no_bbox_fallback(self, transformer):
        """Without bbox, use 256×256 reference size."""
        lms = [{"name": "left_collar", "x": 128, "y": 128, "visible": True}]
        result = transformer._normalise_landmarks(lms, [])
        assert result[0]["x"] == pytest.approx(0.5, abs=0.01)
        assert result[0]["y"] == pytest.approx(0.5, abs=0.01)

    def test_normalise_landmarks_preserves_name(self, transformer):
        """Landmark name must be preserved after normalisation."""
        lms = [{"name": "left_collar", "x": 50, "y": 50, "visible": True}]
        result = transformer._normalise_landmarks(lms, [0, 0, 256, 256])
        assert result[0]["name"] == "left_collar"

    # ── Bounding box normalisation ─────────────────────────────────────────────

    def test_normalise_bbox_basic(self, transformer):
        result = transformer._normalise_bbox([0, 0, 256, 256])
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_normalise_bbox_half(self, transformer):
        result = transformer._normalise_bbox([0, 0, 128, 128])
        assert result[2] == pytest.approx(0.5, abs=0.001)
        assert result[3] == pytest.approx(0.5, abs=0.001)

    def test_normalise_bbox_empty_list(self, transformer):
        assert transformer._normalise_bbox([]) == []

    def test_normalise_bbox_malformed(self, transformer):
        assert transformer._normalise_bbox([1, 2, 3]) == []

    def test_normalise_bbox_clamped_at_one(self, transformer):
        result = transformer._normalise_bbox([0, 0, 512, 512])
        assert all(v <= 1.0 for v in result)

    # ── Image ID generation ───────────────────────────────────────────────────

    def test_build_image_id_prefix(self, transformer):
        image_id = transformer._build_image_id("img/Blouse/img_00000001.jpg")
        assert image_id.startswith("DF_")

    def test_build_image_id_no_extension(self, transformer):
        image_id = transformer._build_image_id("img/Blouse/img_00000001.jpg")
        assert ".jpg" not in image_id

    def test_build_image_id_no_backslashes(self, transformer):
        image_id = transformer._build_image_id("img\\Blouse\\img_00000001.jpg")
        assert "\\" not in image_id

    def test_build_image_id_consistent(self, transformer):
        """Same input must always produce same output."""
        p = "img/Tee/img_00000042.jpg"
        assert transformer._build_image_id(p) == transformer._build_image_id(p)

    # ── Full transform ─────────────────────────────────────────────────────────

    def test_full_transform_returns_record(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert isinstance(result, DeepFashionRecord)

    def test_full_transform_category_correct(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert result.category == "shirts"  # Blouse → shirts

    def test_full_transform_attributes_non_empty(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert len(result.attributes) > 0

    def test_full_transform_landmarks_count(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert len(result.landmarks) == 6

    def test_full_transform_landmarks_normalised(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        for lm in result.landmarks:
            if lm["visible"]:
                assert 0.0 <= lm["x"] <= 1.0
                assert 0.0 <= lm["y"] <= 1.0

    def test_full_transform_image_path_posix(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert "\\" not in result.image_path

    def test_full_transform_dataset_source(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert result.dataset_source == "deepfashion"

    def test_full_transform_split_preserved(self, transformer, raw_record):
        raw_record.split = "val"
        result = transformer.transform(raw_record)
        assert result.split == "val"

    def test_full_transform_bbox_normalised_length(self, transformer, raw_record):
        result = transformer.transform(raw_record)
        assert len(result.bbox_normalised) == 4
        assert all(0.0 <= v <= 1.0 for v in result.bbox_normalised)


# =============================================================================
# ── 6. TestDeepFashionValidator
# =============================================================================

class TestDeepFashionValidator:
    """Tests for all 7 validation layers."""

    def test_valid_record_passes(self, validator, processed_record):
        result = validator.validate(processed_record)
        assert result.is_valid, f"Expected valid. Errors: {result.errors}"
        assert result.errors == []

    def test_missing_image_id_fails(self, validator, processed_record):
        processed_record.image_id = ""
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("image_id" in e for e in result.errors)

    def test_missing_category_fails(self, validator, processed_record):
        processed_record.category = ""
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_missing_image_path_fails(self, validator, processed_record):
        processed_record.image_path = ""
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_invalid_category_fails(self, validator, processed_record):
        processed_record.category = "spacesuit"
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("category" in e.lower() for e in result.errors)

    def test_all_valid_categories_pass(self, validator, processed_record):
        for cat in _VALID_CATEGORIES:
            processed_record.category = cat
            result = validator.validate(processed_record)
            cat_errors = [e for e in result.errors if "category" in e.lower()]
            assert cat_errors == [], f"Category '{cat}' unexpectedly failed: {cat_errors}"

    def test_attributes_not_list_fails(self, validator, processed_record):
        processed_record.attributes = "floral"  # string instead of list
        result = validator.validate(processed_record)
        assert not result.is_valid
        assert any("attributes" in e for e in result.errors)

    def test_landmarks_wrong_count_is_warning(self, validator, processed_record):
        """Wrong number of landmarks generates a warning, not an error."""
        processed_record.landmarks = processed_record.landmarks[:4]  # only 4
        result = validator.validate(processed_record)
        assert any("landmark" in w.lower() or "6" in w for w in result.warnings)

    def test_landmark_missing_key_fails(self, validator, processed_record):
        bad_lm = processed_record.landmarks.copy()
        bad_lm[0] = {"name": "left_collar", "x": 0.3}  # missing y, visible
        processed_record.landmarks = bad_lm
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_landmark_not_dict_fails(self, validator, processed_record):
        processed_record.landmarks[0] = "not_a_dict"
        result = validator.validate(processed_record)
        assert not result.is_valid

    def test_landmark_out_of_range_is_warning(self, validator, processed_record):
        processed_record.landmarks[0]["x"] = 1.5   # > 1.0
        processed_record.landmarks[0]["visible"] = True
        result = validator.validate(processed_record)
        assert result.is_valid   # still valid (just a warning)
        assert any("out of" in w for w in result.warnings)

    def test_invalid_split_is_warning(self, validator, processed_record):
        processed_record.split = "extratest"
        result = validator.validate(processed_record)
        assert result.is_valid
        assert any("split" in w.lower() for w in result.warnings)

    def test_multiple_errors_accumulate(self, validator, processed_record):
        processed_record.image_id  = ""
        processed_record.category  = "galaxy"
        processed_record.image_path = ""
        result = validator.validate(processed_record)
        assert len(result.errors) >= 3


# =============================================================================
# ── 7. TestDeepFashionWriter
# =============================================================================

class TestDeepFashionWriter:
    """Tests for DeepFashionWriter JSON output structure and spec fields."""

    def test_save_records_creates_file(self, tmp_path, processed_record):
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], DFPipelineStats())
        assert path.exists()
        assert path.name == _OUTPUT_FILENAME

    def test_output_json_has_meta_and_records(self, tmp_path, processed_record):
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], DFPipelineStats())

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "_meta"   in data
        assert "records" in data

    def test_spec_fields_in_each_record(self, tmp_path, processed_record):
        """Every record in JSON must contain the 5 required spec fields."""
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_records([processed_record], DFPipelineStats())

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        spec_fields = ["image_id", "category", "attributes", "landmarks", "image_path"]
        for rec_dict in data["records"]:
            for f in spec_fields:
                assert f in rec_dict, f"Spec field '{f}' missing from JSON record"

    def test_meta_total_records_matches(self, tmp_path, processed_record):
        writer  = DeepFashionWriter(output_dir=tmp_path)
        records = [processed_record, processed_record]
        path    = writer.save_records(records, DFPipelineStats())

        with open(path, "r") as f:
            data = json.load(f)

        assert data["_meta"]["total_records"] == 2
        assert len(data["records"]) == 2

    def test_meta_has_stats(self, tmp_path, processed_record):
        writer = DeepFashionWriter(output_dir=tmp_path)
        stats  = DFPipelineStats()
        stats.total_processed = 1
        path   = writer.save_records([processed_record], stats)

        with open(path, "r") as f:
            data = json.load(f)

        assert "stats" in data["_meta"]
        assert data["_meta"]["stats"]["total_processed"] == 1

    def test_save_run_report_creates_file(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_run_report(DFPipelineStats(), tmp_path)
        assert path.exists()
        assert path.name == "deepfashion_run_report.json"

    def test_run_report_json_valid(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_run_report(DFPipelineStats(), tmp_path)

        with open(path, "r") as f:
            data = json.load(f)

        assert "run_info"       in data
        assert "pipeline_stats" in data

    def test_empty_records_list(self, tmp_path):
        writer = DeepFashionWriter(output_dir=tmp_path)
        path   = writer.save_records([], DFPipelineStats())

        with open(path, "r") as f:
            data = json.load(f)

        assert data["_meta"]["total_records"] == 0
        assert data["records"] == []


# =============================================================================
# ── 8. TestDeepFashionLoader (Integration — fully mocked)
# =============================================================================

class TestDeepFashionLoader:
    """Integration tests for DeepFashionLoader orchestrator."""

    def test_init_without_dataset(self, tmp_path):
        """Loader should initialise even when dataset root doesn't exist."""
        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )
        assert loader is not None

    def test_get_dataset_info_missing(self, tmp_path):
        """get_dataset_info() should return available=False for missing root."""
        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )
        info = loader.get_dataset_info()
        assert info.get("available") is False

    def test_run_raises_on_invalid_split(self, tmp_path):
        """run() should raise ValueError for unrecognised split."""
        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )
        with pytest.raises(ValueError, match="Invalid split"):
            loader.run(split="purple")

    def test_run_with_mock_extractor(self, tmp_path, raw_record):
        """Full pipeline run with mock extractor returning 5 records."""
        # Build 5 synthetic raw records
        raw_records = [
            (
                RawDeepFashionRecord(
                    image_rel_path  = f"img/Blouse/img_{i:08d}.jpg",
                    category_raw    = "Blouse",
                    category_id     = 1,
                    attr_vector     = [1, -1, 1, -1, 0],
                    attr_names      = ["floral", "solid", "cotton", "loose", "sleeveless"],
                    landmarks_raw   = [
                        {"name": n, "x": 100, "y": 100, "visible": True}
                        for n in _LANDMARK_NAMES
                    ],
                    bbox            = [50, 30, 206, 230],
                    split           = "train",
                ),
                None,
            )
            for i in range(5)
        ]

        # Patch the extractor on the loader instance
        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )
        mock_extractor = MagicMock()
        mock_extractor.stream.return_value = iter(raw_records)
        mock_extractor.get_dataset_info.return_value = {
            "available": True, "total_images": 5
        }
        loader.extractor = mock_extractor

        # Also mock parser's get_image_paths_for_split (used for tqdm total)
        loader.parser = MagicMock()
        loader.parser.get_image_paths_for_split.return_value = ["x"] * 5

        result = loader.run(split="train", show_progress=False)

        assert "output_path"   in result
        assert "stats"         in result
        assert result["stats"]["total_processed"] == 5
        assert result["total_records"] == 5
        assert Path(result["output_path"]).exists()

    def test_run_skips_transform_failures(self, tmp_path):
        """Pipeline should skip records on catastrophic transform errors."""
        raw_records = [
            (
                RawDeepFashionRecord(
                    image_rel_path  = "img/Blouse/img_00000001.jpg",
                    category_raw    = "Blouse",
                    category_id     = 1,
                    attr_vector     = [1],
                    attr_names      = ["floral"],
                    landmarks_raw   = [{"name": n, "x": 10, "y": 10, "visible": True}
                                       for n in _LANDMARK_NAMES],
                    bbox            = [0, 0, 256, 256],
                    split           = "train",
                ),
                None,
            ),
        ]

        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )

        # Make transformer raise an exception
        loader.transformer = MagicMock()
        loader.transformer.transform.side_effect = RuntimeError("mock transform error")

        mock_extractor = MagicMock()
        mock_extractor.stream.return_value = iter(raw_records)
        loader.extractor = mock_extractor
        loader.parser    = MagicMock()
        loader.parser.get_image_paths_for_split.return_value = ["x"] * 1

        result = loader.run(split="train", show_progress=False)
        assert result["stats"]["total_skipped"] == 1
        assert result["total_records"]          == 0

    def test_run_output_json_spec_fields(self, tmp_path, raw_record):
        """Output JSON must contain all 5 spec fields per record."""
        loader = DeepFashionLoader(
            root_dir   = tmp_path / "missing",
            output_dir = tmp_path,
            use_kb     = False,
        )
        mock_extractor = MagicMock()
        mock_extractor.stream.return_value = iter([(raw_record, None)])
        loader.extractor = mock_extractor
        loader.parser    = MagicMock()
        loader.parser.get_image_paths_for_split.return_value = ["x"]

        result = loader.run(split="train", show_progress=False)
        output_path = Path(result["output_path"])

        with open(output_path, "r") as f:
            data = json.load(f)

        spec_fields = ["image_id", "category", "attributes", "landmarks", "image_path"]
        for rec in data["records"]:
            for f in spec_fields:
                assert f in rec, f"Spec field '{f}' missing from output record"


# =============================================================================
# ── 9. TestCategoryMapping
# =============================================================================

class TestCategoryMapping:
    """Tests that _DF_CATEGORY_MAP covers all required category strings."""

    REQUIRED_STRINGS = [
        "Blouse", "Button-Down", "Dress Shirt", "Shirt",  # shirts
        "Tank", "Tee", "Top",                             # t_shirts
        "Sweater", "Cardigan", "Sweatshirt", "Hoodie",    # hoodies
        "Jacket", "Blazer", "Bomber", "Parka",            # jackets
        "Jeans",                                          # jeans
        "Shorts", "Cutoffs",                              # shorts
        "Dress", "Skirt", "Romper",                       # dresses
        "Shawl",                                          # accessories
    ]

    def test_all_required_strings_mapped(self):
        t = DeepFashionTransformer(kb=None)
        for cat in self.REQUIRED_STRINGS:
            result = t._normalise_category(cat)
            assert result in _VALID_CATEGORIES, \
                f"'{cat}' → '{result}' is not a valid taxonomy key"

    def test_all_map_values_are_valid_taxonomy_keys(self):
        for raw, mapped in _DF_CATEGORY_MAP.items():
            assert mapped in _VALID_CATEGORIES, \
                f"_DF_CATEGORY_MAP['{raw}'] = '{mapped}' is not a valid taxonomy key"


# =============================================================================
# ── 10. TestNormalisationEdgeCases
# =============================================================================

class TestNormalisationEdgeCases:
    """Boundary and edge case tests."""

    @pytest.fixture
    def t(self):
        return DeepFashionTransformer(kb=None)

    def test_category_case_insensitive(self, t):
        assert t._normalise_category("Blouse") == "shirts"
        assert t._normalise_category("blouse") == "shirts"
        assert t._normalise_category("BLOUSE") == "shirts"

    def test_landmark_normalisation_exact_bbox_corner(self, t):
        """Point at bbox bottom-right corner should be exactly (1.0, 1.0)."""
        bbox = [0, 0, 100, 200]
        lms  = [{"name": "left_collar", "x": 100, "y": 200, "visible": True}]
        result = t._normalise_landmarks(lms, bbox)
        assert result[0]["x"] == pytest.approx(1.0)
        assert result[0]["y"] == pytest.approx(1.0)

    def test_bbox_normalised_values_in_range(self, t):
        for bbox in [[0,0,256,256], [10,20,200,240], [0,0,1,1]]:
            result = t._normalise_bbox(bbox)
            assert all(0.0 <= v <= 1.0 for v in result), \
                f"Out-of-range normalised bbox for {bbox}: {result}"

    def test_decode_attrs_all_negative(self, t):
        result = t._decode_attributes([-1, -1, -1], ["a", "b", "c"])
        assert result == []

    def test_decode_attrs_all_positive(self, t):
        result = t._decode_attributes([1, 1, 1], ["a", "b", "c"])
        assert set(result) == {"a", "b", "c"}

    def test_image_id_no_special_chars(self, t):
        image_id = t._build_image_id("img/Category Name/img_00001.jpg")
        assert all(c.isalnum() or c == "_" for c in image_id)

    def test_image_path_forward_slash_only(self, t):
        path = t._build_image_path("img/Blouse/img_00000001.jpg")
        assert "\\" not in path

    def test_transform_empty_attr_vector(self, t):
        """Empty attr_vector should produce attributes=[]."""
        raw = RawDeepFashionRecord(
            image_rel_path  = "img/Blouse/img_00000001.jpg",
            category_raw    = "Blouse",
            category_id     = 1,
            attr_vector     = [],
            attr_names      = [],
            landmarks_raw   = [{"name": n, "x": 0, "y": 0, "visible": False}
                                for n in _LANDMARK_NAMES],
            bbox            = [],
            split           = "train",
        )
        result = t.transform(raw)
        assert result.attributes == []

    def test_transform_empty_bbox_gives_empty_normalised(self, t):
        raw = RawDeepFashionRecord(
            image_rel_path  = "img/Jeans/img_00000001.jpg",
            category_raw    = "Jeans",
            category_id     = 4,
            attr_vector     = [],
            attr_names      = [],
            landmarks_raw   = [{"name": n, "x": 0, "y": 0, "visible": False}
                                for n in _LANDMARK_NAMES],
            bbox            = [],
            split           = "train",
        )
        result = t.transform(raw)
        assert result.bbox_normalised == []

    def test_dfpipeline_stats_to_dict_json_safe(self):
        s = DFPipelineStats()
        s.total_processed = 50; s.total_valid = 48
        s.increment_category("shirts")
        json.dumps(s.to_dict())   # must not raise

    def test_record_attr_count_in_dict(self, processed_record):
        """attr_count should appear in to_dict() output."""
        d = processed_record.to_dict()
        assert "attr_count" in d
        assert d["attr_count"] == processed_record.attr_count

    def test_split_counts_in_stats(self):
        s = DFPipelineStats()
        for _ in range(3): s.increment_split("train")
        for _ in range(2): s.increment_split("val")
        d = s.to_dict()
        assert d["split_counts"]["train"] == 3
        assert d["split_counts"]["val"]   == 2
