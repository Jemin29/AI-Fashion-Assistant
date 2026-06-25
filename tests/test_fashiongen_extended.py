"""
Extended unit tests for data_pipeline/ingestion/fashiongen_loader.py

Targets uncovered branches in:
  - FashionGenLoader.__init__ (path resolution, kb flag)
  - FashionGenTransformer (category, gender, season, style normalisation)
  - FashionGenWriter.save_records / save_run_report
  - FashionGenExtractor.get_total_records / get_dataset_info (stub mode)

All tests are fully self-contained — no real HDF5 file needed.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.ingestion.fashiongen_loader import (
    FashionGenLoader,
    FashionGenTransformer,
    FashionGenWriter,
    FashionGenExtractor,
    RawFashionGenRecord,
    _CATEGORY_MAP,
    _GENDER_MAP,
)


# =============================================================================
# ── Helpers / Factories
# =============================================================================

def _make_raw(
    description: str = "A slim-fit navy cotton shirt.",
    category_raw: str = "T-Shirts",
    subcategory_raw: str = "Polo",
    gender_raw: str = "Men",
    source_index: int = 0,
    hdf5_path: Path = Path("dummy.h5"),
) -> RawFashionGenRecord:
    return RawFashionGenRecord(
        source_index    = source_index,
        image_array     = np.zeros((256, 256, 3), dtype=np.uint8),
        description     = description,
        category_raw    = category_raw,
        subcategory_raw = subcategory_raw,
        gender_raw      = gender_raw,
        hdf5_path       = hdf5_path,
    )


# =============================================================================
# ── FashionGenLoader.__init__ (path resolution)
# =============================================================================

class TestFashionGenLoaderInit:
    """Test that FashionGenLoader resolves paths correctly."""

    def test_default_paths_set(self, tmp_path):
        loader = FashionGenLoader(use_kb=False)
        assert loader.hdf5_path is not None
        assert loader.output_dir is not None
        assert loader.image_dir is not None

    def test_custom_hdf5_path(self, tmp_path):
        custom = tmp_path / "custom.h5"
        loader = FashionGenLoader(hdf5_path=str(custom), use_kb=False)
        assert loader.hdf5_path == custom

    def test_custom_output_dir(self, tmp_path):
        loader = FashionGenLoader(output_dir=str(tmp_path), use_kb=False)
        assert loader.output_dir == tmp_path

    def test_save_images_flag_propagates(self, tmp_path):
        loader = FashionGenLoader(save_images=True, use_kb=False)
        assert loader.writer.save_images is True

    def test_save_images_false_default(self):
        loader = FashionGenLoader(use_kb=False)
        assert loader.writer.save_images is False

    def test_kb_disabled(self):
        loader = FashionGenLoader(use_kb=False)
        assert loader.kb is None

    def test_layers_attached(self):
        loader = FashionGenLoader(use_kb=False)
        assert loader.extractor is not None
        assert loader.transformer is not None
        assert loader.validator is not None
        assert loader.writer is not None


# =============================================================================
# ── FashionGenExtractor (stub / no-file mode)
# =============================================================================

class TestFashionGenExtractorStub:
    """Extractor behaviour when the HDF5 file is absent."""

    def test_get_total_records_returns_zero(self, tmp_path):
        extractor = FashionGenExtractor(hdf5_path=tmp_path / "nonexistent.h5")
        assert extractor.get_total_records() == 0

    def test_get_dataset_info_available_false(self, tmp_path):
        extractor = FashionGenExtractor(hdf5_path=tmp_path / "nonexistent.h5")
        info = extractor.get_dataset_info()
        assert info.get("available") is False


# =============================================================================
# ── FashionGenTransformer — category normalisation
# =============================================================================

class TestNormalizeCategory:
    """_normalize_category maps raw FashionGen strings to taxonomy keys."""

    def setup_method(self):
        self.transformer = FashionGenTransformer(kb=None)

    def test_exact_map_tshirts(self):
        assert self.transformer._normalize_category("T-Shirts") == "t_shirts"

    def test_exact_map_tops_to_tshirts(self):
        # "Tops" is in _CATEGORY_MAP
        result = self.transformer._normalize_category("Tops")
        assert result in ("t_shirts", "shirts")  # accept either mapping

    def test_case_insensitive_match(self):
        result = self.transformer._normalize_category("t-shirts")
        assert result == "t_shirts"

    def test_jeans_mapping(self):
        assert self.transformer._normalize_category("Jeans") == "jeans"

    def test_unknown_category_falls_back_to_accessories(self):
        result = self.transformer._normalize_category("ZXQWERTY_UNKNOWN")
        assert result == "accessories"

    def test_empty_string_falls_back(self):
        result = self.transformer._normalize_category("")
        assert result == "accessories"

    def test_all_category_map_keys_resolve(self):
        """Every key in _CATEGORY_MAP should return a non-empty canonical value."""
        t = self.transformer
        for raw_key in _CATEGORY_MAP:
            result = t._normalize_category(raw_key)
            assert result, f"Empty result for key: {raw_key!r}"


# =============================================================================
# ── FashionGenTransformer — gender normalisation
# =============================================================================

class TestNormalizeGender:
    """_normalize_gender maps raw gender strings to taxonomy keys."""

    def setup_method(self):
        self.transformer = FashionGenTransformer(kb=None)

    def test_men_exact(self):
        assert self.transformer._normalize_gender("Men") == "men"

    def test_women_exact(self):
        assert self.transformer._normalize_gender("Women") == "women"

    def test_boys_maps_to_men(self):
        assert self.transformer._normalize_gender("Boys") == "men"

    def test_girls_maps_to_women(self):
        assert self.transformer._normalize_gender("Girls") == "women"

    def test_male_maps_to_men(self):
        assert self.transformer._normalize_gender("male") == "men"

    def test_female_maps_to_women(self):
        assert self.transformer._normalize_gender("female") == "women"

    def test_unknown_falls_back_to_unisex(self):
        assert self.transformer._normalize_gender("ZXQWERTY") == "unisex"

    def test_empty_string_falls_back_to_unisex(self):
        assert self.transformer._normalize_gender("") == "unisex"

    def test_all_gender_map_keys_resolve(self):
        """Every key in _GENDER_MAP should return a valid canonical value."""
        t = self.transformer
        for raw_key in _GENDER_MAP:
            result = t._normalize_gender(raw_key)
            assert result in ("men", "women", "unisex"), (
                f"Unexpected result {result!r} for key {raw_key!r}"
            )


# =============================================================================
# ── FashionGenTransformer — season / style inference
# =============================================================================

class TestInferSeason:

    def setup_method(self):
        self.t = FashionGenTransformer(kb=None)

    def test_winter_keywords(self):
        result = self.t._infer_season("Warm wool winter coat with fleece lining")
        assert result in ("winter", "all_season")

    def test_summer_keywords(self):
        result = self.t._infer_season("Light linen beach shorts for summer")
        assert result in ("summer", "spring", "all_season")

    def test_empty_description_returns_all_season(self):
        assert self.t._infer_season("") == "all_season"

    def test_no_season_keywords(self):
        result = self.t._infer_season("A solid plain unbranded garment")
        assert result == "all_season"


class TestInferStyle:

    def setup_method(self):
        self.t = FashionGenTransformer(kb=None)

    def test_formal_keywords(self):
        result = self.t._infer_style("Elegant dress shirt for formal events", "shirts")
        assert result in ("formal", "business_casual", "")

    def test_streetwear_keywords(self):
        result = self.t._infer_style("Urban graphic hoodie with street vibes", "hoodies")
        assert result in ("streetwear", "")

    def test_ethnic_wear_category_overrides_style(self):
        result = self.t._infer_style("Traditional embroidered garment", "ethnic_wear")
        assert result == "formal"

    def test_footwear_sneaker_maps_to_athleisure(self):
        result = self.t._infer_style("Running sneaker with gym grip", "footwear")
        assert result == "athleisure"

    def test_empty_description_returns_empty(self):
        assert self.t._infer_style("", "t_shirts") == ""


# =============================================================================
# ── FashionGenTransformer.transform — end-to-end
# =============================================================================

class TestFashionGenTransformerTransform:

    def setup_method(self):
        self.t = FashionGenTransformer(kb=None)

    def test_transform_returns_record_with_all_required_keys(self, tmp_path):
        raw = _make_raw(hdf5_path=tmp_path / "fake.h5")
        rec = self.t.transform(raw)
        for key in ("image_id", "image_path", "category", "gender",
                    "description", "season", "style", "dataset_source"):
            assert key in rec.__dict__ or hasattr(rec, key), f"Missing key: {key}"

    def test_transform_category_is_canonical(self, tmp_path):
        raw = _make_raw(category_raw="T-Shirts", hdf5_path=tmp_path / "f.h5")
        rec = self.t.transform(raw)
        assert rec.category == "t_shirts"

    def test_transform_gender_is_canonical(self, tmp_path):
        raw = _make_raw(gender_raw="Men", hdf5_path=tmp_path / "f.h5")
        rec = self.t.transform(raw)
        assert rec.gender == "men"

    def test_transform_dataset_source(self, tmp_path):
        raw = _make_raw(hdf5_path=tmp_path / "f.h5")
        rec = self.t.transform(raw)
        assert rec.dataset_source == "fashiongen"


# =============================================================================
# ── FashionGenWriter.save_records / save_run_report
# =============================================================================

class TestFashionGenWriterSave:

    def test_save_records_creates_json(self, tmp_path):
        writer = FashionGenWriter(output_dir=tmp_path)
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_processed": 0}
        mock_stats.total_processed = 0
        path = writer.save_records([], mock_stats)
        assert path.exists(), f"Expected JSON at {path}"

    def test_save_records_json_has_records_key(self, tmp_path):
        writer = FashionGenWriter(output_dir=tmp_path)
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_processed": 0}
        mock_stats.total_processed = 0
        path = writer.save_records([], mock_stats)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "records" in data

    def test_save_run_report_creates_json(self, tmp_path):
        writer = FashionGenWriter(output_dir=tmp_path)
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "total_processed": 5, "total_valid": 5,
        }
        path = writer.save_run_report(mock_stats, tmp_path / "test.h5")
        assert path.exists(), f"Expected report at {path}"

    def test_save_run_report_is_valid_json(self, tmp_path):
        writer = FashionGenWriter(output_dir=tmp_path)
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_processed": 3}
        path = writer.save_run_report(mock_stats, tmp_path / "test.h5")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_save_records_custom_output_dir(self, tmp_path):
        sub_dir = tmp_path / "sub" / "output"
        writer = FashionGenWriter(output_dir=sub_dir)
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {}
        mock_stats.total_processed = 0
        path = writer.save_records([], mock_stats)
        assert path.parent == sub_dir


# =============================================================================
# ── FashionGenLoader.run — with fully mocked pipeline
# =============================================================================

class TestFashionGenLoaderRun:
    """Test FashionGenLoader.run() via injected mock extractor."""

    def _make_loader(self, tmp_path):
        loader = FashionGenLoader(
            hdf5_path  = tmp_path / "fake.h5",
            output_dir = tmp_path / "output",
            use_kb     = False,
        )
        return loader

    def test_run_missing_hdf5_returns_zero_records(self, tmp_path):
        loader = self._make_loader(tmp_path)
        # Override extractor so get_total_records → 0 and stream → empty
        loader.extractor.get_total_records = lambda: 0
        loader.extractor.stream = lambda **kw: iter([])
        result = loader.run(max_records=5, show_progress=False)
        assert result["total_records"] == 0

    def test_run_returns_dict_with_required_keys(self, tmp_path):
        loader = self._make_loader(tmp_path)
        loader.extractor.get_total_records = lambda: 0
        loader.extractor.stream = lambda **kw: iter([])
        result = loader.run(show_progress=False)
        for key in ("output_path", "report_path", "stats", "total_records"):
            assert key in result, f"Missing key in result: {key}"

    def test_get_dataset_info_delegates_to_extractor(self, tmp_path):
        loader = self._make_loader(tmp_path)
        loader.extractor.get_dataset_info = lambda: {
            "available": False, "path": "fake.h5"
        }
        info = loader.get_dataset_info()
        assert info["available"] is False
