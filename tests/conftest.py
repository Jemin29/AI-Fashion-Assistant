"""
=============================================================================
tests/conftest.py — Shared Fixtures and Mock Datasets
=============================================================================
Central fixture hub for the complete Fashion AI testing infrastructure.

Provides:
  - Shared synthetic record factories (FashionGen, DeepFashion, unified)
  - Mock HDF5 dataset (in-memory, no files needed)
  - Shared tmp_path-based output dirs
  - Reusable config objects for each module
  - Mock file system helpers
=============================================================================
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

# ─── Ensure project root and scripts are on path ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# =============================================================================
# ── 1. Synthetic Record Factories
# =============================================================================

def make_fg_record(
    image_id: str = "FG_00001",
    category: str = "t_shirts",
    gender: str = "unisex",
    description: str = "A classic black graphic tee with bold print.",
    color: str = "Black",
    **kwargs,
) -> Dict[str, Any]:
    """Build a minimal FashionGen-style processed record."""
    base = {
        "image_id"      : image_id,
        "image_path"    : f"datasets/fashiongen/images/{image_id}.jpg",
        "source_dataset": "fashiongen",
        "category"      : category,
        "subcategory"   : None,
        "gender"        : gender,
        "description"   : description,
        "color"         : [color],
        "fabric"        : ["cotton"],
        "pattern"       : ["solid"],
        "fit"           : "regular_fit",
        "style"         : "streetwear",
        "season"        : "all_season",
        "occasion"      : ["casual"],
        "attributes"    : [],
        "landmarks"     : [],
    }
    base.update(kwargs)
    return base


def make_df_record(
    image_id: str = "DF_00001",
    category: str = "shirts",
    gender: str = "men",
    description: str = "A slim-fit formal white cotton shirt.",
    **kwargs,
) -> Dict[str, Any]:
    """Build a minimal DeepFashion-style processed record."""
    base = {
        "image_id"      : image_id,
        "image_path"    : f"datasets/deepfashion/img/{image_id}.jpg",
        "source_dataset": "deepfashion",
        "category"      : category,
        "subcategory"   : None,
        "gender"        : gender,
        "description"   : description,
        "color"         : ["White"],
        "fabric"        : [],
        "pattern"       : ["solid"],
        "fit"           : "slim_fit",
        "style"         : "formal",
        "season"        : "all_season",
        "occasion"      : ["formal"],
        "attributes"    : ["button-down", "collar"],
        "landmarks"     : [
            {"name": "left_collar", "x": 0.3, "y": 0.1, "visibility": 1},
            {"name": "right_collar", "x": 0.7, "y": 0.1, "visibility": 1},
        ],
    }
    base.update(kwargs)
    return base


def make_valid_record(n: int = 0, source: str = "fashiongen") -> Dict[str, Any]:
    """Build a single fully-valid unified fashion record for validation tests."""
    cats = [
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories",
    ]
    cat = cats[n % len(cats)]
    return {
        "image_id"      : f"REC_{n:05d}",
        "image_path"    : f"datasets/{source}/img/REC_{n:05d}.jpg",
        "source_dataset": source,
        "category"      : cat,
        "gender"        : "unisex",
        "description"   : f"A high-quality {cat} item with modern styling for everyday wear.",
        "color"         : ["Black"],
        "fabric"        : ["cotton"],
        "pattern"       : ["solid"],
        "fit"           : "regular_fit",
        "style"         : "streetwear",
        "season"        : "all_season",
        "occasion"      : ["casual"],
        "attributes"    : [],
        "landmarks"     : [],
    }


def make_record_batch(n: int = 10, source: str = "fashiongen") -> List[Dict[str, Any]]:
    """Build a batch of n valid records alternating sources."""
    sources = ["fashiongen", "deepfashion"]
    return [
        make_valid_record(i, source=sources[i % 2] if source == "mixed" else source)
        for i in range(n)
    ]


# =============================================================================
# ── 2. Mock HDF5 Dataset
# =============================================================================

class MockH5Dataset:
    """
    Minimal mock of an h5py Dataset that supports indexing and iteration.
    Used to simulate FashionGen HDF5 file without needing real h5py files.
    """

    def __init__(self, data: list) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._data[key]
        return self._data[key]


class MockH5File:
    """
    Minimal mock of an h5py File with the FashionGen dataset keys.
    Each key maps to a MockH5Dataset of the given size.
    """

    def __init__(self, n_records: int = 5) -> None:
        self._n = n_records
        self._datasets = {
            "input_image"       : MockH5Dataset([np.zeros((256, 256, 3), dtype=np.uint8)] * n_records),
            "input_description" : MockH5Dataset([
                [f"A stylish clothing item number {i}".encode("utf-8")]
                for i in range(n_records)
            ]),
            "input_category"    : MockH5Dataset([
                b"Tops/T-Shirts" for _ in range(n_records)
            ]),
            "input_gender"      : MockH5Dataset([
                b"men" for _ in range(n_records)
            ]),
            "input_name"        : MockH5Dataset([
                f"FashionItem{i:04d}".encode("utf-8") for i in range(n_records)
            ]),
        }

    def __getitem__(self, key: str):
        return self._datasets.get(key, MockH5Dataset([]))

    def __contains__(self, key: str) -> bool:
        return key in self._datasets

    def keys(self):
        return self._datasets.keys()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# =============================================================================
# ── 3. DeepFashion Mock File System
# =============================================================================

def make_deepfashion_fs(tmp_path: Path, n_records: int = 5) -> Path:
    """
    Create a minimal DeepFashion directory structure under tmp_path.

    Structure:
        tmp_path/
          img/
            MEN/
              Tops_Tees/
                id_00000001/
                  01_1_front.jpg   (zero-byte placeholder)
          Anno_fine/
            list_category_cloth.txt
            list_category_img.txt
            list_attr_cloth.txt
            list_attr_img.txt
            list_bbox.txt
            list_landmarks_align.txt

    Returns:
        tmp_path (the "root_dir" to pass to DeepFashionLoader)
    """
    img_dir = tmp_path / "img" / "MEN" / "Tops_Tees" / "id_00000001"
    img_dir.mkdir(parents=True, exist_ok=True)

    anno_dir = tmp_path / "Anno_fine"
    anno_dir.mkdir(parents=True, exist_ok=True)

    split_dir = tmp_path / "Eval" / "Anno"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Image placeholders
    img_paths = []
    for i in range(n_records):
        fname = f"{i+1:02d}_1_front.jpg"
        p = img_dir / fname
        p.write_bytes(b"\x00" * 10)
        img_paths.append(f"img/MEN/Tops_Tees/id_00000001/{fname}")

    # list_category_cloth.txt
    (anno_dir / "list_category_cloth.txt").write_text(
        "3\ncategory_name category_type\n"
        "Tees 1\nBlouses 1\nJackets 3\n",
        encoding="utf-8",
    )

    # list_category_img.txt
    lines = [str(n_records), "image_name category_label"]
    for i, p in enumerate(img_paths):
        lines.append(f"{p} {(i % 3) + 1}")
    (anno_dir / "list_category_img.txt").write_text("\n".join(lines), encoding="utf-8")

    # list_attr_cloth.txt
    (anno_dir / "list_attr_cloth.txt").write_text(
        "2\nattr_name attr_type\ngraphic 1\nsolid 1\n",
        encoding="utf-8",
    )

    # list_attr_img.txt
    lines = [str(n_records), "image_name " + " ".join(["attr"] * 2)]
    for p in img_paths:
        lines.append(f"{p} 1 0")
    (anno_dir / "list_attr_img.txt").write_text("\n".join(lines), encoding="utf-8")

    # list_bbox.txt
    lines = [str(n_records), "image_name x_1 y_1 x_2 y_2"]
    for p in img_paths:
        lines.append(f"{p} 10 20 100 200")
    (anno_dir / "list_bbox.txt").write_text("\n".join(lines), encoding="utf-8")

    # list_landmarks_align.txt
    lines = [str(n_records), "image_name clothes_type " + " ".join([f"lm_{j}" for j in range(18)])]
    for p in img_paths:
        lm_vals = " ".join(["0 50 50"] * 6)
        lines.append(f"{p} 1 {lm_vals}")
    (anno_dir / "list_landmarks_align.txt").write_text("\n".join(lines), encoding="utf-8")

    # train/val/test splits
    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"list_eval_partition_{split_name}.txt"
        lines = [str(n_records), "image_name eval_status"]
        for p in img_paths:
            lines.append(f"{p} {split_name}")
        split_file.write_text("\n".join(lines), encoding="utf-8")

    return tmp_path


# =============================================================================
# ── 4. pytest Fixtures
# =============================================================================

@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Isolated output directory for each test."""
    out = tmp_path / "processed"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def fg_records() -> List[Dict[str, Any]]:
    """A batch of 10 FashionGen-style records."""
    return [make_fg_record(image_id=f"FG_{i:05d}", category=["t_shirts", "shirts", "hoodies"][i % 3])
            for i in range(10)]


@pytest.fixture
def df_records() -> List[Dict[str, Any]]:
    """A batch of 10 DeepFashion-style records."""
    return [make_df_record(image_id=f"DF_{i:05d}", category=["shirts", "jackets", "pants"][i % 3])
            for i in range(10)]


@pytest.fixture
def valid_batch() -> List[Dict[str, Any]]:
    """20 fully-valid unified records."""
    return make_record_batch(20)


@pytest.fixture
def mock_h5_file() -> MockH5File:
    """A mock HDF5 file with 5 FashionGen records."""
    return MockH5File(n_records=5)


@pytest.fixture
def df_root(tmp_path: Path) -> Path:
    """A minimal DeepFashion file system."""
    return make_deepfashion_fs(tmp_path, n_records=5)


@pytest.fixture
def records_json(tmp_path: Path, valid_batch) -> Path:
    """Write valid_batch to a JSON file and return its path."""
    p = tmp_path / "records.json"
    p.write_text(
        json.dumps({"records": valid_batch}, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def mock_config():
    """Return a standard Week 2 Config model instance for testing."""
    from src.utils.config_manager import get_config
    return get_config()


@pytest.fixture
def tiny_image():
    """Return a tiny 8x8 RGB PIL Image for unit tests."""
    try:
        from PIL import Image
        return Image.new("RGB", (8, 8), color=(255, 0, 0))
    except ImportError:
        return None


@pytest.fixture(autouse=True)
def mock_clip_loading():
    """Mock CLIP model and processor loading globally during tests to avoid downloading or access violations on CPU."""
    from unittest.mock import patch
    try:
        with patch("transformers.CLIPModel.from_pretrained", side_effect=RuntimeError("Mocked CLIP Model Loading")), \
             patch("transformers.CLIPProcessor.from_pretrained", side_effect=RuntimeError("Mocked CLIP Processor Loading")):
            yield
    except ImportError:
        yield



