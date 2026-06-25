"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/fashiongen_ingester.py
=============================================================================
PURPOSE:
    Ingests the FashionGen dataset stored in HDF5 (.h5) format.

DATASET OVERVIEW:
    FashionGen contains 293,000 fashion images (256×256 px) paired with
    human-written natural-language descriptions. Data is stored in a single
    HDF5 file with named datasets (keys) for images, descriptions, categories,
    gender, and sub-categories.

    HDF5 keys:
        - input_image        : (N, 256, 256, 3) uint8 RGB images
        - input_description  : (N,)             byte-string descriptions
        - input_category     : (N,)             coarse category labels
        - input_subcategory  : (N,)             fine-grained labels
        - input_gender       : (N,)             men | women

USAGE:
    >>> from src.data.ingestion import FashionGenIngester
    >>> ingester = FashionGenIngester(hdf5_path="datasets/fashiongen/fashiongen_256_256_train.h5")
    >>> for record in ingester.stream(max_items=100):
    ...     print(record["image_id"], record["category"])

WEEK 1 SCOPE:
    - Stream raw records (image + metadata) to downstream preprocessor.
    - No augmentation or model inference here.
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Generator, Optional, Any

import numpy as np
from loguru import logger

# h5py is used to read the FashionGen HDF5 file efficiently.
# It supports lazy loading — images are only read when accessed.
try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "h5py is required for FashionGen ingestion. "
        "Install it with: pip install h5py"
    ) from exc


# ─── Type aliases ─────────────────────────────────────────────────────────────
FashionRecord = Dict[str, Any]  # One record: image array + metadata fields


class FashionGenIngester:
    """
    Streams records from a FashionGen HDF5 dataset file.

    Each yielded record is a dictionary containing:
        - image_id       : str  — zero-padded index (e.g., "FG_000042")
        - image_array    : np.ndarray  — shape (256, 256, 3), dtype uint8
        - description    : str  — human-written text description
        - category       : str  — coarse category label
        - subcategory    : str  — fine-grained sub-category
        - gender         : str  — "men" | "women" | "unknown"
        - split          : str  — "train" | "val" (inferred from file name)
        - dataset_source : str  — always "fashiongen"
    """

    # ── HDF5 key names in the FashionGen file ─────────────────────────────────
    _HDF5_IMAGE_KEY       = "input_image"
    _HDF5_DESC_KEY        = "input_description"
    _HDF5_CATEGORY_KEY    = "input_category"
    _HDF5_SUBCATEGORY_KEY = "input_subcategory"
    _HDF5_GENDER_KEY      = "input_gender"

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "train",
        id_prefix: str = "FG",
    ) -> None:
        """
        Initialise the ingester.

        Args:
            hdf5_path : Absolute or relative path to the .h5 file.
            split     : Dataset split label ("train" or "val").
            id_prefix : Prefix for generated image IDs (default "FG").
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.id_prefix = id_prefix

        # Validate that the file exists before any processing begins
        if not self.hdf5_path.exists():
            logger.warning(
                f"FashionGen HDF5 file not found at: {self.hdf5_path}\n"
                "Please download the dataset first using:\n"
                "  python data_pipeline/ingestion/download_fashiongen.py"
            )
        else:
            logger.info(f"FashionGenIngester initialised → {self.hdf5_path}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return total number of records in the HDF5 file."""
        if not self.hdf5_path.exists():
            return 0
        with h5py.File(self.hdf5_path, "r") as f:
            return len(f[self._HDF5_IMAGE_KEY])

    def stream(
        self,
        start: int = 0,
        max_items: Optional[int] = None,
    ) -> Generator[FashionRecord, None, None]:
        """
        Lazily yield FashionRecord dicts one at a time.

        The HDF5 file is kept open for the duration of iteration to avoid
        repeated open/close overhead. Records are yielded individually so
        the caller controls memory usage.

        Args:
            start     : Index of the first record to yield (0-based).
            max_items : Maximum records to yield. If None, yields all.

        Yields:
            FashionRecord dict (see class docstring for field descriptions).

        Raises:
            FileNotFoundError : If the HDF5 file does not exist.
            KeyError          : If an expected HDF5 key is missing.
        """
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"FashionGen HDF5 file not found: {self.hdf5_path}"
            )

        logger.info(
            f"Starting FashionGen stream | split={self.split} "
            f"| start={start} | max_items={max_items}"
        )

        records_yielded = 0

        # Open HDF5 in read-only mode; use a context manager for safe closing
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            # Validate required keys exist in file
            self._validate_hdf5_keys(hdf5_file)

            total = len(hdf5_file[self._HDF5_IMAGE_KEY])
            end = total if max_items is None else min(start + max_items, total)

            logger.debug(f"HDF5 total records: {total} | yielding [{start}:{end}]")

            for idx in range(start, end):
                try:
                    record = self._read_record(hdf5_file, idx)
                    yield record
                    records_yielded += 1
                except Exception as exc:
                    # Log corruption and continue — do not crash the pipeline
                    logger.error(f"Failed to read record index {idx}: {exc}")
                    continue

        logger.success(
            f"FashionGen stream complete | yielded {records_yielded} records"
        )

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics about the dataset without streaming all data.

        Returns:
            Dict with keys: total_records, image_shape, categories, genders.
        """
        if not self.hdf5_path.exists():
            return {"error": "HDF5 file not found", "path": str(self.hdf5_path)}

        with h5py.File(self.hdf5_path, "r") as f:
            self._validate_hdf5_keys(f)
            total = len(f[self._HDF5_IMAGE_KEY])
            img_shape = f[self._HDF5_IMAGE_KEY].shape  # (N, H, W, C)

            # Decode a sample of categories to report unique values
            sample_size = min(1000, total)
            categories = set(
                self._decode_bytes(f[self._HDF5_CATEGORY_KEY][i])
                for i in range(sample_size)
            )
            genders = set(
                self._decode_bytes(f[self._HDF5_GENDER_KEY][i])
                for i in range(sample_size)
            )

        stats = {
            "total_records": total,
            "image_shape": img_shape[1:],  # (H, W, C)
            "split": self.split,
            "sampled_categories": sorted(categories),
            "sampled_genders": sorted(genders),
            "file_size_mb": round(self.hdf5_path.stat().st_size / 1e6, 2),
        }
        logger.info(f"Dataset stats: {stats}")
        return stats

    # ── Private helpers ────────────────────────────────────────────────────────

    def _read_record(self, hdf5_file: h5py.File, idx: int) -> FashionRecord:
        """
        Read and decode a single record from the open HDF5 file.

        Args:
            hdf5_file : Open h5py.File object.
            idx       : Integer index of the record to read.

        Returns:
            FashionRecord dictionary with all decoded fields.
        """
        # Read the raw uint8 image array directly (no copy via np.array)
        image_array: np.ndarray = hdf5_file[self._HDF5_IMAGE_KEY][idx]

        # Decode byte-strings stored in HDF5 to Python str
        description  = self._decode_bytes(hdf5_file[self._HDF5_DESC_KEY][idx])
        category     = self._decode_bytes(hdf5_file[self._HDF5_CATEGORY_KEY][idx])
        subcategory  = self._decode_bytes(hdf5_file[self._HDF5_SUBCATEGORY_KEY][idx])
        gender       = self._decode_bytes(hdf5_file[self._HDF5_GENDER_KEY][idx])

        # Build a zero-padded, human-readable image ID
        image_id = f"{self.id_prefix}_{idx:07d}"

        return {
            "image_id"      : image_id,
            "image_array"   : image_array,   # np.ndarray (H, W, 3) uint8
            "description"   : description,
            "category"      : category,
            "subcategory"   : subcategory,
            "gender"        : gender or "unknown",
            "split"         : self.split,
            "dataset_source": "fashiongen",
            "source_index"  : idx,           # Original HDF5 row index
        }

    @staticmethod
    def _decode_bytes(value: Any) -> str:
        """
        Safely decode byte strings stored in HDF5 to Python str.

        HDF5 byte strings may be:
          - bytes           → decode as UTF-8
          - numpy bytes_    → decode as UTF-8
          - numpy ndarray   → take first element and decode
          - str             → return as-is (already decoded)

        Args:
            value : Raw value from an HDF5 dataset element.

        Returns:
            Decoded Python string, stripped of whitespace.
        """
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace").strip()
        if isinstance(value, np.ndarray):
            # Nested arrays (e.g., variable-length strings)
            return value.flat[0].decode("utf-8", errors="replace").strip()
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _validate_hdf5_keys(self, hdf5_file: h5py.File) -> None:
        """
        Assert that all required HDF5 keys exist in the file.

        Args:
            hdf5_file : Open h5py.File object.

        Raises:
            KeyError : If one or more required keys are missing.
        """
        required_keys = [
            self._HDF5_IMAGE_KEY,
            self._HDF5_DESC_KEY,
            self._HDF5_CATEGORY_KEY,
            self._HDF5_SUBCATEGORY_KEY,
            self._HDF5_GENDER_KEY,
        ]
        missing = [k for k in required_keys if k not in hdf5_file]
        if missing:
            raise KeyError(
                f"FashionGen HDF5 file is missing required keys: {missing}\n"
                f"Available keys: {list(hdf5_file.keys())}"
            )
