"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/fashiongen_loader.py
=============================================================================
MODULE: FashionGen Dataset Ingestion Pipeline
WEEK  : 1 — Fashion Domain Research & Dataset Curation
AUTHOR: Fashion AI Team

PURPOSE
-------
A production-grade, clean-architecture pipeline that:

  1. Discovers and validates the FashionGen HDF5 dataset file.
  2. Streams raw records (image bytes + metadata) from the HDF5 store.
  3. Extracts structured attributes via the FashionGen Extractor layer.
  4. Transforms extracted data into canonical FashionGenRecord objects
     aligned with the Fashion Knowledge Base taxonomy.
  5. Validates each record against taxonomy rules (knowledge_base module).
  6. Saves all processed records to processed/fashiongen_processed.json
     with full pipeline statistics.

ARCHITECTURE (Clean / Layered)
-------------------------------
  ┌──────────────────────────────────────────┐
  │  FashionGenLoader          (Orchestrator) │  ← Public API entry point
  ├──────────────────────────────────────────┤
  │  FashionGenExtractor       (Extract)      │  ← HDF5 → raw field dict
  ├──────────────────────────────────────────┤
  │  FashionGenTransformer     (Transform)    │  ← raw dict → canonical record
  ├──────────────────────────────────────────┤
  │  FashionGenValidator       (Validate)     │  ← taxonomy rule checks
  ├──────────────────────────────────────────┤
  │  FashionGenWriter          (Load / Save)  │  ← JSON output + report
  ├──────────────────────────────────────────┤
  │  PipelineStats             (Metrics)      │  ← live counters & timing
  └──────────────────────────────────────────┘

DESIGN PRINCIPLES
-----------------
  • Single Responsibility  — each layer class does exactly one thing.
  • Dependency Injection   — FashionGenLoader accepts all collaborators.
  • Fail-Safe Streams      — corrupted records are logged & skipped, never crash.
  • Progress Tracking      — tqdm progress bar with ETA and throughput.
  • Structured Logging     — loguru with per-record context.
  • Graceful Degradation   — the pipeline runs even without the HDF5 file
                              (mock/stub mode for testing).
  • Fully Type-Annotated   — mypy-clean.

OUTPUT FORMAT (per record)
--------------------------
  {
    "image_id"    : "FG_0000042",
    "image_path"  : "datasets/fashiongen/images/FG_0000042.jpg",
    "description" : "A slim-fit white cotton dress shirt ...",
    "category"    : "shirts",
    "gender"      : "men",
    "season"      : "all_season",
    "style"       : "formal",
    "subcategory" : "formal_shirt",
    "attributes"  : { "colors": [...], "fabrics": [...], ... },
    "is_valid"    : true,
    "errors"      : [],
    "warnings"    : [],
    "source_index": 42,
    "dataset_source": "fashiongen",
    "processed_at": "2026-06-02T14:00:00+00:00"
  }

USAGE
-----
  # Run as a CLI script:
  python data_pipeline/ingestion/fashiongen_loader.py
  python data_pipeline/ingestion/fashiongen_loader.py --max-records 500
  python data_pipeline/ingestion/fashiongen_loader.py --hdf5 path/to/file.h5

  # Use from Python:
  from src.data.ingestion.fashiongen_loader import FashionGenLoader
  loader = FashionGenLoader()
  results = loader.run(max_records=1000)
  print(results["stats"])

=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
from loguru import logger

# Optional tqdm (degrades gracefully to a plain iterator if not installed)
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    logger.warning("tqdm not installed. Install with: pip install tqdm")

# Optional h5py (required for HDF5 reading)
try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False
    logger.warning("h5py not installed. Install with: pip install h5py")

# ─── Internal ─────────────────────────────────────────────────────────────────
# Resolve project root regardless of working directory
_FILE_DIR    = Path(__file__).resolve().parent          # data_pipeline/ingestion/
_PROJECT_ROOT = _FILE_DIR.parent.parent.parent                 # fashion-ai-assistant/

# We add project root to sys.path so the package imports work when run directly
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Knowledge-base imports (may fail on first-time setup — we degrade gracefully)
try:
    from src.data.knowledge_base.fashion_domain_research import (
        FashionDomainResearch,
        normalize_color,
        normalize_fabric,
        normalize_style,
        normalize_season,
        normalize_occasion,
        normalize_fit,
        normalize_pattern,
    )
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    logger.warning(
        "fashion_domain_research not importable — taxonomy normalization disabled. "
        "Run from the project root: cd fashion-ai-assistant && python -m ..."
    )


# =============================================================================
# ── 1. Constants & Default Paths
# =============================================================================

# Default locations (overridable via constructor args)
_DEFAULT_HDF5_PATH  = _PROJECT_ROOT / "datasets" / "fashiongen" / "fashiongen_256_256_train.h5"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "processed"
_DEFAULT_IMAGE_DIR  = _PROJECT_ROOT / "datasets" / "fashiongen" / "images"
_OUTPUT_FILENAME    = "fashiongen_processed.json"

# FashionGen HDF5 dataset keys
_HDF5_KEYS = {
    "image"      : "input_image",
    "description": "input_description",
    "category"   : "input_category",
    "subcategory": "input_subcategory",
    "gender"     : "input_gender",
}

# FashionGen raw-category → taxonomy key mapping
# Source category strings observed in the FashionGen dataset
_CATEGORY_MAP: Dict[str, str] = {
    # T-shirts / tops
    "T-Shirts"          : "t_shirts",
    "T-shirts"          : "t_shirts",
    "Tshirts"           : "t_shirts",
    "Tops"              : "t_shirts",
    "Tank Tops"         : "t_shirts",
    "Polos"             : "t_shirts",
    # Shirts
    "Shirts"            : "shirts",
    "Dress Shirts"      : "shirts",
    "Casual Shirts"     : "shirts",
    "Button Shirts"     : "shirts",
    # Hoodies / sweatshirts
    "Hoodies"           : "hoodies",
    "Sweatshirts"       : "hoodies",
    "Sweaters"          : "hoodies",
    "Pullovers"         : "hoodies",
    # Jackets
    "Jackets"           : "jackets",
    "Coats"             : "jackets",
    "Blazers"           : "jackets",
    "Outerwear"         : "jackets",
    "Vests"             : "jackets",
    "Windbreakers"      : "jackets",
    # Pants / Trousers
    "Pants"             : "pants",
    "Trousers"          : "pants",
    "Slacks"            : "pants",
    "Chinos"            : "pants",
    "Leggings"          : "pants",
    "Joggers"           : "pants",
    # Jeans
    "Jeans"             : "jeans",
    "Denim"             : "jeans",
    "Denim Jeans"       : "jeans",
    # Shorts
    "Shorts"            : "shorts",
    "Bermudas"          : "shorts",
    # Dresses
    "Dresses"           : "dresses",
    "Maxi Dresses"      : "dresses",
    "Mini Dresses"      : "dresses",
    "Skirts"            : "dresses",  # closest category
    # Ethnic Wear
    "Kurtas"            : "ethnic_wear",
    "Salwar Suits"      : "ethnic_wear",
    "Sarees"            : "ethnic_wear",
    "Lehengas"          : "ethnic_wear",
    "Ethnic Sets"       : "ethnic_wear",
    # Footwear
    "Shoes"             : "footwear",
    "Sneakers"          : "footwear",
    "Boots"             : "footwear",
    "Sandals"           : "footwear",
    "Heels"             : "footwear",
    "Loafers"           : "footwear",
    "Flats"             : "footwear",
    "Oxfords"           : "footwear",
    # Accessories
    "Accessories"       : "accessories",
    "Bags"              : "accessories",
    "Watches"           : "accessories",
    "Belts"             : "accessories",
    "Sunglasses"        : "accessories",
    "Hats"              : "accessories",
    "Scarves"           : "accessories",
    "Jewelry"           : "accessories",
    "Wallets"           : "accessories",
}

# FashionGen raw-gender → taxonomy key mapping
_GENDER_MAP: Dict[str, str] = {
    "Men"   : "men",
    "Women" : "women",
    "Boys"  : "men",
    "Girls" : "women",
    "Unisex": "unisex",
    "men"   : "men",
    "women" : "women",
    "male"  : "men",
    "female": "women",
}

# Keyword-based heuristics for inferring season from description text
_SEASON_KEYWORDS: Dict[str, List[str]] = {
    "summer" : ["summer", "beach", "tropical", "lightweight", "linen",
                "sleeveless", "tank", "sundress", "breathable", "cool"],
    "winter" : ["winter", "wool", "fleece", "insulated", "warm", "puffer",
                "down", "heavyweight", "thermal", "cold"],
    "spring" : ["spring", "floral", "pastel", "light", "blossom", "refresh"],
    "autumn" : ["autumn", "fall", "earthy", "layering", "harvest", "rust",
                "mustard", "knit", "corduroy"],
}

# Keyword-based heuristics for inferring style from description text
_STYLE_KEYWORDS: Dict[str, List[str]] = {
    "formal"          : ["formal", "office", "business", "professional", "suit",
                         "dress shirt", "blazer", "tailored", "corporate"],
    "streetwear"      : ["streetwear", "street", "urban", "graphic", "hype",
                         "logo", "oversized", "skate", "hip-hop", "graffiti"],
    "athleisure"      : ["sport", "gym", "workout", "athletic", "performance",
                         "running", "training", "yoga", "active", "fitness"],
    "vintage"         : ["vintage", "retro", "old school", "classic",
                         "nostalgia", "throwback", "70s", "80s", "90s"],
    "luxury"          : ["luxury", "premium", "designer", "couture",
                         "high-end", "exclusive", "silk", "cashmere"],
    "minimalist"      : ["minimalist", "minimal", "clean", "simple",
                         "monochrome", "understated", "neutral"],
    "techwear"        : ["technical", "techwear", "functional", "modular",
                         "waterproof", "gore-tex", "utility", "tactical"],
    "business_casual" : ["smart casual", "office casual", "business casual",
                         "polo", "chinos", "loafers"],
}


# =============================================================================
# ── 2. Data Models
# =============================================================================

@dataclass
class RawFashionGenRecord:
    """
    Holds the raw, unprocessed data extracted directly from one HDF5 row.

    All fields are in their original form — byte strings not yet decoded,
    images as uint8 numpy arrays, indices as integers.

    This model belongs to the Extraction layer and is never persisted.
    """
    source_index  : int
    image_array   : np.ndarray    # shape (H, W, 3), dtype uint8 — may be None on error
    description   : str           # Raw description text
    category_raw  : str           # e.g. "T-Shirts", "Jackets"
    subcategory_raw: str          # e.g. "Graphic Tees"
    gender_raw    : str           # e.g. "Men", "Women"
    hdf5_path     : Path          # Source file path


@dataclass
class FashionGenRecord:
    """
    A fully processed, taxonomy-aligned fashion record ready for persistence.

    This is the canonical output format of the pipeline. All raw strings
    have been normalized to taxonomy keys, images have been saved to disk,
    and the record has been validated against the knowledge base.

    JSON-serialisable via to_dict().
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    image_id      : str            # e.g. "FG_0000042"
    image_path    : str            # Relative path to saved image file
    dataset_source: str = "fashiongen"

    # ── Core Fields (required by spec) ────────────────────────────────────────
    description   : str = ""      # Human-written text description
    category      : str = ""      # Canonical taxonomy key, e.g. "shirts"
    gender        : str = ""      # "men" | "women" | "unisex"
    season        : str = "all_season"   # Inferred from description
    style         : str = ""      # Inferred from description

    # ── Extended Fields ───────────────────────────────────────────────────────
    subcategory   : str = ""      # Normalized subcategory label
    attributes    : Dict[str, Any] = field(default_factory=dict)

    # ── Pipeline Provenance ───────────────────────────────────────────────────
    source_index  : int = -1      # Original HDF5 row index
    is_valid      : bool = False
    errors        : List[str] = field(default_factory=list)
    warnings      : List[str] = field(default_factory=list)
    processed_at  : str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain, JSON-safe dictionary.

        The returned dict matches the output format specification exactly.
        numpy arrays are excluded from output (images are saved as files).
        """
        return {
            "image_id"      : self.image_id,
            "image_path"    : self.image_path,
            "description"   : self.description,
            "category"      : self.category,
            "gender"        : self.gender,
            "season"        : self.season,
            "style"         : self.style,
            "subcategory"   : self.subcategory,
            "attributes"    : self.attributes,
            "is_valid"      : self.is_valid,
            "errors"        : self.errors,
            "warnings"      : self.warnings,
            "source_index"  : self.source_index,
            "dataset_source": self.dataset_source,
            "processed_at"  : self.processed_at,
        }


@dataclass
class PipelineStats:
    """
    Live counters and timing metrics for the ingestion pipeline run.

    Updated in-place as records flow through the pipeline. Thread-safe
    for single-thread use; add a Lock for multi-threaded scenarios.
    """
    # ── Counters ──────────────────────────────────────────────────────────────
    total_read        : int = 0    # Records read from HDF5
    total_processed   : int = 0    # Records successfully transformed
    total_valid       : int = 0    # Records that passed KB validation
    total_invalid     : int = 0    # Records with taxonomy errors
    total_warnings    : int = 0    # Records with taxonomy warnings
    total_skipped     : int = 0    # Records skipped due to extraction errors
    total_corrupted   : int = 0    # Records with corrupt image/data
    total_saved       : int = 0    # Records written to JSON output

    # ── Category breakdown ─────────────────────────────────────────────────────
    category_counts   : Dict[str, int] = field(default_factory=dict)
    gender_counts     : Dict[str, int] = field(default_factory=dict)
    style_counts      : Dict[str, int] = field(default_factory=dict)
    season_counts     : Dict[str, int] = field(default_factory=dict)

    # ── Timing ────────────────────────────────────────────────────────────────
    start_time        : float = field(default_factory=time.time)
    end_time          : float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed wall-clock time in seconds."""
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)

    @property
    def records_per_second(self) -> float:
        """Throughput in records per second."""
        elapsed = self.elapsed_seconds
        return round(self.total_processed / elapsed, 1) if elapsed > 0 else 0.0

    @property
    def valid_rate(self) -> float:
        """Fraction of processed records that are valid (0.0–1.0)."""
        return (
            round(self.total_valid / self.total_processed, 4)
            if self.total_processed > 0 else 0.0
        )

    def increment_category(self, category: str) -> None:
        """Increment the count for a category key."""
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def increment_gender(self, gender: str) -> None:
        self.gender_counts[gender] = self.gender_counts.get(gender, 0) + 1

    def increment_style(self, style: str) -> None:
        self.style_counts[style] = self.style_counts.get(style, 0) + 1

    def increment_season(self, season: str) -> None:
        self.season_counts[season] = self.season_counts.get(season, 0) + 1

    def finalize(self) -> None:
        """Stamp the end time to fix the elapsed seconds."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable stats summary."""
        return {
            "total_read"        : self.total_read,
            "total_processed"   : self.total_processed,
            "total_valid"       : self.total_valid,
            "total_invalid"     : self.total_invalid,
            "total_warnings"    : self.total_warnings,
            "total_skipped"     : self.total_skipped,
            "total_corrupted"   : self.total_corrupted,
            "total_saved"       : self.total_saved,
            "valid_rate"        : self.valid_rate,
            "records_per_second": self.records_per_second,
            "elapsed_seconds"   : self.elapsed_seconds,
            "category_counts"   : self.category_counts,
            "gender_counts"     : self.gender_counts,
            "style_counts"      : self.style_counts,
            "season_counts"     : self.season_counts,
        }

    def log_summary(self) -> None:
        """Log a formatted pipeline summary at INFO level."""
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total read        : {self.total_read:,}")
        logger.info(f"  Total processed   : {self.total_processed:,}")
        logger.info(f"  Total valid       : {self.total_valid:,}")
        logger.info(f"  Total invalid     : {self.total_invalid:,}")
        logger.info(f"  Total skipped     : {self.total_skipped:,}")
        logger.info(f"  Total corrupted   : {self.total_corrupted:,}")
        logger.info(f"  Total saved       : {self.total_saved:,}")
        logger.info(f"  Valid rate        : {self.valid_rate:.1%}")
        logger.info(f"  Throughput        : {self.records_per_second} rec/s")
        logger.info(f"  Elapsed           : {self.elapsed_seconds}s")
        logger.info(f"  Category breakdown: {self.category_counts}")
        logger.info(f"  Gender breakdown  : {self.gender_counts}")
        logger.info("=" * 60)


# =============================================================================
# ── 3. FashionGenExtractor — Extraction Layer
# =============================================================================

class FashionGenExtractor:
    """
    Extraction Layer: reads raw data from the FashionGen HDF5 file.

    Responsibilities:
      - Open and validate the HDF5 file structure.
      - Stream raw records one-by-one (constant memory).
      - Decode HDF5 byte strings to Python str.
      - Detect and flag corrupted records.
      - Report dataset-level statistics (total, shape, sample categories).

    This class does NOT normalize, transform, or validate data.
    It only reads and decodes from HDF5.
    """

    def __init__(self, hdf5_path: Union[str, Path]) -> None:
        """
        Initialise the extractor.

        Args:
            hdf5_path : Path to the FashionGen .h5 file.
        """
        self.hdf5_path = Path(hdf5_path)
        self._validate_file()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_total_records(self) -> int:
        """
        Return the total number of records in the HDF5 file without loading data.

        Returns:
            Integer count, or 0 if file is unavailable.
        """
        if not self.hdf5_path.exists() or not _H5PY_AVAILABLE:
            return 0
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                return int(len(f[_HDF5_KEYS["image"]]))
        except Exception as exc:
            logger.error(f"Failed to get total records: {exc}")
            return 0

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the HDF5 file without streaming all records.

        Returns:
            Dict with total_records, image_shape, file_size_mb, sample data.
        """
        if not self.hdf5_path.exists():
            return {"available": False, "path": str(self.hdf5_path)}

        try:
            with h5py.File(self.hdf5_path, "r") as f:
                self._validate_hdf5_keys(f)
                total      = int(len(f[_HDF5_KEYS["image"]]))
                img_shape  = tuple(f[_HDF5_KEYS["image"]].shape[1:])  # (H, W, C)
                sample_n   = min(500, total)

                categories = sorted({
                    self._decode(f[_HDF5_KEYS["category"]][i])
                    for i in range(sample_n)
                })
                genders = sorted({
                    self._decode(f[_HDF5_KEYS["gender"]][i])
                    for i in range(sample_n)
                })

            return {
                "available"     : True,
                "hdf5_path"     : str(self.hdf5_path),
                "total_records" : total,
                "image_shape"   : img_shape,
                "file_size_mb"  : round(self.hdf5_path.stat().st_size / 1e6, 2),
                "sample_categories": categories,
                "sample_genders"   : genders,
            }
        except Exception as exc:
            logger.error(f"Dataset info error: {exc}")
            return {"available": False, "error": str(exc)}

    def stream(
        self,
        start: int = 0,
        max_records: Optional[int] = None,
    ) -> Generator[Tuple[RawFashionGenRecord, Optional[str]], None, None]:
        """
        Lazily stream raw records from the HDF5 file.

        Keeps the HDF5 file open for the full iteration to avoid repeated
        open/close overhead. Each iteration reads exactly one HDF5 row.

        Args:
            start       : First row index (0-based, inclusive).
            max_records : Maximum rows to yield. None = all rows.

        Yields:
            Tuple of (RawFashionGenRecord, error_message).
            On success, error_message is None.
            On corrupted record, a partial/empty record is yielded
            alongside a descriptive error_message string.

        Raises:
            FileNotFoundError : If the HDF5 file is missing.
            RuntimeError      : If h5py is not installed.
        """
        if not _H5PY_AVAILABLE:
            raise RuntimeError(
                "h5py is required. Install: pip install h5py"
            )
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"FashionGen HDF5 not found: {self.hdf5_path}\n"
                f"Download from: https://fashion-gen.com/"
            )

        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            self._validate_hdf5_keys(hdf5_file)
            total = int(len(hdf5_file[_HDF5_KEYS["image"]]))
            end   = total if max_records is None else min(start + max_records, total)

            logger.info(
                f"FashionGenExtractor streaming | "
                f"file={self.hdf5_path.name} | rows=[{start}:{end}] "
                f"| total_in_file={total:,}"
            )

            for idx in range(start, end):
                raw, err = self._read_row(hdf5_file, idx)
                yield raw, err

    # ── Private ────────────────────────────────────────────────────────────────

    def _read_row(
        self,
        hdf5_file: "h5py.File",  # quoted to avoid forward-ref error if h5py missing
        idx: int,
    ) -> Tuple[RawFashionGenRecord, Optional[str]]:
        """
        Read and decode a single HDF5 row.

        Corruption handling:
          - If the image array is all-zeros or wrong shape → mark corrupted.
          - If any text field decodes to empty → fill with "" and warn.
          - If any unexpected exception occurs → return empty record + error.

        Args:
            hdf5_file : Open h5py.File object.
            idx       : Row index to read.

        Returns:
            Tuple[RawFashionGenRecord, Optional[str]]
        """
        try:
            # ── Read image ─────────────────────────────────────────────────────
            img_arr: np.ndarray = hdf5_file[_HDF5_KEYS["image"]][idx]

            # ── Validate image array ───────────────────────────────────────────
            corruption_error: Optional[str] = None
            if img_arr is None:
                corruption_error = f"Row {idx}: image array is None"
            elif not isinstance(img_arr, np.ndarray):
                corruption_error = f"Row {idx}: image is not ndarray (got {type(img_arr).__name__})"
            elif img_arr.ndim != 3 or img_arr.shape[2] != 3:
                corruption_error = f"Row {idx}: unexpected image shape {img_arr.shape}"
            elif img_arr.max() == 0:
                corruption_error = f"Row {idx}: image is all-zeros (likely corrupted)"

            # ── Read text fields ───────────────────────────────────────────────
            description   = self._decode(hdf5_file[_HDF5_KEYS["description"]][idx])
            category_raw  = self._decode(hdf5_file[_HDF5_KEYS["category"]][idx])
            subcategory_raw = self._decode(hdf5_file[_HDF5_KEYS["subcategory"]][idx])
            gender_raw    = self._decode(hdf5_file[_HDF5_KEYS["gender"]][idx])

            record = RawFashionGenRecord(
                source_index   = idx,
                image_array    = img_arr if corruption_error is None else np.zeros((256, 256, 3), dtype=np.uint8),
                description    = description,
                category_raw   = category_raw,
                subcategory_raw= subcategory_raw,
                gender_raw     = gender_raw,
                hdf5_path      = self.hdf5_path,
            )
            return record, corruption_error

        except Exception as exc:
            # Return a sentinel record so the pipeline can log and continue
            sentinel = RawFashionGenRecord(
                source_index   = idx,
                image_array    = np.zeros((256, 256, 3), dtype=np.uint8),
                description    = "",
                category_raw   = "",
                subcategory_raw= "",
                gender_raw     = "",
                hdf5_path      = self.hdf5_path,
            )
            return sentinel, f"Row {idx}: unexpected read error — {exc}"

    @staticmethod
    def _decode(value: Any) -> str:
        """
        Safely decode an HDF5 byte value to a clean Python str.

        Handles all common HDF5 string storage formats:
          bytes, numpy.bytes_, numpy.ndarray, str.

        Args:
            value : Raw HDF5 field value.

        Returns:
            Stripped UTF-8 string, or "" on decode failure.
        """
        try:
            if isinstance(value, (bytes, np.bytes_)):
                return value.decode("utf-8", errors="replace").strip()
            if isinstance(value, np.ndarray):
                element = value.flat[0]
                if isinstance(element, (bytes, np.bytes_)):
                    return element.decode("utf-8", errors="replace").strip()
                return str(element).strip()
            if isinstance(value, str):
                return value.strip()
            return str(value).strip()
        except Exception:
            return ""

    def _validate_file(self) -> None:
        """Warn (not raise) if the HDF5 file is missing — allows stub mode."""
        if not self.hdf5_path.exists():
            logger.warning(
                f"FashionGen HDF5 not found: {self.hdf5_path}\n"
                f"The pipeline will fail when .stream() is called.\n"
                f"Download dataset from: https://fashion-gen.com/"
            )

    @staticmethod
    def _validate_hdf5_keys(hdf5_file: "h5py.File") -> None:
        """
        Check that all required HDF5 dataset keys exist in the open file.

        Args:
            hdf5_file : Open h5py.File.

        Raises:
            KeyError : If any required key is absent.
        """
        required = list(_HDF5_KEYS.values())
        missing  = [k for k in required if k not in hdf5_file]
        if missing:
            raise KeyError(
                f"HDF5 file is missing required keys: {missing}\n"
                f"Available keys: {list(hdf5_file.keys())}"
            )


# =============================================================================
# ── 4. FashionGenTransformer — Transformation Layer
# =============================================================================

class FashionGenTransformer:
    """
    Transformation Layer: converts RawFashionGenRecord → FashionGenRecord.

    Responsibilities:
      - Map raw category/gender strings to taxonomy keys.
      - Infer season from description text via keyword heuristics.
      - Infer style from description text via keyword heuristics.
      - Build the attributes dict (colors, fabrics, patterns from description).
      - Generate the canonical image_id and image_path.
      - Use Knowledge Base normalizers where available.

    This class does NOT read from HDF5 and does NOT write to disk.
    """

    def __init__(
        self,
        id_prefix  : str = "FG",
        image_dir  : Union[str, Path] = _DEFAULT_IMAGE_DIR,
        kb         : Optional["FashionDomainResearch"] = None,
    ) -> None:
        """
        Initialise the transformer.

        Args:
            id_prefix : Prefix for generated image IDs (default "FG").
            image_dir : Directory where images will be saved.
            kb        : Optional knowledge-base instance for normalization.
        """
        self.id_prefix = id_prefix
        self.image_dir = Path(image_dir)
        self.kb        = kb  # May be None (KB unavailable)

    # ── Public API ─────────────────────────────────────────────────────────────

    def transform(self, raw: RawFashionGenRecord) -> FashionGenRecord:
        """
        Convert a RawFashionGenRecord to a fully structured FashionGenRecord.

        Transformation steps:
          1. Generate image_id and image_path.
          2. Normalize category using _CATEGORY_MAP + KB aliases.
          3. Normalize gender using _GENDER_MAP + KB aliases.
          4. Infer season from description text.
          5. Infer style from description text.
          6. Build attributes dict from description parsing.
          7. Normalize subcategory label.

        Args:
            raw : A RawFashionGenRecord from the extraction layer.

        Returns:
            A FashionGenRecord ready for validation and persistence.
        """
        # ── Step 1: Identity ───────────────────────────────────────────────────
        image_id   = f"{self.id_prefix}_{raw.source_index:07d}"
        image_path = self._build_image_path(image_id)

        # ── Step 2: Category normalization ─────────────────────────────────────
        category = self._normalize_category(raw.category_raw)

        # ── Step 3: Gender normalization ───────────────────────────────────────
        gender   = self._normalize_gender(raw.gender_raw)

        # ── Step 4: Season inference ───────────────────────────────────────────
        season   = self._infer_season(raw.description)

        # ── Step 5: Style inference ────────────────────────────────────────────
        style    = self._infer_style(raw.description, category)

        # ── Step 6: Build attributes from description ──────────────────────────
        attributes = self._extract_attributes(raw.description, category)

        # ── Step 7: Subcategory normalization ──────────────────────────────────
        subcategory = self._normalize_subcategory(raw.subcategory_raw)

        return FashionGenRecord(
            image_id       = image_id,
            image_path     = image_path,
            dataset_source = "fashiongen",
            description    = raw.description,
            category       = category,
            gender         = gender,
            season         = season,
            style          = style,
            subcategory    = subcategory,
            attributes     = attributes,
            source_index   = raw.source_index,
            is_valid       = False,   # Will be set by FashionGenValidator
        )

    # ── Private: Normalization helpers ─────────────────────────────────────────

    def _build_image_path(self, image_id: str) -> str:
        """
        Construct the relative path where the image file will be stored.

        Images are organised in sub-directories of 1000 images each to
        avoid filesystem inode limits on large datasets.
        E.g.: datasets/fashiongen/images/0000/FG_0000042.jpg

        Args:
            image_id : e.g. "FG_0000042"

        Returns:
            Relative path string from project root.
        """
        # Extract numeric part for sub-directory bucketing
        numeric = re.search(r"(\d+)$", image_id)
        if numeric:
            bucket = int(numeric.group(1)) // 1000 * 1000
            bucket_dir = f"{bucket:04d}"
        else:
            bucket_dir = "0000"
        return str(
            Path("datasets") / "fashiongen" / "images" / bucket_dir / f"{image_id}.jpg"
        ).replace("\\", "/")

    def _normalize_category(self, raw_category: str) -> str:
        """
        Map a raw FashionGen category string to a taxonomy key.

        Lookup order:
          1. Exact match in _CATEGORY_MAP.
          2. Case-insensitive exact match.
          3. Knowledge Base alias index lookup.
          4. Substring fuzzy match.
          5. Fallback: "accessories" (safest unknown default).

        Args:
            raw_category : e.g. "T-Shirts", "Jackets", "Denim Jeans"

        Returns:
            Taxonomy category key e.g. "t_shirts", "jackets", "jeans".
        """
        if not raw_category:
            return "accessories"

        # 1. Exact match
        if raw_category in _CATEGORY_MAP:
            return _CATEGORY_MAP[raw_category]

        # 2. Case-insensitive exact match
        for raw_key, mapped in _CATEGORY_MAP.items():
            if raw_key.lower() == raw_category.lower():
                return mapped

        # 3. KB alias lookup
        if self.kb is not None:
            alias_key = raw_category.lower().strip()
            cat_from_kb = self.kb._alias_to_category.get(alias_key)
            if cat_from_kb:
                return cat_from_kb

        # 4. Fuzzy substring match (handles "Graphic T-Shirts" → "t_shirts")
        raw_lower = raw_category.lower()
        for raw_key, mapped in _CATEGORY_MAP.items():
            if raw_key.lower() in raw_lower or raw_lower in raw_key.lower():
                return mapped

        logger.debug(f"Category not mapped: '{raw_category}' → fallback 'accessories'")
        return "accessories"

    def _normalize_gender(self, raw_gender: str) -> str:
        """
        Map a raw FashionGen gender string to a taxonomy key.

        Args:
            raw_gender : e.g. "Men", "Women", "Boys", "Girls"

        Returns:
            "men" | "women" | "unisex"
        """
        if not raw_gender:
            return "unisex"

        # Direct map lookup
        if raw_gender in _GENDER_MAP:
            return _GENDER_MAP[raw_gender]

        # Case-insensitive
        for raw_key, mapped in _GENDER_MAP.items():
            if raw_key.lower() == raw_gender.lower():
                return mapped

        # KB alias lookup
        if self.kb is not None:
            kb_gender = self.kb._alias_to_gender.get(raw_gender.lower())
            if kb_gender:
                return kb_gender

        logger.debug(f"Gender not mapped: '{raw_gender}' → fallback 'unisex'")
        return "unisex"

    def _infer_season(self, description: str) -> str:
        """
        Infer the most likely season from description text.

        Uses keyword frequency scoring. If multiple seasons score equally,
        returns the highest-scoring one. If no keywords match, returns
        "all_season" as a safe fallback.

        Args:
            description : Free-text product description.

        Returns:
            Season key: "spring" | "summer" | "autumn" | "winter" | "all_season"
        """
        if not description:
            return "all_season"

        desc_lower = description.lower()
        scores: Dict[str, int] = {}

        for season, keywords in _SEASON_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[season] = score

        if not scores:
            return "all_season"

        best_season = max(scores, key=lambda s: scores[s])

        # Use KB normalizer if available
        if self.kb is not None and _KB_AVAILABLE:
            kb_season = normalize_season(best_season, kb=self.kb)
            if kb_season:
                return kb_season

        return best_season

    def _infer_style(self, description: str, category: str) -> str:
        """
        Infer the most likely style from description text and category.

        Uses keyword frequency scoring with a category-based prior:
        - footwear → athleisure or formal (based on sub-keywords)
        - ethnic_wear → always "formal" (ethnic formal occasions)

        Args:
            description : Free-text product description.
            category    : Normalized taxonomy category key.

        Returns:
            Style key or empty string if no style can be inferred.
        """
        # Category-based strong priors
        if category == "ethnic_wear":
            return "formal"
        if category == "footwear":
            desc_lower = description.lower()
            if any(kw in desc_lower for kw in ["sneaker", "running", "sport", "gym"]):
                return "athleisure"
            if any(kw in desc_lower for kw in ["formal", "dress", "oxford", "loafer"]):
                return "formal"

        if not description:
            return ""

        desc_lower = description.lower()
        scores: Dict[str, int] = {}

        for style_key, keywords in _STYLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[style_key] = score

        if not scores:
            return ""

        best_style = max(scores, key=lambda s: scores[s])

        # KB normalizer validation
        if self.kb is not None and _KB_AVAILABLE:
            kb_style = normalize_style(best_style, kb=self.kb)
            return kb_style or best_style

        return best_style

    def _extract_attributes(
        self,
        description: str,
        category: str,
    ) -> Dict[str, Any]:
        """
        Extract a structured attributes dict from free-text description.

        Attributes extracted:
          - colors   : Canonical color names mentioned in description.
          - fabrics  : Canonical fabric names mentioned in description.
          - patterns : Canonical pattern keys mentioned in description.

        Uses the KB normalizers when available; falls back to simple
        case-insensitive substring matching against known terms.

        Args:
            description : Free-text product description.
            category    : Normalized category key (for filtering applicable attrs).

        Returns:
            Dict with "colors", "fabrics", "patterns" lists.
        """
        if not description:
            return {"colors": [], "fabrics": [], "patterns": []}

        desc_lower = description.lower()

        # ── Colors ────────────────────────────────────────────────────────────
        found_colors: List[str] = []
        if self.kb is not None:
            # Check all known color aliases
            for alias, canonical in self.kb._alias_to_color.items():
                if len(alias) > 3 and alias in desc_lower:
                    if canonical not in found_colors:
                        found_colors.append(canonical)
        else:
            # Fallback: simple keyword list
            _BASIC_COLORS = [
                "white", "black", "grey", "gray", "beige", "navy",
                "blue", "red", "green", "brown", "pink", "yellow",
                "orange", "purple", "teal", "olive", "burgundy",
            ]
            for color in _BASIC_COLORS:
                if color in desc_lower:
                    found_colors.append(color.capitalize())

        # ── Fabrics ────────────────────────────────────────────────────────────
        found_fabrics: List[str] = []
        if self.kb is not None:
            for alias, canonical in self.kb._alias_to_fabric.items():
                if len(alias) > 3 and alias in desc_lower:
                    if canonical not in found_fabrics:
                        found_fabrics.append(canonical)
        else:
            _BASIC_FABRICS = [
                "cotton", "polyester", "linen", "wool", "silk",
                "denim", "nylon", "spandex", "fleece", "leather",
            ]
            for fabric in _BASIC_FABRICS:
                if fabric in desc_lower:
                    found_fabrics.append(fabric.capitalize())

        # ── Patterns ──────────────────────────────────────────────────────────
        found_patterns: List[str] = []
        _PATTERN_TERMS = {
            "solid": "solid", "plain": "solid", "striped": "stripes",
            "stripe": "stripes", "check": "checks", "plaid": "checks",
            "floral": "floral", "flower": "floral", "leopard": "animal_print",
            "graphic": "graphic", "tie-dye": "tie_dye", "paisley": "paisley",
            "camouflage": "camouflage", "camo": "camouflage",
            "polka dot": "polka_dot", "geometric": "geometric",
        }
        for term, pattern_key in _PATTERN_TERMS.items():
            if term in desc_lower and pattern_key not in found_patterns:
                found_patterns.append(pattern_key)

        return {
            "colors"  : found_colors[:5],    # cap at 5 to avoid noise
            "fabrics" : found_fabrics[:3],
            "patterns": found_patterns[:3],
        }

    def _normalize_subcategory(self, raw_subcategory: str) -> str:
        """
        Normalize a raw subcategory string to a clean label.

        Strips special characters, normalizes case.

        Args:
            raw_subcategory : Raw subcategory string from HDF5.

        Returns:
            Cleaned subcategory string.
        """
        if not raw_subcategory:
            return ""
        # Clean to title case, strip punctuation
        cleaned = re.sub(r"[_\-]+", " ", raw_subcategory).strip()
        return cleaned.title()


# =============================================================================
# ── 5. FashionGenValidator — Validation Layer
# =============================================================================

class FashionGenValidator:
    """
    Validation Layer: validates FashionGenRecord against taxonomy rules.

    Responsibilities:
      - Check that required fields are non-empty.
      - Verify category and gender are valid taxonomy keys.
      - Run knowledge-base conditional rules if KB is available.
      - Assign is_valid, errors, and warnings to each record.

    This class does NOT read from HDF5 and does NOT write to disk.
    """

    # Fields that must be non-empty for a record to be valid
    _REQUIRED_FIELDS = ["image_id", "category", "gender", "dataset_source"]

    # Valid taxonomy values (from knowledge base constants)
    _VALID_CATEGORIES = {
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    }
    _VALID_GENDERS = {"men", "women", "unisex"}
    _VALID_STYLES  = {
        "streetwear", "luxury", "formal", "business_casual",
        "techwear", "minimalist", "vintage", "athleisure", ""
    }
    _VALID_SEASONS = {
        "spring", "summer", "autumn", "winter", "all_season"
    }

    def __init__(
        self,
        kb: Optional["FashionDomainResearch"] = None,
    ) -> None:
        """
        Initialise the validator.

        Args:
            kb : Optional knowledge-base instance for conditional rule checks.
        """
        self.kb = kb

    def validate(self, record: FashionGenRecord) -> FashionGenRecord:
        """
        Run all validation layers on a FashionGenRecord and return it
        with is_valid, errors, and warnings populated.

        Validation layers:
          1. Required fields check.
          2. Category taxonomy membership.
          3. Gender taxonomy membership.
          4. Style taxonomy membership (warning only if unknown).
          5. Season taxonomy membership (warning only if unknown).
          6. Gender-category compatibility (e.g., dresses must be women).
          7. KB conditional rules (if KB is available).

        Args:
            record : A FashionGenRecord from the Transformation layer.

        Returns:
            The same record with is_valid, errors, warnings populated.
        """
        errors:   List[str] = []
        warnings: List[str] = []

        d = record.to_dict()  # Use the dict for rule evaluation

        # ── Layer 1: Required fields ───────────────────────────────────────────
        for req_field in self._REQUIRED_FIELDS:
            val = d.get(req_field, "")
            if not val or (isinstance(val, str) and not val.strip()):
                errors.append(f"Missing required field: '{req_field}'")

        # ── Layer 2: Category validity ─────────────────────────────────────────
        if record.category not in self._VALID_CATEGORIES:
            errors.append(
                f"Invalid category: '{record.category}'. "
                f"Must be one of {sorted(self._VALID_CATEGORIES)}"
            )

        # ── Layer 3: Gender validity ───────────────────────────────────────────
        if record.gender not in self._VALID_GENDERS:
            errors.append(
                f"Invalid gender: '{record.gender}'. "
                f"Must be one of {sorted(self._VALID_GENDERS)}"
            )

        # ── Layer 4: Style validity (warning only) ─────────────────────────────
        if record.style and record.style not in self._VALID_STYLES:
            warnings.append(
                f"Unrecognized style: '{record.style}'. "
                f"Known styles: {sorted(self._VALID_STYLES - {''})}"
            )

        # ── Layer 5: Season validity (warning only) ────────────────────────────
        if record.season not in self._VALID_SEASONS:
            warnings.append(
                f"Unrecognized season: '{record.season}'. "
                f"Known seasons: {sorted(self._VALID_SEASONS)}"
            )

        # ── Layer 6: Gender-category compatibility ────────────────────────────
        if record.category == "dresses" and record.gender == "men":
            errors.append(
                "Category 'dresses' is not valid for gender 'men'"
            )

        # ── Layer 7: KB conditional rules ─────────────────────────────────────
        if self.kb is not None and _KB_AVAILABLE:
            try:
                kb_result = self.kb.validate(d)
                # Merge KB errors and warnings (avoid duplicates)
                for err in kb_result.errors:
                    if err not in errors:
                        errors.append(err)
                for warn in kb_result.warnings:
                    if warn not in warnings:
                        warnings.append(warn)
            except Exception as exc:
                warnings.append(f"KB validation error: {exc}")

        # ── Assign results back to record ──────────────────────────────────────
        record.errors   = errors
        record.warnings = warnings
        record.is_valid = len(errors) == 0

        if record.is_valid:
            logger.debug(f"VALID   {record.image_id}")
        else:
            logger.debug(f"INVALID {record.image_id} | {errors}")

        return record


# =============================================================================
# ── 6. FashionGenWriter — Write / Save Layer
# =============================================================================

class FashionGenWriter:
    """
    Write Layer: persists processed records to disk.

    Responsibilities:
      - Create output directory if needed.
      - Write the JSON output file (fashiongen_processed.json).
      - Optionally save image arrays as JPEG files.
      - Write a pipeline run report (fashiongen_run_report.json).

    This class does NOT read from HDF5 and does NOT transform data.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = _DEFAULT_OUTPUT_DIR,
        image_dir : Union[str, Path] = _DEFAULT_IMAGE_DIR,
        save_images: bool = False,
    ) -> None:
        """
        Initialise the writer.

        Args:
            output_dir  : Directory for JSON output files.
            image_dir   : Directory for saved JPEG image files.
            save_images : If True, save image arrays as JPEGs (slow; ~1 GB+).
        """
        self.output_dir  = Path(output_dir)
        self.image_dir   = Path(image_dir)
        self.save_images = save_images

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)

    def save_records(
        self,
        records: List[FashionGenRecord],
        stats  : PipelineStats,
    ) -> Path:
        """
        Write all processed records to fashiongen_processed.json.

        The output file has the following top-level structure:
        {
          "_meta"  : { pipeline info, stats summary },
          "records": [ { ...FashionGenRecord.to_dict() }, ... ]
        }

        Args:
            records : List of all processed FashionGenRecord objects.
            stats   : Final pipeline statistics.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self.output_dir / _OUTPUT_FILENAME

        payload = {
            "_meta": {
                "dataset"           : "fashiongen",
                "pipeline_version"  : "1.0.0",
                "generated_at"      : datetime.now(timezone.utc).isoformat(),
                "total_records"     : len(records),
                "valid_records"     : sum(1 for r in records if r.is_valid),
                "output_file"       : str(output_path),
                "stats"             : stats.to_dict(),
            },
            "records": [r.to_dict() for r in records],
        }

        logger.info(
            f"Writing {len(records):,} records → {output_path}"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.success(
            f"Saved {len(records):,} records | "
            f"file={output_path.name} | "
            f"size={output_path.stat().st_size / 1024:.1f} KB"
        )
        return output_path

    def save_run_report(
        self,
        stats    : PipelineStats,
        hdf5_path: Path,
    ) -> Path:
        """
        Write a human-readable pipeline run report to fashiongen_run_report.json.

        Args:
            stats     : Final pipeline stats object.
            hdf5_path : Path to the source HDF5 file.

        Returns:
            Path to the saved report file.
        """
        report_path = self.output_dir / "fashiongen_run_report.json"

        report = {
            "run_info": {
                "pipeline"          : "FashionGenLoader",
                "version"           : "1.0.0",
                "run_at"            : datetime.now(timezone.utc).isoformat(),
                "source_hdf5"       : str(hdf5_path),
                "output_file"       : str(self.output_dir / _OUTPUT_FILENAME),
            },
            "pipeline_stats"    : stats.to_dict(),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Run report saved → {report_path}")
        return report_path

    def save_image(
        self,
        image_id  : str,
        image_arr : np.ndarray,
        image_path: str,
    ) -> bool:
        """
        Optionally save a single image array as a JPEG file.

        Skipped if save_images=False or if Pillow is not installed.

        Args:
            image_id  : Record image ID (for logging).
            image_arr : numpy uint8 RGB array.
            image_path: Relative path string (from FashionGenRecord.image_path).

        Returns:
            True if saved, False otherwise.
        """
        if not self.save_images:
            return False

        try:
            from PIL import Image as PILImage
        except ImportError:
            return False

        try:
            abs_path = _PROJECT_ROOT / image_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(image_arr.astype(np.uint8), mode="RGB").save(
                str(abs_path), format="JPEG", quality=90
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to save image {image_id}: {exc}")
            return False


# =============================================================================
# ── 7. FashionGenLoader — Orchestrator (Main Entry Point)
# =============================================================================

class FashionGenLoader:
    """
    Pipeline Orchestrator: coordinates Extractor → Transformer → Validator → Writer.

    This is the single class a caller needs to import. All other classes
    are implementation details of the pipeline.

    Usage:
        loader = FashionGenLoader()
        result = loader.run(max_records=500)
        print(result["stats"]["total_processed"])
        print(result["output_path"])
    """

    def __init__(
        self,
        hdf5_path  : Union[str, Path, None]  = None,
        output_dir : Union[str, Path, None]  = None,
        image_dir  : Union[str, Path, None]  = None,
        id_prefix  : str                     = "FG",
        save_images: bool                    = False,
        use_kb     : bool                    = True,
    ) -> None:
        """
        Initialise the pipeline orchestrator.

        Args:
            hdf5_path   : Path to the FashionGen .h5 file.
                          Defaults to datasets/fashiongen/fashiongen_256_256_train.h5
            output_dir  : Directory for JSON output.
                          Defaults to datasets/processed/
            image_dir   : Directory for optional JPEG saves.
                          Defaults to datasets/fashiongen/images/
            id_prefix   : Image ID prefix (default "FG").
            save_images : Whether to save image arrays as JPEG files.
            use_kb      : Whether to load the Knowledge Base for normalization.
        """
        self.hdf5_path  = Path(hdf5_path)  if hdf5_path  else _DEFAULT_HDF5_PATH
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self.image_dir  = Path(image_dir)  if image_dir  else _DEFAULT_IMAGE_DIR

        # ── Knowledge Base ─────────────────────────────────────────────────────
        self.kb: Optional[FashionDomainResearch] = None
        if use_kb and _KB_AVAILABLE:
            try:
                self.kb = FashionDomainResearch()
                logger.info("Knowledge Base loaded and attached to pipeline")
            except Exception as exc:
                logger.warning(f"KB load failed ({exc}) — proceeding without KB")

        # ── Pipeline layers ────────────────────────────────────────────────────
        self.extractor   = FashionGenExtractor(hdf5_path=self.hdf5_path)
        self.transformer = FashionGenTransformer(
            id_prefix = id_prefix,
            image_dir = self.image_dir,
            kb        = self.kb,
        )
        self.validator   = FashionGenValidator(kb=self.kb)
        self.writer      = FashionGenWriter(
            output_dir  = self.output_dir,
            image_dir   = self.image_dir,
            save_images = save_images,
        )

        logger.info(
            f"FashionGenLoader ready | "
            f"hdf5={self.hdf5_path.name} | "
            f"output={self.output_dir} | "
            f"kb={'yes' if self.kb else 'no'}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        max_records: Optional[int] = None,
        start_index: int = 0,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full ingestion pipeline: Extract → Transform → Validate → Save.

        Progress is reported via tqdm if available; falls back to periodic
        log messages every 1,000 records.

        Args:
            max_records   : Maximum records to process. None = all records.
            start_index   : HDF5 start row (0-based). Useful for resuming.
            show_progress : Whether to display a tqdm progress bar.

        Returns:
            Dict with:
              "output_path"  : Path to the saved JSON file.
              "report_path"  : Path to the run report JSON.
              "stats"        : PipelineStats.to_dict() — full metrics.
              "total_records": int — number of records saved.
        """
        logger.info("=" * 60)
        logger.info("FashionGen Pipeline Starting")
        logger.info(f"  Source  : {self.hdf5_path}")
        logger.info(f"  Output  : {self.output_dir / _OUTPUT_FILENAME}")
        logger.info(f"  Max     : {max_records or 'all'}")
        logger.info("=" * 60)

        stats   = PipelineStats()
        records : List[FashionGenRecord] = []

        # ── Determine total for progress bar ───────────────────────────────────
        total_in_file = self.extractor.get_total_records()
        total_to_process = (
            min(max_records, total_in_file - start_index)
            if max_records is not None
            else total_in_file - start_index
        )

        logger.info(
            f"  HDF5 total records : {total_in_file:,}\n"
            f"  Records to process : {total_to_process:,}"
        )

        # ── Build iterable (with optional tqdm progress bar) ───────────────────
        raw_stream = self.extractor.stream(
            start=start_index,
            max_records=max_records,
        )
        iterable = self._wrap_progress(
            raw_stream,
            total=total_to_process,
            desc="FashionGen Ingestion",
            show=show_progress and _TQDM_AVAILABLE,
        )

        # ── Main processing loop ───────────────────────────────────────────────
        log_interval = max(1, min(1000, total_to_process // 10)) if total_to_process > 0 else 500

        for raw_record, extraction_error in iterable:
            stats.total_read += 1

            # ── Handle extraction errors (corrupted HDF5 row) ─────────────────
            if extraction_error:
                logger.warning(f"Extraction error at index {raw_record.source_index}: {extraction_error}")
                stats.total_corrupted += 1
                stats.total_skipped   += 1
                continue

            try:
                # ── Transform ─────────────────────────────────────────────────
                record: FashionGenRecord = self.transformer.transform(raw_record)

                # ── Validate ──────────────────────────────────────────────────
                record = self.validator.validate(record)

                # ── Update stats ──────────────────────────────────────────────
                stats.total_processed += 1
                if record.is_valid:
                    stats.total_valid += 1
                else:
                    stats.total_invalid += 1
                if record.warnings:
                    stats.total_warnings += 1

                stats.increment_category(record.category or "unknown")
                stats.increment_gender(record.gender or "unknown")
                if record.style:
                    stats.increment_style(record.style)
                stats.increment_season(record.season or "all_season")

                # ── Save image (optional) ──────────────────────────────────────
                if self.writer.save_images:
                    self.writer.save_image(
                        record.image_id,
                        raw_record.image_array,
                        record.image_path,
                    )

                records.append(record)

                # ── Periodic log (non-tqdm fallback) ──────────────────────────
                if not (_TQDM_AVAILABLE and show_progress):
                    if stats.total_processed % log_interval == 0:
                        logger.info(
                            f"Progress: {stats.total_processed:,} / {total_to_process:,} | "
                            f"valid={stats.total_valid:,} | "
                            f"rate={stats.records_per_second} rec/s"
                        )

            except Exception as exc:
                logger.error(
                    f"Transform/Validate error at index {raw_record.source_index}: {exc}",
                    exc_info=True,
                )
                stats.total_skipped += 1
                continue

        # ── Finalise stats ─────────────────────────────────────────────────────
        stats.finalize()
        stats.log_summary()

        # ── Write outputs ──────────────────────────────────────────────────────
        output_path = self.writer.save_records(records, stats)
        report_path = self.writer.save_run_report(stats, self.hdf5_path)
        stats.total_saved = len(records)

        return {
            "output_path"   : output_path,
            "report_path"   : report_path,
            "stats"         : stats.to_dict(),
            "total_records" : len(records),
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the FashionGen HDF5 dataset.

        Delegates to FashionGenExtractor.get_dataset_info().

        Returns:
            Dict with total_records, image_shape, sample categories, etc.
        """
        return self.extractor.get_dataset_info()

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_progress(
        iterable: Iterator,
        total    : int,
        desc     : str,
        show     : bool,
    ) -> Iterator:
        """
        Wrap an iterable with tqdm progress bar if available and requested.

        Args:
            iterable : The raw generator to wrap.
            total    : Expected total item count.
            desc     : Progress bar description string.
            show     : Whether to enable tqdm.

        Returns:
            Wrapped (or original) iterable.
        """
        if show and _TQDM_AVAILABLE:
            return _tqdm(
                iterable,
                total=total,
                desc=desc,
                unit="rec",
                dynamic_ncols=True,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
        return iterable


# =============================================================================
# ── 8. CLI Entry Point
# =============================================================================

def _build_cli_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fashiongen_loader",
        description=(
            "FashionGen Dataset Ingestion Pipeline\n"
            "AI-Powered Fashion Design Assistant — Week 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Process first 500 records\n"
            "  python fashiongen_loader.py --max-records 500\n\n"
            "  # Full pipeline run with custom HDF5 path\n"
            "  python fashiongen_loader.py --hdf5 /data/fashiongen.h5\n\n"
            "  # Show dataset info only\n"
            "  python fashiongen_loader.py --info\n\n"
            "  # Process with image saving enabled\n"
            "  python fashiongen_loader.py --max-records 100 --save-images\n"
        ),
    )

    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Path to FashionGen .h5 file (default: {_DEFAULT_HDF5_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Output directory for processed JSON (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of records to process (default: all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        metavar="N",
        help="HDF5 start index for resumable processing (default: 0)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save image arrays as JPEG files (slow; requires ~1 GB+ disk)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (useful for log files)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset info and exit (no processing)",
    )
    parser.add_argument(
        "--no-kb",
        action="store_true",
        help="Disable Knowledge Base (faster, less normalization)",
    )

    return parser


def main() -> int:
    """
    CLI main entry point.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser  = _build_cli_parser()
    args    = parser.parse_args()

    # ── Build loader ───────────────────────────────────────────────────────────
    loader = FashionGenLoader(
        hdf5_path   = args.hdf5,
        output_dir  = args.output_dir,
        save_images = args.save_images,
        use_kb      = not args.no_kb,
    )

    # ── Info mode ──────────────────────────────────────────────────────────────
    if args.info:
        info = loader.get_dataset_info()
        print("\nFashionGen Dataset Info")
        print("=" * 50)
        for k, v in info.items():
            print(f"  {k:<25} : {v}")
        return 0

    # ── Run pipeline ───────────────────────────────────────────────────────────
    try:
        result = loader.run(
            max_records   = args.max_records,
            start_index   = args.start_index,
            show_progress = not args.no_progress,
        )
        print("\nPipeline Complete")
        print(f"  Records saved : {result['total_records']:,}")
        print(f"  Output        : {result['output_path']}")
        print(f"  Report        : {result['report_path']}")
        return 0
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
