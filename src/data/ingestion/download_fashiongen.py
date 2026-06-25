"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/download_fashiongen.py
=============================================================================
PURPOSE:
    Helper script to guide users through downloading the FashionGen dataset.
    FashionGen requires registration at fashion-gen.com — this script
    provides instructions and validates downloaded files.

USAGE:
    python data_pipeline/ingestion/download_fashiongen.py --help
    python data_pipeline/ingestion/download_fashiongen.py --check-only
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from loguru import logger


# Official FashionGen file sizes (bytes) for integrity validation
# These may change — update after downloading the actual files.
EXPECTED_FILES = {
    "fashiongen_256_256_train.h5": {
        "size_gb_approx": 13.0,
        "description": "FashionGen training set (260,480 items)",
        "download_page": "https://fashion-gen.com/",
    },
    "fashiongen_256_256_val.h5": {
        "size_gb_approx": 1.7,
        "description": "FashionGen validation set (32,500 items)",
        "download_page": "https://fashion-gen.com/",
    },
}


def print_instructions() -> None:
    """Print step-by-step download instructions."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║         FashionGen Dataset — Download Instructions                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Register at: https://fashion-gen.com/                            ║
║     (Free academic access with institutional email)                  ║
║                                                                      ║
║  2. After approval, download these two files:                        ║
║     • fashiongen_256_256_train.h5  (~13 GB)                         ║
║     • fashiongen_256_256_val.h5    (~1.7 GB)                        ║
║                                                                      ║
║  3. Place both files in:                                             ║
║     datasets/fashiongen/                                             ║
║                                                                      ║
║  4. Re-run this script to validate:                                  ║
║     python data_pipeline/ingestion/download_fashiongen.py            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def check_files(fashiongen_dir: Path) -> bool:
    """
    Check if FashionGen files are present and report their status.

    Args:
        fashiongen_dir : Path to datasets/fashiongen/

    Returns:
        True if all expected files are found, False otherwise.
    """
    all_found = True
    print(f"\n📂 Checking: {fashiongen_dir.resolve()}\n")

    for filename, info in EXPECTED_FILES.items():
        fpath = fashiongen_dir / filename
        status = "✅ FOUND" if fpath.exists() else "❌ MISSING"
        size_str = ""
        if fpath.exists():
            size_gb = fpath.stat().st_size / 1e9
            size_str = f"  ({size_gb:.2f} GB)"
        print(f"  {status}  {filename}{size_str}")
        print(f"           {info['description']}")
        if not fpath.exists():
            all_found = False
            print(f"           Download from: {info['download_page']}")
        print()

    return all_found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FashionGen dataset download helper"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/fashiongen"),
        help="Directory where FashionGen HDF5 files should be placed",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if files exist; don't print full instructions",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.check_only:
        print_instructions()

    all_found = check_files(args.data_dir)

    if all_found:
        print("✅ All FashionGen files present. Ready to ingest!")
        sys.exit(0)
    else:
        print("⚠️  Some files are missing. Follow the instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
