"""
=============================================================================
AI-Powered Fashion Design Assistant
notebooks/01_dataset_exploration.py — Week 1 EDA Notebook Script
=============================================================================
This script mirrors the notebook "01_Dataset_Exploration.ipynb" as a
runnable Python file. Convert to notebook with:

    pip install jupytext
    jupytext --to notebook notebooks/01_dataset_exploration.py

CONTENTS:
    1. Environment Setup & Config Loading
    2. FashionGen Dataset Statistics
    3. DeepFashion Dataset Statistics
    4. Category Distribution Visualisation
    5. Sample Image Grid Display
    6. Description Text Analysis
    7. Preprocessing Smoke Test
    8. Validation Report Summary
=============================================================================
"""

# %% [markdown]
# # 🎨 Week 1: Fashion Dataset Exploration
# **AI-Powered Fashion Design Assistant**
# 
# This notebook explores the FashionGen and DeepFashion datasets to:
# - Understand their structure, size, and category distributions
# - Identify preprocessing requirements (resolution, colour space, etc.)
# - Validate data quality before ingestion into the training pipeline

# %% Setup
import sys
import os
from pathlib import Path

# Add project root to path so we can import data_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from loguru import logger

# Load project configuration
CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print("✅ Project config loaded")
print(f"   Project: {config['project']['name']}")
print(f"   Week   : {config['project']['week']}")

# %% [markdown]
# ## 1. FashionGen Dataset Overview

# %% FashionGen stats
from data_pipeline.ingestion import FashionGenIngester

fashiongen_path = os.getenv(
    "FASHIONGEN_HDF5_PATH",
    PROJECT_ROOT / "datasets" / "fashiongen" / "fashiongen_256_256_train.h5"
)

fg_ingester = FashionGenIngester(hdf5_path=fashiongen_path, split="train")
fg_stats    = fg_ingester.get_dataset_stats()

print("\n📦 FashionGen Statistics")
print("=" * 40)
for k, v in fg_stats.items():
    print(f"  {k:<28} : {v}")

# %% [markdown]
# ## 2. DeepFashion Dataset Overview

# %% DeepFashion stats
from data_pipeline.ingestion import DeepFashionIngester

deepfashion_root = os.getenv(
    "DEEPFASHION_ROOT",
    PROJECT_ROOT / "datasets" / "deepfashion"
)

df_ingester  = DeepFashionIngester(root_dir=deepfashion_root)
split_sizes  = df_ingester.get_split_sizes()
cat_dist     = df_ingester.get_category_distribution()

print("\n📦 DeepFashion Split Sizes")
print("=" * 40)
for split, count in split_sizes.items():
    print(f"  {split:<10} : {count:,} images")

# %% [markdown]
# ## 3. Category Distribution

# %% Category distribution plot
if cat_dist:
    # Show top 20 categories
    top_n = 20
    top_cats   = list(cat_dist.keys())[:top_n]
    top_counts = [cat_dist[c] for c in top_cats]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(top_cats[::-1], top_counts[::-1], color=sns.color_palette("viridis", top_n))
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title(f"DeepFashion — Top {top_n} Category Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "category_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n💾 Plot saved to outputs/category_distribution.png")
else:
    print("⚠️  DeepFashion not loaded — skipping category plot.")

# %% [markdown]
# ## 4. Sample Image Grid (FashionGen)

# %% Sample grid
MAX_SAMPLES = 16

print(f"\n🖼  Loading {MAX_SAMPLES} sample images from FashionGen…")
samples = []
try:
    for rec in fg_ingester.stream(max_items=MAX_SAMPLES):
        samples.append(rec)
        if len(samples) >= MAX_SAMPLES:
            break

    if samples:
        cols = 4
        rows = MAX_SAMPLES // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle("FashionGen — Sample Images", fontsize=14, fontweight="bold")

        for i, (ax, rec) in enumerate(zip(axes.flat, samples)):
            ax.imshow(rec["image_array"])
            label = f"{rec['category']}\n{rec['gender']}"
            ax.set_title(label, fontsize=7)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / "outputs" / "fashiongen_samples.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("💾 Saved → outputs/fashiongen_samples.png")
    else:
        print("⚠️  No FashionGen records loaded — is the HDF5 file present?")

except FileNotFoundError:
    print("⚠️  FashionGen HDF5 file not found. Skipping sample grid.")

# %% [markdown]
# ## 5. Preprocessing Smoke Test

# %% Preprocessing smoke test
from data_pipeline.preprocessing import FashionPreprocessor
from data_pipeline.preprocessing.image_preprocessor import PreprocessorConfig

pp_config = PreprocessorConfig(
    target_size=(256, 256),
    blur_threshold=0.0,   # Disabled during EDA
    min_entropy=0.0,
)
preprocessor = FashionPreprocessor(
    output_dir=PROJECT_ROOT / "datasets" / "processed",
    config=pp_config,
)

if samples:
    result = preprocessor.process(samples[0])
    print(f"\n✅ Preprocessing Result")
    print(f"   success          : {result.success}")
    print(f"   processed_path   : {result.processed_path}")
    print(f"   normalized shape : {result.normalized_array.shape if result.normalized_array is not None else 'N/A'}")
    print(f"   processing_time  : {result.processing_time_ms:.1f} ms")
else:
    print("⚠️  No sample records to preprocess.")

# %% [markdown]
# ## 6. Data Validation Summary

# %% Validation summary
from data_pipeline.validation import DataValidator

validator = DataValidator(verify_file_exists=False)

if samples:
    # Build minimal metadata records from raw samples for testing validator
    test_records = []
    for s in samples[:5]:
        test_records.append({
            "image_id"        : s["image_id"],
            "dataset_source"  : s["dataset_source"],
            "processed_path"  : "/tmp/dummy.jpg",  # Placeholder
            "width"           : 256,
            "height"          : 256,
            "color_mode"      : "RGB",
            "file_size_bytes" : 12000,
            "description"     : s.get("description", "No description"),
            "category"        : s.get("category", "UNKNOWN"),
            "split"           : s["split"],
        })

    batch_report = validator.validate_batch(test_records)
    print(f"\n✅ Validation Summary")
    print(f"   {batch_report.summary()}")

    if batch_report.invalid_records():
        print("\n   ⚠️  Invalid records:")
        for r in batch_report.invalid_records():
            for e in r.errors:
                print(f"      [{r.image_id}] {e}")
else:
    print("⚠️  No records to validate.")

# %% [markdown]
# ## ✅ Week 1 EDA Complete
# 
# **Summary:**
# - FashionGen and DeepFashion datasets explored
# - Category distributions visualised
# - Sample images confirmed readable and correct shape
# - Preprocessing pipeline smoke-tested
# - Validation pipeline smoke-tested
# 
# **Next Steps (Week 2):** Full dataset ingestion → preprocessing → metadata generation pipeline run.
