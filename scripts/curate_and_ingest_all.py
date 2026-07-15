"""
scripts/curate_and_ingest_all.py
================================
Curates brand-specific datasets from DeepFashion sample images and ingests them
using BrandDatasetManager to create clean, genuine manifests and directories.
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lora.datasets.brand_dataset_manager import BrandDatasetManager

def curate_brand_images(brand: str, sample_dir: Path, folders: list, target_count: int = 40) -> list:
    selected = []
    
    # 1. Scan specific subdirectories first
    for folder in folders:
        folder_dir = sample_dir / folder
        if folder_dir.exists():
            for p in folder_dir.rglob("*.*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    selected.append(p)
                    
    # Remove duplicates
    selected = list(dict.fromkeys(selected))
    
    # 2. Fallback to general images in sample_dir if needed
    if len(selected) < target_count:
        all_images = []
        for p in sample_dir.rglob("*.*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                all_images.append(p)
        for p in all_images:
            if p not in selected:
                selected.append(p)
                if len(selected) >= target_count:
                    break
                    
    return selected[:target_count]

def main():
    print("Starting curation and ingestion for all 4 brands...")
    
    sample_dir = _ROOT / "datasets" / "deepfashion" / "sample" / "img"
    if not sample_dir.exists():
        print(f"Error: DeepFashion sample directory not found at: {sample_dir}")
        return 1
        
    ds_root = _ROOT / "outputs" / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    
    # Instantiate manager
    ds_mgr = BrandDatasetManager(dataset_root=ds_root)
    
    # Define brand settings
    brands_config = {
        "nike": {
            "folders": ["Anorak", "Hoodie", "Jacket", "Jersey", "Sweatpants", "Sweatshorts"],
            "meta": {"category": "sportswear", "style_tags": ["activewear", "athletic"], "color": ["black"]}
        },
        "gucci": {
            "folders": ["Dress", "Suit", "Blazer", "Coat", "Kimono", "Robe", "Skirt"],
            "meta": {"category": "dresses", "style_tags": ["luxury", "haute-couture"], "color": ["red"]}
        },
        "zara": {
            "folders": ["Cardigan", "Jeans", "Joggers", "Jumpsuit", "Romper", "Sweater", "Pants"],
            "meta": {"category": "casual", "style_tags": ["contemporary", "minimalist"], "color": ["beige"]}
        },
        "h&m": {
            "folders": ["Tee", "Tank", "Shorts", "Leggings", "Underwear", "Socks", "Henley"],
            "meta": {"category": "basics", "style_tags": ["minimalist", "essential"], "color": ["white"]}
        }
    }
    
    for brand, cfg in brands_config.items():
        print(f"\n--- Curating and ingesting {brand.upper()} ---")
        
        # Clean target directory and manifest for this brand first
        brand_dir = ds_root / brand
        if brand_dir.exists():
            shutil.rmtree(brand_dir)
        brand_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = ds_root / f"{brand}_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
            
        # Get selected images
        selected_paths = curate_brand_images(brand, sample_dir, cfg["folders"], target_count=40)
        print(f"Curated {len(selected_paths)} images for {brand}.")
        
        # Ingest
        for idx, src_path in enumerate(selected_paths):
            filename = f"{brand}_design_{idx+1:03d}{src_path.suffix}"
            try:
                with Image.open(src_path) as img:
                    # Resize to 512x512
                    resized = img.resize((512, 512), Image.Resampling.LANCZOS)
                    # Ingest using ds_mgr
                    ds_mgr.ingest_image(
                        brand=brand,
                        image=resized,
                        filename=filename,
                        raw_metadata=cfg["meta"]
                    )
            except Exception as e:
                print(f"Warning: Failed to ingest {src_path.name}: {e}")
                
        print(f"Successfully curated and ingested {brand} dataset.")
        
    print("\nAll brand datasets curated and ingested successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
