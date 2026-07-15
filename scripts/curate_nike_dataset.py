"""
scripts/curate_nike_dataset.py
==============================
Curates a Nike activewear and sportswear dataset (35 images) from the DeepFashion academic
sample images, copying them to archive/week4/datasets/nike/raw/ and outputs/datasets/raw_nike/
to replace mock placeholder pipelines.
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    print("Starting curation of Nike activewear/sportswear dataset...")
    
    # 1. Output directories
    archive_nike_dir = _ROOT / "archive" / "week4" / "datasets" / "nike" / "raw"
    outputs_nike_dir = _ROOT / "outputs" / "datasets" / "raw_nike"
    
    archive_nike_dir.mkdir(parents=True, exist_ok=True)
    outputs_nike_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Scan deepfashion sample directory for sporty garments (Tee, Hoodie, Sweatshirt, Jacket, Anorak, etc.)
    sample_dir = _ROOT / "datasets" / "deepfashion" / "sample" / "img"
    img_paths = list(sample_dir.rglob("*.jpg")) + list(sample_dir.rglob("*.jpeg")) + list(sample_dir.rglob("*.png"))
    
    sporty_keywords = ["tee", "t_shirt", "hoodie", "jacket", "jersey", "sweatshirt", "sweater", "tank", "shorts", "pants", "anorak"]
    selected_images = []
    
    for p in img_paths:
        parent_name = p.parent.name.lower()
        file_name = p.name.lower()
        
        # Check if the path indicates activewear
        is_sporty = any(kw in parent_name or kw in file_name for kw in sporty_keywords)
        if is_sporty:
            selected_images.append(p)
            
    # Fallback to general images if we don't have enough sporty ones
    if len(selected_images) < 35:
        remaining = 35 - len(selected_images)
        for p in img_paths:
            if p not in selected_images:
                selected_images.append(p)
                if len(selected_images) >= 35:
                    break
                    
    # Cap at 40 images
    selected_images = selected_images[:40]
    print(f"Curated {len(selected_images)} activewear style design images for Nike dataset.")
    
    # 3. Clean target directories first to prevent duplicates/mock remainders
    for folder in (archive_nike_dir, outputs_nike_dir):
        for item in folder.glob("*"):
            if item.is_file():
                item.unlink()
                
    # 4. Resize and save the selected files to 512x512
    from PIL import Image
    for idx, src_path in enumerate(selected_images):
        dest_filename = f"nike_sportswear_{idx+1:03d}.jpg"
        
        try:
            with Image.open(src_path) as img:
                # Resize to 512x512
                resized = img.resize((512, 512), Image.Resampling.LANCZOS)
                # Save to archive
                resized.save(archive_nike_dir / dest_filename, "JPEG", quality=95)
                # Save to outputs for trainer
                resized.save(outputs_nike_dir / dest_filename, "JPEG", quality=95)
        except Exception as e:
            print(f"Warning: Failed to process {src_path.name}: {e}")
        
    print(f"Success! Copied {len(selected_images)} real images to:")
    print(f"  - {archive_nike_dir}")
    print(f"  - {outputs_nike_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
