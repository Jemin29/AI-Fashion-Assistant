"""
scripts/train_all_brands.py
===========================
Executes LoraTrainer fine-tuning on Nike, Gucci, Zara, and H&M datasets in real mode
(CPU optimized, resolution 512x512, 1 epoch). Copies the produced .safetensors adapters
to weights/lora/<brand>/, and registers them in LoraRegistry.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from loguru import logger

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lora.trainers.lora_trainer import LoraTrainer
from datasets.fashion_sketch_dataset import FashionSketchDataset
from src.utils.config_manager import get_default_config
from src.lora.style_manager.lora_registry import LoraRegistry

def train_brand(brand: str, epochs: int = 1):
    logger.info(f"\n==================================================")
    logger.info(f"   STARTING LORA TRAINING FOR BRAND: {brand.upper()}   ")
    logger.info(f"==================================================")
    
    cfg = get_default_config()
    
    manifest_file = Path("outputs/datasets") / f"{brand}_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found for brand {brand} at: {manifest_file}")
        
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    # Normalise image_path separators for cross-platform compatibility
    records_list = []
    for rec in manifest_data.values():
        rec = dict(rec)
        rec["image_path"] = rec["image_path"].replace("\\", "/")
        records_list.append(rec)
        
    train_ds = FashionSketchDataset(
        manifest_path=records_list,
        design_dir=Path("outputs/datasets"),
        split="train",
        split_ratio=0.8,
        target_size=(512, 512),
    )
    val_ds = FashionSketchDataset(
        manifest_path=records_list,
        design_dir=Path("outputs/datasets"),
        split="val",
        split_ratio=0.8,
        target_size=(512, 512),
    )
    
    out_dir = Path("outputs") / f"lora_{brand}_real"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    trainer = LoraTrainer(
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=epochs,
        mixed_precision="no",  # CPU-friendly
        output_dir=out_dir,
        dry_run=False,
    )
    
    stats = trainer.train()
    logger.success(f"Finished training brand {brand}. Stats: {stats}")
    
    # ── Post-training: copy the real adapter to weights/lora/<brand>/ ────────
    final_lora_dir = out_dir / "final_lora"
    dest_dir = Path("weights/lora") / brand
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = dest_dir / f"{brand}_lora_adapter.safetensors"
    
    candidate_names = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
    ]
    copied = False
    for name in candidate_names:
        src = final_lora_dir / name
        if src.exists():
            shutil.copy2(src, dest_file)
            logger.success(f"Genuine LoRA adapter for {brand} saved to: {dest_file}")
            copied = True
            break
            
    if not copied:
        # Search recursively for any .safetensors
        found = sorted(final_lora_dir.rglob("*.safetensors"))
        if found:
            shutil.copy2(found[0], dest_file)
            logger.success(f"Genuine LoRA adapter for {brand} saved to: {dest_file}")
        else:
            raise FileNotFoundError(
                f"Real training completed but no .safetensors found under {final_lora_dir}"
            )
            
    # Register the model in the LoraRegistry
    registry = LoraRegistry()
    registry.register_model(
        brand=brand,
        model_path=str(dest_file),
        metadata={"style_description": f"Genuine {brand.capitalize()} fine-tuned LoRA adapter"}
    )
    logger.success(f"Successfully registered genuine LoRA adapter for {brand} in LoraRegistry.")

def main():
    brands = ["nike", "gucci", "zara", "h&m"]
    for brand in brands:
        try:
            train_brand(brand)
        except Exception as e:
            logger.exception(f"Failed to train LoRA adapter for brand: {brand}")
            return 1
            
    logger.success("Real LoRA fine-tuning for all 4 brands completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
