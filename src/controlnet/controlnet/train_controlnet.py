"""
week3/controlnet/train_controlnet.py
====================================
Command Line Interface (CLI) for training/fine-tuning SDXL ControlNet.
Allows launching training sessions locally or via HuggingFace Accelerate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger
from datasets.fashion_sketch_dataset import FashionSketchDataset
from src.controlnet.controlnet.controlnet_trainer import FashionControlNetTrainer
from src.utils.config_manager import get_config


class DummyDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset returning dummy images and prompts for dry-run testing."""
    def __init__(self, size: int = 10) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": torch.randn(3, 64, 64),
            "conditioning_pixel_values": torch.randn(3, 64, 64),
            "prompt": f"A trendy fashion design sample {idx}"
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune SDXL ControlNet on fashion sketch datasets.")
    parser.add_argument("--manifest_path", type=str, default="datasets/processed/final_fashion_dataset.json",
                        help="Path to manifest JSON file.")
    parser.add_argument("--design_dir", type=str, default="datasets/processed",
                        help="Base directory containing design/garment images.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Precision mode.")
    parser.add_argument("--output_dir", type=str, default="outputs/trainer",
                        help="Output directory to save weights and validation artifacts.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run a fast, lightweight CPU simulation for testing purposes.")
    
    args = parser.parse_args()
    
    logger.info("Initializing ControlNet CLI Training Runner...")
    
    try:
        cfg = get_config()
    except Exception as err:
        logger.warning(f"Could not load CentralizedConfig: {err}. Proceeding with default arguments.")
        cfg = None
        
    # Check if manifest exists unless dry_run is set or design_dir is provided
    manifest = Path(args.manifest_path)
    if not manifest.exists() and not args.dry_run and not args.design_dir:
        logger.error(f"Manifest file not found at: {manifest}. Check path or run with --dry-run for testing.")
        return 1
        
    # Load training and validation datasets
    if args.dry_run and not args.design_dir:
        logger.info("Using DummyDatasets for dry-run training mode.")
        train_ds = DummyDataset(size=10)
        val_ds = DummyDataset(size=4)
    else:
        try:
            logger.info("Loading training and validation datasets...")
            manifest_val = None if args.manifest_path in ("None", "") or not Path(args.manifest_path).exists() else args.manifest_path
            train_ds = FashionSketchDataset(
                manifest_path=manifest_val,
                design_dir=args.design_dir,
                split="train",
                augment=True
            )
            val_ds = FashionSketchDataset(
                manifest_path=manifest_val,
                design_dir=args.design_dir,
                split="val",
                augment=False
            )
        except Exception as err:
            logger.exception(f"Dataset loading failed: {err}")
            return 2


    # Initialize trainer
    try:
        trainer = FashionControlNetTrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )
        
        logger.info("Starting training loop...")
        trainer.train()
        logger.success("Training run completed successfully.")
        return 0
    except Exception as err:
        logger.exception(f"Training execution encountered a critical error: {err}")
        return 3

if __name__ == "__main__":
    sys.exit(main())
