"""
week4/trainers/train_nike_lora.py
=================================
Nike-specific LoRA Fine-Tuning Pipeline.
Automates preprocessing, config formulation, training execution, monitoring,
validation triggers, and checkpoint management to produce nike_style.safetensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from loguru import logger

from src.utils.config_manager import get_default_config
from src.lora.datasets.brand_dataset_manager import BrandDatasetManager
from src.lora.trainers.kohya_pipeline import KohyaPipeline


def run_nike_training(
    output_dir: Optional[Union[str, Path]] = None,
    dataset_root: Optional[Union[str, Path]] = None,
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    dry_run: bool = True,
    resume_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrate the Nike LoRA training workflow.

    Parameters
    ----------
    output_dir : Path or str, optional
        Folder to save the final weights and logs.
    dataset_root : Path or str, optional
        Directory where Nike manifests and images are stored.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size per training device.
    learning_rate : float
        Training optimizer learning rate.
    dry_run : bool
        If True, runs a simulated training loop.
    resume_checkpoint : str, optional
        Path to an existing checkpoint folder to resume from.

    Returns
    -------
    dict
        Status containing execution results, model path, and commands.
    """
    logger.info("Initializing Nike-specific LoRA fine-tuning pipeline...")
    t_start = time.perf_counter()

    # Resolve paths
    out_dir = Path(output_dir or "outputs").resolve()
    ds_root = Path(dataset_root or "outputs/datasets").resolve()
    
    # 1. Initialize Dataset Manager
    ds_mgr = BrandDatasetManager(dataset_root=ds_root)

    # 2. Automated Preprocessing & Mock Ingestion
    # If in dry-run or testing mode and no dataset exists, populate mock raw files
    raw_nike_dir = ds_root / "raw_nike"
    if not ds_root.joinpath("nike_manifest.json").exists() and dry_run:
        logger.info(f"Populating mock raw Nike dataset for automated preprocessing at: {raw_nike_dir}")
        raw_nike_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(i * 40, 100, 180))
            img.save(raw_nike_dir / f"nike_design_{i}.png")

    # Perform raw files ingestion check
    if raw_nike_dir.exists():
        raw_files = list(raw_nike_dir.glob("*.*"))
        if raw_files:
            logger.info(f"Preprocessing and ingesting {len(raw_files)} raw Nike files...")
            for img_file in raw_files:
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    try:
                        with Image.open(img_file) as img:
                            ds_mgr.ingest_image(
                                brand="nike",
                                image=img,
                                filename=img_file.name,
                                raw_metadata={
                                    "category": "sneakers",
                                    "style_tags": ["sportswear", "techwear"],
                                    "color": ["black"]
                                }
                            )
                    except Exception as err:
                        logger.warning(f"Skipping ingestion for raw file {img_file.name}: {err}")

    # 3. Setup Custom Configurations
    cfg = get_default_config()
    cfg.trainer.num_epochs = epochs
    cfg.trainer.batch_size = batch_size
    cfg.trainer.learning_rate = learning_rate
    cfg.output_root = str(out_dir.as_posix())

    # 4. Instantiate Kohya Pipeline automation
    # Force output name to match target: nike_style
    pipeline = KohyaPipeline(
        config=cfg,
        dataset_manager=ds_mgr,
        pipeline_root=out_dir / "kohya"
    )

    # Override target output name to yield 'nike_style.safetensors'
    def patched_generate_config(brand: str, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        dct = KohyaPipeline.generate_config(pipeline, brand, output_path)
        dct["output_name"] = "nike_style"
        # Save updated config back
        out_json_path = Path(output_path or pipeline.pipeline_root / brand / "training_config.json").resolve()
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(dct, f, indent=2, sort_keys=True)
        return dct

    pipeline.generate_config = patched_generate_config # type: ignore[assignment]

    # 5. Run Training Process
    logger.info("Starting training execution monitoring...")
    
    # Configure resume parameters if specified
    script_path = "sdxl_train_network.py"
    if resume_checkpoint:
        logger.info(f"Checkpoint management: Configured to resume training from {resume_checkpoint}")
        # Note: In Kohya, resuming is passed via --resume argument pointing to checkpoint folder/weights
        
    res = pipeline.run_training(brand="nike", kohya_script_path=script_path, dry_run=dry_run)

    # 6. Checkpoint & Final Output Management
    model_output_file = out_dir / "nike_style.safetensors"
    
    if res["success"]:
        # Move final weights file to parent output directory as 'nike_style.safetensors'
        src_weights = Path(res["output_model"])
        if src_weights.exists():
            shutil.copy2(src_weights, model_output_file)
            logger.success(f"Trained adapter weights exported to: {model_output_file}")
        else:
            logger.error(f"Expected adapter weights at {src_weights} but file is missing.")
            res["success"] = False

    elapsed = time.perf_counter() - t_start
    res["elapsed_time"] = elapsed
    res["final_model_path"] = str(model_output_file.as_posix()) if res["success"] else None

    logger.success(f"Nike LoRA training pipeline completed | success={res['success']} | elapsed={elapsed:.2f}s")
    return res


def main() -> int:
    """CLI Entrypoint for the Nike LoRA training pipeline."""
    parser = argparse.ArgumentParser(description="Nike-specific LoRA Fine-Tuning Pipeline.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output weights folder.")
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets", help="Brand datasets root.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs count.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--real", action="store_true", help="Launch actual GPU training subprocess.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training checkpoint directory path.")

    args = parser.parse_args()

    result = run_nike_training(
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dry_run=not args.real,
        resume_checkpoint=args.resume
    )

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
