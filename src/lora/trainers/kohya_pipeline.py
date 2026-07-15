"""
week4/trainers/kohya_pipeline.py
================================
Kohya_SS Training Workflow automation.
Provides dataset directory mapping, prompt caption generation, training config serialization,
and subprocess command automation for brand style personalization.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from loguru import logger

from src.lora.datasets.brand_dataset_manager import BrandDatasetManager


class KohyaPipeline:
    """
    Automates the pipeline for Kohya_SS LoRA fine-tuning on brand datasets.
    Supports Nike, Gucci, Zara, and H&M datasets.
    """

    SUPPORTED_BRANDS = {"nike", "gucci", "zara", "h&m"}

    def __init__(
        self,
        config: Any = None,
        dataset_manager: Optional[BrandDatasetManager] = None,
        pipeline_root: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the KohyaPipeline.

        Parameters
        ----------
        config : Week4Config, optional
        dataset_manager : BrandDatasetManager, optional
        pipeline_root : Path or str, optional
            Base directory to output Kohya-structured folders (default: outputs/kohya).
        """
        self.config = config
        self.dataset_manager = dataset_manager or BrandDatasetManager(config=config)
        
        if pipeline_root:
            self.pipeline_root = Path(pipeline_root).resolve()
        elif config and getattr(config, "output_root", None):
            self.pipeline_root = Path(config.output_root).resolve() / "kohya"
        else:
            self.pipeline_root = Path("outputs/kohya").resolve()

        self.pipeline_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized KohyaPipeline | root={self.pipeline_root}")

    def prepare_dataset(
        self,
        brand: str,
        repeats: int = 10,
        activation_word: Optional[str] = None
    ) -> Path:
        """
        Prepare folders and caption files for Kohya training.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        repeats : int
            Number of times to repeat each image.
        activation_word : str, optional
            LoRA trigger word (defaults to brand name).

        Returns
        -------
        Path
            The parent directory of the prepared brand dataset folder.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.SUPPORTED_BRANDS:
            raise ValueError(f"Brand '{brand}' not supported. Choose from: {self.SUPPORTED_BRANDS}")

        act_word = activation_word or brand_key
        img_folder_name = f"{repeats}_{act_word}"
        
        # Kohya expects: <brand_dir>/img/<repeats>_<activation_word>/
        brand_dir = self.pipeline_root / brand_key
        img_dir = brand_dir / "img" / img_folder_name
        
        # Recreate target image folder
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load manifest records
        manifest_path = self.dataset_manager.dataset_root / f"{brand_key}_manifest.json"
        records: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as err:
                logger.error(f"Error loading manifest {manifest_path}: {err}")

        # Fallback to scanning dataset folder if manifest is empty
        brand_src_dir = self.dataset_manager.dataset_root / brand_key
        if not records and brand_src_dir.exists():
            for img_file in brand_src_dir.glob("*.*"):
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    records[img_file.name] = {
                        "image_path": f"{brand_key}/{img_file.name}",
                        "prompt": f"A high-fidelity fashion photo of a {brand_key} design."
                    }

        # Create dummy image in dry run/test scenario if absolutely no files exist
        if not records:
            logger.warning(f"No source images found for brand '{brand_key}' dataset. Creating dummy file.")
            brand_src_dir.mkdir(parents=True, exist_ok=True)
            dummy_img_name = f"dummy_{brand_key}.png"
            dummy_src_path = brand_src_dir / dummy_img_name
            
            img = Image.new("RGB", (512, 512), color=(0, 255, 0))
            img.save(dummy_src_path)
            
            records[dummy_img_name] = {
                "image_path": f"{brand_key}/{dummy_img_name}",
                "prompt": f"A default brand {brand_key} outfit design."
            }

        # 2. Copy images and write sibling text captions
        for filename, record in records.items():
            src_path = self.dataset_manager.dataset_root / record["image_path"]
            if not src_path.exists():
                logger.warning(f"Source file {src_path} listed in manifest but missing on disk.")
                continue

            # Save/Copy to Kohya folder
            dest_image_path = img_dir / filename
            shutil.copy2(src_path, dest_image_path)

            # Create caption .txt file
            prompt = record.get("prompt", f"A beautiful design by {brand_key}.")
            caption_path = img_dir / f"{dest_image_path.stem}.txt"
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(prompt)

        logger.success(f"Prepared Kohya dataset folder structure for '{brand_key}' at: {img_dir}")
        return brand_dir

    def generate_config(
        self,
        brand: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Generate Kohya training configuration parameters.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        output_path : Path or str, optional
            Destination JSON file path (default: <pipeline_root>/<brand>/training_config.json).

        Returns
        -------
        dict
            Configuration dictionary.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.SUPPORTED_BRANDS:
            raise ValueError(f"Brand '{brand}' not supported. Choose from: {self.SUPPORTED_BRANDS}")

        brand_dir = self.pipeline_root / brand_key
        train_data_dir = brand_dir / "img"
        output_dir = brand_dir / "model"
        logging_dir = brand_dir / "log"

        # Create output folders
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        # Default values matching typical Kohya arguments
        defaults = {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "train_data_dir": str(train_data_dir.as_posix()),
            "output_dir": str(output_dir.as_posix()),
            "logging_dir": str(logging_dir.as_posix()),
            "resolution": "1024,1024",
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "network_module": "networks.lora",
            "network_dim": 8,
            "network_alpha": 16,
            "network_dropout": 0.05,
            "train_batch_size": 1,
            "learning_rate": 1e-4,
            "unet_lr": 1e-4,
            "text_encoder_lr": 5e-5,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "max_train_epochs": 10,
            "save_every_n_epochs": 1,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 4,
            "output_name": f"{brand_key}_style_lora"
        }

        # Override defaults if config is supplied
        if self.config:
            # Sync with LoraConfig
            if hasattr(self.config, "lora"):
                lc = self.config.lora
                defaults["network_dim"] = getattr(lc, "r", defaults["network_dim"])
                defaults["network_alpha"] = getattr(lc, "alpha", defaults["network_alpha"])
                defaults["network_dropout"] = getattr(lc, "dropout", defaults["network_dropout"])
            
            # Sync with DatasetConfig
            if hasattr(self.config, "dataset"):
                dc = self.config.dataset
                res = getattr(dc, "resolution", 1024)
                defaults["resolution"] = f"{res},{res}"
            
            # Sync with TrainerConfig
            if hasattr(self.config, "trainer"):
                tc = self.config.trainer
                defaults["train_batch_size"] = getattr(tc, "batch_size", defaults["train_batch_size"])
                defaults["learning_rate"] = getattr(tc, "learning_rate", defaults["learning_rate"])
                defaults["unet_lr"] = getattr(tc, "learning_rate", defaults["unet_lr"])
                defaults["max_train_epochs"] = getattr(tc, "num_epochs", defaults["max_train_epochs"])
                defaults["mixed_precision"] = getattr(tc, "mixed_precision", defaults["mixed_precision"])
                defaults["gradient_accumulation_steps"] = getattr(tc, "gradient_accumulation_steps", defaults["gradient_accumulation_steps"])

        # Save config file
        out_json_path = Path(output_path or brand_dir / "training_config.json").resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2, sort_keys=True)
        
        logger.success(f"Saved Kohya training configuration JSON to: {out_json_path}")
        return defaults

    def run_training(
        self,
        brand: str,
        kohya_script_path: Optional[str] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Automate the Kohya LoRA training subprocess.

        Parameters
        ----------
        brand : str
            Target brand to train (nike, gucci, zara, h&m).
        kohya_script_path : str, optional
            Path to the train_network.py script (default: sdxl_train_network.py).
        dry_run : bool
            If True, simulates execution without executing python subprocesses.

        Returns
        -------
        dict
            Status of training execution containing success flag and CLI commands.
        """
        brand_key = brand.lower().strip()
        
        # 1. Dataset Preparation
        self.prepare_dataset(brand_key)
        
        # 2. Config Generation
        config = self.generate_config(brand_key)
        
        # Resolve script path
        script = kohya_script_path or "sdxl_train_network.py"
        
        # 3. Build Command Line Arguments
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process=1",
            script,
            f"--pretrained_model_name_or_path={config['pretrained_model_name_or_path']}",
            f"--train_data_dir={config['train_data_dir']}",
            f"--output_dir={config['output_dir']}",
            f"--logging_dir={config['logging_dir']}",
            f"--resolution={config['resolution']}",
            f"--min_bucket_reso={config['min_bucket_reso']}",
            f"--max_bucket_reso={config['max_bucket_reso']}",
            f"--network_module={config['network_module']}",
            f"--network_dim={config['network_dim']}",
            f"--network_alpha={config['network_alpha']}",
            f"--network_dropout={config['network_dropout']}",
            f"--train_batch_size={config['train_batch_size']}",
            f"--learning_rate={config['learning_rate']}",
            f"--unet_lr={config['unet_lr']}",
            f"--text_encoder_lr={config['text_encoder_lr']}",
            f"--lr_scheduler={config['lr_scheduler']}",
            f"--lr_warmup_steps={config['lr_warmup_steps']}",
            f"--max_train_epochs={config['max_train_epochs']}",
            f"--save_every_n_epochs={config['save_every_n_epochs']}",
            f"--mixed_precision={config['mixed_precision']}",
            f"--gradient_accumulation_steps={config['gradient_accumulation_steps']}",
            f"--output_name={config['output_name']}"
        ]
        
        if config["enable_bucket"]:
            cmd.append("--enable_bucket")

        logger.info(f"Assembled training command: {' '.join(cmd)}")

        output_model_file = Path(config["output_dir"]) / f"{config['output_name']}.safetensors"

        if dry_run:
            logger.info("DRY-RUN mode active. Simulating Kohya process execution...")
            # Create mock weights file
            output_model_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_model_file, "wb") as f:
                f.write(b"MOCK_LORA_WEIGHTS_KOHYA_SS_WEEK4")
            
            return {
                "success": True,
                "command": cmd,
                "output_model": str(output_model_file.as_posix()),
                "dry_run": True
            }
        else:
            logger.info("Launching actual subprocess execution for Kohya_SS...")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.success(f"Kohya_SS training subprocess completed successfully.")
                if not output_model_file.exists():
                    raise FileNotFoundError(
                        f"Kohya_SS training completed but expected output weights file is missing: {output_model_file}"
                    )
                return {
                    "success": True,
                    "command": cmd,
                    "output_model": str(output_model_file.as_posix()),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "dry_run": False
                }
            except (subprocess.CalledProcessError, FileNotFoundError, Exception) as err:
                logger.error(f"Kohya_SS training subprocess failed: {err}")
                raise RuntimeError(
                    f"Real Kohya training run requested but execution failed: {err}"
                ) from err
