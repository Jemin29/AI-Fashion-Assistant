"""
week4/evaluation/lora_tracker.py
================================
LoRA Experiment Tracker.
Logs and queries style adapter training runs, hyperparameters, loss stats,
validation metrics (CLIP, SSIM), and provides comparison analytics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class LoraExperimentTracker:
    """
    Manages logging, indexing, and statistics queries for LoRA fine-tuning experiments.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the tracker.

        Parameters
        ----------
        output_dir : Path or str, optional
            Base directory to save logs (default: outputs/experiments).
        """
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = Path("outputs/experiments").resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.output_dir / "lora_experiment_registry.json"
        
        # Initialize registry file if not present
        if not self.registry_path.exists():
            self._save_registry({})
        
        logger.info(f"Initialized LoraExperimentTracker | registry={self.registry_path}")

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the registry index dictionary from disk."""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Failed to read registry index: {err}. Returning empty.")
            return {}

    def _save_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        """Persist the registry index dictionary to disk."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
        except Exception as err:
            logger.error(f"Failed to write registry index: {err}")

    def log_experiment(
        self,
        brand: str,
        lora_version: str,
        training_loss: float,
        validation_score: float,
        clip_score: float,
        style_similarity: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an experiment run to disk and central registry index.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        lora_version : str
            LoRA identifier version/tag (e.g. "v1.0", "epoch-10").
        training_loss : float
        validation_score : float
        clip_score : float
        style_similarity : float
        parameters : dict, optional
            Hyperparameters such as rank, alpha, learning_rate.

        Returns
        -------
        dict
            Logged run metadata details.
        """
        brand_key = brand.lower().strip()
        import uuid
        timestamp = int(time.time())
        unique_suffix = uuid.uuid4().hex[:8]
        experiment_id = f"lora_{brand_key}_{timestamp}_{unique_suffix}"

        run_data = {
            "experiment_id": experiment_id,
            "brand": brand_key,
            "lora_version": lora_version,
            "training_loss": round(training_loss, 6),
            "validation_score": round(validation_score, 4),
            "clip_score": round(clip_score, 4),
            "style_similarity": round(style_similarity, 4),
            "parameters": parameters or {},
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

        # 1. Save detailed run file
        run_file = self.output_dir / f"run_{experiment_id}.json"
        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, sort_keys=True)
            logger.debug(f"Saved run log to: {run_file}")
        except Exception as err:
            logger.error(f"Failed to write run log file: {err}")

        # 2. Update central registry index
        registry = self._load_registry()
        registry[experiment_id] = {
            "experiment_id": experiment_id,
            "brand": brand_key,
            "lora_version": lora_version,
            "training_loss": round(training_loss, 6),
            "validation_score": round(validation_score, 4),
            "clip_score": round(clip_score, 4),
            "style_similarity": round(style_similarity, 4),
            "timestamp": timestamp
        }
        self._save_registry(registry)

        logger.success(f"Logged LoRA experiment run | brand={brand_key} | id={experiment_id}")
        return run_data

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed experiment metrics.

        Parameters
        ----------
        experiment_id : str

        Returns
        -------
        dict, optional
        """
        run_file = self.output_dir / f"run_{experiment_id}.json"
        if not run_file.exists():
            return None
        
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.error(f"Failed to read detailed run file {run_file}: {err}")
            return None

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        List all indexed experiments.

        Returns
        -------
        list of dict
        """
        registry = self._load_registry()
        return list(registry.values())

    def get_best_run(self, metric: str, brand: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Query for the best run optimizing a target metric.

        Parameters
        ----------
        metric : str
            Metric key ("training_loss", "validation_score", "clip_score", "style_similarity").
            For "training_loss", lower is better. For others, higher is better.
        brand : str, optional
            Filter for a specific brand style.

        Returns
        -------
        dict, optional
            Summary of the best run registry metadata.
        """
        runs = self.get_all_experiments()
        if not runs:
            return None

        # Filter by brand if requested
        if brand:
            brand_key = brand.lower().strip()
            runs = [r for r in runs if r["brand"] == brand_key]

        if not runs:
            return None

        if metric not in ["training_loss", "validation_score", "clip_score", "style_similarity"]:
            raise ValueError(f"Unsupported metric optimization target: {metric}")

        # Optimize (minimize loss, maximize score metrics)
        if metric == "training_loss":
            best_run = min(runs, key=lambda r: r.get("training_loss", float("inf")))
        else:
            best_run = max(runs, key=lambda r: r.get(metric, -1.0))

        return best_run

    def get_statistics(self) -> Dict[str, Any]:
        """
        Aggregate run statistics grouped globally and by brand.

        Returns
        -------
        dict
        """
        runs = self.get_all_experiments()
        stats: Dict[str, Any] = {
            "total_runs": len(runs),
            "by_brand": {}
        }

        if not runs:
            return stats

        # Helper to compute stats for a list of runs
        def compute_metrics_summary(runs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            losses = [r["training_loss"] for r in runs_list]
            val_scores = [r["validation_score"] for r in runs_list]
            clips = [r["clip_score"] for r in runs_list]
            styles = [r["style_similarity"] for r in runs_list]
            
            count = len(runs_list)
            return {
                "count": count,
                "mean_training_loss": round(sum(losses) / count, 6),
                "mean_validation_score": round(sum(val_scores) / count, 4),
                "mean_clip_score": round(sum(clips) / count, 4),
                "mean_style_similarity": round(sum(styles) / count, 4),
            }

        stats["global"] = compute_metrics_summary(runs)

        # Brand specific aggregation
        brand_groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in runs:
            brand = r["brand"]
            brand_groups.setdefault(brand, []).append(r)

        for brand, brand_runs in brand_groups.items():
            stats["by_brand"][brand] = compute_metrics_summary(brand_runs)

        return stats
