"""
week3/evaluation/controlnet_tracker.py
======================================
Experiment Tracking Framework.
AI-Powered Fashion Design Assistant — Week 3.

Logs and registers ControlNet generation hyperparameters (prompt, control type, scale, seed)
and evaluation metrics (CLIP, SSIM, generation latency).
Saves individual runs to outputs/experiments/ and maintains a central summary registry index.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class ControlNetExperimentTracker:
    """
    Logs, tracks, and analyzes hyperparameters and performance metrics
    across ControlNet experiment runs.
    """

    def __init__(
        self,
        config: Any = None,
        experiment_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Initialize the ControlNetExperimentTracker.

        Parameters
        ----------
        config : Week3Config, optional
        experiment_dir : Path, optional
            Base folder directory to save logs (defaults to week3/outputs/experiments).
        """
        self.config = config
        
        # Resolve target directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir).resolve()
        elif config and getattr(config, "output_dir", None):
            self.experiment_dir = Path(config.output_dir).resolve() / "experiments"
        else:
            self.experiment_dir = Path("week3/outputs/experiments").resolve()

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.experiment_dir / "experiment_registry.json"

        # Initialize registry file if it doesn't exist
        if not self.registry_path.exists():
            self._save_registry([])

    # ── Public APIs: Logging Runs ─────────────────────────────────────────────

    def log_experiment(
        self,
        prompt: str,
        control_type: str,
        control_weight: float,
        seed: int,
        clip_score: float,
        ssim_score: float,
        generation_time: float,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a single experiment run, write its JSON config, and update the central index.

        Returns
        -------
        str
            The unique experiment ID assigned to this run.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        experiment_id = f"exp_{date_str}_{unique_suffix}"

        # 1. Prepare run object
        run_data = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "hyperparameters": {
                "prompt": prompt,
                "control_type": control_type.lower(),
                "control_weight": float(control_weight),
                "seed": int(seed)
            },
            "metrics": {
                "clip_score": float(clip_score),
                "ssim_score": float(ssim_score),
                "generation_time": float(generation_time)
            },
            "metadata": extra_metadata or {}
        }

        # 2. Save individual JSON run file
        run_path = self.experiment_dir / f"run_{experiment_id}.json"
        try:
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved individual experiment run: {run_path}")
        except Exception as exc:
            logger.error(f"Failed to save experiment run file {run_path}: {exc}")
            raise IOError(f"Could not save run file: {exc}") from exc

        # 3. Append to central registry file
        self._append_to_registry(run_data)
        logger.info(f"Logged experiment successfully | id={experiment_id} | ssim={ssim_score:.4f} | clip={clip_score:.4f}")

        return experiment_id

    # ── Public APIs: Analytics and Queries ────────────────────────────────────

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve details of a specific logged run by loading its run file.
        """
        run_path = self.experiment_dir / f"run_{experiment_id}.json"
        if not run_path.exists():
            logger.warning(f"Experiment run not found: {experiment_id}")
            return None
        
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read run file {run_path}: {exc}")
            return None

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Returns all registered experiments from the registry index.
        """
        return self._load_registry()

    def get_best_run(
        self,
        metric: str = "ssim",
        control_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Finds and returns the run that achieved the highest score for the target metric.

        Parameters
        ----------
        metric : str
            Target evaluation metric: "ssim" | "clip" | "time" (finds min time if "time").
        control_type : str, optional
            Filter query by controlnet type (e.g. "sketch", "pose", "depth").
        """
        metric = metric.lower()
        if metric not in ("ssim", "ssim_score", "clip", "clip_score", "time", "generation_time"):
            raise ValueError(f"Unsupported query metric: '{metric}'")

        runs = self._load_registry()
        if not runs:
            return None

        # Filter by controlnet type if requested
        if control_type:
            c_type = control_type.lower()
            runs = [r for r in runs if r["hyperparameters"]["control_type"] == c_type]

        if not runs:
            return None

        # Resolve metric keys
        metric_map = {
            "ssim": "ssim_score",
            "ssim_score": "ssim_score",
            "clip": "clip_score",
            "clip_score": "clip_score",
            "time": "generation_time",
            "generation_time": "generation_time"
        }
        target_key = metric_map[metric]

        # Find best
        if target_key == "generation_time":
            # Best latency means minimum generation time
            best_run = min(runs, key=lambda x: x["metrics"][target_key])
        else:
            best_run = max(runs, key=lambda x: x["metrics"][target_key])

        return best_run

    def get_statistics(self) -> Dict[str, Any]:
        """
        Computes summary statistics aggregated by control_type.
        """
        runs = self._load_registry()
        if not runs:
            return {"total_runs": 0, "by_control_type": {}}

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in runs:
            c_type = r["hyperparameters"]["control_type"]
            by_type.setdefault(c_type, []).append(r)

        stats_by_type = {}
        for c_type, type_runs in by_type.items():
            clips = [r["metrics"]["clip_score"] for r in type_runs]
            ssims = [r["metrics"]["ssim_score"] for r in type_runs]
            times = [r["metrics"]["generation_time"] for r in type_runs]

            stats_by_type[c_type] = {
                "count": len(type_runs),
                "clip_mean": round(float(np.mean(clips)), 4),
                "clip_std": round(float(np.std(clips)), 4),
                "ssim_mean": round(float(np.mean(ssims)), 4),
                "ssim_std": round(float(np.std(ssims)), 4),
                "time_mean": round(float(np.mean(times)), 4),
                "time_std": round(float(np.std(times)), 4)
            }

        all_clips = [r["metrics"]["clip_score"] for r in runs]
        all_ssims = [r["metrics"]["ssim_score"] for r in runs]
        all_times = [r["metrics"]["generation_time"] for r in runs]

        return {
            "total_runs": len(runs),
            "global": {
                "clip_mean": round(float(np.mean(all_clips)), 4),
                "ssim_mean": round(float(np.mean(all_ssims)), 4),
                "time_mean": round(float(np.mean(all_times)), 4)
            },
            "by_control_type": stats_by_type
        }

    # ── Private File Operations ───────────────────────────────────────────────

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Safely reads the registry file from disk."""
        if not self.registry_path.exists():
            return []
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error(f"Failed to read central registry file: {exc}")
            return []

    def _save_registry(self, data: List[Dict[str, Any]]) -> None:
        """Safely writes the registry list back to disk."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Failed to write registry index to disk: {exc}")

    def _append_to_registry(self, run_data: Dict[str, Any]) -> None:
        """Appends a new run summary to the registry list."""
        # Clean record for index lookup (exclude large metadata dicts to keep registry compact)
        summary_record = {
            "experiment_id": run_data["experiment_id"],
            "timestamp": run_data["timestamp"],
            "hyperparameters": run_data["hyperparameters"],
            "metrics": run_data["metrics"]
        }
        
        # Load, append, save
        runs = self._load_registry()
        runs.append(summary_record)
        self._save_registry(runs)
