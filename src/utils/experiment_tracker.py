"""
src/utils/experiment_tracker.py
===============================
Unified Experiment Tracking System.
Supports both Week 2/3/4 SQLite-based experiment tracking and Week 5 JSON-based RAG tracking.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

# Import SQLite-based tracker components from Week 2 evaluation tracker
from src.evaluation.week2_experiment_tracker import (
    ExperimentRecord,
    TRACKED_METRICS,
    CSV_COLUMNS,
    ExperimentTracker as SQLiteExperimentTracker
)


# =============================================================================
# ── Week 5 Experiment Run Model
# =============================================================================

class ExperimentRun(BaseModel):
    """Data model representing a single RAG pipeline experiment run."""
    run_id: str = Field(description="Unique identifier for the experiment run.")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Timestamp of the run."
    )
    query: str = Field(description="The user query processed during the run.")
    retrieved_documents: List[Any] = Field(
        default_factory=list,
        description="Details or IDs of the retrieved documents."
    )
    recommendation_quality: float = Field(
        description="Quantitative score evaluating recommendation relevance/quality."
    )
    confidence_score: float = Field(
        description="Pipeline confidence score of the generated response."
    )
    latency_seconds: float = Field(
        description="Latency duration of the retrieval and generation cycle in seconds."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional custom metadata associated with the run."
    )


# =============================================================================
# ── Unified Experiment Tracker
# =============================================================================

class ExperimentTracker(SQLiteExperimentTracker):
    """
    Unified Experiment Tracker supporting:
    - SQLite-based database tracking for ML image generation models (Week 2-4).
    - JSON-based RAG tracking for RAG and search pipelines (Week 5).
    """

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        experiments_dir: Optional[Union[str, Path]] = None,
        db_name: str = "experiments.db",
        auto_export: bool = False,
    ) -> None:
        """
        Initialize the Experiment Tracker.

        Parameters
        ----------
        log_path : str or Path, optional
            Path to the JSON log database for RAG runs.
        experiments_dir : str or Path, optional
            Directory for the SQLite database.
        db_name : str, optional
            Filename for the SQLite database.
        auto_export : bool, optional
            Auto-export SQLite records to JSON/CSV.
        """
        # Determine and initialize the JSON log path for RAG
        if log_path:
            self.log_path = Path(log_path).resolve()
        else:
            self.log_path = Path("outputs/experiment_runs.json").resolve()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: Dict[str, ExperimentRun] = {}
        self._load_logs()

        # Resolve experiments_dir for SQLite parent tracker
        sqlite_dir = experiments_dir
        if sqlite_dir is None:
            # Fallback to the log_path directory or default
            sqlite_dir = self.log_path.parent / "experiments"

        # Initialize SQLite parent class
        super().__init__(
            experiments_dir=sqlite_dir,
            db_name=db_name,
            auto_export=auto_export
        )

    # ── JSON/RAG logging operations ──────────────────────────────────────────

    def _load_logs(self) -> None:
        """Load experiment runs from the JSON log file."""
        if not self.log_path.exists():
            self.runs = {}
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.runs = {}
            for run_id, run_dict in data.items():
                try:
                    self.runs[run_id] = ExperimentRun(**run_dict)
                except Exception as err:
                    logger.error(f"Failed to parse experiment run '{run_id}': {err}")
            logger.info(f"Loaded {len(self.runs)} experiment runs from log: {self.log_path}")
        except Exception as err:
            logger.error(f"Failed to read experiment log file {self.log_path}: {err}. Starting empty.")
            self.runs = {}

    def _save_logs(self) -> None:
        """Serialize and save all runs to disk in JSON format."""
        try:
            data = {run_id: run.model_dump() for run_id, run in self.runs.items()}
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.debug(f"Saved {len(self.runs)} runs to log file: {self.log_path}")
        except Exception as err:
            logger.error(f"Failed to save experiment log file {self.log_path}: {err}")

    def log_run(
        self,
        query: str,
        retrieved_documents: List[Any],
        recommendation_quality: float,
        confidence_score: float,
        latency_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentRun:
        """Record a new RAG experiment run and persist it to disk."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        run = ExperimentRun(
            run_id=run_id,
            timestamp=timestamp,
            query=query,
            retrieved_documents=retrieved_documents,
            recommendation_quality=recommendation_quality,
            confidence_score=confidence_score,
            latency_seconds=latency_seconds,
            metadata=metadata or {}
        )

        self.runs[run_id] = run
        self._save_logs()
        logger.success(f"Logged experiment run | ID={run_id} | query='{query[:30]}'")
        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Retrieve a specific run by ID."""
        return self.runs.get(run_id)

    def list_runs(self) -> List[ExperimentRun]:
        """List all tracked experiment runs."""
        return list(self.runs.values())

    def clear_logs(self) -> None:
        """Clear all experiment runs from memory and disk."""
        self.runs = {}
        self._save_logs()
        logger.info(f"Cleared all experiment runs at: {self.log_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for JSON-based runs."""
        runs_list = self.list_runs()
        total_runs = len(runs_list)

        if total_runs == 0:
            return {
                "total_runs": 0,
                "average_latency_seconds": 0.0,
                "average_confidence_score": 0.0,
                "average_recommendation_quality": 0.0,
                "min_latency_seconds": 0.0,
                "max_latency_seconds": 0.0
            }

        latencies = [run.latency_seconds for run in runs_list]
        confidences = [run.confidence_score for run in runs_list]
        recs_qualities = [run.recommendation_quality for run in runs_list]

        return {
            "total_runs": total_runs,
            "average_latency_seconds": sum(latencies) / total_runs,
            "average_confidence_score": sum(confidences) / total_runs,
            "average_recommendation_quality": sum(recs_qualities) / total_runs,
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies)
        }

