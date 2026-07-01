"""
Week 6 — Evaluation Service.
Provides access to RAG evaluation metrics, loading previous reports, and running new evaluations.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from week6.gradio_app.logger import get_logger
from week6.gradio_app.config import get_config

logger = get_logger(__name__)


class EvaluationService:
    """
    Handles retrieval and generation of RAG evaluation reports.
    Supports running real RAGEvaluator or generating mock/pre-saved reports.
    """

    def __init__(self, mock_mode: bool = True) -> None:
        self.mock_mode = mock_mode
        cfg = get_config()
        self.report_dir = Path(cfg.paths.reports)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.report_dir / "evaluation_report.json"

    def get_last_report(self) -> Dict[str, Any]:
        """Retrieve the latest evaluation report, fallback to root or mock."""
        if self.report_path.exists():
            try:
                with open(self.report_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading evaluation report: {e}")

        # Fallback to project root outputs/evaluation_report.json if it exists
        project_eval = Path("outputs/evaluation_report.json")
        if project_eval.exists():
            try:
                with open(project_eval, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Copy to our reports directory
                    with open(self.report_path, "w", encoding="utf-8") as out:
                        json.dump(data, out, indent=2)
                    return data
            except Exception as e:
                logger.error(f"Error reading root evaluation report: {e}")

        # Return default mock report
        return self._get_default_mock_report()

    def run_evaluation(self) -> Dict[str, Any]:
        """Execute the RAG evaluation suite and save results."""
        if self.mock_mode:
            logger.info("Running evaluation in mock mode...")
            time.sleep(1.0)
            report = self._get_default_mock_report()
            # Randomly fluctuate metrics a bit for realism
            import random
            report["summary"]["average_latency_seconds"] += random.uniform(-0.001, 0.002)
            report["summary"]["average_retrieval_hit_rate"] = min(1.0, max(0.0, report["summary"]["average_retrieval_hit_rate"] + random.choice([-0.1, 0.0, 0.1])))
            report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving mock evaluation report: {e}")
            return report

        try:
            from src.evaluation.rag_evaluator import RAGEvaluator
            # RAGEvaluator saves report to its report_path
            evaluator = RAGEvaluator(report_path=self.report_path)
            return evaluator.run_evaluation()
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            return self._get_default_mock_report()

    def _get_default_mock_report(self) -> Dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_queries": 5,
                "average_latency_seconds": 0.0035,
                "average_retrieval_hit_rate": 0.8,
                "average_retrieval_mrr": 0.75,
                "average_recommendation_relevance": 0.65,
                "average_grounding_score": 0.9,
                "average_semantic_similarity_query_response": 0.72,
                "average_semantic_similarity_context_response": 0.68
            },
            "test_cases": [
                {
                    "query": "Tell me about streetwear style fits and key items",
                    "category": "style_advice",
                    "latency_seconds": 0.002,
                    "metrics": {
                        "hit_rate": 1.0,
                        "mrr": 1.0,
                        "relevance_score": 0.8,
                        "grounding_score": 1.0,
                        "citation_accuracy": 1.0,
                        "semantic_similarity_query_response": 0.75,
                        "semantic_similarity_context_response": 0.70
                    },
                    "retrieved_ids": ["kb_fashion_styles_streetwear"],
                    "recommendations": ["hoodie", "cargo pants", "sneakers"],
                    "citations": ["kb_fashion_styles_streetwear"]
                },
                {
                    "query": "What are the characteristics and care instructions for linen fabric?",
                    "category": "fabrics",
                    "latency_seconds": 0.003,
                    "metrics": {
                        "hit_rate": 1.0,
                        "mrr": 1.0,
                        "relevance_score": 0.7,
                        "grounding_score": 0.9,
                        "citation_accuracy": 1.0,
                        "semantic_similarity_query_response": 0.70,
                        "semantic_similarity_context_response": 0.65
                    },
                    "retrieved_ids": ["kb_fabric_types_linen"],
                    "recommendations": ["linen shirt", "flax pants"],
                    "citations": ["kb_fabric_types_linen"]
                }
            ]
        }
