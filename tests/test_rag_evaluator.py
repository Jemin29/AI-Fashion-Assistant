"""
week5/tests/test_rag_evaluator.py
=================================
Unit tests for the RAG Evaluator module.
Verifies metrics correctness, calculation edge cases, and report persistence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.rag_evaluator import RAGEvaluator, calculate_cosine_similarity


class TestRAGEvaluator:
    """Validate mathematical helpers, metrics calculations, evaluation pipeline, and report generation."""

    def test_cosine_similarity_calculation(self):
        """Verify vector cosine similarity calculates correctly under multiple conditions."""
        # 1. Equal vectors
        assert calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert calculate_cosine_similarity([0.0, 1.0], [0.0, 1.0]) == pytest.approx(1.0)

        # 2. Opposite vectors
        assert calculate_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

        # 3. Orthogonal vectors
        assert calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

        # 4. Zero/invalid vectors
        assert calculate_cosine_similarity([], [1.0]) == 0.0
        assert calculate_cosine_similarity([1.0], []) == 0.0
        assert calculate_cosine_similarity([1.0], [1.0, 2.0]) == 0.0
        assert calculate_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_evaluator_metrics_calculations(self):
        """Verify metric evaluation functions (retrieval, recommendation, response quality)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = RAGEvaluator(report_path=Path(tmpdir) / "test_report.json")

            # 1. Retrieval Accuracy
            # Expected hit
            res = evaluator.evaluate_retrieval(
                retrieved_items=[{"id": "doc1"}, {"id": "doc2"}],
                expected_ids=["doc2"]
            )
            assert res["hit_rate"] == 1.0
            assert res["mrr"] == 0.5  # Rank 2

            # Expected miss
            res_miss = evaluator.evaluate_retrieval(
                retrieved_items=[{"id": "doc1"}, {"id": "doc2"}],
                expected_ids=["doc3"]
            )
            assert res_miss["hit_rate"] == 0.0
            assert res_miss["mrr"] == 0.0

            # Empty expected matches
            assert evaluator.evaluate_retrieval([], []) == {"hit_rate": 1.0, "mrr": 1.0}
            assert evaluator.evaluate_retrieval([], ["doc1"]) == {"hit_rate": 0.0, "mrr": 0.0}

            # 2. Recommendation Relevance
            rec_res = evaluator.evaluate_recommendations(
                recommendations=["Nike", "Supreme"],
                expected_keywords=["nike", "adidas"]
            )
            # recs = {"nike", "supreme"}, keywords = {"nike", "adidas"}
            # intersection = {"nike"} (size 1)
            # union = {"nike", "supreme", "adidas"} (size 3)
            # Jaccard = 1/3 = 0.3333
            assert rec_res["relevance_score"] == pytest.approx(1.0 / 3.0)

            # Empty recommendation relevance check
            assert evaluator.evaluate_recommendations([], []) == {"relevance_score": 1.0}
            assert evaluator.evaluate_recommendations([], ["tag"]) == {"relevance_score": 0.0}

            # 3. Response Quality (Grounding & Citations)
            # Perfectly grounded
            q_res = evaluator.evaluate_response_quality(
                response_text="According to records [doc1], this is true [doc2].",
                retrieved_items=[{"id": "doc1"}, {"id": "doc2"}]
            )
            assert q_res["grounding_score"] == 1.0
            assert q_res["citation_accuracy"] == 1.0

            # Partially grounded
            q_res_part = evaluator.evaluate_response_quality(
                response_text="According to records [doc1], this is true [doc3].",
                retrieved_items=[{"id": "doc1"}, {"id": "doc2"}]
            )
            assert q_res_part["grounding_score"] == 0.5
            assert q_res_part["citation_accuracy"] == 1.0

            # Missing citations entirely
            q_res_none = evaluator.evaluate_response_quality(
                response_text="No citations here.",
                retrieved_items=[{"id": "doc1"}]
            )
            assert q_res_none["grounding_score"] == 0.0
            assert q_res_none["citation_accuracy"] == 0.0

            # Grounded fallback when empty retrieved items
            assert evaluator.evaluate_response_quality("No citations.", []) == {"grounding_score": 1.0, "citation_accuracy": 1.0}

    def test_run_evaluation_pipeline_and_json_persistence(self):
        """Verify the evaluator runs successfully over test cases and saves JSON reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "evaluation_report.json"

            evaluator = RAGEvaluator(report_path=report_path)

            # Define a small test suite
            test_cases = [
                {
                    "query": "Tell me about streetwear style fits and key items",
                    "category": "style_advice",
                    "expected_retrieval_ids": ["kb_fashion_styles_streetwear"],
                    "expected_recommendation_keywords": ["streetwear", "hoodie"],
                    "user_preferences": {"favorite_style": "streetwear"}
                }
            ]

            report = evaluator.run_evaluation(test_cases=test_cases)

            # Assert keys in output structure
            assert "summary" in report
            assert "test_cases" in report
            assert report["summary"]["total_queries"] == 1
            assert "average_latency_seconds" in report["summary"]
            assert "average_retrieval_hit_rate" in report["summary"]
            assert "average_retrieval_mrr" in report["summary"]
            assert "average_recommendation_relevance" in report["summary"]
            assert "average_grounding_score" in report["summary"]
            assert "average_semantic_similarity_query_response" in report["summary"]
            assert "average_semantic_similarity_context_response" in report["summary"]

            # Assert report file exists and matches output
            assert report_path.exists()
            with open(report_path, "r", encoding="utf-8") as f:
                saved_report = json.load(f)
            assert saved_report["summary"]["total_queries"] == 1
            assert len(saved_report["test_cases"]) == 1
            assert saved_report["test_cases"][0]["query"] == test_cases[0]["query"]
