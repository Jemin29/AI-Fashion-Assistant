"""
week5/rag/rag_evaluator.py
==========================
Fashion RAG Evaluation System.
Measures retrieval accuracy, recommendation relevance, response quality, and
semantic similarity between components in the fashion assistant pipeline.
Generates outputs/evaluation_report.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.fashion_assistant import FashionAssistant


# =============================================================================
# ── Mathematical Helper
# =============================================================================

def calculate_cosine_similarity(v1: Any, v2: Any) -> float:
    """
    Calculate the cosine similarity between two dense vectors.

    Parameters
    ----------
    v1 : Any
    v2 : Any

    Returns
    -------
    float
    """
    if v1 is None or v2 is None:
        return 0.0
    try:
        len1 = len(v1)
        len2 = len(v2)
    except TypeError:
        return 0.0

    if len1 == 0 or len2 == 0 or len1 != len2:
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


# =============================================================================
# ── RAG Evaluator Class
# =============================================================================

class RAGEvaluator:
    """
    Evaluates retrieval accuracy, recommendation relevance, response quality,
    and semantic similarity metrics for the context-aware Fashion Assistant.
    """

    def __init__(
        self,
        assistant: Optional[FashionAssistant] = None,
        report_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the RAG Evaluator.

        Parameters
        ----------
        assistant : FashionAssistant, optional
            The assistant instance to evaluate. If None, instantiates a default.
        report_path : str or Path, optional
            Output path for the evaluation report. Defaults to outputs/evaluation_report.json.
        """
        # If assistant is not provided, initialize a mock-enabled fallback
        if assistant is None:
            logger.info("No assistant instance provided. Initializing default FashionAssistant in mock mode...")
            self.assistant = FashionAssistant(force_mock_embeddings=True)
        else:
            self.assistant = assistant

        if report_path:
            self.report_path = Path(report_path).resolve()
        else:
            self.report_path = Path("outputs/evaluation_report.json").resolve()

        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize default benchmark test cases
        self.default_test_cases = [
            {
                "query": "Tell me about streetwear style fits and key items",
                "category": "style_advice",
                "expected_retrieval_ids": ["kb_fashion_styles_streetwear"],
                "expected_recommendation_keywords": ["streetwear", "hoodie", "sneakers"],
                "user_preferences": {"favorite_style": "streetwear", "favorite_color": "black"}
            },
            {
                "query": "What are the characteristics and care instructions for linen fabric?",
                "category": "fabrics",
                "expected_retrieval_ids": ["kb_fabric_types_linen"],
                "expected_recommendation_keywords": ["linen", "flax", "breathable", "summer"],
                "user_preferences": {}
            },
            {
                "query": "Recommend some luxury brand options for a tailored suit",
                "category": "brands",
                "expected_retrieval_ids": ["kb_brand_profiles_gucci"],
                "expected_recommendation_keywords": ["gucci", "prada", "luxury", "tailoring"],
                "user_preferences": {"favorite_style": "luxury"}
            },
            {
                "query": "Explain the augmented wearable projections trend forecast",
                "category": "trend_advice",
                "expected_retrieval_ids": ["trend_forecast_augmented_wearable_projections"],
                "expected_recommendation_keywords": ["augmented", "wearable", "projection", "ar_tags"],
                "user_preferences": {}
            },
            {
                "query": "Explain what drape means in fashion terminology",
                "category": "fashion_terminology",
                "expected_retrieval_ids": ["kb_fashion_terminology_drape"],
                "expected_recommendation_keywords": ["drape", "fabric", "fall"],
                "user_preferences": {}
            }
        ]

    # ── Evaluation Calculations ──────────────────────────────────────────────

    def evaluate_retrieval(
        self,
        retrieved_items: List[Dict[str, Any]],
        expected_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate retrieval hit rate and Mean Reciprocal Rank (MRR).

        Parameters
        ----------
        retrieved_items : List[Dict[str, Any]]
        expected_ids : List[str]

        Returns
        -------
        Dict[str, float]
        """
        if not expected_ids:
            return {"hit_rate": 1.0, "mrr": 1.0}
        if not retrieved_items:
            return {"hit_rate": 0.0, "mrr": 0.0}

        retrieved_ids = [item.get("id", "") for item in retrieved_items]

        # Calculate Hit Rate @ K (K = len(retrieved_items))
        hit = 0.0
        for exp_id in expected_ids:
            if any(exp_id.lower() in rit.lower() for rit in retrieved_ids):
                hit = 1.0
                break

        # Calculate MRR
        mrr = 0.0
        for rank, rit in enumerate(retrieved_ids, 1):
            if any(exp_id.lower() in rit.lower() for exp_id in expected_ids):
                mrr = 1.0 / rank
                break

        return {"hit_rate": hit, "mrr": mrr}

    def evaluate_recommendations(
        self,
        recommendations: List[str],
        expected_keywords: List[str]
    ) -> Dict[str, float]:
        """
        Calculate recommendation relevance using keyword tag overlap.

        Parameters
        ----------
        recommendations : List[str]
        expected_keywords : List[str]

        Returns
        -------
        Dict[str, float]
        """
        if not expected_keywords:
            return {"relevance_score": 1.0}
        if not recommendations:
            return {"relevance_score": 0.0}

        # Flat clean representations
        clean_recs = set(" ".join(recommendations).lower().split())
        clean_keywords = set(k.lower() for k in expected_keywords)

        intersection = clean_recs & clean_keywords
        union = clean_recs | clean_keywords

        jaccard = len(intersection) / len(union) if union else 0.0
        return {"relevance_score": float(jaccard)}

    def evaluate_response_quality(
        self,
        response_text: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate grounding score and citation correctness of generated response text.

        Parameters
        ----------
        response_text : str
        retrieved_items : List[Dict[str, Any]]

        Returns
        -------
        Dict[str, float]
        """
        # Find citations like [kb_...] or [trend_...] or [item_...]
        citations = re.findall(r"\[([a-zA-Z0-9\_]+)\]", response_text)
        retrieved_ids = {item.get("id", "") for item in retrieved_items}

        if not citations:
            # If nothing was retrieved, response doesn't need citations to be grounded
            if not retrieved_items:
                return {"grounding_score": 1.0, "citation_accuracy": 1.0}
            else:
                # If info was retrieved but not cited, grounding is considered low/incomplete
                return {"grounding_score": 0.0, "citation_accuracy": 0.0}

        # Check citation alignment
        correct_citations = [cit for cit in citations if cit in retrieved_ids]
        grounding_score = len(correct_citations) / len(citations)

        citation_accuracy = 1.0 if len(correct_citations) > 0 else 0.0

        return {
            "grounding_score": float(grounding_score),
            "citation_accuracy": float(citation_accuracy)
        }

    def evaluate_semantic_similarity(
        self,
        query: str,
        response: str,
        retrieved_items: List[Dict[str, Any]],
        embedder: EmbeddingsGenerator
    ) -> Dict[str, float]:
        """
        Calculate query-response and context-response semantic similarity.

        Parameters
        ----------
        query : str
        response : str
        retrieved_items : List[Dict[str, Any]]
        embedder : EmbeddingsGenerator

        Returns
        -------
        Dict[str, float]
        """
        try:
            # 1. Embed query and response
            q_emb = embedder.embed_batch([query])[0]
            r_emb = embedder.embed_batch([response])[0]
            query_response_sim = calculate_cosine_similarity(q_emb, r_emb)

            # 2. Embed retrieved context
            context_text = " ".join([item.get("document", "") for item in retrieved_items])
            if context_text.strip():
                c_emb = embedder.embed_batch([context_text])[0]
                context_response_sim = calculate_cosine_similarity(c_emb, r_emb)
            else:
                context_response_sim = 0.0

            return {
                "semantic_similarity_query_response": query_response_sim,
                "semantic_similarity_context_response": context_response_sim
            }
        except Exception as err:
            logger.error(f"Semantic similarity evaluation failed: {err}")
            return {
                "semantic_similarity_query_response": 0.0,
                "semantic_similarity_context_response": 0.0
            }

    # ── Evaluation Pipeline Execution ────────────────────────────────────────

    def run_evaluation(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline over a series of test queries.
        Generates and saves the evaluation report.

        Parameters
        ----------
        test_cases : list of dict, optional
            Custom test cases. If None, defaults to built-ins.

        Returns
        -------
        Dict[str, Any]
            The evaluation summary report.
        """
        cases = test_cases if test_cases is not None else self.default_test_cases
        logger.info(f"Running RAG evaluation over {len(cases)} test cases...")

        report_cases = []
        total_latency = 0.0

        for idx, case in enumerate(cases, 1):
            query = case["query"]
            expected_ids = case.get("expected_retrieval_ids", [])
            expected_keywords = case.get("expected_recommendation_keywords", [])
            user_prefs = case.get("user_preferences", {})

            logger.info(f"Evaluating Case {idx}/{len(cases)}: '{query}'")
            start_time = time.time()

            # Execute assistant query (RAG pipeline)
            user_id = f"eval_user_{idx}"
            if user_prefs:
                # Seed preferences before running
                self.assistant.user_profile_manager.create_profile(
                    user_id=user_id,
                    favorite_styles=[user_prefs.get("favorite_style")] if "favorite_style" in user_prefs else None,
                    favorite_colors=[user_prefs.get("favorite_color")] if "favorite_color" in user_prefs else None
                )

            # Get generation result
            res = self.assistant.chat(message=query, user_id=user_id)
            latency = time.time() - start_time
            total_latency += latency

            response_text = res.get("response", "")
            
            # Extract retrieved documents
            # General Q&A returns structured retrieved items in "data" or root keys
            retrieved_items = []
            if "data" in res and isinstance(res["data"], dict):
                r_items = res["data"].get("retrieved_items", [])
                if r_items:
                    # Convert to standard dict mapping if list of objects
                    for item in r_items:
                        if hasattr(item, "id"):
                            retrieved_items.append({
                                "id": getattr(item, "id"),
                                "document": getattr(item, "content", getattr(item, "description", "")),
                                "metadata": getattr(item, "metadata", {})
                            })
                        elif isinstance(item, dict):
                            retrieved_items.append(item)
                
                # Check for recommendations inside data
                recs_raw = res["data"].get("brands", res["data"].get("styles", res["data"].get("recommendations", [])))
            else:
                recs_raw = res.get("brands", res.get("recommendations", []))

            # Normalize recommendations list of strings
            recommendations = []
            for r in recs_raw:
                if isinstance(r, dict):
                    recommendations.append(r.get("name", r.get("id", "")))
                elif isinstance(r, str):
                    recommendations.append(r)

            # 1. Evaluate Retrieval Accuracy
            retrieval_metrics = self.evaluate_retrieval(retrieved_items, expected_ids)

            # 2. Evaluate Recommendation Relevance
            rec_metrics = self.evaluate_recommendations(recommendations, expected_keywords)

            # 3. Evaluate Response Quality
            quality_metrics = self.evaluate_response_quality(response_text, retrieved_items)

            # 4. Evaluate Semantic Similarity
            sim_metrics = self.evaluate_semantic_similarity(
                query=query,
                response=response_text,
                retrieved_items=retrieved_items,
                embedder=self.assistant.embeddings_generator
            )

            # Compile test case logs
            case_report = {
                "query": query,
                "category": case.get("category", "general"),
                "latency_seconds": latency,
                "metrics": {
                    **retrieval_metrics,
                    **rec_metrics,
                    **quality_metrics,
                    **sim_metrics
                },
                "retrieved_ids": [item.get("id", "") for item in retrieved_items],
                "recommendations": recommendations,
                "citations": re.findall(r"\[([a-zA-Z0-9\_]+)\]", response_text)
            }
            report_cases.append(case_report)

        # Compute averages across all test cases
        num_cases = len(report_cases)
        avg_hit_rate = sum(c["metrics"]["hit_rate"] for c in report_cases) / num_cases
        avg_mrr = sum(c["metrics"]["mrr"] for c in report_cases) / num_cases
        avg_relevance = sum(c["metrics"]["relevance_score"] for c in report_cases) / num_cases
        avg_grounding = sum(c["metrics"]["grounding_score"] for c in report_cases) / num_cases
        avg_sim_qr = sum(c["metrics"]["semantic_similarity_query_response"] for c in report_cases) / num_cases
        avg_sim_cr = sum(c["metrics"]["semantic_similarity_context_response"] for c in report_cases) / num_cases

        summary_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "summary": {
                "total_queries": num_cases,
                "average_latency_seconds": total_latency / num_cases,
                "average_retrieval_hit_rate": avg_hit_rate,
                "average_retrieval_mrr": avg_mrr,
                "average_recommendation_relevance": avg_relevance,
                "average_grounding_score": avg_grounding,
                "average_semantic_similarity_query_response": avg_sim_qr,
                "average_semantic_similarity_context_response": avg_sim_cr
            },
            "test_cases": report_cases
        }

        # Write to evaluation_report.json
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=2)
            logger.success(f"Evaluation report successfully written to: {self.report_path}")
        except Exception as err:
            logger.error(f"Failed to write evaluation report file: {err}")

        return summary_report
