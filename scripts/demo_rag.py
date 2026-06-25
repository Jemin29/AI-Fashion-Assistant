"""
demo_rag.py
===========
Week 5 Demo Application.
Demonstrates:
  1. Style recommendation
  2. Brand recommendation
  3. Trend search
  4. Semantic search
  5. Fashion Q&A

Ties together FashionAssistant, RAGEvaluator, and ExperimentTracker.
Generates evaluation_report.json and experiment_runs.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# Set up path configurations
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.experiment_tracker import ExperimentTracker
from src.rag.fashion_assistant import FashionAssistant
from src.evaluation.rag_evaluator import RAGEvaluator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def print_header(title: str) -> None:
    """Print section headers."""
    if _RICH:
        console = Console()
        console.print(Panel(title, title="Week 5 RAG System Demo", border_style="magenta", expand=False))
    else:
        print("\n" + "=" * 80)
        print(f" {title.upper()} ")
        print("=" * 80)


def print_info(label: str, content: Any) -> None:
    """Print formatted info block."""
    if _RICH:
        console = Console()
        console.print(f"[bold cyan]{label}:[/bold cyan] {str(content)}")
    else:
        print(f"{label}: {str(content)}")


def run_demo() -> None:
    """Run RAG demonstrations, track experiments, and perform system evaluations."""
    print_header("Initializing Context-Aware Fashion Assistant")

    # Disable extensive debug logs for clean terminal view
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 1. Initialize Components
    assistant = FashionAssistant(force_mock_embeddings=True)
    tracker = ExperimentTracker()
    evaluator = RAGEvaluator(assistant=assistant)

    tracker.clear_logs()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scenario 1: Style Recommendation
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Scenario 1: Style Recommendation")
    query_1 = "I need streetwear style recommendations in black color"
    print_info("Query", query_1)
    
    start_time = time.time()
    res_1 = assistant.chat(message=query_1, user_id="demo_user_1")
    latency_1 = time.time() - start_time
    
    recs_1 = res_1.get("recommendations", res_1.get("data", {}).get("styles", []))
    print_info("Recommendations", recs_1)
    print_info("Response Text", res_1.get("response"))
    print_info("Latency", f"{latency_1:.4f} seconds")

    tracker.log_run(
        query=query_1,
        retrieved_documents=[],
        recommendation_quality=1.0 if recs_1 else 0.0,
        confidence_score=0.8,
        latency_seconds=latency_1,
        metadata={"scenario": "style_recommendation"}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scenario 2: Brand Recommendation
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Scenario 2: Brand Recommendation")
    query_2 = "Recommend brand profiles for a luxury wardrobe"
    print_info("Query", query_2)
    
    start_time = time.time()
    res_2 = assistant.chat(message=query_2, user_id="demo_user_2")
    latency_2 = time.time() - start_time
    
    recs_2 = res_2.get("brands", res_2.get("data", {}).get("brands", []))
    print_info("Recommended Brands", recs_2)
    print_info("Response Text", res_2.get("response"))
    print_info("Latency", f"{latency_2:.4f} seconds")

    tracker.log_run(
        query=query_2,
        retrieved_documents=[],
        recommendation_quality=1.0 if recs_2 else 0.0,
        confidence_score=0.95,
        latency_seconds=latency_2,
        metadata={"scenario": "brand_recommendation"}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scenario 3: Trend Search & Forecasting
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Scenario 3: Trend Search & Forecasting")
    query_3 = "Explain the augmented wearable projections trend forecast"
    print_info("Query", query_3)
    
    start_time = time.time()
    res_3 = assistant.chat(message=query_3, user_id="demo_user_3")
    latency_3 = time.time() - start_time
    
    print_info("Response Text", res_3.get("response"))
    print_info("Latency", f"{latency_3:.4f} seconds")

    tracker.log_run(
        query=query_3,
        retrieved_documents=[],
        recommendation_quality=0.9,
        confidence_score=0.85,
        latency_seconds=latency_3,
        metadata={"scenario": "trend_search"}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scenario 4: Semantic Search
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Scenario 4: Semantic Search")
    query_4 = "Suggest cotton, linen, or lightweight fabrics for spring season"
    print_info("Query", query_4)
    
    start_time = time.time()
    # Call retrieve directly on the retriever to show semantic hits
    hits = assistant.chroma_retriever.retrieve(query=query_4, collection_name="fashion_styles", n_results=3)
    latency_4 = time.time() - start_time
    
    retrieved_ids = [hit.get("id") for hit in hits]
    print_info("Matching Document IDs", retrieved_ids)
    for idx, hit in enumerate(hits, 1):
        print(f"  [{idx}] {hit.get('id')} | Distance: {hit.get('distance'):.4f}")
        print(f"      Text: {hit.get('document')[:100]}...")
    print_info("Latency", f"{latency_4:.4f} seconds")

    tracker.log_run(
        query=query_4,
        retrieved_documents=retrieved_ids,
        recommendation_quality=0.85,
        confidence_score=0.90,
        latency_seconds=latency_4,
        metadata={"scenario": "semantic_search"}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scenario 5: Fashion Q&A
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Scenario 5: Fashion Q&A")
    query_5 = "Explain what drape means in fashion terminology"
    print_info("Query", query_5)
    
    start_time = time.time()
    res_5 = assistant.chat(message=query_5, user_id="demo_user_5")
    latency_5 = time.time() - start_time
    
    citations = res_5.get("citations", [])
    print_info("Grounded Citations", citations)
    print_info("Response Text", res_5.get("response"))
    print_info("Latency", f"{latency_5:.4f} seconds")

    tracker.log_run(
        query=query_5,
        retrieved_documents=citations,
        recommendation_quality=1.0,
        confidence_score=0.95,
        latency_seconds=latency_5,
        metadata={"scenario": "fashion_qa"}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── System Evaluation & Reports Generation
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Generating Audit Reports")
    
    # 1. Run RAG evaluation suite
    eval_report = evaluator.run_evaluation()
    print_info("Evaluation Report Path", str(evaluator.report_path))
    
    # 2. Run tracker stats
    tracker_stats = tracker.get_stats()
    print_info("Experiment Tracking Log Path", str(tracker.log_path))

    # 3. Print dashboard summary
    print_header("System Performance Dashboard")
    if _RICH:
        console = Console()
        table = Table(title="RAG Evaluation & Tracking Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Experiment runs", str(tracker_stats["total_runs"]))
        table.add_row("Average retrieval hit rate", f"{eval_report['summary']['average_retrieval_hit_rate'] * 100:.1f}%")
        table.add_row("Average Mean Reciprocal Rank (MRR)", f"{eval_report['summary']['average_retrieval_mrr']:.4f}")
        table.add_row("Average grounding score", f"{eval_report['summary']['average_grounding_score'] * 100:.1f}%")
        table.add_row("Average recommendation relevance", f"{eval_report['summary']['average_recommendation_relevance'] * 100:.1f}%")
        table.add_row("Average latency", f"{tracker_stats['average_latency_seconds'] * 1000:.2f} ms")
        
        console.print(table)
    else:
        print(f"Total Experiment Runs: {tracker_stats['total_runs']}")
        print(f"Average Retrieval Hit Rate: {eval_report['summary']['average_retrieval_hit_rate'] * 100:.1f}%")
        print(f"Average MRR: {eval_report['summary']['average_retrieval_mrr']:.4f}")
        print(f"Average Grounding Score: {eval_report['summary']['average_grounding_score'] * 100:.1f}%")
        print(f"Average Latency: {tracker_stats['average_latency_seconds'] * 1000:.2f} ms")


if __name__ == "__main__":
    run_demo()
