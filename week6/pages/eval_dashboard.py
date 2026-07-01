"""Week 6 — Evaluation Dashboard Page."""
from __future__ import annotations
from typing import Any, Dict, List
import gradio as gr
import time


def _build_metrics_cards(summary: Dict[str, Any]) -> str:
    """Render metrics as visual HTML cards for a premium look."""
    latency = summary.get("average_latency_seconds", 0) * 1000
    hit_rate = summary.get("average_retrieval_hit_rate", 0)
    mrr = summary.get("average_retrieval_mrr", 0)
    relevance = summary.get("average_recommendation_relevance", 0)
    grounding = summary.get("average_grounding_score", 0)
    sim_qr = summary.get("average_semantic_similarity_query_response", 0)

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Avg Latency</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #ff9f43;">{latency:.1f} ms</div>
            <div style="font-size: 0.75rem; color: #666;">Query execution speed</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Hit Rate @ K</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #2ecc71;">{hit_rate:.1%}</div>
            <div style="font-size: 0.75rem; color: #666;">Retrieval accuracy</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Mean RR (MRR)</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #3498db;">{mrr:.2f}</div>
            <div style="font-size: 0.75rem; color: #666;">Ranked retrieval quality</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Rec Relevance</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #9b59b6;">{relevance:.1%}</div>
            <div style="font-size: 0.75rem; color: #666;">Tag match overlap</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Grounding Score</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #1abc9c;">{grounding:.1%}</div>
            <div style="font-size: 0.75rem; color: #666;">Citation grounding</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;">Semantic Sim</div>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: #e74c3c;">{sim_qr:.2f}</div>
            <div style="font-size: 0.75rem; color: #666;">Query-Response cosine sim</div>
        </div>
    </div>
    """
    return html


def _build_cases_table(cases: List[Dict[str, Any]]) -> str:
    """Build a markdown table of test case details."""
    rows = []
    for c in cases:
        query = c.get("query", "")
        category = c.get("category", "")
        lat = c.get("latency_seconds", 0) * 1000
        metrics = c.get("metrics", {})
        hit = metrics.get("hit_rate", 0)
        mrr = metrics.get("mrr", 0)
        rel = metrics.get("relevance_score", 0)
        grd = metrics.get("grounding_score", 0)

        rows.append(
            f"| **{category.title()}** | `{query}` | {lat:.1f}ms | {hit:.0%} | {mrr:.2f} | {rel:.0%} | {grd:.0%} |"
        )

    return f"""
| Category | Query | Latency | Hit Rate | MRR | Rec Relevance | Grounding |
|:---|:---|:---|:---|:---|:---|:---|
{"".join(rows)}
"""


def build_eval_dashboard_page(eval_service: Any) -> None:
    """Build the Evaluation Dashboard tab."""
    gr.Markdown("## 📊 Evaluation Dashboard — RAG & Pipeline Benchmarks")
    gr.Markdown(
        "Monitor retrieval accuracy, latency, recommendation quality, and LLM grounding scores across our systems.",
        elem_classes="studio-subtitle",
    )

    last_eval = eval_service.get_last_report()
    summary_data = last_eval.get("summary", {})
    cases_data = last_eval.get("test_cases", [])

    with gr.Row():
        timestamp_label = gr.Markdown(f"**Last Run Timestamp**: `{last_eval.get('timestamp', 'Never')}`")
        run_btn = gr.Button("🔄 Run Evaluation Suite", variant="primary", size="sm")

    gr.Markdown("### 📈 Core Summary Metrics")
    metrics_cards_html = gr.HTML(_build_metrics_cards(summary_data))

    gr.Markdown("### 📋 Test Case Details")
    cases_table_md = gr.Markdown(_build_cases_table(cases_data))

    gr.Markdown("### 🔍 Full JSON Report")
    with gr.Accordion("Raw Report Output", open=False):
        raw_json = gr.JSON(last_eval)

    def trigger_evaluation():
        logger_name = "Evaluation Dashboard"
        # Run new evaluation
        new_report = eval_service.run_evaluation()
        new_summary = new_report.get("summary", {})
        new_cases = new_report.get("test_cases", [])
        return (
            f"**Last Run Timestamp**: `{new_report.get('timestamp', 'Never')}`",
            _build_metrics_cards(new_summary),
            _build_cases_table(new_cases),
            new_report
        )

    run_btn.click(
        trigger_evaluation,
        inputs=[],
        outputs=[timestamp_label, metrics_cards_html, cases_table_md, raw_json]
    )
