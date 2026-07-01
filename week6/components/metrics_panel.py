"""Week 6 Metrics Panel Component."""
from __future__ import annotations
from typing import Any, Dict
import gradio as gr


def build_metrics_panel(title: str, metrics: Dict[str, Any]) -> None:
    """Build a visual layout displaying a set of metrics."""
    gr.Markdown(f"### {title}")
    
    html_content = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;'>"
    for label, val in metrics.items():
        val_str = f"{val:.1%}" if isinstance(val, float) and val <= 1.0 else str(val)
        html_content += f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); padding: 1rem; border-radius: 6px; text-align: center;">
            <div style="font-size: 0.8rem; color: #888; text-transform: uppercase;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-top: 0.5rem; color: #ff9f43;">{val_str}</div>
        </div>
        """
    html_content += "</div>"
    
    gr.HTML(html_content)
