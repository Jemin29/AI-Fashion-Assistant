"""
Week 6 — Home Page Dashboard.
Displays project overview, AI features, system status badges, stats cards, and navigation shortcuts.
"""
from __future__ import annotations
import glob
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr


def _check_module(import_path: str) -> bool:
    """Check if a module can be imported, returning True if successful."""
    try:
        import importlib
        importlib.import_module(import_path)
        return True
    except Exception:
        return False


def _get_recent_generations() -> List[str]:
    """Retrieve up to 4 recent images from the outputs directories."""
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    gen_dir = out_dir / "generated"
    sketch_dir = out_dir / "sketches"

    files = []
    if gen_dir.exists():
        files.extend(glob.glob(str(gen_dir / "*.png")))
    if sketch_dir.exists():
        files.extend(glob.glob(str(sketch_dir / "*.png")))

    # Sort by modification time descending
    files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return files[:4]


def _build_stats_html() -> str:
    """Build premium statistic cards using CSS grid."""
    return """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase;">Domain KB</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0; color: #ff9f43;">556 Pairs</div>
            <div style="font-size: 0.7rem; color: #555;">Seeded QA</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase;">Active LoRAs</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0; color: #3498db;">4 Brands</div>
            <div style="font-size: 0.7rem; color: #555;">Nike / Gucci / Zara / HM</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase;">Active Trends</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0; color: #2ecc71;">12 Active</div>
            <div style="font-size: 0.7rem; color: #555;">Velocity Tracked</div>
        </div>
        <div class="metric-card" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase;">Embeddings</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0; color: #9b59b6;">384 Dim</div>
            <div style="font-size: 0.7rem; color: #555;">Dense Vector Index</div>
        </div>
    </div>
    """


def _build_model_status_html() -> str:
    """Build status indicators for AI models and libraries."""
    cuda = _check_module("torch")
    if cuda:
        import torch
        gpu_ok = torch.cuda.is_available()
    else:
        gpu_ok = False

    sdxl_ok = _check_module("diffusers")
    cn_ok = _check_module("src.controlnet")
    peft_ok = _check_module("peft")
    chroma_ok = _check_module("chromadb")

    def _badge(ok: bool) -> str:
        return "<span style='color: #2ecc71; font-weight: bold;'>🟢 ACTIVE</span>" if ok else "<span style='color: #e74c3c; font-weight: bold;'>🔴 MOCK</span>"

    def _gpu_badge(ok: bool) -> str:
        return "<span style='color: #2ecc71; font-weight: bold;'>🟢 AVAILABLE</span>" if ok else "<span style='color: #f1c40f; font-weight: bold;'>🟡 CPU ONLY</span>"

    return f"""
    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.2rem; border-radius: 8px;">
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.6rem; margin-bottom: 0.6rem;">
            <span>🖥️ GPU Inference:</span>
            <span>{_gpu_badge(gpu_ok)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.6rem; margin-bottom: 0.6rem;">
            <span>🎨 SDXL Base Generator:</span>
            <span>{_badge(sdxl_ok)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.6rem; margin-bottom: 0.6rem;">
            <span>✏️ ControlNet Adapters:</span>
            <span>{_badge(cn_ok)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.6rem; margin-bottom: 0.6rem;">
            <span>🏷️ PEFT LoRA Brand Styles:</span>
            <span>{_badge(peft_ok)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding-bottom: 0.2rem;">
            <span>🗄️ ChromaDB RAG Store:</span>
            <span>{_badge(chroma_ok)}</span>
        </div>
    </div>
    """


def build_home_page(nav_component: Optional[gr.Radio] = None) -> None:
    """Build the Home Dashboard layout tab."""
    gr.Markdown("## 🏠 Home Dashboard")
    gr.Markdown(
        "Welcome to the AI Fashion Creative Studio dashboard. View stats, model configurations, and navigate shortcuts.",
        elem_classes="studio-subtitle",
    )

    with gr.Row():
        # ── Left Column: Project Overview & Features ─────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("""
            ### 📖 Project Overview
            The **AI-Powered Fashion Design Assistant** is an end-to-end creative space built to assist designers. 
            It integrates state-of-the-art generative pipelines with structured fashion research data. 
            """)
            
            gr.Markdown("""
            ### ✨ Key AI Capabilities
            - **🎨 Text-to-Fashion (SDXL)**: Turn complex descriptions into editorial fashion photography.
            - **✏️ Sketch2Design (ControlNet)**: Convert layout sketches, body poses, or depth details to fully rendered outfits.
            - **🏷️ Brand Styling (LoRA)**: Infuse brand aesthetics from fine-tuned adapters (Nike, Gucci, Zara, H&M).
            - **💬 RAG Assistant**: Consult design guidelines, fabric properties, and trends via conversational search.
            """)

        # ── Right Column: Statistics & Model Status ──────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Studio Statistics")
            gr.HTML(_build_stats_html())

            gr.Markdown("### ⚙️ System Status")
            status_html = gr.HTML(_build_model_status_html())
            
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            def refresh_status():
                return _build_model_status_html()
            
            refresh_btn.click(refresh_status, outputs=[status_html])

    gr.Markdown("---")

    # ── Quick Actions / Navigation Shortcuts Row ──────────────────────────────
    gr.Markdown("### ⚡ Quick Actions")
    with gr.Row():
        act1 = gr.Button("🎨 Start Generating Styles", variant="secondary")
        act2 = gr.Button("✏️ Render a Sketch", variant="secondary")
        act3 = gr.Button("💬 Ask Design Assistant", variant="secondary")
        act4 = gr.Button("👗 Get Recommendations", variant="secondary")

        # Map shortcut button clicks to update navigation bar if provided
        if nav_component is not None:
            act1.click(lambda: "🎨 Text-to-Fashion", outputs=[nav_component])
            act2.click(lambda: "✏️ Sketch2Design", outputs=[nav_component])
            act3.click(lambda: "💬 Fashion Assistant", outputs=[nav_component])
            act4.click(lambda: "👗 Recommendations", outputs=[nav_component])

    gr.Markdown("---")

    # ── Recent Generations Section ────────────────────────────────────────────
    gr.Markdown("### 🖼️ Recent Session Generations")
    recent_gallery = gr.Gallery(
        label="Recent outputs",
        value=_get_recent_generations(),
        columns=4,
        height=220,
    )

    # Reload gallery on load
    def refresh_recent():
        return _get_recent_generations()
        
    app_loader = gr.Button("🔄 Sync Gallery", size="sm")
    app_loader.click(refresh_recent, outputs=[recent_gallery])
