"""
Week 6 — Main Gradio Application.
Combines all pages, components, and services with a sidebar-based responsive layout.
"""
from __future__ import annotations
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

# Add repository root to system path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Logger and Config
from week6.gradio_app.config import get_config
from week6.gradio_app.logger import setup_logging, get_logger, log_access
from week6.themes.fashion_theme import FashionTheme

# Services
from week6.services import (
    GenerationService,
    ControlNetService,
    LoRAService,
    RAGService,
    RecommendationService,
    TrendService,
    EvaluationService,
)
from week6.pages.home import build_home_page
from week6.pages.text_to_fashion import build_text_to_fashion_page
from week6.pages.sketch_to_design import build_sketch_to_design_page
from week6.pages.style_switcher import build_style_switcher_page
from week6.pages.style_mixer import build_style_mixer_page
from week6.pages.fashion_assistant import build_fashion_assistant_page
from week6.pages.recommendations import build_recommendations_page
from week6.pages.gallery import build_gallery_page
from week6.pages.settings import build_settings_page

logger = get_logger(__name__)





def create_studio_app() -> gr.Blocks:
    """Build the multi-tab AI Fashion Studio application blocks."""
    setup_logging()
    cfg = get_config()
    logger.info("Initializing services for app launch...")

    # Instantiate services based on config
    global_mock = cfg.mock.global_mock
    gen_service = GenerationService(mock_mode=(global_mock or cfg.mock.generation))
    cn_service = ControlNetService(mock_mode=(global_mock or cfg.mock.controlnet))
    lora_service = LoRAService(mock_mode=(global_mock or cfg.mock.lora))
    rag_service = RAGService(mock_mode=(global_mock or cfg.mock.rag))
    rec_service = RecommendationService(mock_mode=(global_mock or cfg.mock.recommendations))
    trend_service = TrendService(mock_mode=(global_mock or cfg.mock.trends))
    eval_service = EvaluationService(mock_mode=global_mock)

    css_content = ""
    css_path = Path(cfg.paths.assets_dir) / "css" / "studio.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

    theme = FashionTheme()

    # Shared list of tab names
    tabs_list = [
        "🏠 Home Dashboard",
        "🎨 Text-to-Fashion",
        "✏️ Sketch2Design",
        "🏷️ Style Switching",
        "🎛️ Style Mixer",
        "💬 Fashion Assistant",
        "👗 Recommendations",
        "🖼️ Gallery",
        "⚙️ Settings",
    ]

    with gr.Blocks(theme=theme, css=css_content, title=cfg.name) as app:
        # Header banner html
        gr.HTML(f"""
        <div style="background: linear-gradient(135deg, #1e1e2f 0%, #0f0f15 100%); border-bottom: 2px solid #ff9f43; padding: 1.5rem; text-align: center; border-radius: 8px 8px 0 0; margin-bottom: 1.5rem;">
            <h1 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: 1px;">
                🎨 <span style="background: linear-gradient(90deg, #ff9f43, #ff5252); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Fashion Creative Studio</span>
            </h1>
            <p style="color: #a0a0b0; font-family: 'Inter', sans-serif; font-size: 0.95rem; margin: 0.4rem 0 0 0; font-weight: 300;">
                Production Creative Workspace — SDXL, ControlNet, LoRA & RAG Intelligence
            </p>
        </div>
        """)

        with gr.Row():
            # ── Sidebar Navigation Panel ──────────────────────────────────────
            with gr.Column(scale=1, min_width=240, elem_id="sidebar-nav-col"):
                gr.Markdown("### 🧭 Navigation")
                nav = gr.Radio(
                    choices=tabs_list,
                    value=tabs_list[0],
                    label="",
                    interactive=True,
                )
                
                gr.Markdown("---")
                # Live status block
                mode_lbl = "Mock Mode (CPU)" if global_mock else "Production (GPU)"
                gr.HTML(f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 6px; font-size: 0.85rem; color: #888;">
                    <div style="margin-bottom: 0.4rem;">🔋 Status: <span style="color: #2ecc71; font-weight: 600;">Active</span></div>
                    <div style="margin-bottom: 0.4rem;">⚡ Engine: <span style="color: #ff9f43;">{mode_lbl}</span></div>
                    <div>🏷️ Version: <span style="color: #3498db;">v{cfg.version}</span></div>
                </div>
                """)

            # ── Main Panel Content ────────────────────────────────────────────
            with gr.Column(scale=4, elem_id="main-panel-col"):

                # ── 1. HOME DASHBOARD ─────────────────────────────────────────
                with gr.Column(visible=True) as home_panel:
                    build_home_page(nav_component=nav)

                # ── 2. TEXT TO FASHION ────────────────────────────────────────
                with gr.Column(visible=False) as style_panel:
                    build_text_to_fashion_page(gen_service)

                # ── 3. SKETCH 2 DESIGN ────────────────────────────────────────
                with gr.Column(visible=False) as sketch_panel:
                    build_sketch_to_design_page(cn_service)

                # ── 4. STYLE SWITCHING ────────────────────────────────────────
                with gr.Column(visible=False) as brand_panel:
                    build_style_switcher_page(lora_service)

                # ── 5. STYLE MIXER ────────────────────────────────────────────
                with gr.Column(visible=False) as mix_panel:
                    build_style_mixer_page(lora_service)

                # ── 6. FASHION ASSISTANT ──────────────────────────────────────
                with gr.Column(visible=False) as assistant_panel:
                    build_fashion_assistant_page(rag_service)

                # ── 7. RECOMMENDATIONS ────────────────────────────────────────
                with gr.Column(visible=False) as rec_panel:
                    build_recommendations_page(rec_service, trend_service)

                # ── 8. GALLERY ────────────────────────────────────────────────
                with gr.Column(visible=False) as gallery_panel:
                    build_gallery_page()

                # ── 9. SETTINGS ───────────────────────────────────────────────
                with gr.Column(visible=False) as settings_panel:
                    build_settings_page()

        # ── Sidebar tab visibility update listener ────────────────────────────
        def update_tab_visibility(selected):
            log_access("tab_switch", f"user switched to {selected}")
            
            # Map index
            return [
                gr.update(visible=(selected == tabs_list[0])), # Home
                gr.update(visible=(selected == tabs_list[1])), # Style Studio (Text-to-Fashion)
                gr.update(visible=(selected == tabs_list[2])), # Sketch2Design
                gr.update(visible=(selected == tabs_list[3])), # Style Switching
                gr.update(visible=(selected == tabs_list[4])), # Style Mixer
                gr.update(visible=(selected == tabs_list[5])), # Fashion Assistant
                gr.update(visible=(selected == tabs_list[6])), # Recommendations
                gr.update(visible=(selected == tabs_list[7])), # Gallery
                gr.update(visible=(selected == tabs_list[8])), # Settings
            ]

        nav.change(
            update_tab_visibility,
            inputs=[nav],
            outputs=[
                home_panel,
                style_panel,
                sketch_panel,
                brand_panel,
                mix_panel,
                assistant_panel,
                rec_panel,
                gallery_panel,
                settings_panel,
            ]
        )



    return app
