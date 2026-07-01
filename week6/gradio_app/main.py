"""
Week 6 — AI Fashion Creative Studio Main Application.
Constructs the Gradio app blocks interface, binds services, and applies custom themes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import gradio as gr

# Add repository root to system path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Logger and Config
from week6.gradio_app.logger import setup_logging, get_logger
from week6.gradio_app.config import get_config

# Themes and CSS
from week6.themes.fashion_theme import FashionTheme

# Reusable Components
from week6.components.header import build_header
from week6.components.status_bar import build_status_bar

# Pages
from week6.pages import (
    build_home_page,
    build_text_to_fashion_page,
    build_sketch_to_design_page,
    build_fashion_assistant_page,
    build_brand_studio_page,
    build_fashion_qa_page,
    build_trend_explorer_page,
    build_recommendations_page,
    build_gallery_page,
    build_eval_dashboard_page,
)

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

logger = get_logger(__name__)


def create_app() -> gr.Blocks:
    """
    App Factory that loads configurations, binds services,
    and returns the fully built gr.Blocks instance.
    """
    # 1. Setup logging & configurations
    setup_logging()
    cfg = get_config()
    logger.info(f"Building {cfg.name} (v{cfg.version})")

    # 2. Determine mock mode per service
    global_mock = cfg.mock.global_mock
    
    mock_gen = global_mock or cfg.mock.generation
    mock_cn = global_mock or cfg.mock.controlnet
    mock_lora = global_mock or cfg.mock.lora
    mock_rag = global_mock or cfg.mock.rag
    mock_rec = global_mock or cfg.mock.recommendations
    mock_trends = global_mock or cfg.mock.trends

    logger.info(
        f"Service mock modes: global={global_mock} | gen={mock_gen} | cn={mock_cn} | "
        f"lora={mock_lora} | rag={mock_rag} | rec={mock_rec} | trends={mock_trends}"
    )

    # 3. Instantiate Services
    gen_service = GenerationService(mock_mode=mock_gen)
    cn_service = ControlNetService(mock_mode=mock_cn)
    lora_service = LoRAService(mock_mode=mock_lora)
    rag_service = RAGService(mock_mode=mock_rag)
    rec_service = RecommendationService(mock_mode=mock_rec)
    trend_service = TrendService(mock_mode=mock_trends)
    eval_service = EvaluationService(mock_mode=global_mock)

    # 4. Load Custom styling (studio.css)
    css_content = ""
    css_path = Path(cfg.paths.assets_dir) / "css" / "studio.css"
    if css_path.exists():
        try:
            css_content = css_path.read_text(encoding="utf-8")
            logger.info("Custom global CSS stylesheet loaded successfully")
        except Exception as e:
            logger.error(f"Failed to read studio.css: {e}")
    else:
        logger.warning(f"Custom CSS stylesheet not found at: {css_path}")

    # 5. Build gr.Blocks layout
    theme = FashionTheme()
    
    with gr.Blocks(theme=theme, css=css_content, title=cfg.name) as app:
        # Header hero component
        build_header()

        # Tabs Layout
        with gr.Tabs() as tabs:
            if cfg.features.home_page:
                with gr.TabItem("🏠 Home", id="home"):
                    build_home_page()
            
            if cfg.features.style_studio:
                with gr.TabItem("🎨 Text-to-Fashion", id="style_studio"):
                    build_text_to_fashion_page(gen_service)
            
            if cfg.features.controlnet_studio:
                with gr.TabItem("✏️ Sketch2Design", id="controlnet"):
                    build_sketch_to_design_page(cn_service)
            
            if cfg.features.brand_studio:
                with gr.TabItem("🏷️ Brand Studio", id="brand_studio"):
                    build_brand_studio_page(lora_service)
            
            if cfg.features.fashion_qa:
                with gr.TabItem("💬 Fashion Assistant", id="fashion_qa"):
                    build_fashion_assistant_page(rag_service)
            
            if cfg.features.trend_explorer:
                with gr.TabItem("📈 Trend Explorer", id="trend_explorer"):
                    build_trend_explorer_page(trend_service)
            
            if cfg.features.recommend_hub:
                with gr.TabItem("👗 Recommend Hub", id="recommend_hub"):
                    build_recommendations_page(rec_service, trend_service)
            
            with gr.TabItem("🖼️ Gallery", id="gallery"):
                build_gallery_page()
            
            if cfg.features.eval_dashboard:
                with gr.TabItem("📊 Eval Dashboard", id="eval_dashboard"):
                    build_eval_dashboard_page(eval_service)

        # Footer Status Bar
        build_status_bar(mock_mode=global_mock)

    return app
