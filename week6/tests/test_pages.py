"""
Week 6 Test Suite — UI Page Builders.
Verifies all page layouts render and compile without runtime block build errors.
"""
from __future__ import annotations
import pytest
import gradio as gr

from week6.pages import (
    build_home_page,
    build_style_studio_page,
    build_text_to_fashion_page,
    build_controlnet_page,
    build_sketch_to_design_page,
    build_style_switcher_page,
    build_style_mixer_page,
    build_fashion_assistant_page,
    build_brand_studio_page,
    build_fashion_qa_page,
    build_trend_explorer_page,
    build_recommend_hub_page,
    build_recommendations_page,
    build_gallery_page,
    build_eval_dashboard_page,
)


def test_pages_render(
    gen_service,
    cn_service,
    lora_service,
    rag_service,
    rec_service,
    trend_service,
    eval_service,
) -> None:
    """Instantiate and check all page builders inside gr.Blocks."""
    with gr.Blocks() as demo:
        # Home
        build_home_page()
        # Style Studio (Old)
        build_style_studio_page(gen_service)
        # Text-to-Fashion (New)
        build_text_to_fashion_page(gen_service)
        # ControlNet (Old)
        build_controlnet_page(cn_service)
        # Sketch2Design (New)
        build_sketch_to_design_page(cn_service)
        # Style Switcher (New)
        build_style_switcher_page(lora_service)
        # Style Mixer (New)
        build_style_mixer_page(lora_service)
        # Fashion Assistant (New)
        build_fashion_assistant_page(rag_service)
        # Brand Studio
        build_brand_studio_page(lora_service)
        # QA
        build_fashion_qa_page(rag_service)
        # Trends
        build_trend_explorer_page(trend_service)
        # Recommend Hub
        build_recommend_hub_page(rec_service)
        # Recommendations
        build_recommendations_page(rec_service, trend_service)
        # Gallery
        build_gallery_page()
        # Eval Dashboard
        build_eval_dashboard_page(eval_service)

    # Check that blocks were successfully created
    assert len(demo.blocks) > 0


def test_studio_app_render() -> None:
    """Test the main sidebar layout Gradio Blocks app rendering."""
    from week6.gradio_app.app import create_studio_app
    app = create_studio_app()
    assert len(app.blocks) > 0

