"""
Week 6 Test Suite — UI Components.
Verifies header, status bar, metrics panel, image gallery, and chat interface builders.
"""
from __future__ import annotations
import pytest
import gradio as gr

from week6.components import (
    build_header,
    build_status_bar,
    build_metrics_panel,
    build_image_gallery,
    build_chat_interface,
)


def test_components_render() -> None:
    """Instantiate and verify component builders inside gr.Blocks."""
    dummy_fn = lambda x: f"Response to {x}"

    with gr.Blocks() as demo:
        # Header
        build_header()
        
        # Status Bar
        build_status_bar(mock_mode=True)
        
        # Metrics Panel
        build_metrics_panel("Test Metrics", {"Metric A": 0.85, "Metric B": 12})
        
        # Image Gallery
        gallery = build_image_gallery()
        assert isinstance(gallery, gr.Gallery)
        
        # Chat Interface
        chat = build_chat_interface(fn=dummy_fn)
        assert isinstance(chat, gr.ChatInterface)

    assert len(demo.blocks) > 0
