"""Week 6 — Brand Studio Page (LoRA brand generation + style mixing)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import gradio as gr
from PIL import Image


from week6.pages.style_switcher import build_style_switcher_page
from week6.pages.style_mixer import build_style_mixer_page


def build_brand_studio_page(lora_service: Any) -> None:
    """Build the Brand Studio tab content, hosting both switcher and mixer."""
    with gr.Tabs():
        with gr.TabItem("🏷️ Style Switching", id="style_switching_tab"):
            build_style_switcher_page(lora_service)
        with gr.TabItem("🎛️ Style Mixer", id="style_mixer_tab"):
            build_style_mixer_page(lora_service)
