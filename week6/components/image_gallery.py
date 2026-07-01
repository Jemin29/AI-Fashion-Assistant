"""Week 6 Image Gallery Component."""
from __future__ import annotations
from typing import List
import gradio as gr


def build_image_gallery(label: str = "Recent Designs", columns: int = 4, rows: int = 2) -> gr.Gallery:
    """Build a styled gr.Gallery with default options."""
    return gr.Gallery(
        label=label,
        columns=columns,
        rows=rows,
        height=280,
        interactive=False,
    )
