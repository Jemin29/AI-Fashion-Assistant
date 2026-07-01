"""Week 6 Header Component."""
from __future__ import annotations
import gradio as gr


def build_header() -> None:
    """Build a premium header with a logo and description."""
    gr.HTML("""
    <div class="studio-header-banner" style="background: linear-gradient(135deg, #1e1e2f 0%, #0f0f15 100%); border-bottom: 2px solid #ff9f43; padding: 1.8rem; text-align: center; border-radius: 8px 8px 0 0; margin-bottom: 1.5rem;">
        <h1 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: 1px;">
            🎨 <span style="background: linear-gradient(90deg, #ff9f43, #ff5252); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Fashion Creative Studio</span>
        </h1>
        <p style="color: #a0a0b0; font-family: 'Inter', sans-serif; font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 300;">
            Week 6 Creative Graduation Suite integrating SDXL, ControlNet, LoRA, and Retrieval-Augmented Generation
        </p>
    </div>
    """)
