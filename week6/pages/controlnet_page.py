"""Week 6 — ControlNet Studio Page."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import gradio as gr
from PIL import Image


def build_controlnet_page(cn_service: Any) -> None:
    """Build the ControlNet Studio tab content."""
    gr.Markdown("## ✏️ ControlNet Studio — Conditioned Fashion Generation")
    gr.Markdown(
        "Upload a sketch, pose image, or depth map to guide SDXL with precise structural control.",
        elem_classes="studio-subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Control Image")
            control_image = gr.Image(
                label="Upload Control Image (sketch / pose / depth)",
                type="pil",
                height=300,
            )
            preprocessed_preview = gr.Image(
                label="Preprocessed Control Signal",
                type="pil",
                height=200,
                interactive=False,
            )
            preview_btn = gr.Button("👁️ Preview Preprocessed", variant="secondary", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Controls")
            mode = gr.Radio(
                choices=["canny", "sketch", "pose", "depth"],
                value="canny",
                label="Conditioning Mode",
            )
            prompt = gr.Textbox(
                label="Fashion Prompt",
                placeholder="A high-fashion model wearing a minimalist ensemble, studio lighting...",
                lines=3,
            )
            conditioning_scale = gr.Slider(
                0.0, 1.5, value=0.7, step=0.05,
                label="ControlNet Conditioning Scale",
                info="Higher = stronger structural guidance",
            )
            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")

            generate_btn = gr.Button("🎨 Generate Conditioned Image", variant="primary", size="lg")

    with gr.Row():
        output_image = gr.Image(label="Generated Output", type="pil", height=400)
        output_meta = gr.JSON(label="Metadata")

    gr.Markdown("---")
    gr.Markdown("""
### 📋 Conditioning Mode Guide

| Mode | Best For | Notes |
|------|----------|-------|
| **Canny** | Precise edge control | Best with clear line drawings |
| **Sketch** | Artistic sketches | Supports rough pencil drawings |
| **Pose** | Body pose control | Uses OpenPose skeleton detection |
| **Depth** | 3D spatial structure | Best with depth map images |
""")

    def on_preview(img, selected_mode):
        if img is None:
            return None
        return cn_service.preprocess_image(img, mode=selected_mode)

    def on_generate(img, prompt_text, selected_mode, cond_scale, n_steps, cfg_scale):
        if img is None:
            return None, {"error": "Please upload a control image."}
        if not prompt_text.strip():
            return None, {"error": "Please enter a prompt."}
        return cn_service.generate_conditioned(
            prompt=prompt_text,
            control_image=img,
            mode=selected_mode,
            conditioning_scale=float(cond_scale),
            num_inference_steps=int(n_steps),
            guidance_scale=float(cfg_scale),
        )

    preview_btn.click(on_preview, inputs=[control_image, mode], outputs=[preprocessed_preview])
    generate_btn.click(
        on_generate,
        inputs=[control_image, prompt, mode, conditioning_scale, steps, cfg],
        outputs=[output_image, output_meta],
    )
