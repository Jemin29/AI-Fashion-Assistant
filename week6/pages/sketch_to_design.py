"""
Week 6 — Sketch2Design Page.
Integrates Week 3 ControlNet Engine with sketch preprocessing (Canny/HED) and preview.
"""
from __future__ import annotations
import random
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from PIL import Image

from src.controlnet.preprocessors.sketch_processor import SketchProcessor


def build_sketch_to_design_page(cn_service: Any) -> None:
    """Build the Sketch2Design tab layout."""
    gr.Markdown("## ✏️ Sketch2Design — Conditioned Studio")
    gr.Markdown(
        "Convert hand-drawn fashion sketches into photorealistic designs with precise control.",
        elem_classes="studio-subtitle",
    )

    # Instantiate preprocessor
    processor = SketchProcessor()

    with gr.Row():
        # ── Left Column: Controls ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input Options")
            
            sketch_input = gr.Image(
                label="Upload Sketch (hand-drawn outlines or CAD contours)",
                type="pil",
                height=300,
            )

            prompt = gr.Textbox(
                label="Fashion Prompt",
                placeholder="A sleek silk evening gown, emerald green, flowing silhouette...",
                lines=3,
            )

            detector = gr.Radio(
                choices=["canny", "hed"],
                value="canny",
                label="Edge Detector Method",
            )

            conditioning_scale = gr.Slider(
                0.1, 1.5, value=0.7, step=0.05,
                label="Conditioning Strength",
                info="Higher = strictly follows the sketch outlines",
            )

            with gr.Accordion("⚙️ Parameters", open=False):
                steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            with gr.Row():
                preview_btn = gr.Button("👁️ Preview Processed Sketch", variant="secondary")
                generate_btn = gr.Button("🎨 Generate Design", variant="primary")

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Preview & Rendered Output")
            
            preview_output = gr.Image(
                label="Processed Sketch Preview",
                type="pil",
                height=250,
                interactive=False,
            )

            output_image = gr.Image(
                label="Rendered Design",
                type="pil",
                height=380,
                interactive=False,
            )

            output_meta = gr.JSON(label="Metadata")

    # ── Event Handlers ────────────────────────────────────────────────────────

    def on_preview(img: Optional[Image.Image], method: str) -> Optional[Image.Image]:
        if img is None:
            return None
        return processor.preprocess_sketch(img, method=method)

    def on_generate(
        img: Optional[Image.Image],
        prompt_text: str,
        method: str,
        cond_scale: float,
        steps_val: int,
        cfg_val: float,
        seed_val: int,
    ) -> Tuple[Optional[Image.Image], Dict[str, Any], Optional[Image.Image]]:
        if img is None:
            return None, {"error": "Please upload a sketch image."}, None
        if not prompt_text.strip():
            return None, {"error": "Prompt cannot be empty."}, None

        # Preprocess sketch first
        preprocessed = processor.preprocess_sketch(img, method=method)

        # Route HED/Canny to sketch/canny mode in engine
        cnet_mode = "sketch" if method == "hed" else "canny"

        # Resolve random seed if -1
        resolved_seed = int(seed_val)
        if resolved_seed == -1:
            resolved_seed = random.randint(0, 1000000)

        # Generate conditioned design
        result = cn_service.generate_conditioned(
            prompt=prompt_text,
            control_image=preprocessed,
            mode=cnet_mode,
            conditioning_scale=float(cond_scale),
            num_inference_steps=int(steps_val),
            guidance_scale=float(cfg_val),
        )
        if not result.success:
            raise gr.Error(result.message)
        
        data_payload = result.data or {}
        res_image = data_payload.get("image")
        meta = result.metadata

        # Enforce resolved seed into metadata
        if meta:
            if "generation" in meta:
                meta["generation"]["seed"] = resolved_seed
                meta["generation"]["preprocessor_method"] = method
            else:
                meta["seed"] = resolved_seed
                meta["preprocessor_method"] = method

        return res_image, meta, preprocessed

    preview_btn.click(
        on_preview,
        inputs=[sketch_input, detector],
        outputs=[preview_output]
    )

    generate_btn.click(
        on_generate,
        inputs=[sketch_input, prompt, detector, conditioning_scale, steps, cfg, seed],
        outputs=[output_image, output_meta, preview_output]
    )
