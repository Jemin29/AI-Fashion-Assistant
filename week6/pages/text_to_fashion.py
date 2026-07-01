"""
Week 6 — Text-to-Fashion Page.
Integrates Week 2 SDXL Generator with support for prompts, batch generation, history, and parameter tuning.
"""
from __future__ import annotations
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

_STYLE_PRESETS = {
    "Streetwear Editorial": "oversized streetwear outfit, urban fashion, bold graphic hoodie, cargo pants, high-top sneakers, editorial photography",
    "Luxury Fashion Week": "haute couture dress, luxury fashion week, Chanel aesthetic, silk fabric, crystal embellishments, runway lighting",
    "Minimalist Studio": "minimalist white outfit, studio photography, clean lines, monochromatic, Celine aesthetic, light background",
    "Athleisure Sport": "performance athleisure outfit, Nike aesthetic, technical fabric, dynamic pose, gym lighting, sport photography",
    "Bohemian Summer": "bohemian summer dress, flowing fabric, floral print, golden hour photography, natural textures",
    "Techwear Utility": "technical techwear outfit, ACRONYM aesthetic, utility pockets, dark palette, futuristic details",
    "Vintage 90s": "90s vintage fashion, retro styling, denim jacket, band tee, high-waisted jeans, film grain",
    "Formal Business": "executive business attire, tailored suit, luxury fabric, professional editorial, power dressing",
}

_NEGATIVE_DEFAULT = (
    "low quality, blurry, watermark, text overlay, distorted anatomy, "
    "bad proportions, amateur photography, oversaturated, plastic look"
)


def build_text_to_fashion_page(gen_service: Any) -> None:
    """Build the Text-to-Fashion tab layout."""
    gr.Markdown("## 🎨 Text-to-Fashion — SDXL Studio")
    gr.Markdown(
        "Generate high-fidelity fashion items or outfits using Stable Diffusion XL. Use presets or construct custom prompts.",
        elem_classes="studio-subtitle",
    )

    # State containers for session history
    prompt_history_state = gr.State([])

    with gr.Row():
        # ── Left Column: Controls ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Prompt Builder")

            preset = gr.Dropdown(
                label="Style Presets",
                choices=["Custom"] + list(_STYLE_PRESETS.keys()),
                value="Custom",
            )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="An oversized techwear cargo pants design, neon straps, technical fabric...",
                lines=4,
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=_NEGATIVE_DEFAULT,
                lines=2,
            )

            # Interactive Prompt History Dropdown
            history_select = gr.Dropdown(
                label="📜 Prompt History",
                choices=[],
                value=None,
                allow_custom_value=False,
                interactive=True,
            )

            with gr.Accordion("⚙️ Parameters", open=False):
                steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                width = gr.Dropdown(choices=[512, 768, 1024], value=512, label="Width")
                height = gr.Dropdown(choices=[512, 768, 1024], value=512, label="Height")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size (Images)")

            generate_btn = gr.Button("✨ Generate Outfits", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear Form", variant="secondary", size="sm")

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Output")
            # Using Gallery to support batch size (1 to 4 images)
            output_gallery = gr.Gallery(
                label="Output Gallery",
                columns=2,
                rows=2,
                height=400,
                show_label=True,
            )
            output_meta = gr.JSON(label="Metadata")

    # ── Event Handlers ────────────────────────────────────────────────────────

    # Preset selection syncs textbox
    def on_preset_change(selected: str) -> str:
        if selected != "Custom":
            return _STYLE_PRESETS.get(selected, "")
        return gr.update()

    preset.change(on_preset_change, inputs=[preset], outputs=[prompt])

    # Prompt History selection updates main textbox
    def on_history_change(selected_hist: Optional[str]) -> str:
        if selected_hist:
            return selected_hist
        return gr.update()

    history_select.change(on_history_change, inputs=[history_select], outputs=[prompt])

    # Generate callback
    def on_generate(
        p: str,
        np: str,
        steps_val: int,
        cfg_val: float,
        w: int,
        h: int,
        seed_val: int,
        batch: int,
        history: List[str],
    ) -> Tuple[List[Image.Image], Dict[str, Any], List[str], Dict[str, Any]]:
        if not p.strip():
            return [], {"error": "Prompt cannot be empty."}, history, gr.update()

        images = []
        metadata_list = []

        # Resolve seeds
        resolved_seed = int(seed_val)
        if resolved_seed == -1:
            resolved_seed = random.randint(0, 1000000)

        # Loop through batch size
        for i in range(batch):
            current_seed = resolved_seed + i
            result = gen_service.generate(
                prompt=p,
                negative_prompt=np,
                num_inference_steps=int(steps_val),
                guidance_scale=float(cfg_val),
                width=int(w),
                height=int(h),
                seed=current_seed,
                style_label="Batch Item" if batch > 1 else "Fashion Design",
            )
            if not result.success:
                raise gr.Error(result.message)
            img = result.data
            meta = result.metadata
            if img is not None:
                images.append(img)
                metadata_list.append(meta)

        # Update prompt history
        if p not in history:
            history = [p] + history

        unified_metadata = {
            "batch_size": batch,
            "resolved_seeds": [resolved_seed + i for i in range(batch)],
            "elapsed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "individual_metadata": metadata_list,
        }

        return images, unified_metadata, history, gr.update(choices=history, value=p)

    generate_btn.click(
        on_generate,
        inputs=[prompt, negative_prompt, steps, cfg, width, height, seed, batch_size, prompt_history_state],
        outputs=[output_gallery, output_meta, prompt_history_state, history_select],
    )

    # Clear callback
    def on_clear():
        return [], {}, "Custom", "", _NEGATIVE_DEFAULT, -1, 1

    clear_btn.click(
        on_clear,
        outputs=[output_gallery, output_meta, preset, prompt, negative_prompt, seed, batch_size],
    )
