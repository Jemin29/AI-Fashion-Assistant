"""Week 6 — Style Studio Page (SDXL Text-to-Fashion Generation)."""
from __future__ import annotations
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


def build_style_studio_page(gen_service: Any) -> None:
    """Build the Style Studio tab content."""
    gr.Markdown("## 🎨 Style Studio — SDXL Fashion Generation")
    gr.Markdown(
        "Generate high-fidelity fashion images using Stable Diffusion XL. "
        "Choose a style preset or craft your own detailed prompt.",
        elem_classes="studio-subtitle",
    )

    with gr.Row():
        # ── Left panel: controls ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Prompt Builder")

            preset = gr.Dropdown(
                label="Style Preset",
                choices=["Custom"] + list(_STYLE_PRESETS.keys()),
                value="Custom",
            )

            prompt = gr.Textbox(
                label="Fashion Prompt",
                placeholder="A stunning model wearing an oversized monochromatic outfit, editorial photography...",
                lines=4,
                max_lines=8,
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=_NEGATIVE_DEFAULT,
                lines=2,
            )

            with gr.Accordion("⚙️ Generation Parameters", open=False):
                steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                width = gr.Dropdown(
                    choices=[512, 768, 1024], value=512, label="Width"
                )
                height = gr.Dropdown(
                    choices=[512, 768, 1024], value=512, label="Height"
                )
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            generate_btn = gr.Button("✨ Generate Fashion Image", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

        # ── Right panel: output ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Output")
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=400,
            )
            output_meta = gr.JSON(label="Generation Metadata", visible=True)

    # ── Gallery of recent generations ────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 🗂️ Session Gallery")
    gallery = gr.Gallery(
        label="Recent Generations",
        columns=4,
        rows=2,
        height=250,
    )
    _gallery_images: List[Image.Image] = []

    # ── Event handlers ────────────────────────────────────────────────────────
    def on_preset_change(selected: str) -> str:
        if selected and selected != "Custom":
            return _STYLE_PRESETS.get(selected, "")
        return gr.update()

    def on_generate(
        prompt_text: str,
        neg_prompt: str,
        n_steps: int,
        cfg_scale: float,
        w: int,
        h: int,
        seed_val: int,
        preset_label: str,
    ) -> Tuple[Optional[Image.Image], Dict, List]:
        if not prompt_text.strip():
            return None, {"error": "Please enter a prompt."}, _gallery_images

        result = gen_service.generate(
            prompt=prompt_text,
            negative_prompt=neg_prompt,
            num_inference_steps=int(n_steps),
            guidance_scale=float(cfg_scale),
            width=int(w),
            height=int(h),
            seed=None if seed_val == -1 else int(seed_val),
            style_label=preset_label if preset_label != "Custom" else "Fashion Generation",
        )
        if not result.success:
            raise gr.Error(result.message)
        
        img = result.data.get("image") if isinstance(result.data, dict) else result.data
        meta = result.metadata
        if img is not None:
            _gallery_images.insert(0, img)
        return img, meta, _gallery_images[:8]

    def on_clear():
        return None, {}, gr.update()

    preset.change(on_preset_change, inputs=[preset], outputs=[prompt])
    generate_btn.click(
        on_generate,
        inputs=[prompt, negative_prompt, steps, cfg, width, height, seed, preset],
        outputs=[output_image, output_meta, gallery],
    )
    clear_btn.click(on_clear, inputs=[], outputs=[output_image, output_meta, gallery])
