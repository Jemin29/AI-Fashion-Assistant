"""
Week 6 — Style Mixer Page.
Integrates Week 4 Style Mixer to blend multiple brand aesthetics (Nike, Gucci, Zara, H&M) with presets and comparison.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image


def build_style_mixer_page(lora_service: Any) -> None:
    """Build the Multi-Brand Style Mixer tab layout."""
    gr.Markdown("## 🎛️ Style Mixer — LoRA Blending")
    gr.Markdown(
        "Blend the aesthetics of multiple fashion brands together using customizable weights or preset combinations.",
        elem_classes="studio-subtitle",
    )

    brands = lora_service.get_brands()
    brand_labels = {"nike": "Nike 🏃", "gucci": "Gucci 💎", "zara": "Zara ✨", "hm": "H&M 🌿"}

    with gr.Row():
        # ── Left Column: Controls ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Blend Configuration")

            preset_dropdown = gr.Dropdown(
                choices=[
                    ("Custom Weights (Manual)", "custom"),
                    ("Nike + Gucci (Sporty Luxury)", "nike_gucci"),
                    ("Nike + Zara (Sporty Minimal)", "nike_zara"),
                    ("Gucci + Zara (Minimal Luxury)", "gucci_zara"),
                ],
                value="custom",
                label="Preset Blends",
            )

            prompt = gr.Textbox(
                label="Base Fashion Prompt",
                placeholder="A luxury winter trench coat, editorial studio photo...",
                lines=3,
            )

            gr.Markdown("#### Brand Weights")
            sliders = {}
            for b in brands:
                sliders[b] = gr.Slider(
                    0.0, 1.0,
                    value=0.5 if b == "nike" else 0.0,
                    step=0.05,
                    label=brand_labels.get(b, b.upper()),
                )

            # Sync sliders with preset selection
            def on_preset_change(preset: str) -> Tuple[Any, ...]:
                if preset == "nike_gucci":
                    return 0.5, 0.5, 0.0, 0.0
                elif preset == "nike_zara":
                    return 0.5, 0.0, 0.5, 0.0
                elif preset == "gucci_zara":
                    return 0.0, 0.5, 0.5, 0.0
                return gr.update(), gr.update(), gr.update(), gr.update()

            preset_dropdown.change(
                on_preset_change,
                inputs=[preset_dropdown],
                outputs=[sliders["nike"], sliders["gucci"], sliders["zara"], sliders["hm"]],
            )

            # If sliders are manually edited, reset preset dropdown to "custom"
            def on_slider_touch() -> str:
                return "custom"

            for s in sliders.values():
                s.change(on_slider_touch, outputs=[preset_dropdown])

            with gr.Row():
                mix_btn = gr.Button("🎨 Blend Aesthetics", variant="primary")
                compare_btn = gr.Button("📊 Generate & Compare Elements", variant="secondary")

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Blended Result")

            with gr.Tabs() as output_tabs:
                with gr.TabItem("🎛️ Blended Design", id="blend_tab"):
                    mixed_output_image = gr.Image(
                        label="Blended Fashion Design",
                        type="pil",
                        height=380,
                        interactive=False,
                    )
                    mixed_output_meta = gr.JSON(label="Blend Metadata")

                with gr.TabItem("📊 Side-by-Side Comparison", id="compare_tab"):
                    gr.Markdown("Showing the blended design side-by-side with its individual source brand aesthetics:")
                    comparison_gallery = gr.Gallery(
                        label="Comparative Output Grid",
                        columns=3,
                        rows=1,
                        height=350,
                        show_label=True,
                    )
                    comparison_meta = gr.JSON(label="Comparison Details")

    # ── Event Handlers ────────────────────────────────────────────────────────

    def on_blend(p: str, *weights) -> Tuple[Optional[Image.Image], Dict[str, Any], Any]:
        if not p.strip():
            return None, {"error": "Prompt cannot be empty."}, gr.update(selected="blend_tab")
        
        brand_w = {b: float(w) for b, w in zip(brands, weights) if float(w) > 0}
        if not brand_w:
            return None, {"error": "At least one brand weight must be greater than 0."}, gr.update(selected="blend_tab")

        result = lora_service.mix_styles(p, brand_w)
        if not result.success:
            raise gr.Error(result.message)
        data_payload = result.data or {}
        img = data_payload.get("image")
        meta = result.metadata
        return img, meta, gr.update(selected="blend_tab")

    def on_compare(p: str, *weights) -> Tuple[List[Tuple[Image.Image, str]], Dict[str, Any], Any]:
        if not p.strip():
            return [], {"error": "Prompt cannot be empty."}, gr.update(selected="compare_tab")

        brand_w = {b: float(w) for b, w in zip(brands, weights) if float(w) > 0}
        if not brand_w:
            return [], {"error": "At least one brand weight must be greater than 0."}, gr.update(selected="compare_tab")

        # 1. Generate blended design
        result = lora_service.mix_styles(p, brand_w)
        if not result.success:
            raise gr.Error(result.message)
        data_payload = result.data or {}
        blend_img = data_payload.get("image")
        blend_meta = result.metadata
        
        gallery_items = []
        if blend_img is not None:
            gallery_items.append((blend_img, "Blended Design (Mixed LoRAs)"))

        # 2. Generate individual pure brands that contributed
        individual_meta = {}
        for b, w in brand_w.items():
            pure_res = lora_service.generate_with_brand(p, b, lora_scale=0.85)
            if not pure_res.success:
                raise gr.Error(pure_res.message)
            pure_data = pure_res.data or {}
            pure_img = pure_data.get("image")
            pure_meta = pure_res.metadata
            if pure_img is not None:
                label_text = f"Pure {brand_labels.get(b, b.title())} (Weight: {w:.1%})"
                gallery_items.append((pure_img, label_text))
                individual_meta[b] = pure_meta

        meta = {
            "blend_weights": brand_w,
            "blend_metadata": blend_meta,
            "individual_metadata": individual_meta,
        }
        return gallery_items, meta, gr.update(selected="compare_tab")

    mix_btn.click(
        on_blend,
        inputs=[prompt] + [sliders[b] for b in brands],
        outputs=[mixed_output_image, mixed_output_meta, output_tabs]
    )

    compare_btn.click(
        on_compare,
        inputs=[prompt] + [sliders[b] for b in brands],
        outputs=[comparison_gallery, comparison_meta, output_tabs]
    )
