"""
Week 6 — Brand Style Switching Page.
Integrates Week 4 LoRA adapters to switch the stylistic aesthetic of outfits (Nike, Gucci, Zara, H&M).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image


def build_style_switcher_page(lora_service: Any) -> None:
    """Build the Brand Style Switching tab layout."""
    gr.Markdown("## 🏷️ Brand Style Switching — LoRA Registry")
    gr.Markdown(
        "Instantly swap the designer aesthetic and brand signature of your outfits using fine-tuned LoRA adapters.",
        elem_classes="studio-subtitle",
    )

    brands = lora_service.get_brands()
    brand_labels = {"nike": "Nike 🏃", "gucci": "Gucci 💎", "zara": "Zara ✨", "hm": "H&M 🌿"}

    with gr.Row():
        # ── Left Column: Controls ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Design Controls")

            brand_dropdown = gr.Dropdown(
                choices=[(brand_labels.get(b, b.title()), b) for b in brands],
                value=brands[0] if brands else "nike",
                label="Primary Brand",
            )

            prompt = gr.Textbox(
                label="Fashion Prompt",
                placeholder="A sports jacket with zipper detailing and technical cargo pants...",
                lines=3,
            )

            lora_scale = gr.Slider(
                0.3, 1.5, value=0.85, step=0.05,
                label="LoRA Adapter Scale",
                info="Higher = stronger brand styling signature",
            )

            with gr.Accordion("⚙️ Parameters", open=False):
                steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")

            with gr.Row():
                generate_btn = gr.Button("🎨 Generate Output", variant="primary")
                compare_btn = gr.Button("📊 Compare All Brands", variant="secondary")

            gr.Markdown("---")
            brand_info_display = gr.Markdown()

            def update_brand_display(b: str) -> str:
                info = lora_service.get_brand_info(b)
                if not info:
                    return "_No description loaded for this brand._"
                return f"""
                ### Brand Card: **{info.get('name', b.upper())}**
                _{info.get('description', '')}_
                
                - **Aesthetic Signature**: {info.get('aesthetic', '')}
                - **Color Theme**: {info.get('color_palette', '')}
                - **Key Style Matches**: {', '.join(info.get('key_styles', []))}
                """

            brand_dropdown.change(update_brand_display, inputs=[brand_dropdown], outputs=[brand_info_display])
            
            # Seed brand description on load
            if brands:
                brand_info_display.value = update_brand_display(brands[0])

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Design Outputs")

            with gr.Tabs() as output_tabs:
                with gr.TabItem("🏷️ Single Brand View", id="single_tab"):
                    single_output_image = gr.Image(
                        label="Selected Brand Output",
                        type="pil",
                        height=380,
                        interactive=False,
                    )
                    single_output_meta = gr.JSON(label="Metadata")

                with gr.TabItem("📊 Comparative Brand View", id="compare_tab"):
                    gr.Markdown("Comparing Nike, Gucci, Zara, and H&M aesthetics for the exact same prompt:")
                    compare_gallery = gr.Gallery(
                        label="Brand Aesthetics Comparison Grid",
                        columns=2,
                        rows=2,
                        height=400,
                        show_label=True,
                    )
                    compare_meta = gr.JSON(label="Comparison Details")

    # ── Event Handlers ────────────────────────────────────────────────────────

    def on_generate(
        p: str,
        b: str,
        scale: float,
        steps_val: int,
        cfg_val: float,
    ) -> Tuple[Optional[Image.Image], Dict[str, Any], Any]:
        if not p.strip():
            return None, {"error": "Prompt cannot be empty."}, gr.update(selected="single_tab")
        
        result = lora_service.generate_with_brand(
            prompt=p,
            brand=b,
            lora_scale=float(scale),
            num_inference_steps=int(steps_val),
            guidance_scale=float(cfg_val),
        )
        if not result.success:
            raise gr.Error(result.message)
        img = result.data
        meta = result.metadata
        return img, meta, gr.update(selected="single_tab")

    def on_compare(
        p: str,
        scale: float,
        steps_val: int,
        cfg_val: float,
    ) -> Tuple[List[Tuple[Image.Image, str]], Dict[str, Any], Any]:
        if not p.strip():
            return [], {"error": "Prompt cannot be empty."}, gr.update(selected="compare_tab")

        comparison_images = []
        meta_dict = {}

        for b in brands:
            result = lora_service.generate_with_brand(
                prompt=p,
                brand=b,
                lora_scale=float(scale),
                num_inference_steps=int(steps_val),
                guidance_scale=float(cfg_val),
            )
            if not result.success:
                raise gr.Error(result.message)
            img = result.data
            meta = result.metadata
            if img is not None:
                label_text = brand_labels.get(b, b.title())
                comparison_images.append((img, label_text))
                meta_dict[b] = meta

        return comparison_images, {"status": "comparison generated", "details": meta_dict}, gr.update(selected="compare_tab")

    generate_btn.click(
        on_generate,
        inputs=[prompt, brand_dropdown, lora_scale, steps, cfg],
        outputs=[single_output_image, single_output_meta, output_tabs]
    )

    compare_btn.click(
        on_compare,
        inputs=[prompt, lora_scale, steps, cfg],
        outputs=[compare_gallery, compare_meta, output_tabs]
    )
