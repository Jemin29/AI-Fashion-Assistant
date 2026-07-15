"""
Week 6 — WardrobeGen Page.
Curates complementary outfits based on style/occasion preferences via the RAG system
and renders visual representations using the SDXL pipeline.
"""
from __future__ import annotations
import random
import time
from typing import Any, Dict, List, Tuple
import gradio as gr
from PIL import Image

def build_wardrobe_gen_page(rec_service: Any, gen_service: Any) -> None:
    """Build the WardrobeGen page tab layout."""
    gr.Markdown("## 👗 WardrobeGen — Intelligent Capsule Studio")
    gr.Markdown(
        "Curate coordinated capsule outfits tailored to your style preferences and visualize the lookbook.",
        elem_classes="studio-subtitle",
    )

    # Status banner
    is_mock = rec_service.mock_mode or gen_service.mock_mode
    if is_mock:
        gr.HTML(
            "<div style='background-color: rgba(255, 193, 7, 0.15); border-left: 4px solid #ffc107; padding: 12px; margin-bottom: 20px; border-radius: 4px;'>"
            "<strong>⚠️ Mock Mode Active:</strong> Some backend models are running in mock simulation. "
            "To enable real inference, set <code>GLOBAL_MOCK=False</code> in your configuration.</div>"
        )
    else:
        gr.HTML(
            "<div style='background-color: rgba(40, 167, 69, 0.15); border-left: 4px solid #28a745; padding: 12px; margin-bottom: 20px; border-radius: 4px;'>"
            "<strong>✅ Real Wardrobe Gen Active:</strong> RAG and SDXL generator are running in production mode. "
            f"Device: <code>cpu</code></div>"
        )

    with gr.Row():
        # ── Left Column: Controls ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Curate Preferences")

            style_pref = gr.Dropdown(
                choices=["streetwear", "luxury", "minimalist", "athleisure", "vintage"],
                value="minimalist",
                label="Style Aesthetic",
            )

            occasion = gr.Dropdown(
                choices=["casual", "formal", "business_casual", "weekend", "sport", "evening"],
                value="casual",
                label="Occasion Profile",
            )

            existing_items = gr.Textbox(
                label="Existing Closet Pieces (Optional)",
                placeholder="e.g. matte black windbreaker, blue jeans...",
                lines=2,
            )

            num_outfits = gr.Slider(
                minimum=2, maximum=4, value=3, step=1,
                label="Number of Outfits to Render",
            )

            with gr.Accordion("⚙️ Parameters", open=False):
                steps = gr.Slider(10, 50, value=25, step=1, label="SDXL Inference Steps")
                cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            generate_btn = gr.Button("🎨 Generate Coordinated Lookbook", variant="primary")

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📖 Capsule Curation Results")
            
            curation_output = gr.Markdown(
                value="*Curation details will appear here after clicking generate.*",
                elem_id="wardrobe-curation-markdown"
            )

            gr.Markdown("### 🖼️ Rendered Lookbook Gallery")
            output_gallery = gr.Gallery(
                label="Rendered Outfits",
                columns=2,
                rows=2,
                height=350,
                show_label=True,
            )
            output_meta = gr.JSON(label="System Details")

    # ── Event Handlers ────────────────────────────────────────────────────────
    from week6.pages.utils import safe_callback

    @safe_callback(3, fallback_values=["", [], {}])
    def on_generate_wardrobe(
        style: str,
        occ: str,
        closet: str,
        num: int,
        steps_val: int,
        cfg_val: float,
        seed_val: int,
    ) -> Tuple[str, List[Image.Image], Dict[str, Any]]:
        # 1. Fetch RAG outfit recommendations
        outfits_res = rec_service.recommend_outfits(style, occ, n=int(num))
        if not outfits_res.success:
            return f"Error fetching recommendations: {outfits_res.message}", [], {}

        outfit_list = outfits_res.data
        if not outfit_list:
            return "No matching outfits found in the recommendation database.", [], {}

        # 2. Format Curation Markdown
        md_lines = []
        md_lines.append(f"### 🛍️ Curated Capsule Suggestions for **{style.title()}** ({occ.replace('_', ' ').title()})")
        if closet.strip():
            md_lines.append(f"*Incorporating existing closet items: '{closet.strip()}'*")
        md_lines.append("")

        generated_images = []
        individual_meta = []

        # Resolve Seed
        base_seed = int(seed_val)
        if base_seed == -1:
            base_seed = random.randint(0, 1000000)

        for idx, outfit in enumerate(outfit_list[:int(num)]):
            name = outfit.get("name", f"Outfit {idx+1}")
            items_dict = outfit.get("items", {})
            score = outfit.get("score", 0.90)

            # Format items text
            items_desc = []
            for k, v in items_dict.items():
                if v:
                    items_desc.append(f"- **{k.capitalize()}**: {v}")

            items_str = "\n".join(items_desc)
            md_lines.append(f"#### {idx+1}. {name} (Match Score: {score:.2f})")
            md_lines.append(items_str)
            md_lines.append("")

            # 3. Construct generation prompt for SDXL
            flat_items = ", ".join([str(v) for v in items_dict.values() if v])
            prompt = f"A coordinated {style} outfit for {occ} wear: {flat_items}."
            if closet.strip():
                prompt += f" Coordinating with user's {closet.strip()}."
            prompt += " High-fashion editorial photography, clean studio catalog background, professional model styling."

            # Generate image (using real SDXL/PEFT model via GenerationService)
            # Specify width/height 64x64 on CPU dry_run or per resolution choice to be fast
            # Let's specify width=512, height=512 but note CPU may take some time (or fast mock if in mock mode)
            result = gen_service.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, extra limbs, bad proportions",
                num_inference_steps=int(steps_val),
                guidance_scale=float(cfg_val),
                width=512,
                height=512,
                seed=base_seed + idx,
                style_label=f"Outfit: {name}"
            )
            if result.success and result.data is not None:
                generated_images.append(result.data.get("image"))
                individual_meta.append(result.metadata)

        # Unified curation markdown text
        markdown_text = "\n".join(md_lines)

        meta = {
            "style": style,
            "occasion": occ,
            "num_outfits_requested": num,
            "base_seed": base_seed,
            "individual_runs": individual_meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return markdown_text, generated_images, meta

    generate_btn.click(
        on_generate_wardrobe,
        inputs=[style_pref, occasion, existing_items, num_outfits, steps, cfg, seed],
        outputs=[curation_output, output_gallery, output_meta],
    )
