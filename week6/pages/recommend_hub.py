"""Week 6 — Recommendation Hub Page."""
from __future__ import annotations
from typing import Any, Dict, List
import gradio as gr


def build_recommend_hub_page(rec_service: Any) -> None:
    """Build the Recommendation Hub tab."""
    gr.Markdown("## 👗 Recommendation Hub — Personalized Fashion Intelligence")
    gr.Markdown(
        "Get AI-powered style and brand recommendations tailored to your preferences.",
        elem_classes="studio-subtitle",
    )

    with gr.Tabs():
        # ── Tab A: Style Recommendations ─────────────────────────────────────
        with gr.TabItem("👗 Style Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Your Preferences")
                    gender = gr.Dropdown(
                        choices=["men", "women", "unisex"],
                        value="unisex",
                        label="Gender",
                    )
                    style = gr.Dropdown(
                        choices=["streetwear", "luxury", "formal", "business_casual",
                                 "techwear", "minimalist", "vintage", "athleisure",
                                 "bohemian", "preppy", "romantic", "avant_garde"],
                        value="minimalist",
                        label="Style Category",
                    )
                    occasion = gr.Dropdown(
                        choices=["casual", "business_casual", "formal", "party",
                                 "sport", "outdoor", "beach", "lounge", "date"],
                        value="casual",
                        label="Occasion",
                    )
                    fit = gr.Dropdown(
                        choices=["slim_fit", "regular_fit", "relaxed_fit", "oversized",
                                 "cropped", "skinny", "straight", "athletic_fit"],
                        value="regular_fit",
                        label="Fit Profile",
                    )
                    n_style = gr.Slider(1, 8, value=5, step=1, label="Number of Recommendations")
                    style_btn = gr.Button("👗 Get Style Recommendations", variant="primary", size="lg")

                with gr.Column(scale=2):
                    style_output = gr.Markdown("_Set your preferences and click 'Get Style Recommendations'._")

            def on_style_rec(g, s, o, f, n):
                result = rec_service.recommend_styles(g, s, o, f, n=int(n))
                if not result.success:
                    raise gr.Error(result.message)
                recs = result.data or []
                if not recs:
                    return "_No recommendations match your preferences._"
                md = f"## 👗 Top {len(recs)} Style Recommendations\n\n"
                for i, r in enumerate(recs, 1):
                    if isinstance(r, dict):
                        score = r.get("score", 0.0)
                        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        md += f"### {i}. {r.get('style', 'Style')}\n"
                        md += f"**Match Score**: `{score_bar}` {score:.0%}\n\n"
                        md += f"{r.get('description', '')}\n\n---\n\n"
                    else:
                        md += f"### {i}.\n{r}\n\n---\n\n"
                return md

            style_btn.click(
                on_style_rec,
                inputs=[gender, style, occasion, fit, n_style],
                outputs=[style_output],
            )

        # ── Tab B: Brand Recommendations ──────────────────────────────────────
        with gr.TabItem("🏷️ Brand Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Your Aesthetic Profile")
                    preferred_styles = gr.CheckboxGroup(
                        choices=["streetwear", "minimalist", "luxury", "athletic",
                                 "vintage", "bohemian", "formal", "techwear"],
                        value=["minimalist"],
                        label="Preferred Styles",
                    )
                    aesthetic_desc = gr.Textbox(
                        label="Describe Your Target Aesthetic",
                        placeholder="Clean, understated elegance with quality fabrics and minimal branding...",
                        lines=3,
                    )
                    n_brand = gr.Slider(1, 6, value=4, step=1, label="Number of Brands")
                    brand_btn = gr.Button("🏷️ Find Matching Brands", variant="primary", size="lg")

                with gr.Column(scale=2):
                    brand_output = gr.Markdown("_Set your aesthetic and click 'Find Matching Brands'._")

            def on_brand_rec(styles, aesthetic, n):
                result = rec_service.recommend_brands(styles, aesthetic, n=int(n))
                if not result.success:
                    raise gr.Error(result.message)
                recs = result.data or []
                if not recs:
                    return "_No brand recommendations match your profile._"
                md = f"## 🏷️ Top {len(recs)} Brand Recommendations\n\n"
                for i, r in enumerate(recs, 1):
                    if isinstance(r, dict):
                        score = r.get("score", 0.0)
                        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        md += f"### {i}. {r.get('brand', 'Brand')}\n"
                        md += f"**Match Score**: `{score_bar}` {score:.0%} | "
                        md += f"**Category**: {r.get('category', '')} | "
                        md += f"**Price Range**: {r.get('price_range', '')}\n\n---\n\n"
                    else:
                        md += f"### {i}.\n{r}\n\n---\n\n"
                return md

            brand_btn.click(
                on_brand_rec,
                inputs=[preferred_styles, aesthetic_desc, n_brand],
                outputs=[brand_output],
            )

        # ── Tab C: User Profile ───────────────────────────────────────────────
        with gr.TabItem("👤 User Profile"):
            gr.Markdown("### 👤 Your Fashion Profile")
            user_id = gr.Textbox(label="User ID", value="demo_user", lines=1)
            view_profile_btn = gr.Button("🔍 View My Profile", variant="secondary")
            profile_output = gr.JSON(label="User Preference Profile")

            def on_view_profile(uid):
                result = rec_service.get_user_profile(uid or "demo_user")
                if not result.success:
                    raise gr.Error(result.message)
                return result.data

            view_profile_btn.click(on_view_profile, inputs=[user_id], outputs=[profile_output])
