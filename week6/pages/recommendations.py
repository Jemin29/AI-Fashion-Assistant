"""
Week 6 — Recommendation Dashboard Page.
Integrates the Week 5 Style and Brand recommendation systems, seasonal trends,
and personalized profile matchers with visual cards and confidence scores.
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple

import gradio as gr
from PIL import Image, ImageDraw


def _generate_recommendation_card_image(title: str, score: float) -> Image.Image:
    """Generate a high-fidelity visual abstract card background for a recommendation."""
    width, height = 300, 300
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Gradient Background based on title hash
    seed = hash(title) % (2**32)
    rng = random.Random(seed)
    hue = rng.randint(0, 360)
    
    # Calculate premium gradient colors (HSL-like mapping)
    r1 = int(25 + 45 * abs(((hue / 60) % 2) - 1))
    g1 = int(20 + 35 * abs(((hue / 60 - 2) % 2) - 1))
    b1 = int(35 + 55 * abs(((hue / 60 - 4) % 2) - 1))

    for y in range(height):
        ratio = y / height
        r = int(r1 * (1 - ratio) + (r1 // 3) * ratio)
        g = int(g1 * (1 - ratio) + (g1 // 3) * ratio)
        b = int(b1 * (1 - ratio) + min(255, b1 + 80) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw premium geometric overlay
    for _ in range(rng.randint(2, 4)):
        shape_type = rng.choice(["ellipse", "rectangle", "line"])
        opacity_color = (rng.randint(180, 255), rng.randint(150, 220), rng.randint(100, 200))
        if shape_type == "ellipse":
            x0 = rng.randint(20, 100)
            y0 = rng.randint(20, 100)
            x1 = rng.randint(150, 280)
            y1 = rng.randint(150, 280)
            draw.ellipse([x0, y0, x1, y1], outline=opacity_color, width=2)
        elif shape_type == "rectangle":
            x0 = rng.randint(30, 80)
            y0 = rng.randint(30, 80)
            x1 = rng.randint(180, 270)
            y1 = rng.randint(180, 270)
            draw.rectangle([x0, y0, x1, y1], outline=opacity_color, width=1)
        else:
            lx0 = rng.randint(0, 300)
            ly0 = rng.randint(0, 300)
            lx1 = rng.randint(0, 300)
            ly1 = rng.randint(0, 300)
            draw.line([(lx0, ly0), (lx1, ly1)], fill=opacity_color, width=2)

    # Draw a stylized icon in the center (e.g. circles or diamonds)
    center_color = (255, 159, 67)
    draw.polygon([
        (width // 2, height // 2 - 40),
        (width // 2 - 40, height // 2),
        (width // 2, height // 2 + 40),
        (width // 2 + 40, height // 2)
    ], outline=center_color, width=3)
    
    # Outer circle border
    draw.ellipse([10, 10, width-11, height-11], outline=(255, 255, 255, 30), width=1)
    return img


def build_recommendations_page(rec_service: Any, trend_service: Any) -> None:
    """Build the modular card-based Recommendations page."""
    gr.Markdown("## 👗 AI Fashion Recommendation Dashboard")
    gr.Markdown(
        "Explore personalized style pairings, brand recommendations, trend predictions, and lookbooks.",
        elem_classes="studio-subtitle",
    )

    with gr.Tabs():
        # ── 1. STYLE RECOMMENDATIONS ─────────────────────────────────────────
        with gr.Tab("👗 Style Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Set Style Preferences")
                    style_gender = gr.Dropdown(
                        choices=["men", "women", "unisex"],
                        value="unisex",
                        label="Gender Focus",
                    )
                    style_pref = gr.Dropdown(
                        choices=["streetwear", "luxury", "minimalist", "athleisure", "vintage"],
                        value="minimalist",
                        label="Style Category",
                    )
                    style_occasion = gr.Dropdown(
                        choices=["casual", "formal", "business casual", "sports"],
                        value="casual",
                        label="Occasion",
                    )
                    style_fit = gr.Dropdown(
                        choices=["regular_fit", "slim_fit", "oversized"],
                        value="regular_fit",
                        label="Fit Profile",
                    )
                    style_btn = gr.Button("Get Style Matches", variant="primary")

                with gr.Column(scale=3):
                    gr.Markdown("### 🌟 Recommended Styles")
                    
                    # 4-card grid
                    with gr.Row():
                        style_cols = []
                        style_images = []
                        style_htmls = []
                        for i in range(4):
                            with gr.Column(visible=False, min_width=200) as col:
                                img = gr.Image(label="", show_label=False, interactive=False, height=220)
                                html = gr.HTML()
                                style_cols.append(col)
                                style_images.append(img)
                                style_htmls.append(html)
                    
                    style_empty_lbl = gr.Markdown("_Submit criteria to view style recommendations._")

            def on_get_styles(g, s, o, f):
                recs = rec_service.recommend_styles(g, s, o, f, n=4)
                if not recs:
                    return [gr.update(visible=False)] * 4 + [None] * 4 + [None] * 4 + [gr.update(value="_No recommendations matches found._", visible=True)]

                updates = []
                for i in range(4):
                    if i < len(recs):
                        r = recs[i]
                        title = r.get("style", "Custom Pair")
                        score = r.get("score", 0.85)
                        desc = r.get("description", "A perfectly balanced clothing ensemble.")
                        category = s.capitalize()
                        
                        card_img = _generate_recommendation_card_image(title, score)
                        card_html = f"""
                        <div style="background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.06); border-top: none; padding: 1rem; border-radius: 0 0 8px 8px; font-family: 'Inter', sans-serif;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <span style="font-size: 0.8rem; font-weight: 500; color: #ff9f43; background: rgba(255,159,67,0.1); padding: 0.2rem 0.5rem; border-radius: 4px;">{category}</span>
                                <span style="font-size: 0.85rem; font-weight: 600; color: #ff5252;">{score:.0%} Match</span>
                            </div>
                            <h4 style="color: #fff; margin: 0.3rem 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.3px;">{title}</h4>
                            <p style="font-size: 0.85rem; color: #a0a0b0; margin: 0.5rem 0 0 0; line-height: 1.4; font-weight: 300;">{desc}</p>
                        </div>
                        """
                        updates.append(gr.update(visible=True))
                        updates.append(card_img)
                        updates.append(card_html)
                    else:
                        updates.append(gr.update(visible=False))
                        updates.append(None)
                        updates.append("")

                updates.append(gr.update(visible=False))
                return updates

            style_btn.click(
                on_get_styles,
                inputs=[style_gender, style_pref, style_occasion, style_fit],
                outputs=style_cols + style_images + style_htmls + [style_empty_lbl],
            )

        # ── 2. BRAND RECOMMENDATIONS ─────────────────────────────────────────
        with gr.Tab("🏷️ Brand Matcher"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Your Aesthetic Profile")
                    brand_styles = gr.CheckboxGroup(
                        choices=["streetwear", "minimalist", "luxury", "athletic", "vintage"],
                        value=["minimalist"],
                        label="Preferred Aesthetics",
                    )
                    brand_aesthetic = gr.Textbox(
                        label="Describe target aesthetic",
                        placeholder="Clean silhouettes with high-quality fabrics...",
                        lines=2,
                    )
                    brand_btn = gr.Button("Match Brands", variant="primary")

                with gr.Column(scale=3):
                    gr.Markdown("### 🌟 Matching Brands")
                    
                    # 4-card grid
                    with gr.Row():
                        brand_cols = []
                        brand_images = []
                        brand_htmls = []
                        for i in range(4):
                            with gr.Column(visible=False, min_width=200) as col:
                                img = gr.Image(label="", show_label=False, interactive=False, height=220)
                                html = gr.HTML()
                                brand_cols.append(col)
                                brand_images.append(img)
                                brand_htmls.append(html)
                    
                    brand_empty_lbl = gr.Markdown("_Submit aesthetic details to view brand matches._")

            def on_get_brands(styles, aesthetic):
                recs = rec_service.recommend_brands(styles, aesthetic, n=4)
                if not recs:
                    return [gr.update(visible=False)] * 4 + [None] * 4 + [None] * 4 + [gr.update(value="_No matching brands found._", visible=True)]

                updates = []
                for i in range(4):
                    if i < len(recs):
                        r = recs[i]
                        brand_name = r.get("brand", "Unknown Brand")
                        score = r.get("score", 0.85)
                        category = r.get("category", "Fashion")
                        price = r.get("price_range", "$$")
                        
                        card_img = _generate_recommendation_card_image(brand_name, score)
                        card_html = f"""
                        <div style="background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.06); border-top: none; padding: 1rem; border-radius: 0 0 8px 8px; font-family: 'Inter', sans-serif;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <span style="font-size: 0.8rem; font-weight: 500; color: #3498db; background: rgba(52,152,219,0.1); padding: 0.2rem 0.5rem; border-radius: 4px;">{category}</span>
                                <span style="font-size: 0.85rem; font-weight: 600; color: #ff9f43;">{score:.0%} Match</span>
                            </div>
                            <h4 style="color: #fff; margin: 0.3rem 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.3px;">{brand_name}</h4>
                            <p style="font-size: 0.85rem; color: #a0a0b0; margin: 0.5rem 0 0 0; font-weight: 300;">Price Rating: <span style="color: #2ecc71; font-weight: 500;">{price}</span></p>
                        </div>
                        """
                        updates.append(gr.update(visible=True))
                        updates.append(card_img)
                        updates.append(card_html)
                    else:
                        updates.append(gr.update(visible=False))
                        updates.append(None)
                        updates.append("")

                updates.append(gr.update(visible=False))
                return updates

            brand_btn.click(
                on_get_brands,
                inputs=[brand_styles, brand_aesthetic],
                outputs=brand_cols + brand_images + brand_htmls + [brand_empty_lbl],
            )

        # ── 3. TREND RECOMMENDATIONS ─────────────────────────────────────────
        with gr.Tab("📈 Trend Predictions"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Select Season")
                    trend_season = gr.Dropdown(
                        choices=["spring_summer", "autumn_winter"],
                        value="spring_summer",
                        label="Forecast Season",
                    )
                    trend_btn = gr.Button("Forecast Trends", variant="primary")

                with gr.Column(scale=3):
                    gr.Markdown("### 🌟 Trending Elements")
                    
                    # 4-card grid
                    with gr.Row():
                        trend_cols = []
                        trend_images = []
                        trend_htmls = []
                        for i in range(4):
                            with gr.Column(visible=False, min_width=200) as col:
                                img = gr.Image(label="", show_label=False, interactive=False, height=220)
                                html = gr.HTML()
                                trend_cols.append(col)
                                trend_images.append(img)
                                trend_htmls.append(html)
                    
                    trend_empty_lbl = gr.Markdown("_Submit season details to view trend forecasts._")

            def on_get_trends(season):
                recs = trend_service.forecast_season(season)
                if not recs:
                    return [gr.update(visible=False)] * 4 + [None] * 4 + [None] * 4 + [gr.update(value="_No trend forecasts found._", visible=True)]

                updates = []
                for i in range(4):
                    if i < len(recs):
                        r = recs[i]
                        trend_name = r.get("name", "Unknown Trend")
                        growth = r.get("velocity", "+15% YoY")
                        score = 0.88 if "+" in str(growth) else 0.75
                        desc = r.get("description", "A fast-moving fashion movement.")
                        cat = season.replace("_", " ").title()
                        
                        card_img = _generate_recommendation_card_image(trend_name, score)
                        card_html = f"""
                        <div style="background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.06); border-top: none; padding: 1rem; border-radius: 0 0 8px 8px; font-family: 'Inter', sans-serif;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <span style="font-size: 0.8rem; font-weight: 500; color: #ff5252; background: rgba(255,82,82,0.1); padding: 0.2rem 0.5rem; border-radius: 4px;">{cat}</span>
                                <span style="font-size: 0.85rem; font-weight: 600; color: #2ecc71;">{growth}</span>
                            </div>
                            <h4 style="color: #fff; margin: 0.3rem 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.3px;">{trend_name}</h4>
                            <p style="font-size: 0.85rem; color: #a0a0b0; margin: 0.5rem 0 0 0; line-height: 1.4; font-weight: 300;">{desc}</p>
                        </div>
                        """
                        updates.append(gr.update(visible=True))
                        updates.append(card_img)
                        updates.append(card_html)
                    else:
                        updates.append(gr.update(visible=False))
                        updates.append(None)
                        updates.append("")

                updates.append(gr.update(visible=False))
                return updates

            trend_btn.click(
                on_get_trends,
                inputs=[trend_season],
                outputs=trend_cols + trend_images + trend_htmls + [trend_empty_lbl],
            )

        # ── 4. PERSONALIZED LOOKBOOK ──────────────────────────────────────────
        with gr.Tab("👤 Personalized Suggestions"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 User Preference Model")
                    user_id = gr.Textbox(label="User ID", value="demo_user", lines=1)
                    lookbook_btn = gr.Button("Generate Personalized Lookbook", variant="primary")
                    profile_json = gr.JSON(label="Active Preference Profile")

                with gr.Column(scale=2):
                    gr.Markdown("### 🧥 Your Coordinated Lookbook")
                    lookbook_col = gr.Column(visible=False)
                    with lookbook_col:
                        with gr.Row():
                            lookbook_img = gr.Image(show_label=False, interactive=False, height=280)
                            lookbook_details = gr.HTML()
                    
                    lookbook_empty = gr.Markdown("_Click 'Generate Personalized Lookbook' to compile your profile-matching outfit card._")

            def on_generate_lookbook(uid):
                uid_clean = uid.strip() or "demo_user"
                profile = rec_service.get_user_profile(uid_clean)
                
                # Derive styling parameters
                top_styles = profile.get("top_styles", ["minimalist"])
                colors = profile.get("favorite_colors", ["black"])
                occasions = profile.get("favorite_occasions", ["casual"])
                
                style_query = top_styles[0] if top_styles else "minimalist"
                color_query = colors[0] if colors else "black"
                occasion_query = occasions[0] if occasions else "casual"
                
                # Fetch mock style matching
                recs = rec_service.recommend_styles("unisex", style_query, occasion_query, "regular_fit", n=1)
                best_match = recs[0] if recs else {"style": "Quiet Luxury", "description": "Coordinated minimalist ensemble."}
                
                outfit_title = f"{style_query.title()} {occasion_query.title()} Set"
                card_img = _generate_recommendation_card_image(outfit_title, 0.95)
                
                lookbook_html = f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.08); padding: 1.5rem; border-radius: 8px; font-family: 'Inter', sans-serif;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #ff9f43; background: rgba(255,159,67,0.15); padding: 0.3rem 0.6rem; border-radius: 4px;">{best_match.get('style', 'Tailored Look').upper()}</span>
                        <span style="font-size: 0.9rem; font-weight: 700; color: #ff5252;">95% Compatibility Score</span>
                    </div>
                    <h3 style="color: #fff; margin: 0 0 0.5rem 0; font-size: 1.3rem;">{outfit_title}</h3>
                    <p style="font-size: 0.95rem; color: #d0d0d0; line-height: 1.5; margin: 0.5rem 0; font-weight: 300;">{best_match.get('description', '')}</p>
                    <div style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.8rem; font-size: 0.85rem; color: #888;">
                        <div style="margin-bottom: 0.3rem;">🎨 Preferred Palette: <strong style="color: #ff9f43;">{", ".join(colors)}</strong></div>
                        <div>🧥 Occasion Matcher: <strong style="color: #3498db;">{", ".join(occasions)}</strong></div>
                    </div>
                </div>
                """
                return (
                    profile,
                    gr.update(visible=True),
                    card_img,
                    lookbook_html,
                    gr.update(visible=False)
                )

            lookbook_btn.click(
                on_generate_lookbook,
                inputs=[user_id],
                outputs=[profile_json, lookbook_col, lookbook_img, lookbook_details, lookbook_empty]
            )
