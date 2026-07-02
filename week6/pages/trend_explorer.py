"""Week 6 — Trend Explorer Page."""
from __future__ import annotations
from typing import Any
import gradio as gr


def build_trend_explorer_page(trend_service: Any) -> None:
    """Build the Trend Explorer tab."""
    gr.Markdown("## 📈 Trend Explorer — Fashion Trend Intelligence")
    gr.Markdown(
        "Explore velocity-ranked trend forecasts and seasonal fashion intelligence powered by the Week 5 trend engine.",
        elem_classes="studio-subtitle",
    )

    seasons = trend_service.get_seasons()

    with gr.Tabs():
        # ── Tab A: All Trends Dashboard ───────────────────────────────────────
        with gr.TabItem("📊 Trend Dashboard"):
            gr.Markdown("### 🔥 Current Trend Velocity Rankings")

            refresh_btn = gr.Button("🔄 Load Trends", variant="primary")
            trends_table = gr.Markdown("_Click 'Load Trends' to fetch data._")

            from week6.pages.utils import safe_callback

            @safe_callback(1, fallback_values=["_Failed to load trends._"])
            def load_all_trends() -> str:
                result = trend_service.get_all_trends()
                if not result.success:
                    raise gr.Error(result.message)
                trends = result.data or []
                if not trends:
                    return "No trends available."
                md = "| Rank | Trend | Velocity | Growth | Season | Confidence |\n"
                md += "|------|-------|----------|--------|--------|------------|\n"
                for i, t in enumerate(trends[:8], 1):
                    vel_bar = "█" * int(t["velocity"] * 10) + "░" * (10 - int(t["velocity"] * 10))
                    md += (
                        f"| **{i}** | {t['name']} | `{vel_bar}` {t['velocity']:.2f} | "
                        f"{t['growth']} | {t['season'].replace('_', ' ').title()} | "
                        f"{t['confidence']:.0%} |\n"
                    )
                return md

            refresh_btn.click(load_all_trends, inputs=[], outputs=[trends_table])

        # ── Tab B: Trend Deep Dive ────────────────────────────────────────────
        with gr.TabItem("🔬 Trend Analysis"):
            gr.Markdown("### 🔬 Deep Dive — Trend Analysis")
            with gr.Row():
                with gr.Column(scale=1):
                    trend_input = gr.Textbox(
                        label="Trend Name",
                        placeholder="Quiet Luxury, Chrome Metallics, Utility Minimalism...",
                        lines=1,
                    )
                    analyze_btn = gr.Button("🔍 Analyze Trend", variant="primary", size="lg")

                    gr.Markdown("**Quick Select:**")
                    quick_trends = ["Quiet Luxury", "Chrome Metallics", "Utility Minimalism", "Biomorphic Prints"]
                    for qt in quick_trends:
                        qb = gr.Button(qt, variant="secondary", size="sm")
                        qb.click(fn=lambda t=qt: t, outputs=[trend_input])

                with gr.Column(scale=2):
                    trend_detail = gr.Markdown("_Select or type a trend name to analyze._")

            @safe_callback(1, fallback_values=["_Analysis failed._"])
            def on_analyze(trend_name: str) -> str:
                if not trend_name.strip():
                    return "_Please enter a trend name._"
                result = trend_service.explain_trend(trend_name)
                if not result.success:
                    raise gr.Error(result.message)
                info = result.data or {}
                vel = info.get("velocity", 0.0)
                vel_bar = "█" * int(vel * 10) + "░" * (10 - int(vel * 10))
                conf = info.get("confidence", 0.0)
                influences = info.get("key_influences", [])
                demographics = info.get("target_demographics", [])

                return f"""
## 📈 {info.get('trend', trend_name)}

{info.get('explanation', '')}

---
| Metric | Value |
|--------|-------|
| **Velocity Score** | `{vel_bar}` {vel:.2f} |
| **Confidence** | {conf:.0%} |
| **Growth Rate** | {info.get('growth_rate', 'N/A')} |
| **Target Season** | {info.get('target_season', 'N/A').replace('_', ' ').title()} |

**Key Influences**: {', '.join(influences) if influences else 'N/A'}  
**Target Demographics**: {', '.join(demographics) if demographics else 'N/A'}
"""

            analyze_btn.click(on_analyze, inputs=[trend_input], outputs=[trend_detail])

        # ── Tab C: Seasonal Forecast ──────────────────────────────────────────
        with gr.TabItem("🔮 Seasonal Forecast"):
            gr.Markdown("### 🔮 Seasonal Trend Forecasts")
            with gr.Row():
                with gr.Column(scale=1):
                    season_select = gr.Radio(
                        choices=seasons,
                        value=seasons[0],
                        label="Target Season",
                    )
                    forecast_btn = gr.Button("🔮 Generate Forecast", variant="primary", size="lg")

                with gr.Column(scale=2):
                    forecast_output = gr.Markdown("_Select a season and click 'Generate Forecast'._")

            @safe_callback(1, fallback_values=["_Forecast failed._"])
            def on_forecast(season: str) -> str:
                result = trend_service.forecast_season(season)
                if not result.success:
                    raise gr.Error(result.message)
                trends = result.data or []
                if not trends:
                    return f"No forecasts for '{season}'."

                season_label = season.replace("_", " ").title()
                md = f"## 🔮 Trend Forecast — {season_label}\n\n"
                for i, t in enumerate(trends, 1):
                    md += f"### {i}. {t.get('name', t.get('trend', 'Trend'))}\n"
                    md += f"- **Growth**: {t.get('growth', t.get('growth_rate', 'N/A'))}\n"
                    md += f"- **Explanation**: {t.get('explanation', '')}\n"
                    if t.get("reasoning"):
                        md += f"- **Reasoning**: {t.get('reasoning')}\n"
                    md += "\n---\n\n"
                return md

            forecast_btn.click(on_forecast, inputs=[season_select], outputs=[forecast_output])
