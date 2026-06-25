import gradio as gr
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.fashion_assistant import FashionAssistant

# Initialize the Fashion Assistant with mock embeddings to run lightweight and offline
assistant = FashionAssistant(force_mock_embeddings=True)

# Custom CSS for modern, premium styling (dark mode, HSL accents, clean typography)
CUSTOM_CSS = """
body {
    background-color: #0f111a;
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    color: #e3e6f3;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}
h1 {
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #ff7e5f, #feb47b, #86a8e7, #7f7fd5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-anim 8s ease infinite;
    text-align: center;
    margin-bottom: 0.2em;
}
p.description {
    text-align: center;
    color: #8b92b6;
    font-size: 1.1em;
    margin-bottom: 2em;
}
.tabs {
    border-radius: 12px;
    background: rgba(22, 28, 45, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}
.tabitem {
    padding: 20px !important;
}
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}
.input-box, .output-box {
    border-radius: 8px !important;
    background-color: #1a1e2e !important;
    border: 1px solid #2e344e !important;
}
@keyframes gradient-anim {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
"""

def qa_func(question):
    try:
        res = assistant.answer_question(question)
        ans = res.get("response", "No response generated.")
        docs = res.get("source_documents", [])
        doc_str = ""
        if docs:
            doc_str = "\n\n### 📚 Retrieved Context\n"
            for i, d in enumerate(docs[:3]):
                name = d.get('metadata', {}).get('name', d.get('metadata', {}).get('brand', 'Document'))
                doc_str += f"- **{name}**: {d.get('document', '')[:200]}...\n"
        return ans + doc_str
    except Exception as e:
        return f"Error running RAG pipeline: {str(e)}"

def style_recs(gender, style, occasion, fit):
    try:
        prefs = {
            "gender": gender,
            "style": style,
            "occasion": occasion,
            "fit": fit
        }
        recs = assistant.recommend_styles(prefs, n_results=3)
        if not recs:
            return "No style recommendations match those parameters."
        out = "## 👗 Recommended Styles\n\n"
        for i, r in enumerate(recs):
            out += f"### {i+1}. Style Profile\n{r}\n\n---\n\n"
        return out
    except Exception as e:
        return f"Error recommending styles: {str(e)}"

def brand_recs(preferred_styles, target_brand_profile):
    try:
        profile = {
            "styles": [s.strip() for s in preferred_styles.split(",") if s.strip()],
            "target_aesthetic": target_brand_profile
        }
        recs = assistant.recommend_brands(profile, n_results=3)
        if not recs:
            return "No brand recommendations match those parameters."
        out = "## 🏷️ Recommended Brands\n\n"
        for i, r in enumerate(recs):
            out += f"### {i+1}. Brand\n{r}\n\n---\n\n"
        return out
    except Exception as e:
        return f"Error recommending brands: {str(e)}"

def explain_trend_func(trend_name):
    try:
        res = assistant.explain_trend(trend_name)
        out = f"## 📈 Trend Analysis: {res.get('trend')}\n\n"
        out += f"**Explanation**: {res.get('explanation')}\n\n"
        out += f"- **Confidence Index**: {res.get('confidence', 0.5):.2f}\n"
        out += f"- **Forecast Reasoning**: {res.get('reasoning')}\n"
        return out
    except Exception as e:
        return f"Error retrieving trend: {str(e)}"

def forecast_trends_func(season):
    try:
        recs = assistant.get_trend_forecast(season)
        if not recs:
            return f"No forecasts available for season '{season}'."
        out = f"## 🔮 Trend Forecasts for {season.replace('_', ' ').title()}\n\n"
        for i, r in enumerate(recs):
            out += f"### {i+1}. {r.get('trend', 'Trend')}\n"
            out += f"- **Target Season**: {r.get('season')}\n"
            out += f"- **Growth Potential**: {r.get('growth_rate')}\n"
            out += f"- **Aesthetic Description**: {r.get('explanation')}\n"
            out += f"- **Reasoning**: {r.get('reasoning')}\n\n"
        return out
    except Exception as e:
        return f"Error forecasting trends: {str(e)}"

def semantic_search_func(query):
    try:
        results = assistant.chroma_retriever.retrieve(query, n_results=5)
        if not results:
            return "No matching documents found in vector index."
        out = "## 🔍 ChromaDB Vector Matches\n\n"
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items() if k not in ["document"])
            out += f"### Match {i+1} (Collection: {r.get('collection')} | Similarity Distance: {r.get('distance', 0.0):.4f})\n"
            out += f"> {r.get('document')}\n\n"
            out += f"*Metadata: {meta_str}*\n\n---\n\n"
        return out
    except Exception as e:
        return f"Error during semantic search: {str(e)}"

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")) as demo:
        gr.Markdown("# AI-Powered Fashion Design Assistant")
        gr.Markdown(
            "Welcome to the Week 5 graduation app! Access domain intelligence, personalized design recommendations, trend forecasts, and semantic search queries from a single unified workspace.",
            elem_classes="description"
        )
        
        with gr.Tabs(elem_classes="tabs"):
            
            # Tab 1: Fashion Q&A
            with gr.TabItem("💬 Fashion Q&A", elem_classes="tabitem"):
                gr.Markdown("### Ask anything about fabric qualities, style aesthetics, and fashion theory.")
                with gr.Row():
                    with gr.Column():
                        q_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Why is linen popular in Spring/Summer? Explain its drape and weave.",
                            lines=3,
                            elem_classes="input-box"
                        )
                        q_btn = gr.Button("Submit Question", variant="primary", elem_classes="primary")
                    with gr.Column():
                        q_output = gr.Markdown(label="Answer", elem_classes="output-box")
                q_btn.click(qa_func, inputs=[q_input], outputs=[q_output])

            # Tab 2: Style Recommendation
            with gr.TabItem("👗 Style Recommendation", elem_classes="tabitem"):
                gr.Markdown("### Generate personalized style combinations based on fit, occasion, and preferences.")
                with gr.Row():
                    with gr.Column():
                        gender = gr.Dropdown(
                            choices=["men", "women", "unisex"],
                            value="unisex",
                            label="Gender"
                        )
                        style = gr.Dropdown(
                            choices=["streetwear", "luxury", "formal", "business_casual", "techwear", "minimalist", "vintage", "athleisure"],
                            value="streetwear",
                            label="Style Category"
                        )
                        occasion = gr.Dropdown(
                            choices=["casual", "business_casual", "formal", "party", "sport", "outdoor", "beach", "lounge"],
                            value="casual",
                            label="Occasion"
                        )
                        fit = gr.Dropdown(
                            choices=["slim_fit", "regular_fit", "relaxed_fit", "oversized", "cropped", "skinny", "straight", "athletic_fit"],
                            value="regular_fit",
                            label="Fit Profile"
                        )
                        style_btn = gr.Button("Recommend Styles", variant="primary", elem_classes="primary")
                    with gr.Column():
                        style_output = gr.Markdown(label="Recommendations", elem_classes="output-box")
                style_btn.click(style_recs, inputs=[gender, style, occasion, fit], outputs=[style_output])

            # Tab 3: Brand Recommendation
            with gr.TabItem("🏷️ Brand Recommendation", elem_classes="tabitem"):
                gr.Markdown("### Identify target apparel brands matching your aesthetic profile and preference coordinates.")
                with gr.Row():
                    with gr.Column():
                        preferred_styles = gr.Textbox(
                            label="Preferred Style Themes (comma-separated)",
                            placeholder="streetwear, minimalist, athletic",
                            elem_classes="input-box"
                        )
                        target_brand_profile = gr.Textbox(
                            label="Target Brand Aesthetic Profile",
                            placeholder="Techwear and functional activewear with high-durability fabrics.",
                            lines=3,
                            elem_classes="input-box"
                        )
                        brand_btn = gr.Button("Recommend Brands", variant="primary", elem_classes="primary")
                    with gr.Column():
                        brand_output = gr.Markdown(label="Brand Profiles", elem_classes="output-box")
                brand_btn.click(brand_recs, inputs=[preferred_styles, target_brand_profile], outputs=[brand_output])

            # Tab 4: Trend Forecasting
            with gr.TabItem("📈 Trend Forecasting", elem_classes="tabitem"):
                gr.Markdown("### Explore active design element trends and future fashion forecasts.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Search & Analyze Active Trend")
                        trend_name = gr.Textbox(
                            label="Trend Name",
                            placeholder="Velvet Gown Opulence",
                            elem_classes="input-box"
                        )
                        trend_btn = gr.Button("Analyze Trend", variant="primary", elem_classes="primary")
                        
                        gr.Markdown("#### Forecast Future Trends")
                        season = gr.Dropdown(
                            choices=["spring_summer", "autumn_winter"],
                            value="spring_summer",
                            label="Target Forecast Season"
                        )
                        forecast_btn = gr.Button("Forecast Trends", variant="primary", elem_classes="primary")
                    with gr.Column():
                        trend_output = gr.Markdown(label="Trend Output", elem_classes="output-box")
                trend_btn.click(explain_trend_func, inputs=[trend_name], outputs=[trend_output])
                forecast_btn.click(forecast_trends_func, inputs=[season], outputs=[trend_output])

            # Tab 5: Semantic Search
            with gr.TabItem("🔍 Semantic Search", elem_classes="tabitem"):
                gr.Markdown("### Perform high-dimensional vector search queries directly over our unified ChromaDB index.")
                with gr.Row():
                    with gr.Column():
                        search_input = gr.Textbox(
                            label="Query Vector Index",
                            placeholder="oversized black utility hoodies for skateboarding",
                            lines=2,
                            elem_classes="input-box"
                        )
                        search_btn = gr.Button("Execute Vector Search", variant="primary", elem_classes="primary")
                    with gr.Column():
                        search_output = gr.Markdown(label="Vector Matches", elem_classes="output-box")
                search_btn.click(semantic_search_func, inputs=[search_input], outputs=[search_output])
                
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
