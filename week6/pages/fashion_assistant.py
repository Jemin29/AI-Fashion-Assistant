"""
Week 6 — Fashion RAG Chat Assistant Page.
Integrates the Week 5 RAG coordinator with a chatbot interface, grounding citations,
and interactive capability builders for fabric guidance, trend forecasting, styling advice,
and brand suggestions.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

_SUGGESTIONS = [
    ("🌾 Linen vs Cotton", "What makes linen better than cotton in warm climates?"),
    ("📈 Quiet Luxury Trend", "Explain the quiet luxury trend and its key elements."),
    ("👔 Capsule Wardrobe", "How do I style a capsule wardrobe for business casual?"),
    ("💎 Gucci Aesthetics", "What are the core aesthetics and palettes of Gucci?"),
    ("🏃 Streetwear vs Athleisure", "Explain the difference between streetwear and athleisure."),
]


def build_fashion_assistant_page(rag_service: Any) -> None:
    """Build the conversational RAG Fashion Assistant tab layout with interactive widgets."""
    gr.Markdown("## 💬 Fashion Assistant — RAG-Grounded Chat")
    gr.Markdown(
        "Interact with our conversational guide grounded in the fashion intelligence knowledge base using ChromaDB.",
        elem_classes="studio-subtitle",
    )

    with gr.Row():
        # ── Left Column: Conversation Chat ──────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversational Fashion Guide",
                height=480,
                show_label=True,
                avatar_images=(None, "🤖"),
                render_markdown=True,
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask about fabric care, streetwear history, or quiet luxury...",
                    container=False,
                    scale=4,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")

        # ── Right Column: Citations & Contextual Info ────────────────────────
        with gr.Column(scale=1):
            with gr.Tabs() as control_tabs:
                with gr.Tab("📚 Grounded Citations", id="citations_tab"):
                    citations_display = gr.Markdown("_Submit a question or use the widgets below to view grounding citations._")

                with gr.Tab("🛠️ Interactive Capabilities", id="widgets_tab"):
                    gr.Markdown("### Instant RAG Assistant Features")
                    
                    # 1. Fabric Guidance Accordion
                    with gr.Accordion("🌾 Fabric Guidance", open=False):
                        fabric_drop = gr.Dropdown(
                            choices=["linen", "denim", "silk", "cotton", "wool", "cashmere"],
                            value="linen",
                            label="Select Fabric",
                        )
                        fabric_btn = gr.Button("Get Fabric Guidance", size="sm", variant="secondary")

                    # 2. Trend Explanation Accordion
                    with gr.Accordion("📈 Trend Radar", open=False):
                        trend_drop = gr.Dropdown(
                            choices=["Quiet Luxury", "Cyber Utility Wear", "Patterned Silk Minimalism", "Organic Linen Loungewear", "Active Loungewear Tech"],
                            value="Quiet Luxury",
                            label="Select Trend",
                        )
                        custom_trend = gr.Textbox(
                            placeholder="Or type a custom trend...",
                            label="Custom Trend",
                        )
                        trend_btn = gr.Button("Explain Trend", size="sm", variant="secondary")

                    # 3. Styling Advice Accordion
                    with gr.Accordion("👔 Style Advisor", open=False):
                        style_drop = gr.Dropdown(
                            choices=["streetwear", "athleisure", "minimalist", "luxury", "vintage"],
                            value="minimalist",
                            label="Style Category",
                        )
                        occasion_drop = gr.Dropdown(
                            choices=["casual", "formal", "business casual", "sports"],
                            value="business casual",
                            label="Occasion",
                        )
                        color_drop = gr.Dropdown(
                            choices=["neutral tones", "all black", "earthy brown", "vibrant accents"],
                            value="neutral tones",
                            label="Color Palette",
                        )
                        style_btn = gr.Button("Get Styling Advice", size="sm", variant="secondary")

                    # 4. Brand Suggestions Accordion
                    with gr.Accordion("💎 Brand Matcher", open=False):
                        brand_style_drop = gr.Dropdown(
                            choices=["streetwear", "minimalist", "luxury", "athleisure"],
                            value="streetwear",
                            label="Brand Style Aesthetic",
                        )
                        brand_btn = gr.Button("Suggest Brands", size="sm", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### 💡 Quick Suggestions")
            with gr.Row():
                for label, q_text in _SUGGESTIONS:
                    sug_btn = gr.Button(label, variant="secondary", size="sm")
                    sug_btn.click(lambda q=q_text: q, outputs=[chat_input])

    # ── Event Handlers ────────────────────────────────────────────────────────

    def on_send(question: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str]:
        if not question.strip():
            return history, "", "_No question entered._"

        # Use the chat method to run intent routing and retrieve citations
        res = rag_service.chat(question)
        answer = res.get("response", "No answer found.")
        docs = res.get("source_documents", [])

        if docs:
            sources_md = "#### 📚 Retrieved Citations\n\n"
            for i, d in enumerate(docs[:3], 1):
                meta = d.get("metadata", {})
                name = meta.get("name") or meta.get("brand") or meta.get("trend") or meta.get("style") or f"Document {i}"
                snippet = d.get("document", "")[:180]
                dist = d.get("distance", 0.0)
                sources_md += f"**{i}. {str(name).title()}** _(similarity: {1-dist:.2f})_\n"
                sources_md += f"> {snippet}...\n\n"
        else:
            sources_md = "_No specific grounding documents retrieved from vector index._"

        history = history or []
        history.append((question, answer))
        return history, "", sources_md

    def on_fabric_click(fabric: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str]:
        query = f"Explain the properties, durability, breathability, and weight of {fabric} fabric."
        return on_send(query, history)

    def on_trend_click(trend: str, custom: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str]:
        selected_trend = custom.strip() if custom.strip() else trend
        query = f"Provide a trend explanation and growth forecast details for {selected_trend}."
        return on_send(query, history)

    def on_style_click(style: str, occasion: str, color: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str]:
        query = f"Provide styling advice for a {color} look in {style} style suitable for a {occasion} occasion."
        return on_send(query, history)

    def on_brand_click(style: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str]:
        query = f"Recommend brand suggestions that match a {style} style aesthetic."
        return on_send(query, history)

    def on_clear():
        return [], "", "_Conversation cleared._"

    # Chat Events
    send_btn.click(
        on_send,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
    chat_input.submit(
        on_send,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
    clear_btn.click(
        on_clear,
        outputs=[chatbot, chat_input, citations_display],
    )

    # Capability Widget Events
    fabric_btn.click(
        on_fabric_click,
        inputs=[fabric_drop, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
    trend_btn.click(
        on_trend_click,
        inputs=[trend_drop, custom_trend, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
    style_btn.click(
        on_style_click,
        inputs=[style_drop, occasion_drop, color_drop, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
    brand_btn.click(
        on_brand_click,
        inputs=[brand_style_drop, chatbot],
        outputs=[chatbot, chat_input, citations_display],
    )
