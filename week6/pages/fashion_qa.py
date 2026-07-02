"""Week 6 — Fashion Q&A Page (RAG-powered conversational assistant)."""
from __future__ import annotations
from typing import Any, List, Tuple
import gradio as gr

_EXAMPLE_QUESTIONS = [
    "What fabrics are best for summer fashion?",
    "Explain the difference between streetwear and athleisure.",
    "What are the key characteristics of minimalist fashion?",
    "Which color palettes are trending for Autumn/Winter 2025?",
    "How do I style a capsule wardrobe for business casual?",
    "What makes linen better than cotton in warm climates?",
]


def build_fashion_qa_page(rag_service: Any) -> None:
    """Build the Fashion Q&A tab content."""
    gr.Markdown("## 💬 Fashion Q&A — RAG-Powered Fashion Intelligence")
    gr.Markdown(
        "Ask any question about fashion, fabrics, styles, or trends. "
        "Answers are grounded in the 556-item fashion knowledge base using ChromaDB retrieval.",
        elem_classes="studio-subtitle",
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Fashion Assistant",
                height=450,
                show_label=True,
                avatar_images=(None, "🤖"),
                render_markdown=True,
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="",
                    placeholder="Ask a fashion question...",
                    lines=1,
                    scale=4,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            clear_chat_btn = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
            for q in _EXAMPLE_QUESTIONS:
                example_btn = gr.Button(q, variant="secondary", size="sm")
                example_btn.click(
                    fn=lambda q=q: q,
                    outputs=[question_input],
                )

            gr.Markdown("---")
            gr.Markdown("### 📚 Retrieved Sources")
            sources_display = gr.Markdown("_Ask a question to see retrieved knowledge base sources._")

    # ── Semantic Search section ───────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 🔍 Vector Similarity Search")
    gr.Markdown("Query the ChromaDB vector index directly for raw semantic similarity results.")

    with gr.Row():
        with gr.Column(scale=2):
            search_query = gr.Textbox(
                label="Search Query",
                placeholder="oversized black utility jacket streetwear...",
                lines=1,
            )
            search_btn = gr.Button("🔍 Execute Search", variant="primary")
        with gr.Column(scale=1):
            n_results = gr.Slider(1, 10, value=5, step=1, label="Number of Results")

    search_results = gr.Markdown(label="Search Results")

    # ── Event handlers ────────────────────────────────────────────────────────
    from week6.pages.utils import safe_callback
    _chat_history: List[Tuple[str, str]] = []

    @safe_callback(3, fallback_values=[[], "", "_Error occurred._"])
    def on_send(question: str, history: List):
        if not question.strip():
            return history, "", "_No question entered._"

        result = rag_service.answer_question(question)
        if not result.success:
            raise gr.Error(result.message)
        data_payload = result.data or {}
        answer = data_payload.get("response", "No response generated.")
        docs = data_payload.get("source_documents", [])

        # Format citations
        if docs:
            sources_md = "**Retrieved from knowledge base:**\n\n"
            for i, d in enumerate(docs[:3], 1):
                meta = d.get("metadata", {})
                name = meta.get("name") or meta.get("brand") or meta.get("trend") or f"Document {i}"
                snippet = d.get("document", "")[:150]
                dist = d.get("distance", 0.0)
                sources_md += f"**{i}. {name}** _(similarity: {1-dist:.2f})_\n> {snippet}...\n\n"
        else:
            sources_md = "_No specific documents retrieved._"

        history = history or []
        history.append((question, answer))
        return history, "", sources_md

    @safe_callback(1, fallback_values=["_Search failed._"])
    def on_search(query: str, n: int) -> str:
        if not query.strip():
            return "_Enter a search query._"
        result = rag_service.semantic_search(query, n_results=int(n))
        if not result.success:
            raise gr.Error(result.message)
        results = result.data or []
        if not results:
            return "No results found."

        md = f"**{len(results)} results for:** _{query}_\n\n---\n\n"
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            name = meta.get("name") or meta.get("brand") or f"Result {i}"
            dist = r.get("distance", 0.0)
            coll = r.get("collection", "fashion_knowledge")
            doc = r.get("document", "")[:200]
            md += f"### {i}. {name}\n"
            md += f"> {doc}...\n\n"
            md += f"*Collection: `{coll}` | Similarity Distance: `{dist:.4f}`*\n\n---\n\n"
        return md

    @safe_callback(3, fallback_values=[[], "", "_Conversation cleared._"])
    def on_clear():
        return [], "", "_Conversation cleared._"

    send_btn.click(on_send, inputs=[question_input, chatbot], outputs=[chatbot, question_input, sources_display])
    question_input.submit(on_send, inputs=[question_input, chatbot], outputs=[chatbot, question_input, sources_display])
    search_btn.click(on_search, inputs=[search_query, n_results], outputs=[search_results])
    clear_chat_btn.click(on_clear, inputs=[], outputs=[chatbot, question_input, sources_display])
