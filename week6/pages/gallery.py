"""
Week 6 — Generated Image Gallery (History-Powered).

Displays all creative images generated during the session or past sessions.
Supports searching, filtering, sorting, rating, editing notes, exporting,
downloading, and deleting entries from both database and disk.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from week6.services.history_manager import HistoryManager, HistoryEntry

try:
    from week6.gradio_app.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ── Paths & Manager ───────────────────────────────────────────────────────────
_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
_MGR = HistoryManager()


def sync_history_with_disk(mgr: HistoryManager) -> None:
    """Scan outputs folders and register any unregistered PNG files in the history store."""
    gen_dir = _OUTPUTS_DIR / "generated"
    sketch_dir = _OUTPUTS_DIR / "sketches"

    png_files = []
    if gen_dir.exists():
        png_files.extend(list(gen_dir.glob("*.png")))
    if sketch_dir.exists():
        png_files.extend(list(sketch_dir.glob("*.png")))

    # Get set of all currently indexed image paths (normalized to absolute paths)
    with mgr._lock:
        indexed_paths = {
            str(Path(e.image_path).resolve())
            for e in mgr._records.values()
            if not e.deleted and e.image_path
        }

    logger.info("Scanning outputs to sync with history store...")
    synced_count = 0
    for p in png_files:
        abs_path = str(p.resolve())
        if abs_path in indexed_paths:
            continue

        # Register this file
        json_path = p.with_suffix(".json")
        meta = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning("Failed to load sidecar JSON metadata for %s: %s", p.name, e)

        # Parse basic fields
        prompt = meta.get("prompt") or meta.get("enriched_prompt")
        if not prompt:
            name = p.stem
            is_cn = name.startswith("cn") or "sketch" in str(p.parent)
            is_lora = "lora" in name or name.startswith("brand")
            prompt = "Streetwear fashion editorial design photoshoot"
            if is_cn:
                prompt = "Conditioned design render from input sketch template"
            elif is_lora:
                prompt = "Nike/Gucci brand personalized design study"

        try:
            name = p.stem
            is_cn = name.startswith("cn") or "sketch" in str(p.parent)
            is_lora = "lora" in name or name.startswith("brand")
            service = "controlnet" if is_cn else ("lora" if is_lora else "generation")
            
            # Format model name
            model_default = "SDXL v1.0 (Mock)"
            if is_cn:
                model_default = "ControlNet Canny (Mock)"
            elif is_lora:
                model_default = "LoRA Adapter (Mock)"
                
            mgr.record(
                prompt=prompt,
                image_path=abs_path,
                negative_prompt=meta.get("negative_prompt", ""),
                model=meta.get("model") or model_default,
                style=meta.get("style_preset") or meta.get("mode") or "",
                mode=meta.get("run_mode") or meta.get("mode") or "mock",
                service=service,
                brand=meta.get("brand", ""),
                conditioning_mode=meta.get("mode") if is_cn else "",
                resolution=f"{meta.get('width', 512)}x{meta.get('height', 512)}",
                seed=meta.get("seed"),
                steps=meta.get("steps", 0),
                guidance_scale=meta.get("cfg", 7.5),
                lora_scale=meta.get("lora_scale", 0.0),
                latency_ms=meta.get("latency_ms", 0.0),
            )
            synced_count += 1
        except Exception as e:
            logger.error("Failed to register existing file %s in history: %s", p.name, e)

    if synced_count > 0:
        logger.info("Successfully registered %d existing disk designs in history manager", synced_count)


# Synchronize on load
try:
    sync_history_with_disk(_MGR)
except Exception as e:
    logger.error("Error during startup history synchronization: %s", e)


def query_history(
    search_q: str, filter_type: str, sort_by: str, page: int, page_size: int = 12
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], str, gr.update, gr.update]:
    """Helper to query the HistoryManager and return formatted components."""
    filters: Dict[str, Any] = {}
    if filter_type != "all":
        filters["service"] = filter_type

    entries, total = _MGR.search(
        search_q, page=page, page_size=page_size, sort=sort_by, filters=filters
    )
    
    # Format entries for gr.State
    entries_dicts = [e.to_dict() for e in entries]
    
    # Format for gr.Gallery: (image_path, short_caption)
    gallery_data = []
    for e in entries:
        if e.image_exists:
            gallery_data.append((e.image_path, e.short_prompt))
        else:
            # Fallback placeholder if file was deleted externally
            gallery_data.append((
                str(Path(__file__).parent.parent / "assets" / "placeholder.png"),
                f"[Missing] {e.short_prompt}"
            ))

    # Pagination calculation
    max_page = max(1, (total + page_size - 1) // page_size)
    page_str = f"**Page {page} of {max_page}** (Total: `{total}` concepts)"
    
    prev_update = gr.update(interactive=(page > 1))
    next_update = gr.update(interactive=(page < max_page))
    
    return entries_dicts, gallery_data, page_str, prev_update, next_update


def build_gallery_page() -> None:
    """Build the history-powered session & persistent Generated Image Gallery tab."""
    gr.Markdown("## 🖼️ AI Creative Studio Design Gallery")
    gr.Markdown(
        "Browse, filter, and review all fashion concepts generated. Write reviews, add ratings, and export reports.",
        elem_classes="studio-subtitle",
    )

    # ── State Trackers ────────────────────────────────────────────────────────
    current_entries_state = gr.State(value=[])
    current_page_state = gr.State(value=1)
    page_size_val = 12

    with gr.Row():
        # ── Left Column: Grid & Filters ───────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="🔍 Search prompt text or tags...",
                    label="",
                    container=False,
                    scale=3
                )
                filter_dropdown = gr.Dropdown(
                    choices=[
                        ("All Creative Types", "all"),
                        ("🎨 Text-to-Fashion", "generation"),
                        ("✏️ Sketch-to-Design", "controlnet"),
                        ("🏷️ Brand & Mixer LoRAs", "lora")
                    ],
                    value="all",
                    label="",
                    container=False,
                    scale=2
                )
                sort_dropdown = gr.Dropdown(
                    choices=[
                        ("📅 Newest First", "newest"),
                        ("⏳ Oldest First", "oldest"),
                        ("⭐ Highest Rating", "rating"),
                        ("⚡ Fastest Latency", "latency"),
                        ("🔤 Prompt (A-Z)", "prompt")
                    ],
                    value="newest",
                    label="",
                    container=False,
                    scale=2
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            gallery = gr.Gallery(
                label="Creative Studio Gallery Grid",
                columns=3,
                rows=4,
                height=650,
                allow_preview=False,
                elem_id="studio-gallery-grid"
            )

            # Pagination row
            with gr.Row(elem_classes="pagination-row"):
                prev_btn = gr.Button("⬅️ Previous Page", variant="secondary", interactive=False)
                page_indicator = gr.Markdown("**Page 1 of 1** (Total: `0` concepts)", elem_id="page-indicator")
                next_btn = gr.Button("Next Page ➡️", variant="secondary", interactive=False)

        # ── Right Column: Selected Design Inspector ───────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Concept Inspector")
            
            with gr.Column(visible=False) as details_card:
                selected_entry_id = gr.Textbox(visible=False)
                
                selected_image = gr.Image(
                    label="",
                    show_label=False,
                    interactive=False,
                    height=280
                )
                
                prompt_text = gr.Markdown()
                
                # Review section
                with gr.Group():
                    rating_input = gr.Slider(
                        minimum=0,
                        maximum=5,
                        step=1,
                        label="Rating (1-5 Stars, 0 = Unrated)"
                    )
                    notes_input = gr.Textbox(
                        label="Design Notes / Review comments",
                        placeholder="Add design review notes...",
                        lines=2
                    )
                    save_review_btn = gr.Button("💾 Save Review", variant="primary")
                    review_status = gr.Markdown("")

                # Technical accordion
                with gr.Accordion("⚙️ Detailed Parameters", open=False):
                    technical_params_html = gr.HTML()

                gr.Markdown("---")
                with gr.Row():
                    download_file = gr.File(label="📥 Download Concept Image")
                    delete_btn = gr.Button("🗑️ Delete Concept", variant="stop")

            empty_details_lbl = gr.Markdown(
                "_Select a concept from the gallery grid to inspect generation details, add reviews, and download assets._"
            )

            # History Export Card
            with gr.Group(elem_classes="export-card"):
                gr.Markdown("### 📤 Export Studio Dataset")
                export_format = gr.Radio(
                    choices=[("JSON Backup", "json"), ("CSV Dataset", "csv"), ("Markdown Report", "md")],
                    value="json",
                    label="Format Type"
                )
                export_btn = gr.Button("📤 Generate Export", variant="secondary")
                export_file = gr.File(label="📥 Download Exported Dataset", visible=False)

    # ── Event Handlers ────────────────────────────────────────────────────────
    from week6.pages.utils import safe_callback

    @safe_callback(7, fallback_values=[[], [], "Page 1", gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=True)])
    def handle_refresh(search_q, filt, sort_by, page_num):
        entries_dicts, gallery_data, page_str, prev_up, next_up = query_history(
            search_q, filt, sort_by, page_num, page_size_val
        )
        return (
            entries_dicts,
            gallery_data,
            page_str,
            prev_up,
            next_up,
            gr.update(visible=False),
            gr.update(visible=True)
        )

    @safe_callback(8, fallback_values=[1, [], [], "Page 1", gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=True)])
    def handle_prev_page(search_q, filt, sort_by, current_page):
        new_page = max(1, current_page - 1)
        entries_dicts, gallery_data, page_str, prev_up, next_up = query_history(
            search_q, filt, sort_by, new_page, page_size_val
        )
        return (
            new_page,
            entries_dicts,
            gallery_data,
            page_str,
            prev_up,
            next_up,
            gr.update(visible=False),
            gr.update(visible=True)
        )

    @safe_callback(8, fallback_values=[1, [], [], "Page 1", gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=True)])
    def handle_next_page(search_q, filt, sort_by, current_page):
        new_page = current_page + 1
        entries_dicts, gallery_data, page_str, prev_up, next_up = query_history(
            search_q, filt, sort_by, new_page, page_size_val
        )
        return (
            new_page,
            entries_dicts,
            gallery_data,
            page_str,
            prev_up,
            next_up,
            gr.update(visible=False),
            gr.update(visible=True)
        )

    @safe_callback(9, fallback_values=[gr.update()] * 9)
    def handle_select(evt: gr.SelectData, entries_dicts: List[Dict[str, Any]]):
        index = evt.index
        if index >= len(entries_dicts):
            return [gr.update()] * 9

        entry_dict = entries_dicts[index]
        entry = HistoryEntry.from_dict(entry_dict)
        
        prompt_val = f"### 📝 Prompt\n> {entry.prompt}"
        
        # Build technical parameters summary
        params_html = f"""
        <div style="font-family: monospace; font-size: 0.85rem; line-height: 1.5; color: #a0a0b0;">
            <div><b>Record ID</b> : {entry.id}</div>
            <div><b>Model</b>     : <span style="color: #ff9f43;">{entry.model}</span></div>
            <div><b>Resolution</b>: {entry.resolution}</div>
            <div><b>Seed</b>       : {entry.seed if entry.seed is not None else 'N/A'}</div>
            <div><b>Steps / CFG</b>: {entry.steps} / {entry.guidance_scale}</div>
            <div><b>Latency</b>   : {entry.latency_ms} ms</div>
            <div><b>Service</b>   : <code>{entry.service}</code></div>
            <div><b>Created At</b> : {entry.created_at}</div>
        """
        if entry.brand:
            params_html += f"<div><b>LoRA Brand</b>: {entry.brand.upper()}</div>"
        if entry.conditioning_mode:
            params_html += f"<div><b>Control</b>   : {entry.conditioning_mode}</div>"
        params_html += f"<div><b>Image Path</b>: <small style='word-break: break-all;'>{entry.image_path}</small></div></div>"
        
        return (
            gr.update(visible=True),       # details_card visible
            gr.update(visible=False),      # empty_details_lbl invisible
            entry.image_path,              # selected_image
            prompt_val,                    # prompt_text
            int(entry.rating),             # rating_input
            entry.notes,                   # notes_input
            params_html,                   # technical_params_html
            entry.image_path,              # download_file
            entry.id,                      # selected_entry_id
        )

    @safe_callback(2, fallback_values=["❌ Action failed.", gr.update()])
    def handle_save_review(entry_id, rating, notes):
        if not entry_id:
            return "⚠️ No design selected.", gr.update()
        
        try:
            _MGR.update_entry(entry_id, rating=int(rating), notes=notes)
            return "✅ Review saved successfully!", gr.update()
        except Exception as e:
            return f"❌ Save failed: {e}", gr.update()

    @safe_callback(7, fallback_values=[[], [], "Page 1", gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=True)])
    def handle_delete(entry_id, search_q, filt, sort_by, current_page):
        if not entry_id:
            return [gr.update()] * 7
            
        try:
            # Delete from history manager
            _MGR.delete_entry(entry_id, delete_file=True)
        except Exception as e:
            logger.error("Failed to delete history entry %s: %s", entry_id, e)
            
        # Refresh current page view
        entries_dicts, gallery_data, page_str, prev_up, next_up = query_history(
            search_q, filt, sort_by, current_page, page_size_val
        )
        return (
            entries_dicts,
            gallery_data,
            page_str,
            prev_up,
            next_up,
            gr.update(visible=False),
            gr.update(visible=True)
        )

    @safe_callback(1, fallback_values=[gr.update(visible=False)])
    def handle_export(export_format_type):
        try:
            if export_format_type == "json":
                out_path = _MGR.export_json()
            elif export_format_type == "csv":
                out_path = _MGR.export_csv()
            else:
                out_path = _MGR.export_markdown()
            
            if out_path and Path(out_path).exists():
                return gr.update(value=out_path, visible=True)
            else:
                return gr.update(visible=False)
        except Exception as e:
            logger.error("Failed to export history dataset: %s", e)
            return gr.update(visible=False)

    # ── Event Binding ─────────────────────────────────────────────────────────

    # Page loads and changes
    load_events = [search_box.change, filter_dropdown.change, sort_dropdown.change, refresh_btn.click]
    for event in load_events:
        event(
            handle_refresh,
            inputs=[search_box, filter_dropdown, sort_dropdown, current_page_state],
            outputs=[
                current_entries_state,
                gallery,
                page_indicator,
                prev_btn,
                next_btn,
                details_card,
                empty_details_lbl
            ]
        )

    prev_btn.click(
        handle_prev_page,
        inputs=[search_box, filter_dropdown, sort_dropdown, current_page_state],
        outputs=[
            current_page_state,
            current_entries_state,
            gallery,
            page_indicator,
            prev_btn,
            next_btn,
            details_card,
            empty_details_lbl
        ]
    )

    next_btn.click(
        handle_next_page,
        inputs=[search_box, filter_dropdown, sort_dropdown, current_page_state],
        outputs=[
            current_page_state,
            current_entries_state,
            gallery,
            page_indicator,
            prev_btn,
            next_btn,
            details_card,
            empty_details_lbl
        ]
    )

    # Gallery selection
    gallery.select(
        handle_select,
        inputs=[current_entries_state],
        outputs=[
            details_card,
            empty_details_lbl,
            selected_image,
            prompt_text,
            rating_input,
            notes_input,
            technical_params_html,
            download_file,
            selected_entry_id
        ]
    )

    # Save review notes
    save_review_btn.click(
        handle_save_review,
        inputs=[selected_entry_id, rating_input, notes_input],
        outputs=[review_status, review_status]
    )

    # Delete entry
    delete_btn.click(
        handle_delete,
        inputs=[selected_entry_id, search_box, filter_dropdown, sort_dropdown, current_page_state],
        outputs=[
            current_entries_state,
            gallery,
            page_indicator,
            prev_btn,
            next_btn,
            details_card,
            empty_details_lbl
        ]
    )

    # Export history triggers
    export_btn.click(
        handle_export,
        inputs=[export_format],
        outputs=[export_file]
    )

    # Initial startup query on load
    dummy_btn = gr.Button("dummy_load", visible=False)
    dummy_btn.click(
        handle_refresh,
        inputs=[search_box, filter_dropdown, sort_dropdown, current_page_state],
        outputs=[
            current_entries_state,
            gallery,
            page_indicator,
            prev_btn,
            next_btn,
            details_card,
            empty_details_lbl
        ]
    )
