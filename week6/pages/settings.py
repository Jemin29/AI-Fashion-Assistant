"""
Week 6 — Settings Page.

Allows users to configure generation defaults (resolution, model, style, brand),
infrastructure preferences (output folder, GPU/CPU), and application behaviour
(mock toggles, server config).  All settings are persisted to
``week6/outputs/settings.json`` and reloaded automatically on startup.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
_WEEK6_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR = _WEEK6_ROOT / "outputs"
_SETTINGS_FILE = _OUTPUTS_DIR / "settings.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
_RESOLUTION_CHOICES = [
    "512×512",
    "768×768",
    "1024×1024",
    "1024×1536  (portrait)",
    "1536×1024  (landscape)",
]

_MODEL_CHOICES = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "Custom (specify below)",
]

_STYLE_CHOICES = [
    "streetwear",
    "high-fashion / couture",
    "casual / everyday",
    "sportswear / athleisure",
    "vintage / retro",
    "minimalist",
    "bohemian",
    "formal / business",
    "avant-garde",
    "sustainable / eco",
]

_BRAND_CHOICES = [
    "Nike",
    "Gucci",
    "Zara",
    "H&M",
    "Prada",
    "Supreme",
    "Levi's",
    "Uniqlo",
    "Balenciaga",
    "Custom",
]

_DEVICE_CHOICES = ["auto (recommended)", "cuda (GPU)", "cpu (CPU)", "mps (Apple Silicon)"]

_SCHEDULER_CHOICES = [
    "DPMSolverMultistepScheduler (fast)",
    "EulerAncestralDiscreteScheduler (creative)",
    "PNDMScheduler (classic)",
    "DDIMScheduler (deterministic)",
]

_DEFAULT_SETTINGS: Dict[str, Any] = {
    # Generation
    "resolution": "1024×1024",
    "model": _MODEL_CHOICES[0],
    "custom_model": "",
    "style": "streetwear",
    "brand": "Nike",
    "inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": _SCHEDULER_CHOICES[0],
    "seed": -1,
    # Infrastructure
    "device": "auto (recommended)",
    "output_folder": str(_OUTPUTS_DIR / "generated"),
    "output_format": "PNG",
    "output_quality": 95,
    "auto_save": True,
    # Mock / debug
    "global_mock": True,
    "mock_generation": True,
    "mock_controlnet": True,
    "mock_lora": True,
    "mock_rag": False,
    "mock_recommendations": False,
    "mock_trends": False,
    # Server
    "server_host": "0.0.0.0",
    "server_port": 7860,
    "server_share": False,
    # UI
    "chat_history_limit": 20,
    "gallery_max_images": 50,
    "enable_animations": True,
}


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_settings() -> Dict[str, Any]:
    """Load persisted settings from JSON, falling back to defaults for any missing key."""
    settings = dict(_DEFAULT_SETTINGS)
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            settings.update(saved)
        except Exception:
            pass
    return settings


def save_settings(settings: Dict[str, Any]) -> Tuple[bool, str]:
    """Persist settings dict to JSON.  Returns (success, message)."""
    try:
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        # Add a last-saved timestamp
        settings["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
        return True, settings["_saved_at"]
    except Exception as exc:
        return False, str(exc)


def reset_settings() -> Dict[str, Any]:
    """Wipe persisted settings file and return factory defaults."""
    if _SETTINGS_FILE.exists():
        try:
            _SETTINGS_FILE.unlink()
        except Exception:
            pass
    return dict(_DEFAULT_SETTINGS)


# ── Status badge helper ───────────────────────────────────────────────────────

def _status_html(ok: bool, msg: str) -> str:
    colour = "#2ecc71" if ok else "#e74c3c"
    icon = "✅" if ok else "❌"
    return (
        f'<div style="padding:0.6rem 1rem;background:rgba(0,0,0,0.25);border-radius:6px;'
        f'border-left:4px solid {colour};font-size:0.9rem;color:{colour};">'
        f'{icon} {msg}</div>'
    )


# ── Page builder ─────────────────────────────────────────────────────────────

def build_settings_page() -> None:
    """Render the full Settings page inside the current gr.Blocks context."""

    # Load persisted values once at build time
    s = load_settings()

    gr.Markdown("## ⚙️ Application Settings")
    gr.Markdown(
        "Configure generation defaults, hardware preferences, and app behaviour.  "
        "Changes are persisted to `week6/outputs/settings.json` and loaded on restart.",
        elem_classes="studio-subtitle",
    )

    # ── Status bar ──────────────────────────────────────────────────────────
    status_bar = gr.HTML(
        _status_html(True, f"Settings loaded from file — last saved: {s.get('_saved_at', 'never')}.")
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1 — Generation Defaults
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("🎨 Generation Defaults", open=True):
        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                inp_resolution = gr.Dropdown(
                    choices=_RESOLUTION_CHOICES,
                    value=s["resolution"],
                    label="Default Resolution",
                    info="Output canvas size for image generation",
                )
                inp_model = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=s["model"],
                    label="Default Diffusion Model",
                    info="Base checkpoint used by the generation engine",
                )
                inp_custom_model = gr.Textbox(
                    value=s.get("custom_model", ""),
                    label="Custom Model Path / HuggingFace ID",
                    placeholder="e.g. my-org/my-custom-sdxl-lora",
                    info="Only used when 'Custom' is selected above",
                    visible=(s["model"] == "Custom (specify below)"),
                )
                inp_style = gr.Dropdown(
                    choices=_STYLE_CHOICES,
                    value=s["style"],
                    label="Preferred Style",
                    info="Pre-fills the style field on generation pages",
                )
                inp_brand = gr.Dropdown(
                    choices=_BRAND_CHOICES,
                    value=s["brand"],
                    label="Preferred Brand",
                    info="Pre-fills the brand field on style-switching pages",
                )

            # Right column
            with gr.Column(scale=1):
                inp_steps = gr.Slider(
                    minimum=5, maximum=100, step=5,
                    value=s["inference_steps"],
                    label="Default Inference Steps",
                    info="More steps → higher quality but slower (20–50 is typical)",
                )
                inp_cfg = gr.Slider(
                    minimum=1.0, maximum=20.0, step=0.5,
                    value=s["guidance_scale"],
                    label="Default Guidance Scale (CFG)",
                    info="Higher = more prompt-adherent; 7–9 recommended",
                )
                inp_scheduler = gr.Dropdown(
                    choices=_SCHEDULER_CHOICES,
                    value=s["scheduler"],
                    label="Default Scheduler",
                    info="Noise schedule algorithm",
                )
                inp_seed = gr.Number(
                    value=s["seed"],
                    label="Default Seed",
                    info="Set to -1 for random seed on each generation",
                    precision=0,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 2 — Hardware & Output
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("💾 Hardware & Output", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                inp_device = gr.Radio(
                    choices=_DEVICE_CHOICES,
                    value=s["device"],
                    label="Compute Device",
                    info="Select GPU/CPU backend for inference",
                )
                inp_output_folder = gr.Textbox(
                    value=s["output_folder"],
                    label="Output Folder",
                    placeholder="Absolute or relative path to save generated images",
                    info="Folder where all generated images are stored",
                    lines=1,
                )
                open_folder_btn = gr.Button("📂 Open Output Folder", size="sm")

            with gr.Column(scale=1):
                inp_format = gr.Radio(
                    choices=["PNG", "JPEG", "WEBP"],
                    value=s["output_format"],
                    label="Output Image Format",
                )
                inp_quality = gr.Slider(
                    minimum=50, maximum=100, step=5,
                    value=s["output_quality"],
                    label="JPEG/WEBP Compression Quality",
                    info="Only applies to JPEG/WEBP formats",
                )
                inp_auto_save = gr.Checkbox(
                    value=s["auto_save"],
                    label="Auto-save all generated images",
                    info="If unchecked, images are only shown in the UI",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3 — Mock Mode Toggles
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("🧪 Mock / Debug Mode", open=False):
        gr.Markdown(
            "_Mock mode replaces real model inference with fast placeholder responses — "
            "useful when running without a GPU._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp_global_mock = gr.Checkbox(
                    value=s["global_mock"],
                    label="🌐 Global Mock Switch (overrides all below)",
                )
                inp_mock_gen = gr.Checkbox(value=s["mock_generation"], label="Mock SDXL Generation Engine")
                inp_mock_cn = gr.Checkbox(value=s["mock_controlnet"], label="Mock ControlNet Engine")
                inp_mock_lora = gr.Checkbox(value=s["mock_lora"], label="Mock LoRA Personalisation")
            with gr.Column(scale=1):
                inp_mock_rag = gr.Checkbox(value=s["mock_rag"], label="Mock RAG Assistant")
                inp_mock_rec = gr.Checkbox(value=s["mock_recommendations"], label="Mock Recommendations")
                inp_mock_trends = gr.Checkbox(value=s["mock_trends"], label="Mock Trend Forecaster")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 4 — Server & UI
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("🌐 Server & UI", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Gradio Server**")
                inp_host = gr.Textbox(value=s["server_host"], label="Server Host")
                inp_port = gr.Number(
                    value=s["server_port"], label="Server Port", precision=0
                )
                inp_share = gr.Checkbox(
                    value=s["server_share"],
                    label="Enable Gradio Public Share Link",
                    info="Creates a temporary tunnelled URL (requires internet)",
                )
            with gr.Column(scale=1):
                gr.Markdown("**UI Preferences**")
                inp_chat_limit = gr.Slider(
                    minimum=5, maximum=100, step=5,
                    value=s["chat_history_limit"],
                    label="Chat History Limit (messages)",
                )
                inp_gallery_max = gr.Slider(
                    minimum=10, maximum=200, step=10,
                    value=s["gallery_max_images"],
                    label="Gallery Max Images Shown",
                )
                inp_animations = gr.Checkbox(
                    value=s["enable_animations"],
                    label="Enable UI Micro-animations",
                )

    # ── Settings file preview ────────────────────────────────────────────────
    with gr.Accordion("🗂️ Settings File Preview", open=False):
        settings_json_preview = gr.JSON(
            value=s,
            label=f"Current contents of  {_SETTINGS_FILE}",
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Row():
        save_btn = gr.Button("💾 Save Settings", variant="primary", scale=2)
        reset_btn = gr.Button("🔄 Reset to Defaults", variant="secondary", scale=1)
        export_btn = gr.Button("📤 Export settings.json", scale=1)

    export_file = gr.File(label="Download settings.json", visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # Event Handlers
    # ══════════════════════════════════════════════════════════════════════════

    # Show / hide custom model text box
    def _toggle_custom_model(model_val: str) -> gr.update:
        return gr.update(visible=(model_val == "Custom (specify below)"))

    inp_model.change(_toggle_custom_model, inputs=[inp_model], outputs=[inp_custom_model])

    # Open output folder in OS file explorer (best-effort)
    def _open_folder(folder: str) -> str:
        path = Path(folder)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                return _status_html(False, f"Could not create folder: {exc}")
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
            return _status_html(True, f"Opened: {path}")
        except Exception as exc:
            return _status_html(False, f"Could not open folder: {exc}")

    open_folder_btn.click(_open_folder, inputs=[inp_output_folder], outputs=[status_bar])

    # Collect all inputs into an ordered list for save / reset handlers
    _all_inputs = [
        inp_resolution, inp_model, inp_custom_model, inp_style, inp_brand,
        inp_steps, inp_cfg, inp_scheduler, inp_seed,
        inp_device, inp_output_folder, inp_format, inp_quality, inp_auto_save,
        inp_global_mock, inp_mock_gen, inp_mock_cn, inp_mock_lora,
        inp_mock_rag, inp_mock_rec, inp_mock_trends,
        inp_host, inp_port, inp_share,
        inp_chat_limit, inp_gallery_max, inp_animations,
    ]

    def _collect_settings(
        resolution, model, custom_model, style, brand,
        steps, cfg_scale, scheduler, seed,
        device, output_folder, fmt, quality, auto_save,
        global_mock, mock_gen, mock_cn, mock_lora,
        mock_rag, mock_rec, mock_trends,
        host, port, share,
        chat_limit, gallery_max, animations,
    ) -> Dict[str, Any]:
        return {
            "resolution": resolution,
            "model": model,
            "custom_model": custom_model,
            "style": style,
            "brand": brand,
            "inference_steps": int(steps),
            "guidance_scale": float(cfg_scale),
            "scheduler": scheduler,
            "seed": int(seed),
            "device": device,
            "output_folder": output_folder,
            "output_format": fmt,
            "output_quality": int(quality),
            "auto_save": auto_save,
            "global_mock": global_mock,
            "mock_generation": mock_gen,
            "mock_controlnet": mock_cn,
            "mock_lora": mock_lora,
            "mock_rag": mock_rag,
            "mock_recommendations": mock_rec,
            "mock_trends": mock_trends,
            "server_host": host,
            "server_port": int(port),
            "server_share": share,
            "chat_history_limit": int(chat_limit),
            "gallery_max_images": int(gallery_max),
            "enable_animations": animations,
        }

    def on_save(*args) -> Tuple[str, Any]:
        settings_dict = _collect_settings(*args)
        ok, msg = save_settings(settings_dict)
        if ok:
            html = _status_html(True, f"Settings saved successfully at {msg}.")
        else:
            html = _status_html(False, f"Save failed: {msg}")
        return html, settings_dict

    def on_reset() -> list:
        defaults = reset_settings()
        return [
            _status_html(True, "Settings reset to factory defaults. Click 'Save Settings' to persist."),
            defaults["resolution"],
            defaults["model"],
            defaults["custom_model"],
            defaults["style"],
            defaults["brand"],
            float(defaults["inference_steps"]),
            float(defaults["guidance_scale"]),
            defaults["scheduler"],
            float(defaults["seed"]),
            defaults["device"],
            defaults["output_folder"],
            defaults["output_format"],
            float(defaults["output_quality"]),
            defaults["auto_save"],
            defaults["global_mock"],
            defaults["mock_generation"],
            defaults["mock_controlnet"],
            defaults["mock_lora"],
            defaults["mock_rag"],
            defaults["mock_recommendations"],
            defaults["mock_trends"],
            defaults["server_host"],
            float(defaults["server_port"]),
            defaults["server_share"],
            float(defaults["chat_history_limit"]),
            float(defaults["gallery_max_images"]),
            defaults["enable_animations"],
            defaults,  # JSON preview
        ]

    def on_export(*args) -> Tuple[gr.update, str]:
        settings_dict = _collect_settings(*args)
        ok, msg = save_settings(settings_dict)
        if ok and _SETTINGS_FILE.exists():
            return gr.update(value=str(_SETTINGS_FILE), visible=True), \
                   _status_html(True, "Settings saved and ready for download.")
        return gr.update(visible=False), _status_html(False, f"Export failed: {msg}")

    save_btn.click(
        on_save,
        inputs=_all_inputs,
        outputs=[status_bar, settings_json_preview],
    )

    reset_btn.click(
        on_reset,
        inputs=[],
        outputs=[
            status_bar,
            inp_resolution, inp_model, inp_custom_model, inp_style, inp_brand,
            inp_steps, inp_cfg, inp_scheduler, inp_seed,
            inp_device, inp_output_folder, inp_format, inp_quality, inp_auto_save,
            inp_global_mock, inp_mock_gen, inp_mock_cn, inp_mock_lora,
            inp_mock_rag, inp_mock_rec, inp_mock_trends,
            inp_host, inp_port, inp_share,
            inp_chat_limit, inp_gallery_max, inp_animations,
            settings_json_preview,
        ],
    )

    export_btn.click(
        on_export,
        inputs=_all_inputs,
        outputs=[export_file, status_bar],
    )
