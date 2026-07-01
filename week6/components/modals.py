"""
Week 6 — Reusable Modal / Dialog Components.

Gradio doesn't have a native modal widget, so modals are implemented as
hidden ``gr.Column`` panels that slide into view on demand.  Each factory
function returns a tuple of Gradio components plus the visibility-toggle
callbacks needed to open / close the modal.

Modal families
--------------
- ``confirm_modal``        — Yes / No confirmation dialog
- ``info_modal``           — Read-only information overlay
- ``form_modal``           — Form inputs inside a dialog overlay
- ``image_preview_modal``  — Full-size image preview dialog
- ``progress_modal``       — Task-in-progress blocking overlay
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import gradio as gr

# ── Design tokens ─────────────────────────────────────────────────────────────
_OVERLAY_BG  = "rgba(0,0,0,0.65)"
_SURFACE     = "#16162a"
_BORDER      = "rgba(255,255,255,0.09)"
_ACCENT      = "#ff9f43"
_DANGER      = "#ff5252"
_TEXT_PRIM   = "#f0f0f5"
_TEXT_SEC    = "#a0a0b0"

_MODAL_WRAP_STYLE = (
    "background:#16162a;"
    "border:1px solid rgba(255,255,255,0.09);"
    "border-radius:14px;"
    "padding:2rem;"
    "max-width:520px;"
    "margin:auto;"
    "box-shadow:0 24px 60px rgba(0,0,0,0.5);"
)

_OVERLAY_STYLE = (
    "position:fixed;top:0;left:0;width:100%;height:100%;"
    f"background:{_OVERLAY_BG};"
    "display:flex;align-items:center;justify-content:center;"
    "z-index:9999;"
    "backdrop-filter:blur(4px);"
)


# ── Confirm modal ─────────────────────────────────────────────────────────────

def confirm_modal(
    *,
    title: str = "Are you sure?",
    body: str = "This action cannot be undone.",
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    confirm_variant: str = "stop",
    elem_id_prefix: str = "confirm",
) -> Tuple[gr.Column, gr.Button, gr.Button, Callable, Callable]:
    """Render a hidden Yes / No confirmation dialog.

    The dialog is initially hidden.  Call ``open_fn()`` (connected to a
    trigger button) to make it visible, and ``close_fn()`` to hide it.

    Args:
        title:            Dialog heading text.
        body:             Description message shown below the title.
        confirm_label:    Label for the confirm / destructive action button.
        cancel_label:     Label for the cancel button.
        confirm_variant:  Gradio variant for the confirm button.
        elem_id_prefix:   Prefix used to generate unique element IDs.

    Returns:
        Tuple of:
        - ``modal_col``     — the ``gr.Column`` wrapping the whole modal
        - ``confirm_btn``   — the positive action button
        - ``cancel_btn``    — the cancel / close button
        - ``open_modal``    — zero-arg callable returning ``gr.update(visible=True)``
        - ``close_modal``   — zero-arg callable returning ``gr.update(visible=False)``

    Example::

        modal_col, yes_btn, no_btn, open_fn, close_fn = confirm_modal(
            title="Delete Design?",
            body="The generated image will be permanently removed.",
            confirm_label="Delete",
        )
        # Wire trigger button:
        delete_trigger.click(open_fn, outputs=[modal_col])
        # Wire confirm / cancel:
        yes_btn.click(my_delete_handler, ...).then(close_fn, outputs=[modal_col])
        no_btn.click(close_fn, outputs=[modal_col])
    """
    with gr.Column(visible=False, elem_id=f"{elem_id_prefix}-modal") as modal_col:
        gr.HTML(f"""
<div style="{_MODAL_WRAP_STYLE}">
    <div style="font-size:1.3rem;font-weight:800;color:{_TEXT_PRIM};
                margin-bottom:0.6rem;">⚠️ {title}</div>
    <div style="font-size:0.92rem;color:{_TEXT_SEC};line-height:1.6;
                margin-bottom:1.5rem;">{body}</div>
</div>
""")
        with gr.Row():
            confirm_btn = gr.Button(
                value=f"✔ {confirm_label}",
                variant=confirm_variant,
                elem_id=f"{elem_id_prefix}-confirm-btn",
                scale=2,
            )
            cancel_btn = gr.Button(
                value=f"✖ {cancel_label}",
                variant="secondary",
                elem_id=f"{elem_id_prefix}-cancel-btn",
                scale=1,
            )

    def open_modal() -> gr.update:
        """Return a gr.update that makes this modal visible."""
        return gr.update(visible=True)

    def close_modal() -> gr.update:
        """Return a gr.update that hides this modal."""
        return gr.update(visible=False)

    # Wire cancel directly
    cancel_btn.click(close_modal, outputs=[modal_col])

    return modal_col, confirm_btn, cancel_btn, open_modal, close_modal


# ── Info modal ────────────────────────────────────────────────────────────────

def info_modal(
    *,
    title: str = "Information",
    body: str = "",
    close_label: str = "Got it",
    icon: str = "ℹ️",
    elem_id_prefix: str = "info",
) -> Tuple[gr.Column, gr.Button, Callable, Callable]:
    """Render a read-only information overlay.

    Args:
        title:          Dialog heading.
        body:           Body text / HTML shown in the dialog.
        close_label:    Label for the dismiss button.
        icon:           Icon displayed next to the title.
        elem_id_prefix: Prefix for unique element IDs.

    Returns:
        Tuple of:
        - ``modal_col``  — the wrapping ``gr.Column``
        - ``close_btn``  — the dismiss button
        - ``open_modal`` — callable returning ``gr.update(visible=True)``
        - ``close_modal``— callable returning ``gr.update(visible=False)``
    """
    with gr.Column(visible=False, elem_id=f"{elem_id_prefix}-info-modal") as modal_col:
        gr.HTML(f"""
<div style="{_MODAL_WRAP_STYLE}">
    <div style="font-size:1.3rem;font-weight:800;color:{_TEXT_PRIM};margin-bottom:0.7rem;">
        {icon} {title}
    </div>
    <div style="font-size:0.92rem;color:{_TEXT_SEC};line-height:1.65;margin-bottom:1.5rem;">
        {body}
    </div>
</div>
""")
        close_btn = gr.Button(
            value=f"✔ {close_label}",
            variant="primary",
            elem_id=f"{elem_id_prefix}-close-btn",
        )

    def open_modal() -> gr.update:
        return gr.update(visible=True)

    def close_modal() -> gr.update:
        return gr.update(visible=False)

    close_btn.click(close_modal, outputs=[modal_col])
    return modal_col, close_btn, open_modal, close_modal


# ── Form modal ────────────────────────────────────────────────────────────────

def form_modal(
    fields: Sequence[Tuple[str, str, Any]],
    *,
    title: str = "Enter Details",
    submit_label: str = "Submit",
    cancel_label: str = "Cancel",
    elem_id_prefix: str = "form",
) -> Tuple[gr.Column, list, gr.Button, gr.Button, Callable, Callable]:
    """Render a modal containing a set of form input fields.

    Args:
        fields:         Sequence of ``(label, input_type, default_value)`` tuples.
                        Supported *input_type* values: ``"text"``, ``"number"``,
                        ``"dropdown"``, ``"checkbox"``, ``"slider"``.
                        For ``"dropdown"``, pass a ``list`` as *default_value*.
        title:          Modal heading.
        submit_label:   Label for the submit button.
        cancel_label:   Label for the cancel button.
        elem_id_prefix: Prefix for unique element IDs.

    Returns:
        Tuple of:
        - ``modal_col``   — the wrapping ``gr.Column``
        - ``inputs``      — list of Gradio input components (in field order)
        - ``submit_btn``  — the submit button
        - ``cancel_btn``  — the cancel button
        - ``open_modal``  — callable returning ``gr.update(visible=True)``
        - ``close_modal`` — callable returning ``gr.update(visible=False)``
    """
    inputs: list = []
    with gr.Column(visible=False, elem_id=f"{elem_id_prefix}-form-modal") as modal_col:
        gr.Markdown(f"### {title}")
        for label, itype, default in fields:
            itype = itype.lower()
            if itype == "text":
                inp = gr.Textbox(label=label, value=str(default))
            elif itype == "number":
                inp = gr.Number(label=label, value=float(default))
            elif itype == "slider":
                lo, hi = (0, 100)
                val = float(default)
                inp = gr.Slider(minimum=lo, maximum=hi, value=val, label=label)
            elif itype == "checkbox":
                inp = gr.Checkbox(label=label, value=bool(default))
            elif itype == "dropdown":
                choices = default if isinstance(default, list) else [str(default)]
                inp = gr.Dropdown(choices=choices, label=label, value=choices[0] if choices else None)
            else:
                inp = gr.Textbox(label=label, value=str(default))
            inputs.append(inp)

        with gr.Row():
            submit_btn = gr.Button(
                value=f"✔ {submit_label}",
                variant="primary",
                scale=2,
                elem_id=f"{elem_id_prefix}-submit-btn",
            )
            cancel_btn = gr.Button(
                value=f"✖ {cancel_label}",
                variant="secondary",
                scale=1,
                elem_id=f"{elem_id_prefix}-cancel-btn",
            )

    def open_modal() -> gr.update:
        return gr.update(visible=True)

    def close_modal() -> gr.update:
        return gr.update(visible=False)

    cancel_btn.click(close_modal, outputs=[modal_col])
    return modal_col, inputs, submit_btn, cancel_btn, open_modal, close_modal


# ── Image preview modal ───────────────────────────────────────────────────────

def image_preview_modal(
    *,
    title: str = "Image Preview",
    close_label: str = "Close",
    elem_id_prefix: str = "imgpreview",
) -> Tuple[gr.Column, gr.Image, gr.Markdown, gr.Button, Callable, Callable]:
    """Render a full-screen image preview modal with caption support.

    Args:
        title:          Modal heading.
        close_label:    Dismiss button label.
        elem_id_prefix: Prefix for unique element IDs.

    Returns:
        Tuple of:
        - ``modal_col``   — the wrapping ``gr.Column``
        - ``img_widget``  — the ``gr.Image`` to set ``.value`` on
        - ``caption``     — ``gr.Markdown`` for the caption text
        - ``close_btn``   — the dismiss button
        - ``open_modal``  — callable; connect to trigger with image path + caption as inputs
        - ``close_modal`` — callable returning ``gr.update(visible=False)``

    Example::

        preview_col, prev_img, prev_cap, close_btn, open_fn, close_fn = image_preview_modal()

        thumbnail.select(
            lambda files, evt: (open_fn(), files[evt.index], "My Caption"),
            inputs=[files_state],
            outputs=[preview_col, prev_img, prev_cap],
        )
        close_btn.click(close_fn, outputs=[preview_col])
    """
    with gr.Column(visible=False, elem_id=f"{elem_id_prefix}-modal") as modal_col:
        gr.Markdown(f"### 🖼️ {title}")
        img_widget = gr.Image(
            label="",
            show_label=False,
            interactive=False,
            height=480,
            elem_id=f"{elem_id_prefix}-img",
        )
        caption = gr.Markdown("", elem_id=f"{elem_id_prefix}-caption")
        close_btn = gr.Button(
            value=f"✖ {close_label}",
            variant="secondary",
            elem_id=f"{elem_id_prefix}-close-btn",
        )

    def open_modal() -> gr.update:
        return gr.update(visible=True)

    def close_modal() -> gr.update:
        return gr.update(visible=False)

    close_btn.click(close_modal, outputs=[modal_col])
    return modal_col, img_widget, caption, close_btn, open_modal, close_modal


# ── Progress modal ────────────────────────────────────────────────────────────

def progress_modal(
    *,
    title: str = "Processing…",
    subtitle: str = "Please wait while we generate your design.",
    elem_id_prefix: str = "progress",
) -> Tuple[gr.Column, gr.HTML, Callable, Callable]:
    """Render a blocking progress / loading overlay.

    The overlay shows a CSS spinner and a customisable message.  Update the
    ``status_html`` widget to reflect live progress text.

    Args:
        title:          Main loading heading.
        subtitle:       Helper text shown below the spinner.
        elem_id_prefix: Prefix for unique element IDs.

    Returns:
        Tuple of:
        - ``modal_col``    — the wrapping ``gr.Column``
        - ``status_html``  — ``gr.HTML`` widget for dynamic status text
        - ``open_modal``   — callable returning ``gr.update(visible=True)``
        - ``close_modal``  — callable returning ``gr.update(visible=False)``
    """
    _spinner_css = """
<style>
@keyframes spin { to { transform: rotate(360deg); } }
.studio-spinner {
    width: 48px; height: 48px;
    border: 4px solid rgba(255,159,67,0.2);
    border-top-color: #ff9f43;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
    margin: 0 auto 1.2rem;
}
</style>
"""
    with gr.Column(visible=False, elem_id=f"{elem_id_prefix}-progress-modal") as modal_col:
        gr.HTML(f"""
{_spinner_css}
<div style="{_MODAL_WRAP_STYLE}text-align:center;">
    <div class="studio-spinner"></div>
    <div style="font-size:1.2rem;font-weight:700;color:{_TEXT_PRIM};
                margin-bottom:0.4rem;">{title}</div>
    <div style="font-size:0.88rem;color:{_TEXT_SEC};">{subtitle}</div>
</div>
""")
        status_html = gr.HTML("", elem_id=f"{elem_id_prefix}-status")

    def open_modal() -> gr.update:
        return gr.update(visible=True)

    def close_modal() -> gr.update:
        return gr.update(visible=False)

    return modal_col, status_html, open_modal, close_modal
