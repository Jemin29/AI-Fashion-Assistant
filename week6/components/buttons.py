"""
Week 6 — Reusable Button Components.

Provides factory functions for consistently styled Gradio buttons across
the studio.  Each function returns a ``gr.Button`` (or a group of buttons
inside a ``gr.Row``) that can be wired to event handlers by the caller.

Design tokens
-------------
- Primary accent : #ff9f43  (warm amber)
- Danger accent  : #ff5252  (vivid red)
- Success accent : #2ecc71  (emerald)
- Info accent    : #3498db  (sky blue)
- Surface dark   : #1e1e2f
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gradio as gr


# ── Primitive builders ────────────────────────────────────────────────────────

def primary_button(
    label: str,
    *,
    icon: str = "",
    size: str = "lg",
    elem_id: Optional[str] = None,
    scale: int = 1,
    min_width: int = 120,
    interactive: bool = True,
) -> gr.Button:
    """Return a Gradio *primary* button (amber gradient, bold).

    Args:
        label:       Visible button text.
        icon:        Optional emoji / unicode icon prepended to the label.
        size:        Gradio size token – ``"sm"``, ``"md"``, or ``"lg"``.
        elem_id:     HTML element id for CSS targeting.
        scale:       Flex-grow scale inside a parent ``gr.Row``.
        min_width:   Minimum pixel width.
        interactive: Whether the button is clickable on render.

    Returns:
        A configured ``gr.Button`` instance.
    """
    full_label = f"{icon} {label}".strip() if icon else label
    return gr.Button(
        value=full_label,
        variant="primary",
        size=size,
        elem_id=elem_id or f"btn-primary-{label.lower().replace(' ', '-')}",
        scale=scale,
        min_width=min_width,
        interactive=interactive,
    )


def secondary_button(
    label: str,
    *,
    icon: str = "",
    size: str = "lg",
    elem_id: Optional[str] = None,
    scale: int = 1,
    min_width: int = 120,
    interactive: bool = True,
) -> gr.Button:
    """Return a Gradio *secondary* button (subtle outline style).

    Args:
        label:       Visible button text.
        icon:        Optional emoji / unicode icon prepended to the label.
        size:        Gradio size token.
        elem_id:     HTML element id for CSS targeting.
        scale:       Flex-grow scale inside a parent ``gr.Row``.
        min_width:   Minimum pixel width.
        interactive: Whether the button is clickable on render.

    Returns:
        A configured ``gr.Button`` instance.
    """
    full_label = f"{icon} {label}".strip() if icon else label
    return gr.Button(
        value=full_label,
        variant="secondary",
        size=size,
        elem_id=elem_id or f"btn-secondary-{label.lower().replace(' ', '-')}",
        scale=scale,
        min_width=min_width,
        interactive=interactive,
    )


def danger_button(
    label: str,
    *,
    icon: str = "🗑️",
    size: str = "md",
    elem_id: Optional[str] = None,
    scale: int = 1,
    min_width: int = 100,
    interactive: bool = True,
) -> gr.Button:
    """Return a *danger / destructive* button (Gradio ``stop`` variant, red).

    Args:
        label:       Visible button text.
        icon:        Optional emoji (defaults to 🗑️).
        size:        Gradio size token.
        elem_id:     HTML element id for CSS targeting.
        scale:       Flex-grow scale inside a parent ``gr.Row``.
        min_width:   Minimum pixel width.
        interactive: Whether the button is clickable on render.

    Returns:
        A configured ``gr.Button`` instance.
    """
    full_label = f"{icon} {label}".strip() if icon else label
    return gr.Button(
        value=full_label,
        variant="stop",
        size=size,
        elem_id=elem_id or f"btn-danger-{label.lower().replace(' ', '-')}",
        scale=scale,
        min_width=min_width,
        interactive=interactive,
    )


def icon_button(
    icon: str,
    tooltip: str = "",
    *,
    size: str = "sm",
    variant: str = "secondary",
    elem_id: Optional[str] = None,
    scale: int = 0,
    min_width: int = 48,
) -> gr.Button:
    """Return a compact *icon-only* button.

    Args:
        icon:    Emoji or unicode glyph displayed as the button label.
        tooltip: Accessible description (used as the label for screen readers).
        size:    Gradio size token.
        variant: Gradio variant string.
        elem_id: HTML element id.
        scale:   Flex-grow scale.
        min_width: Minimum pixel width.

    Returns:
        A configured ``gr.Button`` instance.
    """
    return gr.Button(
        value=icon,
        variant=variant,
        size=size,
        elem_id=elem_id or f"btn-icon-{tooltip.lower().replace(' ', '-')}",
        scale=scale,
        min_width=min_width,
    )


# ── Compound builders ─────────────────────────────────────────────────────────

def action_button_row(
    actions: Sequence[Tuple[str, str, str]],
) -> list[gr.Button]:
    """Render a horizontal row of labelled action buttons and return them.

    Each element of ``actions`` is a ``(label, icon, variant)`` triple where
    *variant* is one of ``"primary"``, ``"secondary"``, or ``"stop"``.

    Example::

        generate_btn, clear_btn, save_btn = action_button_row([
            ("Generate", "🎨", "primary"),
            ("Clear",    "🔄", "secondary"),
            ("Save",     "💾", "secondary"),
        ])

    Args:
        actions: Sequence of ``(label, icon, variant)`` tuples.

    Returns:
        List of ``gr.Button`` objects in declaration order.
    """
    buttons: list[gr.Button] = []
    with gr.Row():
        for label, icon, variant in actions:
            full_label = f"{icon} {label}".strip() if icon else label
            btn = gr.Button(
                value=full_label,
                variant=variant,
                elem_id=f"btn-{label.lower().replace(' ', '-')}",
                scale=1,
            )
            buttons.append(btn)
    return buttons


def generate_clear_row(
    generate_label: str = "Generate",
    clear_label: str = "Clear",
    generate_icon: str = "🎨",
    clear_icon: str = "🔄",
) -> Tuple[gr.Button, gr.Button]:
    """Convenience wrapper: primary *Generate* + secondary *Clear* button row.

    Args:
        generate_label: Label for the primary action button.
        clear_label:    Label for the secondary clear button.
        generate_icon:  Icon for the primary action button.
        clear_icon:     Icon for the clear button.

    Returns:
        Tuple of ``(generate_btn, clear_btn)``.
    """
    with gr.Row():
        gen_btn = gr.Button(
            value=f"{generate_icon} {generate_label}",
            variant="primary",
            scale=3,
            elem_id="btn-generate",
        )
        clr_btn = gr.Button(
            value=f"{clear_icon} {clear_label}",
            variant="secondary",
            scale=1,
            elem_id="btn-clear",
        )
    return gen_btn, clr_btn


def copy_download_row(
    copy_label: str = "Copy",
    download_label: str = "Download",
) -> Tuple[gr.Button, gr.Button]:
    """Render a compact *Copy* + *Download* icon-button pair.

    Args:
        copy_label:     Label text for the copy button.
        download_label: Label text for the download button.

    Returns:
        Tuple of ``(copy_btn, download_btn)``.
    """
    with gr.Row():
        copy_btn = gr.Button(
            value=f"📋 {copy_label}",
            variant="secondary",
            size="sm",
            scale=1,
            elem_id="btn-copy",
        )
        dl_btn = gr.Button(
            value=f"⬇️ {download_label}",
            variant="secondary",
            size="sm",
            scale=1,
            elem_id="btn-download",
        )
    return copy_btn, dl_btn


def toggle_button(
    label_on: str,
    label_off: str,
    *,
    icon_on: str = "✅",
    icon_off: str = "⭕",
    initial_state: bool = False,
    elem_id: Optional[str] = None,
) -> Tuple[gr.Button, gr.State]:
    """Render a stateful toggle button that flips its label on each click.

    Args:
        label_on:      Visible label when the toggle is *active*.
        label_off:     Visible label when the toggle is *inactive*.
        icon_on:       Icon prepended when *active*.
        icon_off:      Icon prepended when *inactive*.
        initial_state: Starting toggle state (``True`` = on).
        elem_id:       HTML element id.

    Returns:
        Tuple of ``(toggle_button, state_component)``.  Wire the button's
        ``click`` event to ``on_toggle_click`` (defined below) with the state
        as both input and output.
    """
    _state = gr.State(value=initial_state)
    _label = f"{icon_on} {label_on}" if initial_state else f"{icon_off} {label_off}"
    _btn = gr.Button(
        value=_label,
        variant="primary" if initial_state else "secondary",
        elem_id=elem_id or "btn-toggle",
    )

    def on_toggle_click(current: bool) -> Tuple[gr.update, bool]:
        """Flip the toggle state and update the button label/variant."""
        new = not current
        new_label = f"{icon_on} {label_on}" if new else f"{icon_off} {label_off}"
        return gr.update(value=new_label, variant="primary" if new else "secondary"), new

    _btn.click(on_toggle_click, inputs=[_state], outputs=[_btn, _state])
    return _btn, _state
