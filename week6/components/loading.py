"""
Week 6 — Reusable Loading / Progress Components.

Provides factory functions for animated loading states, progress bars,
skeleton placeholders, and spinners — all implemented via ``gr.HTML`` and
inline CSS animations so they work without JavaScript frameworks.

Component families
------------------
- ``spinner``            — circular CSS spinner
- ``progress_bar``       — horizontal labelled progress bar
- ``multi_step_progress``— multi-step pipeline indicator
- ``skeleton_card``      — shimmering content-placeholder skeleton
- ``skeleton_gallery``   — row of skeleton image placeholders
- ``loading_overlay``    — full-panel loading overlay with spinner
- ``dots_loader``        — animated three-dot ellipsis loader
- ``generation_progress``— specialised generation status tracker
"""
from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import gradio as gr

# ── Design tokens ─────────────────────────────────────────────────────────────
_ACCENT  = "#ff9f43"
_ACCENT2 = "#ff5252"
_SUCCESS = "#2ecc71"
_INFO    = "#3498db"
_WARNING = "#f39c12"
_SURFACE = "rgba(255,255,255,0.03)"
_BORDER  = "rgba(255,255,255,0.07)"
_TEXT_PRIM = "#f0f0f5"
_TEXT_SEC  = "#a0a0b0"

# ── Shared CSS (injected once per page) ───────────────────────────────────────
_CSS_INJECTED = False
_SHARED_CSS = """
<style>
/* ── Studio Loading Animations ──────────────────────────────────────── */
@keyframes studio-spin   { to { transform: rotate(360deg); } }
@keyframes studio-pulse  { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
@keyframes studio-shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
@keyframes studio-bounce {
    0%, 80%, 100% { transform: scale(0); }
    40%           { transform: scale(1.0); }
}
@keyframes studio-bar-fill {
    from { width: 0%; }
    to   { width: var(--target-w, 100%); }
}
.studio-spinner-ring {
    display: inline-block;
    width: var(--sz, 40px); height: var(--sz, 40px);
    border: calc(var(--sz, 40px) * 0.1) solid rgba(255,159,67,0.18);
    border-top-color: #ff9f43;
    border-radius: 50%;
    animation: studio-spin 0.8s linear infinite;
}
.studio-skeleton {
    background: linear-gradient(
        90deg,
        rgba(255,255,255,0.04) 25%,
        rgba(255,255,255,0.10) 50%,
        rgba(255,255,255,0.04) 75%
    );
    background-size: 200% 100%;
    animation: studio-shimmer 1.6s infinite;
    border-radius: 6px;
}
.studio-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #ff9f43;
    border-radius: 50%;
    animation: studio-bounce 1.4s ease-in-out infinite both;
}
</style>
"""


def _inject_css() -> str:
    """Return the shared CSS block (call once per page build context)."""
    global _CSS_INJECTED
    if not _CSS_INJECTED:
        _CSS_INJECTED = True
        return _SHARED_CSS
    return ""


# ── Spinner ────────────────────────────────────────────────────────────────────

def spinner(
    *,
    size: int = 40,
    label: str = "",
    color: str = _ACCENT,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a circular CSS loading spinner.

    Args:
        size:   Diameter in pixels.
        label:  Optional text shown below the spinner.
        color:  Ring colour.
        render: If ``True`` renders via ``gr.HTML``; else returns HTML string.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    label_html = (
        f'<div style="font-size:0.82rem;color:{_TEXT_SEC};margin-top:0.6rem;">{label}</div>'
        if label else ""
    )
    html = f"""
{_inject_css()}
<div style="display:flex;flex-direction:column;align-items:center;
            justify-content:center;padding:1rem;">
    <div class="studio-spinner-ring" style="
        --sz:{size}px;
        border-color:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18);
        border-top-color:{color};
    "></div>
    {label_html}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Progress bar ───────────────────────────────────────────────────────────────

def progress_bar(
    value: float,
    *,
    label: str = "",
    max_value: float = 100.0,
    color: str = _ACCENT,
    show_percent: bool = True,
    animate: bool = True,
    height: int = 8,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a horizontal labelled progress bar.

    Args:
        value:       Current progress value.
        label:       Label displayed above the bar.
        max_value:   Maximum value (determines 100% width).
        color:       Bar fill colour.
        show_percent:Whether to show the percentage text.
        animate:     Whether to animate the fill width on render.
        height:      Bar height in pixels.
        render:      If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    pct = min(100.0, max(0.0, (value / max_value) * 100)) if max_value else 0
    pct_text = f"{pct:.0f}%" if show_percent else ""
    label_row = ""
    if label or pct_text:
        label_row = (
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:0.35rem;">'
            f'<span style="font-size:0.8rem;color:{_TEXT_SEC};">{label}</span>'
            f'<span style="font-size:0.78rem;font-weight:700;color:{color};">{pct_text}</span>'
            f'</div>'
        )
    anim_css = (
        f"animation:studio-bar-fill 0.6s ease-out forwards;--target-w:{pct}%;"
        if animate else f"width:{pct}%;"
    )
    html = f"""
{_inject_css()}
{label_row}
<div style="background:rgba(255,255,255,0.06);border-radius:{height}px;
            height:{height}px;overflow:hidden;margin-bottom:0.5rem;">
    <div style="height:100%;background:{color};border-radius:{height}px;
                {anim_css}"></div>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Multi-step progress ────────────────────────────────────────────────────────

def multi_step_progress(
    steps: Sequence[str],
    current_step: int,
    *,
    accent_color: str = _ACCENT,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a linear pipeline / step-indicator.

    Args:
        steps:        Ordered list of step names.
        current_step: 0-indexed index of the currently active step.
                      Steps before *current_step* are shown as completed;
                      the current step pulses; future steps are dimmed.
        accent_color: Colour for completed / active indicators.
        render:       If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    step_items: list[str] = []
    for i, name in enumerate(steps):
        if i < current_step:
            # Completed
            circle = (
                f'<div style="width:28px;height:28px;border-radius:50%;'
                f'background:{accent_color};color:#000;display:flex;'
                f'align-items:center;justify-content:center;font-size:0.8rem;'
                f'font-weight:800;flex-shrink:0;">✓</div>'
            )
            text_color = accent_color
        elif i == current_step:
            # Active
            circle = (
                f'<div style="width:28px;height:28px;border-radius:50%;'
                f'border:2px solid {accent_color};background:rgba(255,159,67,0.15);'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:0.75rem;font-weight:800;color:{accent_color};'
                f'flex-shrink:0;animation:studio-pulse 1.2s ease-in-out infinite;">'
                f'{i+1}</div>'
            )
            text_color = _TEXT_PRIM
        else:
            # Pending
            circle = (
                f'<div style="width:28px;height:28px;border-radius:50%;'
                f'border:1px solid {_BORDER};background:rgba(255,255,255,0.03);'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:0.75rem;font-weight:600;color:{_TEXT_SEC};'
                f'flex-shrink:0;">{i+1}</div>'
            )
            text_color = _TEXT_SEC

        connector = (
            '<div style="flex:1;height:2px;background:'
            + (accent_color if i < current_step else _BORDER)
            + ';margin:0 0.4rem;"></div>'
            if i < len(steps) - 1 else ""
        )
        step_items.append(
            f'<div style="display:flex;align-items:center;">'
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.3rem;">'
            f'{circle}'
            f'<span style="font-size:0.7rem;color:{text_color};white-space:nowrap;'
            f'max-width:70px;text-align:center;line-height:1.2;">{name}</span>'
            f'</div>'
            f'{connector}'
            f'</div>'
        )

    html = f"""
{_inject_css()}
<div style="display:flex;align-items:flex-start;justify-content:center;
            padding:0.8rem;margin:0.8rem 0;">
    {''.join(step_items)}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Skeleton card ──────────────────────────────────────────────────────────────

def skeleton_card(
    *,
    lines: int = 3,
    show_image: bool = True,
    image_height: int = 140,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a shimmering skeleton-loader card.

    Args:
        lines:        Number of text-line skeletons below the image area.
        show_image:   Whether to include an image-placeholder block at the top.
        image_height: Height in pixels for the image placeholder.
        render:       If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    img_html = (
        f'<div class="studio-skeleton" style="height:{image_height}px;'
        f'border-radius:8px;margin-bottom:0.8rem;"></div>'
        if show_image else ""
    )
    widths = ["100%", "80%", "60%"]
    line_html = "".join(
        f'<div class="studio-skeleton" style="height:12px;border-radius:6px;'
        f'width:{widths[i % len(widths)]};margin-bottom:0.5rem;"></div>'
        for i in range(lines)
    )
    html = f"""
{_inject_css()}
<div style="background:{_SURFACE};border:1px solid {_BORDER};border-radius:10px;
            padding:1rem;margin-bottom:0.8rem;">
    {img_html}
    {line_html}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Skeleton gallery ───────────────────────────────────────────────────────────

def skeleton_gallery(
    count: int = 4,
    *,
    columns: int = 4,
    image_height: int = 160,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a row of skeleton image placeholders.

    Args:
        count:        Number of skeleton tiles to render.
        columns:      Number of CSS grid columns.
        image_height: Height in pixels for each placeholder tile.
        render:       If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    tiles = "".join(
        f'<div class="studio-skeleton" style="height:{image_height}px;'
        f'border-radius:10px;"></div>'
        for _ in range(count)
    )
    html = f"""
{_inject_css()}
<div style="display:grid;grid-template-columns:repeat({columns},1fr);
            gap:0.8rem;margin-bottom:1rem;">
    {tiles}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Loading overlay ────────────────────────────────────────────────────────────

def loading_overlay(
    *,
    title: str = "Generating…",
    subtitle: str = "Please wait — your design is being crafted.",
    size: int = 56,
    visible: bool = False,
    elem_id: str = "studio-loading-overlay",
) -> gr.HTML:
    """Render a full-panel loading overlay and return the widget.

    The widget is initially hidden (``visible=False``).  Update its ``.value``
    to the full HTML string to show it, or set it to ``""`` to hide it.

    Args:
        title:    Bold heading displayed below the spinner.
        subtitle: Smaller helper text below the heading.
        size:     Spinner diameter in pixels.
        visible:  Initial visibility (default ``False``).
        elem_id:  HTML element id for CSS targeting.

    Returns:
        The ``gr.HTML`` widget (update its value to show / hide).
    """
    def _html(show: bool) -> str:
        if not show:
            return ""
        return f"""
{_inject_css()}
<div style="position:absolute;top:0;left:0;width:100%;height:100%;
            background:rgba(10,10,20,0.75);backdrop-filter:blur(6px);
            display:flex;flex-direction:column;align-items:center;
            justify-content:center;z-index:999;border-radius:10px;">
    <div class="studio-spinner-ring" style="--sz:{size}px;"></div>
    <div style="font-size:1.1rem;font-weight:700;color:{_TEXT_PRIM};margin-top:1.2rem;">{title}</div>
    <div style="font-size:0.85rem;color:{_TEXT_SEC};margin-top:0.3rem;">{subtitle}</div>
</div>
"""

    return gr.HTML(value=_html(visible), elem_id=elem_id)


# ── Dots loader ────────────────────────────────────────────────────────────────

def dots_loader(
    label: str = "Loading",
    *,
    color: str = _ACCENT,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a bouncing three-dot ellipsis loader.

    Args:
        label:  Text shown to the left of the dots.
        color:  Dot colour.
        render: If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    delays = ["0s", "0.16s", "0.32s"]
    dots = "".join(
        f'<div class="studio-dot" style="background:{color};'
        f'animation-delay:{d};"></div>'
        for d in delays
    )
    html = f"""
{_inject_css()}
<div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0;">
    <span style="font-size:0.88rem;color:{_TEXT_SEC};">{label}</span>
    <div style="display:flex;gap:0.3rem;align-items:center;">{dots}</div>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Generation progress tracker ────────────────────────────────────────────────

_GEN_STEPS = [
    "Encode Prompt",
    "Apply LoRA",
    "Denoise (DDIM)",
    "VAE Decode",
    "Post-process",
    "Save Output",
]


def generation_progress(
    current_step: int = 0,
    *,
    elapsed_secs: float = 0.0,
    step_names: Optional[Sequence[str]] = None,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a specialised generation pipeline progress tracker.

    Args:
        current_step: 0-indexed current pipeline stage.
        elapsed_secs: Elapsed seconds displayed in the header.
        step_names:   Custom stage names (defaults to ``_GEN_STEPS``).
        render:       If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    steps = list(step_names) if step_names else _GEN_STEPS
    pct = int((current_step / len(steps)) * 100)
    elapsed_html = (
        f'<span style="font-size:0.75rem;color:{_TEXT_SEC};">'
        f'⏱ {elapsed_secs:.1f}s elapsed</span>'
    )
    step_list = multi_step_progress(steps, current_step, render=False) or ""
    bar = progress_bar(pct, label="Overall Progress", render=False) or ""
    html = f"""
{_inject_css()}
<div style="background:{_SURFACE};border:1px solid {_BORDER};border-radius:10px;
            padding:1.2rem;margin-bottom:1rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:0.8rem;">
        <span style="font-size:0.92rem;font-weight:700;color:{_TEXT_PRIM};">
            🎨 Generation Pipeline
        </span>
        {elapsed_html}
    </div>
    {bar}
    {step_list}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html
