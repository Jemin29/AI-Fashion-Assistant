"""
Week 6 — Reusable Card Components.

Provides pure-HTML card factory functions that render consistently styled
information blocks inside Gradio ``gr.HTML`` widgets.  All cards follow the
studio's dark glassmorphic design language (deep navy surfaces, amber accents,
subtle borders, hover lift animations).

Card families
-------------
- ``stat_card``          — single KPI / metric display
- ``stat_card_row``      — horizontal grid of KPI cards
- ``info_card``          — titled body-text card with optional icon
- ``image_caption_card`` — thumbnail with overlay caption
- ``brand_card``         — brand logo + colour swatch + description
- ``trend_card``         — trend entry with percentage badge
- ``recommendation_card``— recommended item with confidence bar
- ``feature_card``       — feature highlight with icon and description
- ``alert_card``         — colour-coded message card (info/success/warning/error)
- ``tag_list_card``      — list of pill tags inside a card
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr

# ── Design tokens (shared across all cards) ───────────────────────────────────
_SURFACE = "rgba(255,255,255,0.03)"
_BORDER  = "rgba(255,255,255,0.07)"
_HOVER_BORDER = "rgba(255,159,67,0.35)"
_TEXT_PRIMARY   = "#f0f0f5"
_TEXT_SECONDARY = "#a0a0b0"
_ACCENT  = "#ff9f43"
_DANGER  = "#ff5252"
_SUCCESS = "#2ecc71"
_INFO    = "#3498db"
_WARNING = "#f39c12"

_CARD_BASE = (
    f"background:{_SURFACE};"
    f"border:1px solid {_BORDER};"
    "border-radius:10px;"
    "padding:1.2rem 1.4rem;"
    "transition:transform 0.2s ease,box-shadow 0.2s ease,border-color 0.2s ease;"
)


# ── Stat / KPI cards ──────────────────────────────────────────────────────────

def stat_card(
    label: str,
    value: Any,
    *,
    icon: str = "",
    unit: str = "",
    delta: Optional[str] = None,
    accent_color: str = _ACCENT,
) -> str:
    """Return HTML string for a single metric / KPI card.

    Args:
        label:        Metric label (shown above the value in small caps).
        value:        The numeric or string value to display prominently.
        icon:         Optional emoji shown in the top-right corner.
        unit:         Unit suffix appended to *value* (e.g. ``"%"``, ``"ms"``).
        delta:        Optional change indicator, e.g. ``"+2.3%"`` or ``"-0.5"``.
        accent_color: Hex / CSS colour used for the value text.

    Returns:
        HTML string that can be embedded in ``gr.HTML``.
    """
    delta_html = ""
    if delta is not None:
        delta_col = _SUCCESS if delta.startswith("+") else _DANGER
        delta_html = (
            f'<div style="font-size:0.78rem;color:{delta_col};margin-top:0.25rem;">'
            f'{delta}</div>'
        )
    icon_html = (
        f'<div style="position:absolute;top:0.8rem;right:1rem;font-size:1.5rem;opacity:0.6;">{icon}</div>'
        if icon else ""
    )
    return f"""
<div class="metric-card" style="{_CARD_BASE}position:relative;min-width:120px;">
    {icon_html}
    <div style="font-size:0.72rem;color:{_TEXT_SECONDARY};text-transform:uppercase;
                letter-spacing:0.08em;font-weight:600;margin-bottom:0.5rem;">{label}</div>
    <div style="font-size:2rem;font-weight:800;color:{accent_color};
                font-family:'Outfit',sans-serif;line-height:1;">{value}{unit}</div>
    {delta_html}
</div>
"""


def stat_card_row(
    metrics: Dict[str, Any],
    *,
    icons: Optional[Dict[str, str]] = None,
    units: Optional[Dict[str, str]] = None,
    deltas: Optional[Dict[str, str]] = None,
    columns: int = 4,
) -> None:
    """Render a responsive grid of ``stat_card`` tiles via ``gr.HTML``.

    Args:
        metrics: Mapping of ``{label: value}`` pairs.
        icons:   Optional mapping of ``{label: emoji}`` for card icons.
        units:   Optional mapping of ``{label: unit_string}``.
        deltas:  Optional mapping of ``{label: delta_string}``.
        columns: Number of columns in the CSS grid (default 4).
    """
    icons  = icons  or {}
    units  = units  or {}
    deltas = deltas or {}

    cards_html = "".join(
        stat_card(
            label=lbl,
            value=val,
            icon=icons.get(lbl, ""),
            unit=units.get(lbl, ""),
            delta=deltas.get(lbl),
        )
        for lbl, val in metrics.items()
    )
    grid_html = (
        f'<div style="display:grid;grid-template-columns:repeat({columns},1fr);'
        f'gap:1rem;margin-bottom:1.5rem;">{cards_html}</div>'
    )
    gr.HTML(grid_html)


# ── Info card ─────────────────────────────────────────────────────────────────

def info_card(
    title: str,
    body: str,
    *,
    icon: str = "ℹ️",
    accent_color: str = _INFO,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a titled informational card.

    Args:
        title:        Bold card title.
        body:         Main content text (plain text or HTML).
        icon:         Emoji prepended to the title.
        accent_color: Left-border accent colour.
        render:       If ``True`` (default) render via ``gr.HTML``; otherwise
                      return the raw HTML string for composition.

    Returns:
        ``None`` when *render* is ``True``; HTML string otherwise.
    """
    html = f"""
<div style="{_CARD_BASE}border-left:3px solid {accent_color};margin-bottom:1rem;">
    <div style="font-size:1rem;font-weight:700;color:{_TEXT_PRIMARY};margin-bottom:0.5rem;">
        {icon} {title}
    </div>
    <div style="font-size:0.9rem;color:{_TEXT_SECONDARY};line-height:1.6;">
        {body}
    </div>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Brand card ────────────────────────────────────────────────────────────────

def brand_card(
    name: str,
    description: str,
    *,
    color: str = _ACCENT,
    emoji: str = "🏷️",
    tags: Sequence[str] = (),
    selected: bool = False,
) -> str:
    """Return HTML for a brand identity card with colour swatch and tag pills.

    Args:
        name:        Brand name.
        description: Short brand description.
        color:       Brand primary colour (used for swatch and border).
        emoji:       Brand emoji / logo glyph.
        tags:        Sequence of style / category tag strings.
        selected:    If ``True``, renders with a highlighted active border.

    Returns:
        HTML string.
    """
    border_style = f"border:2px solid {color};" if selected else f"border:1px solid {_BORDER};"
    tag_pills = "".join(
        f'<span style="background:rgba(255,255,255,0.06);border:1px solid {_BORDER};'
        f'border-radius:20px;padding:0.15rem 0.6rem;font-size:0.72rem;'
        f'color:{_TEXT_SECONDARY};margin-right:0.3rem;">{t}</span>'
        for t in tags
    )
    return f"""
<div class="metric-card" style="background:{_SURFACE};{border_style}border-radius:12px;
     padding:1.2rem;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:60px;height:60px;
                background:{color};opacity:0.12;border-radius:0 12px 0 60px;"></div>
    <div style="font-size:2.2rem;margin-bottom:0.5rem;">{emoji}</div>
    <div style="font-size:1.1rem;font-weight:700;color:{_TEXT_PRIMARY};margin-bottom:0.3rem;">
        {name}
    </div>
    <div style="font-size:0.85rem;color:{_TEXT_SECONDARY};margin-bottom:0.8rem;line-height:1.5;">
        {description}
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.3rem;">{tag_pills}</div>
</div>
"""


# ── Trend card ────────────────────────────────────────────────────────────────

def trend_card(
    trend_name: str,
    category: str,
    score: float,
    *,
    direction: str = "↑",
    description: str = "",
    rank: int = 0,
) -> str:
    """Return HTML for a fashion trend entry card with a momentum badge.

    Args:
        trend_name:  Name of the trend.
        category:    Category label (e.g. ``"Streetwear"``).
        score:       Momentum score in ``[0, 1]``.
        direction:   Arrow glyph indicating direction (``"↑"`` / ``"↓"`` / ``"→"``).
        description: Optional short description.
        rank:        Optional rank badge number (0 = hidden).

    Returns:
        HTML string.
    """
    pct = int(score * 100)
    dir_color = _SUCCESS if direction == "↑" else (_DANGER if direction == "↓" else _WARNING)
    rank_html = (
        f'<div style="position:absolute;top:0.8rem;right:0.8rem;'
        f'background:{_ACCENT};color:#000;border-radius:50%;width:24px;height:24px;'
        f'display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:800;">'
        f'#{rank}</div>'
        if rank > 0 else ""
    )
    bar_html = (
        f'<div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;margin-top:0.8rem;">'
        f'<div style="width:{pct}%;height:100%;background:{_ACCENT};border-radius:4px;'
        f'transition:width 0.4s ease;"></div></div>'
    )
    desc_html = (
        f'<div style="font-size:0.8rem;color:{_TEXT_SECONDARY};margin-top:0.4rem;">{description}</div>'
        if description else ""
    )
    return f"""
<div class="metric-card" style="{_CARD_BASE}position:relative;">
    {rank_html}
    <div style="font-size:0.7rem;color:{_TEXT_SECONDARY};text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.3rem;">{category}</div>
    <div style="font-size:1rem;font-weight:700;color:{_TEXT_PRIMARY};">{trend_name}</div>
    <div style="font-size:1.4rem;font-weight:800;color:{dir_color};margin-top:0.4rem;">
        {direction} {pct}%
    </div>
    {desc_html}
    {bar_html}
</div>
"""


# ── Recommendation card ───────────────────────────────────────────────────────

def recommendation_card(
    title: str,
    subtitle: str,
    confidence: float,
    *,
    tags: Sequence[str] = (),
    reasoning: str = "",
    icon: str = "👗",
    rank: int = 0,
) -> str:
    """Return HTML for a personalised recommendation card.

    Args:
        title:      Item / style name.
        subtitle:   Brand or category label.
        confidence: Confidence score in ``[0, 1]``.
        tags:       Style / season / fabric tags.
        reasoning:  Short rationale sentence.
        icon:       Fashion emoji icon.
        rank:       Optional ordinal rank badge.

    Returns:
        HTML string.
    """
    pct = int(confidence * 100)
    conf_color = _SUCCESS if pct >= 80 else (_WARNING if pct >= 60 else _DANGER)
    tag_pills = "".join(
        f'<span style="background:rgba(255,159,67,0.12);border:1px solid rgba(255,159,67,0.25);'
        f'border-radius:20px;padding:0.15rem 0.55rem;font-size:0.7rem;color:{_ACCENT};'
        f'margin-right:0.3rem;">{t}</span>'
        for t in tags
    )
    rank_badge = (
        f'<div style="background:{_ACCENT};color:#000;border-radius:50%;'
        f'width:22px;height:22px;display:inline-flex;align-items:center;'
        f'justify-content:center;font-size:0.7rem;font-weight:800;margin-right:0.4rem;">#{rank}</div>'
        if rank > 0 else ""
    )
    reasoning_html = (
        f'<div style="font-size:0.78rem;color:{_TEXT_SECONDARY};'
        f'border-top:1px solid {_BORDER};margin-top:0.7rem;padding-top:0.5rem;">'
        f'💡 {reasoning}</div>'
        if reasoning else ""
    )
    return f"""
<div class="metric-card" style="{_CARD_BASE}">
    <div style="display:flex;align-items:flex-start;gap:0.8rem;">
        <div style="font-size:2rem;">{icon}</div>
        <div style="flex:1;">
            <div style="font-size:1rem;font-weight:700;color:{_TEXT_PRIMARY};">
                {rank_badge}{title}
            </div>
            <div style="font-size:0.82rem;color:{_TEXT_SECONDARY};margin-top:0.15rem;">{subtitle}</div>
        </div>
        <div style="font-size:1.2rem;font-weight:800;color:{conf_color};white-space:nowrap;">
            {pct}%
        </div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border-radius:4px;height:5px;margin:0.7rem 0 0.5rem;">
        <div style="width:{pct}%;height:100%;background:{conf_color};border-radius:4px;"></div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.3rem;">{tag_pills}</div>
    {reasoning_html}
</div>
"""


# ── Feature card ──────────────────────────────────────────────────────────────

def feature_card(
    title: str,
    description: str,
    *,
    icon: str = "✨",
    accent_color: str = _ACCENT,
    badge: str = "",
) -> str:
    """Return HTML for a feature highlight card.

    Args:
        title:        Feature title.
        description:  Short description paragraph.
        icon:         Large icon displayed at the top.
        accent_color: Top-border accent line colour.
        badge:        Optional small badge text (e.g. ``"NEW"``).

    Returns:
        HTML string.
    """
    badge_html = (
        f'<span style="background:{accent_color};color:#000;font-size:0.65rem;'
        f'font-weight:800;border-radius:20px;padding:0.1rem 0.55rem;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-left:0.4rem;">{badge}</span>'
        if badge else ""
    )
    return f"""
<div class="metric-card" style="{_CARD_BASE}border-top:3px solid {accent_color};text-align:center;">
    <div style="font-size:2.4rem;margin-bottom:0.6rem;">{icon}</div>
    <div style="font-size:1rem;font-weight:700;color:{_TEXT_PRIMARY};margin-bottom:0.3rem;">
        {title}{badge_html}
    </div>
    <div style="font-size:0.85rem;color:{_TEXT_SECONDARY};line-height:1.6;">{description}</div>
</div>
"""


# ── Alert card ────────────────────────────────────────────────────────────────

_ALERT_PALETTE = {
    "info":    (_INFO,    "ℹ️",  "rgba(52,152,219,0.08)"),
    "success": (_SUCCESS, "✅",  "rgba(46,204,113,0.08)"),
    "warning": (_WARNING, "⚠️", "rgba(243,156,18,0.08)"),
    "error":   (_DANGER,  "❌", "rgba(255,82,82,0.08)"),
}


def alert_card(
    message: str,
    *,
    level: str = "info",
    title: str = "",
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a colour-coded alert / notification card.

    Args:
        message: Alert body text (HTML supported).
        level:   One of ``"info"``, ``"success"``, ``"warning"``, ``"error"``.
        title:   Optional bold title above the message.
        render:  If ``True`` render via ``gr.HTML``; else return HTML string.

    Returns:
        ``None`` when rendered; HTML string when *render* is ``False``.
    """
    level = level if level in _ALERT_PALETTE else "info"
    color, icon, bg = _ALERT_PALETTE[level]
    title_html = (
        f'<div style="font-weight:700;margin-bottom:0.25rem;color:{color};">{title}</div>'
        if title else ""
    )
    html = f"""
<div style="background:{bg};border:1px solid {color};border-left:4px solid {color};
            border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.8rem;">
    <div style="display:flex;gap:0.6rem;align-items:flex-start;">
        <div style="font-size:1.1rem;line-height:1.4;">{icon}</div>
        <div style="flex:1;">
            {title_html}
            <div style="font-size:0.88rem;color:{_TEXT_PRIMARY};line-height:1.55;">{message}</div>
        </div>
    </div>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Tag list card ─────────────────────────────────────────────────────────────

def tag_list_card(
    title: str,
    tags: Sequence[str],
    *,
    color: str = _ACCENT,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a card containing a list of pill-shaped tags.

    Args:
        title:  Card title text.
        tags:   Sequence of tag strings to display as pills.
        color:  Pill border / text accent colour.
        render: If ``True`` render via ``gr.HTML``; else return HTML string.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    pill_html = "".join(
        f'<span style="background:rgba(255,255,255,0.04);border:1px solid {color};'
        f'border-radius:20px;padding:0.25rem 0.75rem;font-size:0.82rem;'
        f'color:{color};margin:0.2rem;">{t}</span>'
        for t in tags
    )
    html = f"""
<div style="{_CARD_BASE}margin-bottom:1rem;">
    <div style="font-size:0.88rem;font-weight:700;color:{_TEXT_PRIMARY};
                margin-bottom:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">
        {title}
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.3rem;">{pill_html}</div>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Divider card ──────────────────────────────────────────────────────────────

def section_divider(label: str = "", *, color: str = _BORDER) -> None:
    """Render a styled horizontal divider with optional centred label.

    Args:
        label: Optional text displayed at the centre of the divider.
        color: Line colour.
    """
    if label:
        html = f"""
<div style="display:flex;align-items:center;gap:1rem;margin:1.2rem 0;">
    <div style="flex:1;height:1px;background:{color};"></div>
    <div style="font-size:0.78rem;color:{_TEXT_SECONDARY};text-transform:uppercase;
                letter-spacing:0.08em;white-space:nowrap;">{label}</div>
    <div style="flex:1;height:1px;background:{color};"></div>
</div>
"""
    else:
        html = f'<div style="height:1px;background:{color};margin:1.2rem 0;"></div>'
    gr.HTML(html)
