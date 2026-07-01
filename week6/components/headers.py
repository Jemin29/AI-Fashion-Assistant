"""
Week 6 — Reusable Header Components.

Provides factory functions for consistently styled page / section headers
rendered via ``gr.HTML``.  All headers follow the studio's dark glassmorphic
design language (deep navy gradient, amber accents, Outfit / Inter fonts).

Header families
---------------
- ``app_header``         — top-of-page application masthead
- ``page_header``        — per-tab page title block with subtitle
- ``section_header``     — inline section divider with optional badge
- ``breadcrumb_header``  — breadcrumb navigation trail
- ``hero_banner``        — full-width hero banner with CTA text
- ``compact_header``     — minimal one-liner header strip
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gradio as gr

# ── Design tokens ─────────────────────────────────────────────────────────────
_ACCENT  = "#ff9f43"
_ACCENT2 = "#ff5252"
_SUCCESS = "#2ecc71"
_INFO    = "#3498db"
_SURFACE = "#1e1e2f"
_BORDER  = "rgba(255,255,255,0.07)"
_TEXT_PRIM  = "#ffffff"
_TEXT_SEC   = "#a0a0b0"
_GRAD_BG    = "linear-gradient(135deg,#1e1e2f 0%,#0f0f15 100%)"
_GRAD_TEXT  = f"linear-gradient(90deg,{_ACCENT},{_ACCENT2})"


# ── Application masthead ───────────────────────────────────────────────────────

def app_header(
    title: str = "AI Fashion Creative Studio",
    subtitle: str = "Week 6 Creative Graduation Suite",
    *,
    version: str = "v1.0.0",
    icon: str = "🎨",
    show_version: bool = True,
) -> None:
    """Render the top-of-page application masthead banner.

    Args:
        title:        Application name displayed prominently.
        subtitle:     Short tagline / description rendered below.
        version:      Version string shown in the top-right corner.
        icon:         Large icon prepended to the title.
        show_version: Whether to render the version badge.
    """
    version_html = (
        f'<div style="position:absolute;top:1rem;right:1.2rem;'
        f'background:rgba(255,255,255,0.07);border:1px solid {_BORDER};'
        f'border-radius:20px;padding:0.25rem 0.75rem;font-size:0.72rem;'
        f'color:{_TEXT_SEC};font-weight:600;">{version}</div>'
        if show_version else ""
    )
    gr.HTML(f"""
<div class="studio-header-banner" style="background:{_GRAD_BG};border-bottom:2px solid {_ACCENT};
     padding:1.8rem 2rem;text-align:center;border-radius:10px 10px 0 0;
     margin-bottom:1.5rem;position:relative;">
    {version_html}
    <h1 style="color:{_TEXT_PRIM};font-family:'Outfit',sans-serif;font-size:2.3rem;
               font-weight:800;margin:0;letter-spacing:-0.5px;">
        {icon}
        <span style="background:{_GRAD_TEXT};-webkit-background-clip:text;
                     -webkit-text-fill-color:transparent;">&nbsp;{title}</span>
    </h1>
    <p style="color:{_TEXT_SEC};font-family:'Inter',sans-serif;font-size:0.95rem;
              margin:0.5rem 0 0;font-weight:300;">{subtitle}</p>
</div>
""")


# ── Per-page header ────────────────────────────────────────────────────────────

def page_header(
    title: str,
    subtitle: str = "",
    *,
    icon: str = "",
    badge: str = "",
    badge_color: str = _ACCENT,
    accent_color: str = _ACCENT,
) -> None:
    """Render a per-tab page title block with optional subtitle and badge.

    Args:
        title:        Page title (rendered as a large heading).
        subtitle:     Optional short description rendered below.
        icon:         Emoji prepended to the title.
        badge:        Optional pill badge text (e.g. ``"NEW"`` or ``"Week 5"``).
        badge_color:  Badge background colour.
        accent_color: Gradient / underline accent colour.
    """
    icon_html = f"{icon}&nbsp;" if icon else ""
    badge_html = (
        f'<span style="background:{badge_color};color:#000;font-size:0.68rem;'
        f'font-weight:800;border-radius:20px;padding:0.15rem 0.6rem;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-left:0.6rem;'
        f'vertical-align:middle;">{badge}</span>'
        if badge else ""
    )
    subtitle_html = (
        f'<p style="color:{_TEXT_SEC};font-family:\'Inter\',sans-serif;font-size:0.95rem;'
        f'margin:0.4rem 0 0;font-weight:300;line-height:1.6;">{subtitle}</p>'
        if subtitle else ""
    )
    gr.HTML(f"""
<div style="margin-bottom:1.6rem;padding-bottom:1rem;
            border-bottom:2px solid {accent_color}22;">
    <h2 style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
               color:{_TEXT_PRIM};margin:0;line-height:1.2;">
        {icon_html}{title}{badge_html}
    </h2>
    {subtitle_html}
</div>
""")


# ── Section header ─────────────────────────────────────────────────────────────

def section_header(
    label: str,
    *,
    icon: str = "",
    badge: str = "",
    badge_color: str = _INFO,
    level: int = 3,
    accent_color: str = _ACCENT,
    divider: bool = False,
) -> None:
    """Render an inline section divider header.

    Args:
        label:        Section label text.
        icon:         Optional emoji prepended to the label.
        badge:        Optional small count / status badge.
        badge_color:  Badge accent colour.
        level:        Heading level 2–5 (visual size only, uses ``<div>``).
        accent_color: Left-border accent colour.
        divider:      If ``True``, render a top horizontal rule first.
    """
    sizes = {2: "1.3rem", 3: "1.1rem", 4: "0.95rem", 5: "0.85rem"}
    font_size = sizes.get(level, "1.1rem")
    icon_html = f"{icon}&nbsp;" if icon else ""
    badge_html = (
        f'<span style="background:{badge_color};color:#fff;font-size:0.65rem;'
        f'font-weight:700;border-radius:10px;padding:0.1rem 0.45rem;'
        f'margin-left:0.4rem;">{badge}</span>'
        if badge else ""
    )
    divider_html = f'<hr style="border:none;border-top:1px solid {_BORDER};margin:1rem 0 0.8rem;"/>' if divider else ""
    gr.HTML(f"""
{divider_html}
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.9rem;
            padding-left:0.75rem;border-left:3px solid {accent_color};">
    <span style="font-family:'Outfit',sans-serif;font-size:{font_size};
                 font-weight:700;color:{_TEXT_PRIM};">
        {icon_html}{label}{badge_html}
    </span>
</div>
""")


# ── Breadcrumb header ──────────────────────────────────────────────────────────

def breadcrumb_header(
    crumbs: Sequence[str],
    *,
    separator: str = "›",
    accent_color: str = _ACCENT,
) -> None:
    """Render a breadcrumb navigation trail.

    The *last* crumb is highlighted in the accent colour; all others are muted.

    Args:
        crumbs:       Ordered sequence of breadcrumb strings.
        separator:    Separator glyph between crumbs.
        accent_color: Colour for the active (last) crumb.
    """
    parts: list[str] = []
    for i, crumb in enumerate(crumbs):
        is_last = (i == len(crumbs) - 1)
        colour = accent_color if is_last else _TEXT_SEC
        weight = "700" if is_last else "400"
        parts.append(
            f'<span style="color:{colour};font-weight:{weight};">{crumb}</span>'
        )
    sep_html = f' <span style="color:{_BORDER};margin:0 0.4rem;">{separator}</span> '
    gr.HTML(f"""
<div style="font-family:'Inter',sans-serif;font-size:0.82rem;
            margin-bottom:1rem;display:flex;align-items:center;flex-wrap:wrap;">
    {sep_html.join(parts)}
</div>
""")


# ── Hero banner ────────────────────────────────────────────────────────────────

def hero_banner(
    headline: str,
    tagline: str = "",
    *,
    cta_text: str = "",
    gradient_start: str = "rgba(255,159,67,0.12)",
    gradient_end: str = "rgba(255,82,82,0.08)",
    icon: str = "✨",
    extra_html: str = "",
) -> None:
    """Render a full-width hero banner with headline, tagline, and optional CTA.

    Args:
        headline:        Large headline text (supports HTML).
        tagline:         Subtitle / tagline below the headline.
        cta_text:        Optional call-to-action text (displayed as a styled pill).
        gradient_start:  CSS colour for the top-left gradient stop.
        gradient_end:    CSS colour for the bottom-right gradient stop.
        icon:            Decorative icon above the headline.
        extra_html:      Any additional raw HTML appended inside the banner.
    """
    cta_html = (
        f'<div style="display:inline-block;background:{_ACCENT};color:#000;'
        f'font-weight:700;font-size:0.88rem;border-radius:25px;'
        f'padding:0.5rem 1.5rem;margin-top:1rem;letter-spacing:0.04em;">'
        f'{cta_text}</div>'
        if cta_text else ""
    )
    tagline_html = (
        f'<p style="color:{_TEXT_SEC};font-size:1rem;font-weight:300;'
        f'margin:0.5rem 0 0;line-height:1.7;">{tagline}</p>'
        if tagline else ""
    )
    gr.HTML(f"""
<div class="studio-hero" style="background:linear-gradient(135deg,{gradient_start} 0%,{gradient_end} 100%);
     border:1px solid {_BORDER};border-radius:12px;padding:2.5rem 2rem;
     text-align:center;margin-bottom:2rem;
     box-shadow:0 8px 32px rgba(0,0,0,0.35);
     backdrop-filter:blur(4px);">
    <div style="font-size:2.8rem;margin-bottom:0.5rem;">{icon}</div>
    <h1 style="font-family:'Outfit',sans-serif;font-size:2.2rem;font-weight:800;margin:0;
               background:{_GRAD_TEXT};-webkit-background-clip:text;
               -webkit-text-fill-color:transparent;letter-spacing:-0.5px;">
        {headline}
    </h1>
    {tagline_html}
    {cta_html}
    {extra_html}
</div>
""")


# ── Compact header ─────────────────────────────────────────────────────────────

def compact_header(
    title: str,
    right_text: str = "",
    *,
    icon: str = "",
    accent_color: str = _ACCENT,
) -> None:
    """Render a minimal one-liner header strip with optional right-aligned text.

    Args:
        title:        Header text (left-aligned).
        right_text:   Optional text displayed flush-right (e.g. a count or timestamp).
        icon:         Optional emoji prepended to the title.
        accent_color: Bottom-border underline colour.
    """
    icon_html = f"{icon}&nbsp;" if icon else ""
    right_html = (
        f'<span style="font-size:0.78rem;color:{_TEXT_SEC};font-weight:400;">'
        f'{right_text}</span>'
        if right_text else ""
    )
    gr.HTML(f"""
<div style="display:flex;align-items:baseline;justify-content:space-between;
            border-bottom:1px solid {accent_color}44;padding-bottom:0.5rem;
            margin-bottom:1rem;">
    <span style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;
                 color:{_TEXT_PRIM};">{icon_html}{title}</span>
    {right_html}
</div>
""")


# ── Tab strip header ───────────────────────────────────────────────────────────

def tab_strip_header(
    tabs: Sequence[str],
    active: str,
    *,
    icon_map: Optional[dict] = None,
    accent_color: str = _ACCENT,
) -> Tuple[gr.HTML, gr.State]:
    """Render a visual pill-tab strip that tracks the active selection via state.

    .. note::
        This is a **display-only** component; it does not drive Gradio tab
        switching by itself.  Connect the returned ``gr.State`` as an output of
        a ``gr.Radio`` (nav selector) to keep them in sync.

    Args:
        tabs:         Ordered list of tab names.
        active:       Currently active tab name (must be in *tabs*).
        icon_map:     Optional ``{tab_name: emoji}`` mapping.
        accent_color: Active pill highlight colour.

    Returns:
        Tuple of ``(html_widget, active_state)``.
    """
    icon_map = icon_map or {}
    _active_state = gr.State(value=active)

    def _render(active_tab: str) -> str:
        pills = []
        for tab in tabs:
            ico = icon_map.get(tab, "")
            label = f"{ico}&nbsp;{tab}".strip() if ico else tab
            is_active = (tab == active_tab)
            bg  = accent_color if is_active else "transparent"
            col = "#000" if is_active else _TEXT_SEC
            bdr = accent_color if is_active else _BORDER
            pills.append(
                f'<div style="background:{bg};color:{col};border:1px solid {bdr};'
                f'border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;'
                f'font-weight:{"700" if is_active else "400"};white-space:nowrap;'
                f'cursor:pointer;">{label}</div>'
            )
        return (
            '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;'
            'margin-bottom:1rem;">' + "".join(pills) + "</div>"
        )

    html_widget = gr.HTML(value=_render(active))
    return html_widget, _active_state
