"""
Week 6 — Reusable Footer Components.

Provides factory functions for consistently styled page footers and bottom
bars rendered via ``gr.HTML``.  All footers follow the studio's dark
glassmorphic design language.

Footer families
---------------
- ``app_footer``         — full application footer with links and copyright
- ``status_footer``      — live system status bar
- ``mini_footer``        — compact single-line footer strip
- ``session_info_footer``— session metadata (model, resolution, timestamp)
- ``attribution_footer`` — technology stack attribution row
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import gradio as gr

# ── Design tokens ─────────────────────────────────────────────────────────────
_ACCENT  = "#ff9f43"
_ACCENT2 = "#ff5252"
_SUCCESS = "#2ecc71"
_INFO    = "#3498db"
_WARNING = "#f39c12"
_SURFACE = "#111118"
_BORDER  = "rgba(255,255,255,0.05)"
_TEXT_PRIM  = "#f0f0f5"
_TEXT_SEC   = "#888899"
_GRAD_TEXT  = f"linear-gradient(90deg,{_ACCENT},{_ACCENT2})"


# ── App footer ─────────────────────────────────────────────────────────────────

def app_footer(
    project_name: str = "AI Fashion Creative Studio",
    *,
    version: str = "v1.0.0",
    links: Optional[Dict[str, str]] = None,
    copyright_year: int = 2024,
    show_built_with: bool = True,
) -> None:
    """Render a full application footer with optional navigation links.

    Args:
        project_name:   Application name displayed in the copyright line.
        version:        Version string shown next to the project name.
        links:          Optional ``{label: url}`` dict for footer navigation links.
        copyright_year: Year in the copyright statement.
        show_built_with:Whether to show a 'Built with' technology row.
    """
    links = links or {
        "Documentation": "#",
        "GitHub": "#",
        "Report Issue": "#",
    }
    link_items = "".join(
        f'<a href="{url}" style="color:{_TEXT_SEC};text-decoration:none;font-size:0.8rem;'
        f'transition:color 0.2s;" onmouseover="this.style.color=\'{_ACCENT}\'"'
        f' onmouseout="this.style.color=\'{_TEXT_SEC}\'">{label}</a>'
        for label, url in links.items()
    )
    built_html = (
        f'<div style="font-size:0.75rem;color:{_TEXT_SEC};margin-top:0.6rem;">'
        f'Built with&nbsp;'
        f'<span style="color:{_ACCENT};">Gradio</span> &middot; '
        f'<span style="color:{_INFO};">PyTorch</span> &middot; '
        f'<span style="color:{_SUCCESS};">Diffusers</span> &middot; '
        f'<span style="color:#9b59b6;">ChromaDB</span>'
        f'</div>'
        if show_built_with else ""
    )
    gr.HTML(f"""
<div style="background:{_SURFACE};border-top:1px solid {_BORDER};
            padding:1.4rem 2rem;margin-top:2.5rem;border-radius:0 0 10px 10px;">
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;
                align-items:center;gap:1rem;">
        <div>
            <span style="font-family:'Outfit',sans-serif;font-size:0.9rem;
                         font-weight:700;background:{_GRAD_TEXT};
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                {project_name}
            </span>
            <span style="color:{_TEXT_SEC};font-size:0.75rem;margin-left:0.4rem;">{version}</span>
        </div>
        <div style="display:flex;gap:1.5rem;align-items:center;">{link_items}</div>
    </div>
    <div style="border-top:1px solid {_BORDER};margin-top:0.9rem;padding-top:0.7rem;
                display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;">
        <div style="font-size:0.75rem;color:{_TEXT_SEC};">
            &copy; {copyright_year} {project_name}. All rights reserved.
        </div>
        {built_html}
    </div>
</div>
""")


# ── Status footer ──────────────────────────────────────────────────────────────

def status_footer(
    *,
    mock_mode: bool = True,
    version: str = "v1.0.0",
    environment: str = "Development",
    show_timestamp: bool = True,
) -> gr.HTML:
    """Render a live system status bar and return the ``gr.HTML`` widget.

    Returning the widget allows callers to update it dynamically via event
    handlers (e.g. update the GPU / CPU label after settings are changed).

    Args:
        mock_mode:       If ``True`` shows a mock-mode indicator; else GPU-ready.
        version:         Application version string.
        environment:     Deployment environment label.
        show_timestamp:  Whether to display the current date.

    Returns:
        The rendered ``gr.HTML`` widget (can be passed to ``.update()``).
    """
    def _render(is_mock: bool) -> str:
        mode_text  = "🟡 Mock Mode (CPU)" if is_mock else "🟢 GPU Production"
        mode_color = _WARNING if is_mock else _SUCCESS
        ts = f"📅 {time.strftime('%Y-%m-%d')}" if show_timestamp else ""
        return f"""
<div style="background:{_SURFACE};border:1px solid {_BORDER};
            border-radius:6px;padding:0.75rem 1.4rem;margin-top:1.8rem;
            display:flex;flex-wrap:wrap;justify-content:space-between;
            align-items:center;gap:0.5rem;font-size:0.82rem;color:{_TEXT_SEC};">
    <div>⚡ Status:&nbsp;<span style="color:{mode_color};font-weight:600;">{mode_text}</span></div>
    <div>🌐 Env:&nbsp;<span style="color:{_INFO};">{environment}</span></div>
    {f'<div>{ts}</div>' if ts else ''}
    <div>🏷️ Version:&nbsp;<span style="color:{_SUCCESS};">{version}</span></div>
</div>
"""

    widget = gr.HTML(value=_render(mock_mode))
    return widget


# ── Mini footer ────────────────────────────────────────────────────────────────

def mini_footer(
    text: str = "AI Fashion Creative Studio — Week 6",
    *,
    right_text: str = "",
    accent_color: str = _ACCENT,
) -> None:
    """Render a compact single-line footer strip.

    Args:
        text:         Left-side text content.
        right_text:   Optional right-aligned text (e.g. a quick status note).
        accent_color: Top border accent colour.
    """
    right_html = (
        f'<span style="font-size:0.75rem;color:{_TEXT_SEC};">{right_text}</span>'
        if right_text else ""
    )
    gr.HTML(f"""
<div style="border-top:1px solid {accent_color}33;padding:0.6rem 0;margin-top:1.5rem;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:0.76rem;color:{_TEXT_SEC};">{text}</span>
    {right_html}
</div>
""")


# ── Session info footer ────────────────────────────────────────────────────────

def session_info_footer(
    model: str = "SDXL v1.0 (Mock)",
    resolution: str = "1024×1024",
    *,
    style: str = "",
    brand: str = "",
    generation_count: int = 0,
) -> gr.HTML:
    """Render a session metadata footer and return the widget for dynamic updates.

    Args:
        model:            Active generation model name.
        resolution:       Current output resolution string.
        style:            Active style preset (optional).
        brand:            Active brand filter (optional).
        generation_count: Number of images generated in this session.

    Returns:
        The rendered ``gr.HTML`` widget.
    """
    def _render(m: str, res: str, sty: str, br: str, cnt: int) -> str:
        items: list[str] = [
            f'🤖 Model: <span style="color:{_ACCENT};">{m}</span>',
            f'📐 Resolution: <span style="color:{_INFO};">{res}</span>',
        ]
        if sty:
            items.append(f'🎨 Style: <span style="color:#9b59b6;">{sty}</span>')
        if br:
            items.append(f'🏷️ Brand: <span style="color:{_SUCCESS};">{br}</span>')
        items.append(f'🖼️ Generated: <span style="color:{_ACCENT};">{cnt}</span>')

        pills = "&nbsp;&bull;&nbsp;".join(items)
        return f"""
<div style="background:{_SURFACE};border:1px solid {_BORDER};border-radius:6px;
            padding:0.6rem 1.2rem;margin-top:0.5rem;font-size:0.8rem;color:{_TEXT_SEC};
            white-space:nowrap;overflow-x:auto;">
    {pills}
</div>
"""

    widget = gr.HTML(value=_render(model, resolution, style, brand, generation_count))
    return widget


# ── Attribution footer ─────────────────────────────────────────────────────────

def attribution_footer(
    technologies: Optional[Sequence[Dict[str, str]]] = None,
) -> None:
    """Render a technology-stack attribution row.

    Args:
        technologies: Optional list of ``{"name": ..., "color": ..., "icon": ...}``
                      dicts.  Defaults to the studio's standard stack.
    """
    if technologies is None:
        technologies = [
            {"name": "Gradio 4",      "color": _ACCENT,  "icon": "🖥️"},
            {"name": "Diffusers",     "color": _SUCCESS,  "icon": "🎨"},
            {"name": "PyTorch",       "color": _ACCENT2,  "icon": "🔥"},
            {"name": "ChromaDB",      "color": "#9b59b6", "icon": "🗄️"},
            {"name": "LangChain",     "color": _INFO,     "icon": "🔗"},
            {"name": "Transformers",  "color": _WARNING,  "icon": "🤗"},
        ]
    pills = "".join(
        f'<span style="background:rgba(255,255,255,0.03);border:1px solid {t["color"]}44;'
        f'border-radius:20px;padding:0.2rem 0.7rem;font-size:0.75rem;'
        f'color:{t["color"]};margin:0.2rem;">{t["icon"]}&nbsp;{t["name"]}</span>'
        for t in technologies
    )
    gr.HTML(f"""
<div style="text-align:center;margin-top:1rem;padding:0.8rem;
            border-top:1px solid {_BORDER};">
    <div style="font-size:0.72rem;color:{_TEXT_SEC};margin-bottom:0.5rem;
                text-transform:uppercase;letter-spacing:0.06em;">Powered By</div>
    <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0.4rem;">
        {pills}
    </div>
</div>
""")
