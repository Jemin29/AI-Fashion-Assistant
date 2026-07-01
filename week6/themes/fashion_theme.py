"""
Week 6 — Fashion Theme (Aurora Luxe) — Enhanced Edition.

The primary production theme for the AI Fashion Creative Studio.  Deep indigo-
slate backgrounds, warm amber-coral accents, and fluid gradient animations
inspired by the aurora borealis — a nod to fashion's seasonal colour cycles.

Design language
---------------
- Background   : hsl(225, 25%, 7%)   (deep indigo-navy)
- Surface       : hsl(225, 20%, 10%)  (elevated slate)
- Accent #1     : hsl(245, 70%, 60%)  (electric indigo)
- Accent #2     : hsl(15, 88%, 65%)   (coral-amber)
- Accent #3     : hsl(175, 65%, 50%)  (teal)
- Typography    : Outfit (headings) + Inter (body) + JetBrains Mono (code)

Backward compatibility
----------------------
``FashionTheme`` and ``get_theme()`` remain importable as before so
``app.py`` requires no changes.
"""
from __future__ import annotations

import gradio as gr
from gradio.themes import Base, GoogleFont
from gradio.themes.utils import colors, sizes, fonts


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Theme Class
# ══════════════════════════════════════════════════════════════════════════════

class FashionTheme(Base):
    """
    Aurora Luxe — the flagship Gradio theme for the AI Fashion Creative Studio.

    Design principles
    -----------------
    - Deep indigo-navy backgrounds (editorial dark mode)
    - Warm amber + electric indigo dual-accent palette
    - Outfit / Inter premium typography
    - Glassmorphism card surfaces with glow hovers
    - Smooth multi-stop gradient animations
    """

    NAME = "aurora_luxe"
    DESCRIPTION = "Deep indigo-navy with amber-coral aurora accents."
    PREVIEW_ACCENT = "#ff9f43"

    def __init__(self) -> None:
        super().__init__(
            primary_hue=colors.indigo,
            secondary_hue=colors.slate,
            neutral_hue=colors.slate,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            text_size=sizes.text_md,
            font=[
                GoogleFont("Outfit"),
                GoogleFont("Inter"),
                fonts.Font("system-ui"),
                fonts.Font("sans-serif"),
            ],
            font_mono=[
                GoogleFont("JetBrains Mono"),
                fonts.Font("Consolas"),
                fonts.Font("monospace"),
            ],
        )
        self.set(
            # ── Backgrounds ──────────────────────────────────────────────
            body_background_fill="hsl(225, 25%, 7%)",
            body_background_fill_dark="hsl(225, 25%, 7%)",
            block_background_fill="hsl(225, 20%, 10%)",
            block_background_fill_dark="hsl(225, 20%, 10%)",
            panel_background_fill="hsl(225, 20%, 12%)",
            panel_background_fill_dark="hsl(225, 20%, 12%)",
            # ── Borders ───────────────────────────────────────────────────
            block_border_color="hsl(225, 20%, 18%)",
            block_border_color_dark="hsl(225, 20%, 18%)",
            block_border_width="1px",
            panel_border_color="hsl(225, 18%, 22%)",
            # ── Typography ───────────────────────────────────────────────
            body_text_color="hsl(225, 30%, 88%)",
            body_text_color_dark="hsl(225, 30%, 88%)",
            body_text_color_subdued="hsl(225, 20%, 55%)",
            body_text_color_subdued_dark="hsl(225, 20%, 55%)",
            block_title_text_color="hsl(225, 35%, 92%)",
            block_label_text_color="hsl(225, 18%, 62%)",
            # ── Buttons ───────────────────────────────────────────────────
            button_primary_background_fill=(
                "linear-gradient(135deg, hsl(245,70%,55%) 0%, hsl(275,65%,45%) 100%)"
            ),
            button_primary_background_fill_hover=(
                "linear-gradient(135deg, hsl(245,75%,62%) 0%, hsl(275,70%,52%) 100%)"
            ),
            button_primary_text_color="white",
            button_primary_border_color="transparent",
            button_primary_border_color_hover="transparent",
            button_secondary_background_fill="hsl(225, 20%, 15%)",
            button_secondary_background_fill_hover="hsl(225, 20%, 20%)",
            button_secondary_text_color="hsl(225, 30%, 80%)",
            button_secondary_border_color="hsl(225, 18%, 25%)",
            button_cancel_background_fill="hsl(0, 65%, 42%)",
            button_cancel_background_fill_hover="hsl(0, 68%, 50%)",
            button_cancel_text_color="white",
            # ── Inputs ───────────────────────────────────────────────────
            input_background_fill="hsl(225, 22%, 13%)",
            input_background_fill_focus="hsl(225, 22%, 16%)",
            input_border_color="hsl(225, 20%, 22%)",
            input_border_color_focus="hsl(245, 60%, 55%)",
            input_border_color_hover="hsl(225, 20%, 28%)",
            input_placeholder_color="hsl(225, 15%, 38%)",
            input_shadow="none",
            input_shadow_focus="0 0 0 3px rgba(102,126,234,0.18)",
            # ── Slider ───────────────────────────────────────────────────
            slider_color="hsl(245, 70%, 60%)",
            slider_color_dark="hsl(245, 70%, 60%)",
            # ── Checkbox / Radio ─────────────────────────────────────────
            checkbox_background_color="hsl(225, 22%, 13%)",
            checkbox_background_color_focus="hsl(225, 22%, 16%)",
            checkbox_background_color_selected="hsl(245, 70%, 55%)",
            checkbox_border_color="hsl(225, 18%, 26%)",
            checkbox_border_color_focus="hsl(245, 60%, 55%)",
            # ── Shadows ──────────────────────────────────────────────────
            block_shadow="0 4px 24px rgba(0,0,0,0.4)",
            block_shadow_dark="0 4px 24px rgba(0,0,0,0.6)",
        )


def get_theme() -> FashionTheme:
    """Return a fresh ``FashionTheme`` instance (backward-compatible)."""
    return FashionTheme()


# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS — Aurora Luxe design system
# ══════════════════════════════════════════════════════════════════════════════

STUDIO_CSS = """
/* ── Aurora Luxe — AI Fashion Creative Studio CSS ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ─────────────────────────────────────────────────────────── */
:root {
    --bg-deep:       hsl(225, 25%, 7%);
    --bg-card:       hsl(225, 20%, 10%);
    --bg-elevated:   hsl(225, 20%, 14%);
    --bg-input:      hsl(225, 22%, 13%);
    --bg-overlay:    rgba(10, 12, 28, 0.85);

    --accent-primary:   hsl(245, 70%, 60%);
    --accent-glow:      rgba(102, 126, 234, 0.30);
    --accent-coral:     hsl(15, 88%, 65%);
    --accent-teal:      hsl(175, 65%, 50%);
    --accent-gold:      hsl(42, 90%, 60%);
    --accent-emerald:   hsl(155, 65%, 50%);
    --accent-rose:      hsl(330, 70%, 62%);

    --text-primary:  hsl(225, 30%, 90%);
    --text-secondary:hsl(225, 20%, 60%);
    --text-muted:    hsl(225, 15%, 40%);

    --border:        hsl(225, 20%, 18%);
    --border-accent: var(--accent-primary);

    --radius-xs: 4px;
    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;

    --shadow-sm:   0 2px 8px rgba(0,0,0,0.3);
    --shadow-md:   0 4px 20px rgba(0,0,0,0.45);
    --shadow-lg:   0 8px 40px rgba(0,0,0,0.6);
    --shadow-glow: 0 0 30px var(--accent-glow);

    --transition: 200ms cubic-bezier(0.4,0,0.2,1);
    --transition-slow: 400ms cubic-bezier(0.4,0,0.2,1);
}

/* ── Body & Container ─────────────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Outfit','Inter',system-ui,sans-serif !important;
    color: var(--text-primary) !important;
}
.gradio-container { max-width:1340px !important; margin:0 auto; padding:0 16px; }

/* ── Hero Banner ──────────────────────────────────────────────────────────── */
@keyframes aurora {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.studio-hero {
    text-align: center;
    padding: 40px 20px 24px;
    position: relative;
}
.studio-hero h1 {
    font-size: clamp(2rem,5vw,3.2rem) !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px !important;
    background: linear-gradient(120deg,
        hsl(15,88%,65%) 0%,
        hsl(42,90%,60%) 20%,
        hsl(245,70%,65%) 50%,
        hsl(175,65%,50%) 80%,
        hsl(15,88%,65%) 100%) !important;
    background-size: 400% 400% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: aurora 10s ease infinite !important;
}
.studio-subtitle {
    color: var(--text-secondary) !important;
    font-size: 1.05rem !important;
    margin-top: 8px !important;
    font-weight: 300 !important;
}

/* ── Header Banner ─────────────────────────────────────────────────────────── */
.studio-header-banner {
    background: linear-gradient(135deg, hsl(225,25%,9%) 0%, hsl(225,22%,6%) 100%);
    border-bottom: 2px solid var(--accent-coral);
    border-radius: 10px 10px 0 0;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
}

/* ── Panels & Cards ────────────────────────────────────────────────────────── */
.panel, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color var(--transition) !important;
}
.metric-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-elevated)) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}
.metric-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: var(--shadow-glow) !important;
    border-color: var(--accent-primary) !important;
}
.metric-value { font-size: 2rem !important; font-weight: 700 !important; color: var(--accent-primary) !important; }
.metric-label { color: var(--text-secondary) !important; font-size: 0.85rem !important; margin-top: 4px !important; }

/* ── Glassmorphism Panel ───────────────────────────────────────────────────── */
.glass-panel {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, hsl(245,70%,55%) 0%, hsl(275,65%,45%) 100%) !important;
    border: none !important; color: #fff !important; font-weight: 600 !important;
    border-radius: var(--radius-md) !important; padding: 12px 24px !important;
    box-shadow: 0 4px 16px rgba(102,126,234,0.35) !important;
    transition: all var(--transition) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(102,126,234,0.5) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* ── Inputs ────────────────────────────────────────────────────────────────── */
input, textarea, select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}
input:focus, textarea:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
    outline: none !important;
}

/* ── Tab Navigation ────────────────────────────────────────────────────────── */
.tab-nav {
    background: var(--bg-card) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    padding: 6px !important;
    gap: 4px !important;
}
.tab-nav button {
    border-radius: var(--radius-md) !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    transition: all var(--transition) !important;
}
.tab-nav button:hover { color: var(--text-primary) !important; background: var(--bg-elevated) !important; }
.tab-nav button.selected {
    background: var(--accent-primary) !important;
    color: white !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ── Gallery ───────────────────────────────────────────────────────────────── */
.gallery-item {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 2px solid transparent !important;
    transition: all var(--transition) !important;
}
.gallery-item:hover {
    border-color: var(--accent-primary) !important;
    transform: scale(1.02) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ── Chat ──────────────────────────────────────────────────────────────────── */
.chatbot .message.bot {
    background: var(--bg-elevated) !important;
    border-left: 3px solid var(--accent-primary) !important;
    border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
}
.chatbot .message.user {
    background: linear-gradient(135deg, hsl(245,50%,22%) 0%, hsl(275,45%,18%) 100%) !important;
    border-radius: var(--radius-md) 0 0 var(--radius-md) !important;
}

/* ── Code / Markdown ───────────────────────────────────────────────────────── */
code, pre {
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 0.88rem !important;
}
pre { padding: 16px !important; overflow-x: auto !important; border: 1px solid var(--border) !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* ── Slider ────────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: var(--accent-primary) !important; }

/* ── Accordion ─────────────────────────────────────────────────────────────── */
.accordion-header {
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ── Status Badges ─────────────────────────────────────────────────────────── */
.badge-active   { background:rgba(52,211,153,0.14)!important; color:hsl(155,65%,52%)!important;
                  border:1px solid rgba(52,211,153,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
.badge-mock     { background:rgba(251,191,36,0.14)!important; color:hsl(42,90%,60%)!important;
                  border:1px solid rgba(251,191,36,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
.badge-inactive { background:rgba(107,114,128,0.14)!important; color:hsl(220,10%,55%)!important;
                  border:1px solid rgba(107,114,128,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
.badge-new      { background:rgba(102,126,234,0.18)!important; color:hsl(245,70%,72%)!important;
                  border:1px solid rgba(102,126,234,0.35)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }

/* ── Confidence Bars ───────────────────────────────────────────────────────── */
.confidence-bar-wrap { background:rgba(255,255,255,0.05); border-radius:4px; height:5px; overflow:hidden; }
.confidence-bar-fill { height:100%; border-radius:4px; transition:width 0.5s ease; }

/* ── Sidebar Nav ───────────────────────────────────────────────────────────── */
#sidebar-nav-col { background: var(--bg-card); border-right: 1px solid var(--border); border-radius: var(--radius-md); }
#main-panel-col  { animation: fadeIn 0.3s ease-out; }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes slideInLeft { from { opacity:0; transform:translateX(-16px); } to { opacity:1; transform:translateX(0); } }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
.gradio-container { animation: fadeIn 0.4s ease-out; }
"""
