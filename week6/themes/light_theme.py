"""
Week 6 — Light Theme (Ivory Atelier).

A crisp editorial-light Gradio theme inspired by the clean aesthetic of
premium fashion lookbooks and luxury white-space design:  warm ivory
backgrounds, deep charcoal type, and rich jewel-tone accents.

Design language
---------------
- Background   : hsl(40, 30%, 97%)   (warm white / ivory)
- Surface       : hsl(40, 20%, 99%)   (pure white panel)
- Accent        : hsl(245, 68%, 55%)  (deep indigo)
- Secondary     : hsl(330, 65%, 55%)  (raspberry rose)
- Success       : hsl(160, 55%, 40%)  (deep emerald)
- Typography    : Cormorant Garamond (headings) + Inter (body) + JetBrains Mono (code)
"""
from __future__ import annotations

import gradio as gr
from gradio.themes import Base, GoogleFont
from gradio.themes.utils import colors, sizes, fonts


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Theme Class
# ══════════════════════════════════════════════════════════════════════════════

class LightTheme(Base):
    """
    Ivory Atelier — a clean, editorial light theme for the AI Fashion Creative Studio.

    Inspired by minimalist luxury fashion branding:
    warm ivory, deep indigo accents, and razor-sharp typography.
    """

    NAME = "ivory_atelier"
    DESCRIPTION = "Clean editorial ivory with deep indigo fashion accents."
    PREVIEW_ACCENT = "#5046e4"

    def __init__(self) -> None:
        super().__init__(
            primary_hue=colors.indigo,
            secondary_hue=colors.pink,
            neutral_hue=colors.stone,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            text_size=sizes.text_md,
            font=[
                GoogleFont("Cormorant Garamond"),
                GoogleFont("Inter"),
                fonts.Font("Georgia"),
                fonts.Font("serif"),
            ],
            font_mono=[
                GoogleFont("JetBrains Mono"),
                fonts.Font("Consolas"),
                fonts.Font("monospace"),
            ],
        )
        self.set(
            # ── Backgrounds ──────────────────────────────────────────────
            body_background_fill="hsl(40, 30%, 97%)",
            body_background_fill_dark="hsl(40, 20%, 94%)",
            block_background_fill="hsl(0, 0%, 100%)",
            block_background_fill_dark="hsl(40, 15%, 98%)",
            panel_background_fill="hsl(40, 18%, 98%)",
            panel_background_fill_dark="hsl(40, 18%, 98%)",
            # ── Borders ───────────────────────────────────────────────────
            block_border_color="hsl(40, 18%, 88%)",
            block_border_color_dark="hsl(40, 18%, 88%)",
            block_border_width="1px",
            panel_border_color="hsl(40, 14%, 85%)",
            # ── Typography ───────────────────────────────────────────────
            body_text_color="hsl(240, 15%, 18%)",
            body_text_color_dark="hsl(240, 15%, 18%)",
            body_text_color_subdued="hsl(240, 8%, 46%)",
            body_text_color_subdued_dark="hsl(240, 8%, 46%)",
            block_title_text_color="hsl(240, 20%, 14%)",
            block_label_text_color="hsl(240, 10%, 38%)",
            # ── Buttons ───────────────────────────────────────────────────
            button_primary_background_fill=(
                "linear-gradient(135deg, hsl(245,68%,55%) 0%, hsl(270,60%,48%) 100%)"
            ),
            button_primary_background_fill_hover=(
                "linear-gradient(135deg, hsl(245,72%,62%) 0%, hsl(270,65%,55%) 100%)"
            ),
            button_primary_text_color="white",
            button_primary_border_color="transparent",
            button_primary_border_color_hover="transparent",
            button_secondary_background_fill="hsl(40, 18%, 95%)",
            button_secondary_background_fill_hover="hsl(40, 18%, 90%)",
            button_secondary_text_color="hsl(240, 15%, 25%)",
            button_secondary_border_color="hsl(40, 18%, 82%)",
            button_cancel_background_fill="hsl(0, 60%, 52%)",
            button_cancel_background_fill_hover="hsl(0, 62%, 58%)",
            button_cancel_text_color="white",
            # ── Inputs ───────────────────────────────────────────────────
            input_background_fill="hsl(0, 0%, 100%)",
            input_background_fill_focus="hsl(245, 30%, 98%)",
            input_border_color="hsl(40, 18%, 80%)",
            input_border_color_focus="hsl(245, 68%, 55%)",
            input_border_color_hover="hsl(40, 18%, 70%)",
            input_placeholder_color="hsl(240, 8%, 58%)",
            input_shadow="0 1px 3px rgba(0,0,0,0.06)",
            input_shadow_focus="0 0 0 3px rgba(80,70,228,0.14)",
            # ── Slider ───────────────────────────────────────────────────
            slider_color="hsl(245, 68%, 55%)",
            slider_color_dark="hsl(245, 68%, 55%)",
            # ── Checkbox / Radio ─────────────────────────────────────────
            checkbox_background_color="hsl(0, 0%, 100%)",
            checkbox_background_color_focus="hsl(245, 30%, 97%)",
            checkbox_background_color_selected="hsl(245, 68%, 55%)",
            checkbox_border_color="hsl(40, 18%, 76%)",
            checkbox_border_color_focus="hsl(245, 68%, 55%)",
            # ── Shadows ──────────────────────────────────────────────────
            block_shadow="0 2px 12px rgba(0,0,0,0.06)",
            block_shadow_dark="0 2px 12px rgba(0,0,0,0.08)",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

LIGHT_CSS = """
/* ── Light Theme (Ivory Atelier) ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:        hsl(40, 30%, 97%);
    --bg-card:        hsl(0, 0%, 100%);
    --bg-elevated:    hsl(40, 18%, 96%);
    --bg-input:       hsl(0, 0%, 100%);

    --accent-indigo:  hsl(245, 68%, 55%);
    --accent-indigo-g:rgba(80, 70, 228, 0.18);
    --accent-rose:    hsl(330, 65%, 55%);
    --accent-gold:    hsl(42, 85%, 48%);
    --accent-emerald: hsl(160, 55%, 40%);

    --text-primary:   hsl(240, 15%, 18%);
    --text-secondary: hsl(240, 8%, 46%);
    --text-muted:     hsl(240, 6%, 62%);

    --border:         hsl(40, 18%, 88%);
    --border-accent:  var(--accent-indigo);

    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;

    --shadow-sm:   0 1px 4px rgba(0,0,0,0.06);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.10);
    --shadow-lg:   0 8px 32px rgba(0,0,0,0.14);
    --shadow-glow: 0 0 24px rgba(80,70,228,0.20);

    --transition: 200ms cubic-bezier(0.4,0,0.2,1);
}

body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Cormorant Garamond','Inter',Georgia,serif !important;
    color: var(--text-primary) !important;
}
.gradio-container { max-width:1340px !important; margin:0 auto; padding:0 16px; }

/* ── Animated headline ─────────────────────────────────────────────────────── */
@keyframes atelier-shine {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.studio-hero h1 {
    background: linear-gradient(120deg, hsl(245,68%,50%) 0%, hsl(330,65%,50%) 40%, hsl(42,85%,48%) 80%) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: atelier-shine 8s ease infinite !important;
    font-size: clamp(2rem,5vw,3rem) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
.studio-subtitle { color: var(--text-secondary) !important; font-weight: 300 !important; }

/* ── Panels & Cards ────────────────────────────────────────────────────────── */
.panel, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-sm) !important;
}
.metric-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}
.metric-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-glow) !important;
    border-color: var(--accent-indigo) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, hsl(245,68%,55%) 0%, hsl(270,60%,48%) 100%) !important;
    border: none !important; color: #fff !important; font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 3px 12px rgba(80,70,228,0.28) !important;
    transition: all var(--transition) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(80,70,228,0.42) !important;
}

/* ── Inputs ────────────────────────────────────────────────────────────────── */
input, textarea, select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}
input:focus, textarea:focus {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 3px rgba(80,70,228,0.14) !important;
    outline: none !important;
}

/* ── Tab nav ───────────────────────────────────────────────────────────────── */
.tab-nav { background: var(--bg-card) !important; border-radius: var(--radius-lg) !important;
           border: 1px solid var(--border) !important; padding: 6px !important; }
.tab-nav button { border-radius: var(--radius-md) !important; color: var(--text-secondary) !important;
                  background: transparent !important; border: none !important;
                  font-weight: 500 !important; transition: all var(--transition) !important; }
.tab-nav button:hover { color: var(--text-primary) !important; background: var(--bg-elevated) !important; }
.tab-nav button.selected { background: var(--accent-indigo) !important; color: white !important;
                            box-shadow: var(--shadow-glow) !important; }

/* ── Gallery ───────────────────────────────────────────────────────────────── */
.gallery-item { border-radius: var(--radius-md) !important; border: 2px solid transparent !important;
                transition: all var(--transition) !important; }
.gallery-item:hover { border-color: var(--accent-indigo) !important;
                       box-shadow: var(--shadow-glow) !important; transform: scale(1.02) !important; }

/* ── Chat ──────────────────────────────────────────────────────────────────── */
.chatbot .message.bot { background: var(--bg-elevated) !important;
                         border-left: 3px solid var(--accent-indigo) !important; }
.chatbot .message.user { background: hsl(245, 50%, 94%) !important; color: var(--text-primary) !important; }

/* ── Code ──────────────────────────────────────────────────────────────────── */
code, pre { background: var(--bg-elevated) !important; border-radius: var(--radius-sm) !important;
            font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: 0.87rem !important; }
pre { padding: 16px !important; border: 1px solid var(--border) !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-indigo); }

/* ── Slider ────────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: var(--accent-indigo) !important; }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
.gradio-container { animation: fadeInUp 0.35s ease-out; }

/* ── Status badges ─────────────────────────────────────────────────────────── */
.badge-active   { background:rgba(22,163,74,0.12)!important; color:hsl(160,55%,35%)!important;
                  border:1px solid rgba(22,163,74,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
.badge-mock     { background:rgba(202,138,4,0.12)!important; color:hsl(42,85%,40%)!important;
                  border:1px solid rgba(202,138,4,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
"""


def get_theme() -> LightTheme:
    """Return a fresh ``LightTheme`` instance."""
    return LightTheme()
