"""
Week 6 — Dark Theme (Obsidian Noir).

A premium midnight-dark Gradio theme inspired by luxury editorial fashion:
deep obsidian backgrounds, rose-gold accents, charcoal surfaces, and
razor-sharp typography.  Suitable for professional studio environments and
AI generation workflows.

Design language
---------------
- Background  : #0a0a0f  (pure obsidian)
- Surface      : #13131f  (elevated charcoal)
- Accent       : hsl(330, 70%, 62%)  (rose-gold)
- Secondary    : hsl(200, 60%, 55%)  (sapphire blue)
- Success      : hsl(155, 65%, 50%)  (emerald)
- Typography   : Outfit (headings) + Inter (body) + JetBrains Mono (code)
"""
from __future__ import annotations

import gradio as gr
from gradio.themes import Base, GoogleFont
from gradio.themes.utils import colors, sizes, fonts


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Theme Class
# ══════════════════════════════════════════════════════════════════════════════

class DarkTheme(Base):
    """
    Obsidian Noir — a deep dark theme for the AI Fashion Creative Studio.

    Inspired by the backstage of a luxury fashion week runway:
    pure black surfaces, rose-gold edges, and high-contrast type.
    """

    NAME = "obsidian_noir"
    DESCRIPTION = "Deep midnight dark with rose-gold fashion accents."
    PREVIEW_ACCENT = "#e84393"

    def __init__(self) -> None:
        super().__init__(
            primary_hue=colors.pink,
            secondary_hue=colors.blue,
            neutral_hue=colors.zinc,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_lg,
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
            body_background_fill="hsl(240, 18%, 5%)",
            body_background_fill_dark="hsl(240, 18%, 5%)",
            block_background_fill="hsl(240, 16%, 8%)",
            block_background_fill_dark="hsl(240, 16%, 8%)",
            panel_background_fill="hsl(240, 15%, 10%)",
            panel_background_fill_dark="hsl(240, 15%, 10%)",
            # ── Borders ───────────────────────────────────────────────────
            block_border_color="hsl(240, 14%, 16%)",
            block_border_color_dark="hsl(240, 14%, 16%)",
            block_border_width="1px",
            panel_border_color="hsl(240, 12%, 20%)",
            # ── Typography ───────────────────────────────────────────────
            body_text_color="hsl(240, 20%, 88%)",
            body_text_color_dark="hsl(240, 20%, 88%)",
            body_text_color_subdued="hsl(240, 12%, 55%)",
            body_text_color_subdued_dark="hsl(240, 12%, 55%)",
            block_title_text_color="hsl(240, 25%, 92%)",
            block_label_text_color="hsl(240, 15%, 65%)",
            # ── Buttons ───────────────────────────────────────────────────
            button_primary_background_fill=(
                "linear-gradient(135deg, hsl(330,70%,55%) 0%, hsl(300,60%,45%) 100%)"
            ),
            button_primary_background_fill_hover=(
                "linear-gradient(135deg, hsl(330,75%,62%) 0%, hsl(300,65%,52%) 100%)"
            ),
            button_primary_text_color="white",
            button_primary_border_color="transparent",
            button_primary_border_color_hover="transparent",
            button_secondary_background_fill="hsl(240, 16%, 13%)",
            button_secondary_background_fill_hover="hsl(240, 16%, 18%)",
            button_secondary_text_color="hsl(240, 20%, 78%)",
            button_secondary_border_color="hsl(240, 14%, 22%)",
            button_cancel_background_fill="hsl(0, 65%, 35%)",
            button_cancel_background_fill_hover="hsl(0, 68%, 42%)",
            button_cancel_text_color="white",
            # ── Inputs ───────────────────────────────────────────────────
            input_background_fill="hsl(240, 16%, 10%)",
            input_background_fill_focus="hsl(240, 16%, 13%)",
            input_border_color="hsl(240, 14%, 20%)",
            input_border_color_focus="hsl(330, 70%, 55%)",
            input_border_color_hover="hsl(240, 14%, 26%)",
            input_placeholder_color="hsl(240, 10%, 38%)",
            input_shadow="none",
            input_shadow_focus="0 0 0 3px rgba(232,67,147,0.18)",
            # ── Slider ───────────────────────────────────────────────────
            slider_color="hsl(330, 70%, 58%)",
            slider_color_dark="hsl(330, 70%, 58%)",
            # ── Checkbox / Radio ─────────────────────────────────────────
            checkbox_background_color="hsl(240, 16%, 10%)",
            checkbox_background_color_focus="hsl(240, 16%, 13%)",
            checkbox_background_color_selected="hsl(330, 70%, 55%)",
            checkbox_border_color="hsl(240, 14%, 22%)",
            checkbox_border_color_focus="hsl(330, 70%, 55%)",
            # ── Shadows ──────────────────────────────────────────────────
            block_shadow="0 4px 28px rgba(0,0,0,0.6)",
            block_shadow_dark="0 4px 28px rgba(0,0,0,0.8)",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
/* ── Dark Theme (Obsidian Noir) ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:       hsl(240, 18%, 5%);
    --bg-card:       hsl(240, 16%, 8%);
    --bg-elevated:   hsl(240, 15%, 11%);
    --bg-input:      hsl(240, 16%, 10%);
    --bg-overlay:    rgba(10, 10, 18, 0.85);

    --accent-rose:   hsl(330, 70%, 62%);
    --accent-rose-g: rgba(232, 67, 147, 0.22);
    --accent-sapph:  hsl(200, 60%, 55%);
    --accent-gold:   hsl(42, 88%, 60%);
    --accent-emerald:hsl(155, 65%, 50%);

    --text-primary:  hsl(240, 20%, 90%);
    --text-secondary:hsl(240, 12%, 58%);
    --text-muted:    hsl(240, 8%, 38%);

    --border:        hsl(240, 14%, 16%);
    --border-accent: var(--accent-rose);

    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;

    --shadow-sm:  0 2px 8px rgba(0,0,0,0.4);
    --shadow-md:  0 4px 20px rgba(0,0,0,0.6);
    --shadow-lg:  0 8px 40px rgba(0,0,0,0.8);
    --shadow-glow:0 0 28px var(--accent-rose-g);

    --transition: 200ms cubic-bezier(0.4,0,0.2,1);
}

body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Outfit','Inter',system-ui,sans-serif !important;
    color: var(--text-primary) !important;
}
.gradio-container { max-width:1340px !important; margin:0 auto; padding:0 16px; }

/* ── Animated gradient header ─────────────────────────────────────────────── */
@keyframes noir-gradient {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.studio-hero h1 {
    background: linear-gradient(120deg, #e84393 0%, #bd10e0 30%, #4facfe 70%, #00f2fe 100%) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: noir-gradient 7s ease infinite !important;
    font-size: clamp(2rem,5vw,3rem) !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
}
.studio-subtitle { color: var(--text-secondary) !important; font-weight: 300 !important; }

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
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}
.metric-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: var(--shadow-glow) !important;
    border-color: var(--accent-rose) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, hsl(330,70%,55%) 0%, hsl(300,60%,45%) 100%) !important;
    border: none !important; color: #fff !important; font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 4px 16px rgba(232,67,147,0.35) !important;
    transition: all var(--transition) !important;
}
button.primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(232,67,147,0.5) !important; }

/* ── Inputs ────────────────────────────────────────────────────────────────── */
input, textarea, select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}
input:focus, textarea:focus {
    border-color: var(--accent-rose) !important;
    box-shadow: 0 0 0 3px rgba(232,67,147,0.18) !important;
    outline: none !important;
}

/* ── Tab navigation ────────────────────────────────────────────────────────── */
.tab-nav { background: var(--bg-card) !important; border-radius: var(--radius-lg) !important;
           border: 1px solid var(--border) !important; padding: 6px !important; }
.tab-nav button { border-radius: var(--radius-md) !important; color: var(--text-secondary) !important;
                  background: transparent !important; border: none !important;
                  transition: all var(--transition) !important; font-weight: 500 !important; }
.tab-nav button:hover { color: var(--text-primary) !important; background: var(--bg-elevated) !important; }
.tab-nav button.selected { background: var(--accent-rose) !important; color: white !important;
                            box-shadow: var(--shadow-glow) !important; }

/* ── Gallery ───────────────────────────────────────────────────────────────── */
.gallery-item { border-radius: var(--radius-md) !important; overflow: hidden !important;
                border: 2px solid transparent !important; transition: all var(--transition) !important; }
.gallery-item:hover { border-color: var(--accent-rose) !important; transform: scale(1.02) !important;
                       box-shadow: var(--shadow-glow) !important; }

/* ── Chat ──────────────────────────────────────────────────────────────────── */
.chatbot .message.bot { background: var(--bg-elevated) !important;
                         border-left: 3px solid var(--accent-rose) !important; }
.chatbot .message.user { background: linear-gradient(135deg, hsl(330,40%,18%), hsl(300,35%,14%)) !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-rose); }

/* ── Slider ────────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: var(--accent-rose) !important; }

/* ── Code ──────────────────────────────────────────────────────────────────── */
code, pre { background: var(--bg-elevated) !important; border-radius: var(--radius-sm) !important;
            font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: 0.87rem !important; }
pre { padding: 16px !important; border: 1px solid var(--border) !important; }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
.gradio-container { animation: fadeInUp 0.35s ease-out; }

/* ── Status badges ─────────────────────────────────────────────────────────── */
.badge-active   { background:rgba(52,211,153,0.14)!important; color:hsl(155,65%,52%)!important;
                  border:1px solid rgba(52,211,153,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
.badge-mock     { background:rgba(251,191,36,0.14)!important; color:hsl(42,88%,62%)!important;
                  border:1px solid rgba(251,191,36,0.3)!important; border-radius:20px!important;
                  padding:3px 12px!important; font-size:0.8rem!important; font-weight:600!important; }
"""


def get_theme() -> DarkTheme:
    """Return a fresh ``DarkTheme`` instance."""
    return DarkTheme()
