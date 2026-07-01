"""
Week 6 — Themes Package.

Exports all three AI Fashion Studio themes and provides a runtime
theme-switching system.

Available themes
----------------
- ``FashionTheme``  — Aurora Luxe (dark indigo + amber coral)  [DEFAULT]
- ``DarkTheme``     — Obsidian Noir (midnight black + rose-gold)
- ``LightTheme``    — Ivory Atelier (warm ivory + deep indigo)

Runtime switching
-----------------
Use ``ThemeManager`` to resolve a theme by name and retrieve the matching
Gradio ``Base`` instance + custom CSS string at runtime::

    from week6.themes import ThemeManager

    # Resolve by string key (e.g. from settings.json)
    theme_obj, css = ThemeManager.resolve("dark")

    # Build the app with the resolved theme
    with gr.Blocks(theme=theme_obj, css=css) as app:
        ...

    # Or use the Gradio updater helper in a settings-save event:
    def on_theme_change(theme_name):
        return ThemeManager.css_for(theme_name)

    theme_radio.change(on_theme_change, inputs=[theme_radio], outputs=[css_state])

Available keys: ``"fashion"`` / ``"aurora"``, ``"dark"`` / ``"noir"``,
``"light"`` / ``"ivory"``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from gradio.themes import Base

from week6.themes.fashion_theme import FashionTheme, STUDIO_CSS
from week6.themes.dark_theme import DarkTheme, DARK_CSS
from week6.themes.light_theme import LightTheme, LIGHT_CSS


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Theme classes
    "FashionTheme",
    "DarkTheme",
    "LightTheme",
    # CSS strings
    "STUDIO_CSS",
    "DARK_CSS",
    "LIGHT_CSS",
    # Helpers
    "get_theme",
    "get_all_themes",
    "ThemeManager",
    "THEME_NAMES",
    "DEFAULT_THEME",
]

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_THEME: str = "fashion"

#: Human-readable display names → canonical key
THEME_NAMES: Dict[str, str] = {
    "🎨 Aurora Luxe (Default)": "fashion",
    "🌑 Obsidian Noir (Dark)":  "dark",
    "☀️ Ivory Atelier (Light)": "light",
}

#: All canonical aliases that resolve to each theme
_ALIAS_MAP: Dict[str, str] = {
    # fashion / aurora luxe
    "fashion": "fashion",
    "aurora":  "fashion",
    "luxe":    "fashion",
    "default": "fashion",
    # dark / obsidian noir
    "dark":    "dark",
    "noir":    "dark",
    "obsidian":"dark",
    "black":   "dark",
    # light / ivory atelier
    "light":   "light",
    "ivory":   "light",
    "atelier": "light",
    "white":   "light",
}

_THEME_REGISTRY: Dict[str, Tuple[Type[Base], str]] = {
    "fashion": (FashionTheme, STUDIO_CSS),
    "dark":    (DarkTheme,    DARK_CSS),
    "light":   (LightTheme,   LIGHT_CSS),
}


# ── Module-level helpers ───────────────────────────────────────────────────────

def get_theme(name: str = DEFAULT_THEME) -> Base:
    """Return an instantiated Gradio ``Base`` theme object by name or alias.

    Args:
        name: Theme name / alias (e.g. ``"dark"``, ``"noir"``, ``"light"``).
              Defaults to ``"fashion"``.

    Returns:
        An instantiated Gradio theme object.

    Raises:
        ValueError: If *name* does not match any known theme or alias.
    """
    canonical = _ALIAS_MAP.get(name.lower().strip())
    if canonical is None:
        valid = sorted(set(_ALIAS_MAP.keys()))
        raise ValueError(
            f"Unknown theme '{name}'.  Valid names / aliases: {valid}"
        )
    cls, _ = _THEME_REGISTRY[canonical]
    return cls()


def get_all_themes() -> Dict[str, Tuple[Base, str]]:
    """Return a dict of ``{canonical_name: (theme_instance, css_string)}``.

    Useful for building theme preview galleries or dropdown selectors.

    Returns:
        Dictionary mapping canonical name to ``(theme, css)`` pairs.
    """
    return {
        name: (cls(), css)
        for name, (cls, css) in _THEME_REGISTRY.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# ThemeManager — runtime theme switching engine
# ══════════════════════════════════════════════════════════════════════════════

class ThemeManager:
    """
    Centralised runtime theme manager for the AI Fashion Creative Studio.

    Supports theme resolution by name/alias, CSS-only retrieval, theme
    metadata listing, and Gradio event-handler helpers for runtime switching.

    Class attributes
    ----------------
    ``THEMES``       — ``{canonical: (ThemeClass, css)}`` registry
    ``ALIASES``      — ``{alias: canonical}`` mapping
    ``DISPLAY_NAMES``— ordered ``{display_label: canonical}`` for dropdowns

    Examples
    --------
    Resolve a theme from a settings file::

        theme_obj, css = ThemeManager.resolve("dark")
        with gr.Blocks(theme=theme_obj, css=css) as app:
            ...

    Use in a settings-save callback::

        def on_save(theme_name, ...):
            css = ThemeManager.css_for(theme_name)
            ...

    List themes for a dropdown::

        gr.Dropdown(choices=ThemeManager.display_choices(), ...)
    """

    THEMES: Dict[str, Tuple[Type[Base], str]] = _THEME_REGISTRY
    ALIASES: Dict[str, str] = _ALIAS_MAP
    DISPLAY_NAMES: Dict[str, str] = THEME_NAMES

    # ── Resolution ────────────────────────────────────────────────────────────

    @classmethod
    def resolve(cls, name: str = DEFAULT_THEME) -> Tuple[Base, str]:
        """Resolve a theme name/alias to ``(theme_instance, css_string)``.

        Args:
            name: Theme name or alias (case-insensitive).

        Returns:
            Tuple of ``(Gradio Base theme instance, CSS string)``.

        Raises:
            ValueError: On unknown theme name.
        """
        canonical = cls._canonical(name)
        theme_cls, css = cls.THEMES[canonical]
        return theme_cls(), css

    @classmethod
    def theme_for(cls, name: str = DEFAULT_THEME) -> Base:
        """Return only the Gradio theme object for *name*.

        Args:
            name: Theme name or alias.

        Returns:
            Instantiated Gradio ``Base`` theme.
        """
        theme, _ = cls.resolve(name)
        return theme

    @classmethod
    def css_for(cls, name: str = DEFAULT_THEME) -> str:
        """Return only the CSS string for *name*.

        Useful when you want to update only the injected CSS without
        rebuilding the entire ``gr.Blocks`` app.

        Args:
            name: Theme name or alias.

        Returns:
            CSS string.
        """
        _, css = cls.resolve(name)
        return css

    # ── Metadata ──────────────────────────────────────────────────────────────

    @classmethod
    def list_themes(cls) -> Dict[str, Dict[str, str]]:
        """Return metadata about every available theme.

        Returns:
            Dict of ``{canonical_name: {name, description, preview_accent}}``.
        """
        result: Dict[str, Dict[str, str]] = {}
        for canonical, (theme_cls, _) in cls.THEMES.items():
            result[canonical] = {
                "name":           getattr(theme_cls, "NAME", canonical),
                "description":    getattr(theme_cls, "DESCRIPTION", ""),
                "preview_accent": getattr(theme_cls, "PREVIEW_ACCENT", "#888"),
                "canonical_key":  canonical,
            }
        return result

    @classmethod
    def display_choices(cls) -> list[str]:
        """Return a list of human-readable display labels for use in ``gr.Dropdown``.

        Returns:
            Ordered list of display name strings.
        """
        return list(cls.DISPLAY_NAMES.keys())

    @classmethod
    def canonical_from_display(cls, display_label: str) -> str:
        """Convert a display label back to its canonical theme key.

        Args:
            display_label: One of the strings from ``display_choices()``.

        Returns:
            Canonical theme key (e.g. ``"dark"``).

        Raises:
            ValueError: If the label is not found.
        """
        if display_label not in cls.DISPLAY_NAMES:
            raise ValueError(
                f"Unknown display label '{display_label}'.  "
                f"Valid: {list(cls.DISPLAY_NAMES.keys())}"
            )
        return cls.DISPLAY_NAMES[display_label]

    # ── Gradio event-handler helpers ──────────────────────────────────────────

    @classmethod
    def make_css_switcher(cls) -> callable:
        """Return a Gradio event-handler function that maps a theme name to CSS.

        The returned function accepts a single *theme_name* string argument
        (matching a canonical key or alias) and returns the corresponding CSS
        string.  Wire it to a ``gr.Radio`` or ``gr.Dropdown`` ``change`` event
        with a ``gr.HTML`` output that injects a ``<style>`` block::

            theme_picker = gr.Dropdown(
                choices=ThemeManager.display_choices(), label="Theme"
            )
            style_inject = gr.HTML(visible=False, elem_id="theme-css-inject")

            def _apply_theme(display_label):
                key = ThemeManager.canonical_from_display(display_label)
                css = ThemeManager.css_for(key)
                return f"<style>{css}</style>"

            theme_picker.change(_apply_theme,
                                inputs=[theme_picker],
                                outputs=[style_inject])

        .. note::
            Gradio currently does not support live theme-object swapping after
            ``gr.Blocks`` is built.  CSS injection is the only supported
            runtime switch for styling changes.  Full Gradio theme tokens
            (e.g. button colours) require an app restart.

        Returns:
            A ``(display_label: str) -> str`` callable.
        """
        def _switcher(display_label: str) -> str:
            try:
                key = cls.canonical_from_display(display_label)
            except ValueError:
                # Fallback: try direct key/alias
                key = cls._canonical(display_label)
            css = cls.css_for(key)
            return f"<style>\n{css}\n</style>"

        return _switcher

    @classmethod
    def build_theme_switcher_ui(cls) -> tuple:
        """Build a complete theme-switcher UI inside the current gr.Blocks context.

        Creates a ``gr.Dropdown`` for theme selection and a hidden ``gr.HTML``
        widget for injecting the CSS.  Wires the change event automatically.

        Returns:
            Tuple of ``(dropdown_widget, css_inject_html_widget)``.

        Example::

            with gr.Blocks(theme=FashionTheme(), css=STUDIO_CSS) as app:
                with gr.Accordion("🎨 Theme", open=False):
                    picker, injector = ThemeManager.build_theme_switcher_ui()
        """
        import gradio as gr

        choices = cls.display_choices()
        default_display = next(
            k for k, v in cls.DISPLAY_NAMES.items() if v == DEFAULT_THEME
        )

        picker = gr.Dropdown(
            choices=choices,
            value=default_display,
            label="🎨 UI Theme",
            info="Select a theme.  CSS updates instantly; full token changes require restart.",
            elem_id="theme-switcher-dropdown",
        )
        css_inject = gr.HTML(value="", visible=False, elem_id="theme-css-inject")
        status_md  = gr.Markdown(
            value=f"✅ **{default_display}** active.",
            elem_id="theme-switcher-status",
        )

        _switcher = cls.make_css_switcher()

        def _on_change(display_label: str):
            style_html = _switcher(display_label)
            return style_html, f"✅ **{display_label}** applied."

        picker.change(
            _on_change,
            inputs=[picker],
            outputs=[css_inject, status_md],
        )

        return picker, css_inject

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _canonical(cls, name: str) -> str:
        """Resolve an alias or canonical name; raise ``ValueError`` on miss."""
        key = name.lower().strip()
        canonical = cls.ALIASES.get(key)
        if canonical is None:
            valid = sorted(set(cls.ALIASES.keys()))
            raise ValueError(
                f"Unknown theme '{name}'.  Valid keys/aliases: {valid}"
            )
        return canonical
