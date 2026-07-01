"""
Week 6 — Reusable Notification Components.

Provides factory functions for inline toast-style notifications, banner
alerts, dismissible messages, and a notification queue — all implemented
via Gradio ``gr.HTML`` and ``gr.State``.

Notification families
---------------------
- ``toast``                — brief auto-hiding in-panel toast strip
- ``banner_notification``  — sticky top-of-section banner (info/success/warning/error)
- ``inline_status``        — inline status line with coloured dot + text
- ``notification_list``    — scrollable log of accumulated notification messages
- ``dismissible_alert``    — alert that can be hidden by the user
- ``field_validation_msg`` — per-field validation feedback message
- ``notification_queue``   — stateful queue of notifications with append/clear helpers
"""
from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import gradio as gr

# ── Design tokens ─────────────────────────────────────────────────────────────
_ACCENT  = "#ff9f43"
_ACCENT2 = "#ff5252"
_SUCCESS = "#2ecc71"
_INFO    = "#3498db"
_WARNING = "#f39c12"
_DANGER  = "#ff5252"
_SURFACE = "rgba(255,255,255,0.03)"
_BORDER  = "rgba(255,255,255,0.07)"
_TEXT_PRIM = "#f0f0f5"
_TEXT_SEC  = "#a0a0b0"

_LEVEL_PALETTE = {
    "info":    (_INFO,    "ℹ️",  "rgba(52,152,219,0.10)"),
    "success": (_SUCCESS, "✅",  "rgba(46,204,113,0.10)"),
    "warning": (_WARNING, "⚠️", "rgba(243,156,18,0.10)"),
    "error":   (_DANGER,  "❌", "rgba(255,82,82,0.10)"),
}

_TOAST_CSS = """
<style>
@keyframes toast-slide-in {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.studio-toast { animation: toast-slide-in 0.3s ease-out forwards; }
@keyframes toast-fade-out {
    from { opacity: 1; }
    to   { opacity: 0; }
}
</style>
"""


# ── Toast ──────────────────────────────────────────────────────────────────────

def toast(
    message: str,
    *,
    level: str = "info",
    title: str = "",
    render: bool = True,
) -> Optional[str]:
    """Return (or render) a compact animated toast notification strip.

    Args:
        message: Notification body text (HTML supported).
        level:   Severity: ``"info"``, ``"success"``, ``"warning"``, or ``"error"``.
        title:   Optional bold title prepended to the message.
        render:  If ``True`` renders via ``gr.HTML``; else returns HTML string.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    level = level if level in _LEVEL_PALETTE else "info"
    color, icon, bg = _LEVEL_PALETTE[level]
    title_span = (
        f'<strong style="margin-right:0.4rem;color:{color};">{title}:</strong>'
        if title else ""
    )
    html = f"""
{_TOAST_CSS}
<div class="studio-toast" style="background:{bg};border:1px solid {color};
     border-left:4px solid {color};border-radius:8px;padding:0.7rem 1rem;
     margin-bottom:0.6rem;display:flex;gap:0.6rem;align-items:flex-start;">
    <span style="font-size:1rem;line-height:1.4;flex-shrink:0;">{icon}</span>
    <span style="font-size:0.88rem;color:{_TEXT_PRIM};line-height:1.55;">
        {title_span}{message}
    </span>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Banner notification ────────────────────────────────────────────────────────

def banner_notification(
    message: str,
    *,
    level: str = "info",
    title: str = "",
    action_text: str = "",
    action_url: str = "#",
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a full-width sticky-style banner notification.

    Banners are more prominent than toasts and suited for page-level messages
    (e.g. "Model loading failed — running in mock mode").

    Args:
        message:     Body text (HTML supported).
        level:       Severity: ``"info"`` / ``"success"`` / ``"warning"`` / ``"error"``.
        title:       Optional bold heading for the banner.
        action_text: Optional call-to-action link text.
        action_url:  URL for the call-to-action link.
        render:      If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    level = level if level in _LEVEL_PALETTE else "info"
    color, icon, bg = _LEVEL_PALETTE[level]
    title_html = (
        f'<div style="font-weight:700;font-size:0.95rem;color:{color};'
        f'margin-bottom:0.25rem;">{icon}&nbsp;{title}</div>'
        if title else f'<span style="font-size:1rem;margin-right:0.5rem;">{icon}</span>'
    )
    action_html = (
        f'<a href="{action_url}" style="color:{color};font-weight:600;'
        f'font-size:0.82rem;text-decoration:underline;white-space:nowrap;">'
        f'{action_text}</a>'
        if action_text else ""
    )
    html = f"""
<div style="background:{bg};border:1px solid {color};border-radius:8px;
            padding:1rem 1.2rem;margin-bottom:1rem;
            display:flex;justify-content:space-between;align-items:center;gap:1rem;">
    <div style="flex:1;">
        {title_html}
        <div style="font-size:0.88rem;color:{_TEXT_PRIM};line-height:1.55;">{message}</div>
    </div>
    {action_html}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Inline status ──────────────────────────────────────────────────────────────

def inline_status(
    message: str,
    *,
    level: str = "info",
    icon: str = "",
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a compact inline status indicator with a coloured dot.

    Args:
        message: Status description text.
        level:   Severity: ``"info"`` / ``"success"`` / ``"warning"`` / ``"error"``.
        icon:    Optional emoji prepended (overrides the default dot if provided).
        render:  If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    level = level if level in _LEVEL_PALETTE else "info"
    color, default_icon, _ = _LEVEL_PALETTE[level]
    indicator = icon if icon else f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};"></span>'
    html = f"""
<div style="display:flex;align-items:center;gap:0.45rem;
            font-size:0.85rem;color:{_TEXT_SEC};padding:0.2rem 0;">
    {indicator}
    <span style="color:{_TEXT_PRIM};">{message}</span>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Notification list ──────────────────────────────────────────────────────────

def notification_list(
    entries: Sequence[Tuple[str, str]],
    *,
    max_visible: int = 8,
    height: int = 220,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a scrollable log of timestamped notification entries.

    Args:
        entries:     Sequence of ``(level, message)`` tuples in chronological order.
                     *level* is one of ``"info"`` / ``"success"`` / ``"warning"`` / ``"error"``.
        max_visible: Maximum number of items shown before scrolling.
        height:      Fixed container height in pixels.
        render:      If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    visible = list(entries)[-max_visible:]
    rows = []
    for i, (lvl, msg) in enumerate(reversed(visible)):
        lvl = lvl if lvl in _LEVEL_PALETTE else "info"
        color, icon, _ = _LEVEL_PALETTE[lvl]
        ts = time.strftime("%H:%M:%S")
        rows.append(f"""
<div style="display:flex;gap:0.5rem;align-items:flex-start;
            padding:0.45rem 0;border-bottom:1px solid {_BORDER};
            {'opacity:0.55;' if i > 0 else ''}">
    <span style="font-size:0.9rem;flex-shrink:0;">{icon}</span>
    <span style="font-size:0.82rem;color:{_TEXT_PRIM};flex:1;line-height:1.4;">{msg}</span>
    <span style="font-size:0.72rem;color:{_TEXT_SEC};flex-shrink:0;">{ts}</span>
</div>
""")

    if not rows:
        rows = [f'<div style="color:{_TEXT_SEC};font-size:0.82rem;padding:0.5rem 0;">No notifications yet.</div>']

    html = f"""
<div style="background:{_SURFACE};border:1px solid {_BORDER};border-radius:8px;
            padding:0.8rem;height:{height}px;overflow-y:auto;">
    {''.join(rows)}
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Dismissible alert ──────────────────────────────────────────────────────────

def dismissible_alert(
    message: str,
    *,
    level: str = "warning",
    title: str = "",
    dismiss_label: str = "Dismiss",
    elem_id_prefix: str = "dismiss-alert",
) -> Tuple[gr.Column, gr.Button]:
    """Render a dismissible alert panel with a close button.

    Args:
        message:          Alert body text.
        level:            Severity level.
        title:            Optional bold title.
        dismiss_label:    Close button label.
        elem_id_prefix:   Prefix for unique element IDs.

    Returns:
        Tuple of:
        - ``alert_col`` — the ``gr.Column`` wrapping the alert (hide to dismiss)
        - ``dismiss_btn`` — the button that hides the alert

    Wire the dismiss button::

        alert_col, dismiss_btn = dismissible_alert("GPU not available — using mock mode.")
        dismiss_btn.click(lambda: gr.update(visible=False), outputs=[alert_col])
    """
    level = level if level in _LEVEL_PALETTE else "warning"
    color, icon, bg = _LEVEL_PALETTE[level]
    title_html = (
        f'<div style="font-weight:700;color:{color};margin-bottom:0.2rem;">{title}</div>'
        if title else ""
    )
    with gr.Column(visible=True, elem_id=f"{elem_id_prefix}-col") as alert_col:
        gr.HTML(f"""
<div style="background:{bg};border:1px solid {color};border-left:4px solid {color};
            border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.6rem;">
    {title_html}
    <div style="display:flex;align-items:flex-start;gap:0.5rem;">
        <span style="font-size:1rem;flex-shrink:0;">{icon}</span>
        <span style="font-size:0.88rem;color:{_TEXT_PRIM};line-height:1.55;">{message}</span>
    </div>
</div>
""")
        dismiss_btn = gr.Button(
            value=f"✖ {dismiss_label}",
            variant="secondary",
            size="sm",
            elem_id=f"{elem_id_prefix}-btn",
        )

    dismiss_btn.click(lambda: gr.update(visible=False), outputs=[alert_col])
    return alert_col, dismiss_btn


# ── Field validation message ───────────────────────────────────────────────────

def field_validation_msg(
    message: str = "",
    *,
    valid: Optional[bool] = None,
    render: bool = True,
) -> Optional[str]:
    """Render (or return) a small per-field validation feedback message.

    Args:
        message: Validation message text (empty string renders nothing visible).
        valid:   ``True`` = success, ``False`` = error, ``None`` = neutral/info.
        render:  If ``True`` renders via ``gr.HTML``; else returns HTML.

    Returns:
        ``None`` when rendered; HTML string otherwise.
    """
    if not message:
        html = '<div style="height:1.2rem;"></div>'  # placeholder to avoid layout jump
    else:
        if valid is True:
            color, icon = _SUCCESS, "✔"
        elif valid is False:
            color, icon = _DANGER, "✘"
        else:
            color, icon = _INFO, "ℹ"
        html = f"""
<div style="display:flex;align-items:center;gap:0.35rem;
            font-size:0.78rem;color:{color};margin-top:0.2rem;line-height:1.4;">
    <span style="font-weight:700;">{icon}</span>
    <span>{message}</span>
</div>
"""
    if render:
        gr.HTML(html)
        return None
    return html


# ── Notification queue ─────────────────────────────────────────────────────────

class NotificationQueue:
    """Stateful in-memory notification queue backed by a ``gr.State``.

    Maintains a list of ``(level, message, timestamp)`` triples and provides
    helper methods to build Gradio event-handler callbacks.

    Usage::

        nq = NotificationQueue(max_size=20)
        notif_state, notif_html = nq.build_component()

        # In an event handler:
        def on_generate(...):
            ...
            update = nq.push("success", "Image generated in 2.3s")
            return image, update  # notif_html is in outputs

    Args:
        max_size: Maximum number of notifications retained in the queue.
    """

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._state: Optional[gr.State] = None
        self._html_widget: Optional[gr.HTML] = None

    def build_component(
        self,
        *,
        height: int = 220,
        label: str = "Notifications",
    ) -> Tuple[gr.State, gr.HTML]:
        """Create and return the backing ``gr.State`` and display ``gr.HTML`` widget.

        Call this once inside a ``gr.Blocks`` context.

        Args:
            height: Height of the scrollable log area in pixels.
            label:  Section label rendered above the log.

        Returns:
            Tuple of ``(state, html_widget)``.
        """
        self._state = gr.State(value=[])

        if label:
            gr.Markdown(f"**{label}**")

        empty_html = notification_list([], height=height, render=False) or ""
        self._html_widget = gr.HTML(value=empty_html)
        return self._state, self._html_widget

    def push(
        self,
        level: str,
        message: str,
        *,
        height: int = 220,
    ) -> Any:  # returns a Gradio update dict suitable for gr.HTML
        """Return an event-handler function that appends one notification.

        The returned function accepts the current ``gr.State`` list and returns
        ``(new_state, html_update)`` — wire it as::

            some_btn.click(nq.push("success", "Done!"),
                           inputs=[notif_state],
                           outputs=[notif_state, notif_html])

        Args:
            level:   Notification severity.
            message: Notification message text.
            height:  Display height in pixels.

        Returns:
            A two-argument callable compatible with Gradio event wiring.
        """
        _level = level
        _msg = message
        _height = height
        _max = self.max_size

        def _handler(current: List[Tuple[str, str]]) -> Tuple[List, str]:
            new_entries = list(current) + [(_level, _msg)]
            if len(new_entries) > _max:
                new_entries = new_entries[-_max:]
            html = notification_list(new_entries, height=_height, render=False) or ""
            return new_entries, html

        return _handler

    def clear(self, *, height: int = 220) -> Any:
        """Return an event-handler function that clears all notifications.

        Wire it as::

            clear_btn.click(nq.clear(),
                            inputs=[],
                            outputs=[notif_state, notif_html])

        Args:
            height: Display height in pixels.

        Returns:
            A zero-argument callable compatible with Gradio event wiring.
        """
        _height = height

        def _handler() -> Tuple[List, str]:
            html = notification_list([], height=_height, render=False) or ""
            return [], html

        return _handler
