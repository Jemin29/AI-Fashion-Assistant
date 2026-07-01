"""Week 6 Reusable Components Package.

Exports all UI component factory functions grouped by category:
- Existing: header, status_bar, metrics_panel, image_gallery, chat_interface
- Buttons:  primary, secondary, danger, icon, compound rows, toggle
- Cards:    stat, info, brand, trend, recommendation, feature, alert, tag-list
- Modals:   confirm, info, form, image-preview, progress overlay
- Headers:  app masthead, page header, section header, breadcrumb, hero banner
- Footers:  app footer, status bar, mini footer, session info, attribution
- Loading:  spinner, progress bar, multi-step, skeleton, dots, generation tracker
- Notifications: toast, banner, inline status, dismissible alert, queue
"""

# ── Original components ───────────────────────────────────────────────────────
from week6.components.header import build_header
from week6.components.status_bar import build_status_bar
from week6.components.metrics_panel import build_metrics_panel
from week6.components.image_gallery import build_image_gallery
from week6.components.chat_interface import build_chat_interface

# ── Buttons ───────────────────────────────────────────────────────────────────
from week6.components.buttons import (
    primary_button,
    secondary_button,
    danger_button,
    icon_button,
    action_button_row,
    generate_clear_row,
    copy_download_row,
    toggle_button,
)

# ── Cards ─────────────────────────────────────────────────────────────────────
from week6.components.cards import (
    stat_card,
    stat_card_row,
    info_card,
    brand_card,
    trend_card,
    recommendation_card,
    feature_card,
    alert_card,
    tag_list_card,
    section_divider,
)

# ── Modals ────────────────────────────────────────────────────────────────────
from week6.components.modals import (
    confirm_modal,
    info_modal,
    form_modal,
    image_preview_modal,
    progress_modal,
)

# ── Headers ───────────────────────────────────────────────────────────────────
from week6.components.headers import (
    app_header,
    page_header,
    section_header,
    breadcrumb_header,
    hero_banner,
    compact_header,
    tab_strip_header,
)

# ── Footers ───────────────────────────────────────────────────────────────────
from week6.components.footers import (
    app_footer,
    status_footer,
    mini_footer,
    session_info_footer,
    attribution_footer,
)

# ── Loading ───────────────────────────────────────────────────────────────────
from week6.components.loading import (
    spinner,
    progress_bar,
    multi_step_progress,
    skeleton_card,
    skeleton_gallery,
    loading_overlay,
    dots_loader,
    generation_progress,
)

# ── Notifications ─────────────────────────────────────────────────────────────
from week6.components.notifications import (
    toast,
    banner_notification,
    inline_status,
    notification_list,
    dismissible_alert,
    field_validation_msg,
    NotificationQueue,
)

__all__ = [
    # original
    "build_header",
    "build_status_bar",
    "build_metrics_panel",
    "build_image_gallery",
    "build_chat_interface",
    # buttons
    "primary_button",
    "secondary_button",
    "danger_button",
    "icon_button",
    "action_button_row",
    "generate_clear_row",
    "copy_download_row",
    "toggle_button",
    # cards
    "stat_card",
    "stat_card_row",
    "info_card",
    "brand_card",
    "trend_card",
    "recommendation_card",
    "feature_card",
    "alert_card",
    "tag_list_card",
    "section_divider",
    # modals
    "confirm_modal",
    "info_modal",
    "form_modal",
    "image_preview_modal",
    "progress_modal",
    # headers
    "app_header",
    "page_header",
    "section_header",
    "breadcrumb_header",
    "hero_banner",
    "compact_header",
    "tab_strip_header",
    # footers
    "app_footer",
    "status_footer",
    "mini_footer",
    "session_info_footer",
    "attribution_footer",
    # loading
    "spinner",
    "progress_bar",
    "multi_step_progress",
    "skeleton_card",
    "skeleton_gallery",
    "loading_overlay",
    "dots_loader",
    "generation_progress",
    # notifications
    "toast",
    "banner_notification",
    "inline_status",
    "notification_list",
    "dismissible_alert",
    "field_validation_msg",
    "NotificationQueue",
]
