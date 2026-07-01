"""Week 6 Pages Package."""
from week6.pages.home import build_home_page
from week6.pages.style_studio import build_style_studio_page
from week6.pages.text_to_fashion import build_text_to_fashion_page
from week6.pages.controlnet_page import build_controlnet_page
from week6.pages.sketch_to_design import build_sketch_to_design_page
from week6.pages.style_switcher import build_style_switcher_page
from week6.pages.style_mixer import build_style_mixer_page
from week6.pages.fashion_assistant import build_fashion_assistant_page
from week6.pages.brand_studio import build_brand_studio_page
from week6.pages.fashion_qa import build_fashion_qa_page
from week6.pages.trend_explorer import build_trend_explorer_page
from week6.pages.recommend_hub import build_recommend_hub_page
from week6.pages.recommendations import build_recommendations_page
from week6.pages.gallery import build_gallery_page
from week6.pages.eval_dashboard import build_eval_dashboard_page
from week6.pages.settings import build_settings_page

__all__ = [
    "build_home_page",
    "build_style_studio_page",
    "build_text_to_fashion_page",
    "build_controlnet_page",
    "build_sketch_to_design_page",
    "build_style_switcher_page",
    "build_style_mixer_page",
    "build_fashion_assistant_page",
    "build_brand_studio_page",
    "build_fashion_qa_page",
    "build_trend_explorer_page",
    "build_recommend_hub_page",
    "build_recommendations_page",
    "build_gallery_page",
    "build_eval_dashboard_page",
    "build_settings_page",
]
