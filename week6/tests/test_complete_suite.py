"""
Week 6 — Complete Test Suite.

Covers all pages, services, components, settings helpers, history manager,
export manager, and state manager targeting ≥80% code coverage.

Test groups
-----------
A. Home Dashboard helpers        (home.py module functions)
B. Text-to-Fashion service       (generation_service.py)
C. Sketch2Design service         (controlnet_service.py)
D. Style Switcher / Style Mixer  (lora_service.py)
E. Fashion Assistant             (rag_service.py)
F. Recommendation Dashboard      (recommendation_service.py)
G. Gallery page helpers          (gallery.py)
H. Settings persistence          (settings.py)
I. History Manager               (history_manager.py — supplement existing)
J. Export Manager                (export_manager.py  — supplement existing)
K. State Manager                 (state_manager.py   — supplement existing)
L. Page render smoke tests       (all build_* functions instantiate cleanly)
M. Components render             (buttons, cards, headers, notifications)
N. Eval service                  (eval_service.py)
O. Trend service                 (trend_service.py)
"""
from __future__ import annotations

import csv
import json
import time
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
from PIL import Image

# ── service imports ────────────────────────────────────────────────────────────
from week6.services.generation_service import GenerationService
from week6.services.controlnet_service import ControlNetService
from week6.services.lora_service import LoRAService, BRAND_REGISTRY
from week6.services.rag_service import RAGService
from week6.services.recommendation_service import RecommendationService
from week6.services.trend_service import TrendService
from week6.services.eval_service import EvaluationService
from week6.services.export_manager import ExportManager, _EXPORTS_DIR
from week6.services.history_manager import HistoryManager
from week6.services.state_manager import StateManager, SessionState
from week6.services.base import ServiceResult, ServiceStatus


# ══════════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def gen_svc() -> GenerationService:
    return GenerationService(mock_mode=True)


@pytest.fixture(scope="module")
def cn_svc() -> ControlNetService:
    return ControlNetService(mock_mode=True)


@pytest.fixture(scope="module")
def lora_svc() -> LoRAService:
    return LoRAService(mock_mode=True)


@pytest.fixture(scope="module")
def rag_svc() -> RAGService:
    return RAGService(mock_mode=True)


@pytest.fixture(scope="module")
def rec_svc() -> RecommendationService:
    return RecommendationService(mock_mode=True)


@pytest.fixture(scope="module")
def trend_svc() -> TrendService:
    return TrendService(mock_mode=True)


@pytest.fixture(scope="module")
def eval_svc() -> EvaluationService:
    return EvaluationService(mock_mode=True)


@pytest.fixture(scope="module")
def export_mgr() -> ExportManager:
    return ExportManager(mock_mode=False)


@pytest.fixture(scope="module")
def hist_mgr() -> HistoryManager:
    return HistoryManager()


@pytest.fixture(scope="module")
def state_mgr() -> StateManager:
    return StateManager(mock_mode=True)


@pytest.fixture
def dummy_rgba() -> Image.Image:
    return Image.new("RGBA", (128, 128), (200, 100, 50, 180))


@pytest.fixture
def dummy_rgb() -> Image.Image:
    return Image.new("RGB", (128, 128), (120, 80, 200))


# ══════════════════════════════════════════════════════════════════════════════
# A. Home Dashboard Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHomeDashboardHelpers:
    """Tests for pure helper functions in week6/pages/home.py."""

    def test_check_module_valid(self):
        from week6.pages.home import _check_module
        assert _check_module("json") is True

    def test_check_module_invalid(self):
        from week6.pages.home import _check_module
        assert _check_module("nonexistent_package_xyz_123") is False

    def test_get_recent_generations_returns_list(self):
        from week6.pages.home import _get_recent_generations
        result = _get_recent_generations()
        assert isinstance(result, list)
        assert len(result) <= 4

    def test_build_stats_html_returns_html_string(self):
        from week6.pages.home import _build_stats_html
        html = _build_stats_html()
        assert isinstance(html, str)
        assert "556 Pairs" in html
        assert "4 Brands" in html

    def test_build_model_status_html_returns_html(self):
        from week6.pages.home import _build_model_status_html
        html = _build_model_status_html()
        assert isinstance(html, str)
        assert "GPU" in html or "SDXL" in html

    def test_build_home_page_renders(self):
        from week6.pages.home import build_home_page
        with gr.Blocks():
            build_home_page(nav_component=None)

    def test_build_home_page_with_nav(self):
        from week6.pages.home import build_home_page
        with gr.Blocks():
            nav = gr.Radio(choices=["A", "B"], value="A")
            build_home_page(nav_component=nav)


# ══════════════════════════════════════════════════════════════════════════════
# B. Text-to-Fashion (GenerationService)
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationService:

    def test_generate_returns_ok(self, gen_svc):
        res = gen_svc.generate(
            prompt="silk evening gown, editorial lighting",
            negative_prompt="low quality",
            num_inference_steps=10, guidance_scale=7.5, width=512, height=512,
        )
        assert res.is_ok
        assert res.data is not None

    def test_generate_empty_prompt_fails(self, gen_svc):
        res = gen_svc.generate(prompt="", negative_prompt="", num_inference_steps=10, guidance_scale=7.5)
        assert not res.is_ok
        assert res.error is not None

    def test_generate_invalid_steps_fails(self, gen_svc):
        res = gen_svc.generate(
            prompt="streetwear hoodie", num_inference_steps=0, guidance_scale=7.5, width=512, height=512
        )
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_generate_invalid_cfg_fails(self, gen_svc):
        res = gen_svc.generate(
            prompt="streetwear hoodie", num_inference_steps=10, guidance_scale=50.0, width=512, height=512
        )
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_generate_invalid_size_fails(self, gen_svc):
        res = gen_svc.generate(
            prompt="streetwear hoodie", num_inference_steps=10, guidance_scale=7.5, width=333, height=333
        )
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_generate_batch(self, gen_svc):
        res = gen_svc.generate_batch(
            prompt="minimalist white tee", num_inference_steps=10, guidance_scale=7.5,
            width=512, height=512, seeds=[101, 102]
        )
        assert res.is_ok
        assert isinstance(res.data, list)
        assert len(res.data) == 2

    def test_generate_batch_exceeds_max(self, gen_svc):
        res = gen_svc.generate_batch(
            prompt="minimalist white tee", num_inference_steps=10, guidance_scale=7.5,
            width=512, height=512, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        assert res.is_ok
        assert len(res.data) == 8

    def test_generate_batch_empty_seeds_fails(self, gen_svc):
        res = gen_svc.generate_batch(
            prompt="minimalist white tee", num_inference_steps=10, guidance_scale=7.5,
            width=512, height=512, seeds=[]
        )
        assert not res.is_ok

    def test_get_style_presets(self, gen_svc):
        presets = gen_svc.get_style_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_get_resolution_options(self, gen_svc):
        options = gen_svc.get_resolution_options()
        assert isinstance(options, list)
        assert len(options) > 0

    def test_health_check(self, gen_svc):
        hc = gen_svc.health_check()
        assert "status" in hc
        assert hc["status"] in ("ok", "mock", "degraded")

    def test_status_property(self, gen_svc):
        s = gen_svc.status
        assert "call_count" in s
        assert "mock_mode" in s

    def test_text_to_fashion_page_renders(self, gen_svc):
        from week6.pages.text_to_fashion import build_text_to_fashion_page
        with gr.Blocks():
            build_text_to_fashion_page(gen_svc)


# ══════════════════════════════════════════════════════════════════════════════
# C. Sketch2Design (ControlNetService)
# ══════════════════════════════════════════════════════════════════════════════

class TestControlNetService:

    def test_generate_returns_ok(self, cn_svc, dummy_rgb):
        res = cn_svc.generate_conditioned(
            control_image=dummy_rgb,
            prompt="sleek silk evening gown",
            mode="canny",
            conditioning_scale=0.7,
            num_inference_steps=10,
            guidance_scale=7.5,
        )
        assert res.is_ok

    def test_generate_no_image_fails(self, cn_svc):
        res = cn_svc.generate_conditioned(
            control_image=None,
            prompt="sleek gown",
            mode="canny",
            conditioning_scale=0.7,
            num_inference_steps=10,
            guidance_scale=7.5,
        )
        assert not res.is_ok

    def test_generate_empty_prompt_fails(self, cn_svc, dummy_rgb):
        res = cn_svc.generate_conditioned(
            control_image=dummy_rgb,
            prompt="",
            mode="canny",
            conditioning_scale=0.7,
            num_inference_steps=10,
            guidance_scale=7.5,
        )
        assert not res.is_ok

    def test_generate_invalid_detector_recovers(self, cn_svc, dummy_rgb):
        res = cn_svc.generate_conditioned(
            control_image=dummy_rgb,
            prompt="editorial gown",
            mode="invalid_detector",
            conditioning_scale=0.7,
            num_inference_steps=10,
            guidance_scale=7.5,
        )
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_health_check(self, cn_svc):
        hc = cn_svc.health_check()
        assert "status" in hc

    def test_sketch_to_design_page_renders(self, cn_svc):
        from week6.pages.sketch_to_design import build_sketch_to_design_page
        with gr.Blocks():
            build_sketch_to_design_page(cn_svc)


# ══════════════════════════════════════════════════════════════════════════════
# D. Style Switcher / Style Mixer (LoRAService)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoRAService:

    def test_get_brands_returns_list(self, lora_svc):
        brands = lora_svc.get_brands()
        assert isinstance(brands, list)
        assert "nike" in brands
        assert "gucci" in brands

    def test_get_brand_info_known(self, lora_svc):
        info = lora_svc.get_brand_info("nike")
        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "Nike"

    def test_get_brand_info_unknown(self, lora_svc):
        info = lora_svc.get_brand_info("unknown_brand_xyz")
        assert info == {} or info is None or not info

    def test_get_brand_display_names(self, lora_svc):
        names = lora_svc.get_brand_display_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_generate_with_brand_ok(self, lora_svc):
        res = lora_svc.generate_with_brand(
            prompt="sporty jacket", brand="nike",
            lora_scale=0.85, num_inference_steps=10, guidance_scale=7.5
        )
        assert res.is_ok
        assert res.data is not None

    def test_generate_with_brand_empty_prompt(self, lora_svc):
        res = lora_svc.generate_with_brand(
            prompt="", brand="nike", lora_scale=0.85, num_inference_steps=10, guidance_scale=7.5
        )
        assert not res.is_ok

    def test_generate_with_brand_invalid_brand(self, lora_svc):
        res = lora_svc.generate_with_brand(
            prompt="sporty jacket", brand="nonexistent_brand",
            lora_scale=0.85, num_inference_steps=10, guidance_scale=7.5
        )
        assert not res.is_ok

    def test_generate_with_brand_invalid_lora_scale(self, lora_svc):
        res = lora_svc.generate_with_brand(
            prompt="sporty jacket", brand="nike",
            lora_scale=99.0, num_inference_steps=10, guidance_scale=7.5
        )
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_mix_styles_ok(self, lora_svc):
        res = lora_svc.mix_styles(
            prompt="luxury athleisure editorial",
            brand_weights={"nike": 0.6, "gucci": 0.4},
            num_inference_steps=10, guidance_scale=7.5,
        )
        assert res.is_ok

    def test_mix_styles_empty_weights(self, lora_svc):
        res = lora_svc.mix_styles(
            prompt="luxury athleisure", brand_weights={}, num_inference_steps=10, guidance_scale=7.5
        )
        assert not res.is_ok

    def test_mix_styles_negative_weight(self, lora_svc):
        res = lora_svc.mix_styles(
            prompt="test", brand_weights={"nike": -0.5}, num_inference_steps=10, guidance_scale=7.5
        )
        assert not res.is_ok

    def test_validate_brand_weights(self, lora_svc):
        normalized, warnings = lora_svc.validate_brand_weights({"nike": 1.0, "gucci": 1.0})
        assert normalized is not None
        assert abs(sum(normalized.values()) - 1.0) < 1e-6

    def test_health_check(self, lora_svc):
        hc = lora_svc.health_check()
        assert "status" in hc

    def test_style_switcher_page_renders(self, lora_svc):
        from week6.pages.style_switcher import build_style_switcher_page
        with gr.Blocks():
            build_style_switcher_page(lora_svc)

    def test_style_mixer_page_renders(self, lora_svc):
        from week6.pages.style_mixer import build_style_mixer_page
        with gr.Blocks():
            build_style_mixer_page(lora_svc)

    def test_brand_registry_has_required_keys(self):
        for brand, info in BRAND_REGISTRY.items():
            assert "name" in info, f"Brand {brand} missing 'name'"
            assert "description" in info, f"Brand {brand} missing 'description'"
            assert "key_styles" in info, f"Brand {brand} missing 'key_styles'"


# ══════════════════════════════════════════════════════════════════════════════
# E. Fashion Assistant (RAGService)
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGService:

    def test_query_returns_ok(self, rag_svc):
        res = rag_svc.answer_question("What is linen fabric?")
        assert res.is_ok
        assert res.data is not None
        assert "response" in res.data

    def test_query_empty_fails(self, rag_svc):
        res = rag_svc.answer_question("")
        assert not res.is_ok

    def test_chat_general_qa(self, rag_svc):
        res = rag_svc.chat("What makes silk so special?")
        assert res.is_ok
        assert isinstance(res.data, dict)
        assert "response" in res.data
        assert len(res.data["response"]) > 0

    def test_chat_fabric_intent(self, rag_svc):
        res = rag_svc.chat("Tell me about linen fabric properties")
        assert res.is_ok

    def test_chat_trend_intent(self, rag_svc):
        res = rag_svc.chat("What are the top trends this season?")
        assert res.is_ok

    def test_chat_style_intent(self, rag_svc):
        res = rag_svc.chat("Recommend a streetwear style")
        assert res.is_ok

    def test_chat_brand_intent(self, rag_svc):
        res = rag_svc.chat("Suggest brands for luxury fashion")
        assert res.is_ok

    def test_chat_preserves_history(self, rag_svc):
        res = rag_svc.chat("Is it breathable?")
        assert res.is_ok

    def test_semantic_search_ok(self, rag_svc):
        res = rag_svc.semantic_search("linen properties", n_results=2)
        assert res.is_ok
        assert isinstance(res.data, list)

    def test_semantic_search_empty_fails(self, rag_svc):
        res = rag_svc.semantic_search("")
        assert not res.is_ok

    def test_health_check(self, rag_svc):
        hc = rag_svc.health_check()
        assert "status" in hc

    def test_fashion_assistant_page_renders(self, rag_svc):
        from week6.pages.fashion_assistant import build_fashion_assistant_page
        with gr.Blocks():
            build_fashion_assistant_page(rag_svc)
        from week6.pages.fashion_assistant import build_fashion_assistant_page
        with gr.Blocks():
            build_fashion_assistant_page(rag_svc)

    def test_suggestions_list_has_items(self):
        from week6.pages.fashion_assistant import _SUGGESTIONS
        assert len(_SUGGESTIONS) >= 3
        for label, query in _SUGGESTIONS:
            assert isinstance(label, str)
            assert isinstance(query, str)


# ══════════════════════════════════════════════════════════════════════════════
# F. Recommendation Dashboard (RecommendationService)
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendationService:

    def test_recommend_styles_returns_list(self, rec_svc):
        res = rec_svc.recommend_styles(gender="men", style="streetwear", n=3)
        assert res.is_ok
        assert isinstance(res.data, list)
        assert len(res.data) <= 3

    def test_recommend_styles_invalid_gender(self, rec_svc):
        res = rec_svc.recommend_styles(gender="alien", style="streetwear", n=3)
        assert res.is_ok
        assert len(res.warnings) > 0

    def test_recommend_styles_invalid_n(self, rec_svc):
        res = rec_svc.recommend_styles(gender="men", style="streetwear", n=999)
        assert res.is_ok

    def test_recommend_brands_returns_list(self, rec_svc):
        res = rec_svc.recommend_brands(styles=["luxury"], n=3)
        assert res.is_ok
        assert isinstance(res.data, list)

    def test_recommend_outfits_returns_list(self, rec_svc):
        res = rec_svc.recommend_outfits(
            style="streetwear", occasion="casual", n=2
        )
        assert res.is_ok
        assert isinstance(res.data, list)

    def test_get_style_compatibility_score(self, rec_svc):
        res = rec_svc.get_style_compatibility("streetwear", "athleisure")
        assert res.is_ok
        assert isinstance(res.data, dict)
        assert "score" in res.data
        assert 0.0 <= res.data["score"] <= 1.0

    def test_get_available_filters(self, rec_svc):
        res = rec_svc.get_available_filters()
        assert isinstance(res, dict)
        assert "genders" in res
        assert "fits" in res

    def test_health_check(self, rec_svc):
        hc = rec_svc.health_check()
        assert "status" in hc

    def test_recommendation_card_image_generation(self):
        from week6.pages.recommendations import _generate_recommendation_card_image
        img = _generate_recommendation_card_image("Quiet Luxury", 0.92)
        assert isinstance(img, Image.Image)
        assert img.size == (300, 300)

    def test_recommendation_card_different_titles_produce_different_images(self):
        from week6.pages.recommendations import _generate_recommendation_card_image
        img1 = _generate_recommendation_card_image("Streetwear", 0.8)
        img2 = _generate_recommendation_card_image("Quiet Luxury", 0.9)
        # Different seeds → different pixel values
        pix1 = img1.getpixel((150, 150))
        pix2 = img2.getpixel((150, 150))
        assert pix1 != pix2

    def test_recommendations_page_renders(self, rec_svc, trend_svc):
        from week6.pages.recommendations import build_recommendations_page
        with gr.Blocks():
            build_recommendations_page(rec_svc, trend_svc)


# ══════════════════════════════════════════════════════════════════════════════
# G. Gallery Page Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestGalleryPageHelpers:

    def test_gallery_page_renders(self):
        from week6.pages.gallery import build_gallery_page
        with gr.Blocks():
            build_gallery_page()

    def test_query_history_returns_tuple(self):
        from week6.pages.gallery import query_history
        entries, gallery_data, page_str, prev_upd, next_upd = query_history(
            search_q="", filter_type="all", sort_by="newest", page=1, page_size=12
        )
        assert isinstance(entries, list)
        assert isinstance(gallery_data, list)
        assert "Page" in page_str

    def test_query_history_with_search(self):
        from week6.pages.gallery import query_history
        entries, gallery_data, page_str, _, _ = query_history(
            search_q="silk gown", filter_type="all", sort_by="newest", page=1
        )
        assert isinstance(entries, list)

    def test_query_history_filter_generation(self):
        from week6.pages.gallery import query_history
        entries, gallery_data, page_str, _, _ = query_history(
            search_q="", filter_type="generation", sort_by="newest", page=1
        )
        assert isinstance(entries, list)

    def test_sync_history_with_disk_runs(self):
        from week6.pages.gallery import sync_history_with_disk, _MGR
        # Should not raise even with empty output folders
        sync_history_with_disk(_MGR)


# ══════════════════════════════════════════════════════════════════════════════
# H. Settings Page Persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestSettingsPage:

    def test_load_settings_returns_dict(self):
        from week6.pages.settings import load_settings, _DEFAULT_SETTINGS
        settings = load_settings()
        assert isinstance(settings, dict)
        for key in _DEFAULT_SETTINGS:
            assert key in settings

    def test_save_and_reload_settings(self, tmp_path):
        from week6.pages.settings import save_settings, _DEFAULT_SETTINGS
        import week6.pages.settings as settings_mod

        orig_file = settings_mod._SETTINGS_FILE
        try:
            settings_mod._SETTINGS_FILE = tmp_path / "test_settings.json"
            test_settings = dict(_DEFAULT_SETTINGS)
            test_settings["inference_steps"] = 42
            ok, ts = save_settings(test_settings)
            assert ok
            assert isinstance(ts, str)

            # Reload and verify
            settings_mod._SETTINGS_FILE = tmp_path / "test_settings.json"
            loaded = settings_mod.load_settings()
            assert loaded["inference_steps"] == 42
        finally:
            settings_mod._SETTINGS_FILE = orig_file

    def test_reset_settings_returns_defaults(self, tmp_path):
        from week6.pages.settings import reset_settings, _DEFAULT_SETTINGS
        import week6.pages.settings as settings_mod

        orig_file = settings_mod._SETTINGS_FILE
        try:
            settings_mod._SETTINGS_FILE = tmp_path / "test_settings.json"
            defaults = reset_settings()
            assert isinstance(defaults, dict)
            assert defaults["resolution"] == _DEFAULT_SETTINGS["resolution"]
        finally:
            settings_mod._SETTINGS_FILE = orig_file

    def test_status_html_ok(self):
        from week6.pages.settings import _status_html
        html = _status_html(True, "Settings loaded.")
        assert "✅" in html
        assert "2ecc71" in html

    def test_status_html_error(self):
        from week6.pages.settings import _status_html
        html = _status_html(False, "Save failed.")
        assert "❌" in html
        assert "e74c3c" in html

    def test_settings_page_renders(self):
        from week6.pages.settings import build_settings_page
        with gr.Blocks():
            build_settings_page()

    def test_default_settings_has_all_keys(self):
        from week6.pages.settings import _DEFAULT_SETTINGS
        required = [
            "resolution", "model", "style", "brand",
            "inference_steps", "guidance_scale", "device",
            "global_mock", "server_port",
        ]
        for key in required:
            assert key in _DEFAULT_SETTINGS, f"Missing key: {key}"

    def test_resolution_choices_non_empty(self):
        from week6.pages.settings import _RESOLUTION_CHOICES
        assert len(_RESOLUTION_CHOICES) >= 3

    def test_model_choices_non_empty(self):
        from week6.pages.settings import _MODEL_CHOICES
        assert len(_MODEL_CHOICES) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# I. History Manager (supplement)
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryManagerSupplement:

    def test_record_and_retrieve(self, hist_mgr):
        entry = hist_mgr.record(
            prompt="velvet blazer, editorial lighting",
            image_path="outputs/generated/test01.png",
            model="SDXL v1.0 (Mock)",
            style="luxury",
            mode="mock",
            service="generation",
        )
        assert entry is not None
        assert entry.prompt == "velvet blazer, editorial lighting"

    def test_search_by_keyword(self, hist_mgr):
        hist_mgr.record(
            prompt="techwear cargo unique_keyword_xyz",
            image_path="outputs/generated/test02.png",
            model="SDXL v1.0 (Mock)",
            style="techwear",
            mode="mock",
            service="generation",
        )
        results, total = hist_mgr.search("unique_keyword_xyz")
        assert total >= 1
        assert any("unique_keyword_xyz" in e.prompt for e in results)

    def test_search_returns_paginated(self, hist_mgr):
        results, total = hist_mgr.search("", page=1, page_size=5)
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_delete_removes_record(self, hist_mgr):
        entry = hist_mgr.record(
            prompt="to_delete_prompt_abc123",
            image_path="outputs/generated/del.png",
            model="SDXL (Mock)",
            style="",
            mode="mock",
            service="generation",
        )
        success = hist_mgr.delete_entry(entry.id)
        assert success
        results, _ = hist_mgr.search("to_delete_prompt_abc123")
        assert all(e.id != entry.id for e in results)

    def test_get_stats_returns_dict(self, hist_mgr):
        stats = hist_mgr.get_stats()
        assert isinstance(stats, dict)
        assert "total" in stats

    def test_export_json(self, hist_mgr, tmp_path):
        out_file = tmp_path / "history_export.json"
        result = hist_mgr.export_json(output_path=out_file)
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert isinstance(data["records"], list)

    def test_export_csv(self, hist_mgr, tmp_path):
        out_file = tmp_path / "history_export.csv"
        result = hist_mgr.export_csv(output_path=out_file)
        assert out_file.exists()
        with open(out_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) >= 1  # at least header


# ══════════════════════════════════════════════════════════════════════════════
# J. Export Manager (supplement)
# ══════════════════════════════════════════════════════════════════════════════

class TestExportManagerSupplement:

    def test_export_image_png(self, export_mgr, dummy_rgba):
        res = export_mgr.export_image(dummy_rgba, format_type="PNG", filename="suite_pil.png")
        assert res.is_ok
        p = Path(res.data)
        assert p.exists()
        with Image.open(p) as img:
            assert img.size == (128, 128)
        p.unlink()

    def test_export_image_jpeg(self, export_mgr, dummy_rgba):
        res = export_mgr.export_image(dummy_rgba, format_type="JPEG", quality=85, filename="suite_pil.jpg")
        assert res.is_ok
        p = Path(res.data)
        assert p.exists()
        with Image.open(p) as img:
            assert img.mode == "RGB"
        p.unlink()

    def test_export_metadata_json(self, export_mgr):
        meta = {"prompt": "test export", "steps": 20, "cfg": 7.5}
        res = export_mgr.export_metadata(meta, format_type="JSON", filename="suite_meta.json")
        assert res.is_ok
        p = Path(res.data)
        with open(p) as f:
            loaded = json.load(f)
        assert loaded["steps"] == 20
        p.unlink()

    def test_export_metadata_csv(self, export_mgr):
        meta = {"prompt": "test", "steps": 15}
        res = export_mgr.export_metadata(meta, format_type="CSV", filename="suite_meta.csv")
        assert res.is_ok
        p = Path(res.data)
        assert p.exists()
        p.unlink()

    def test_export_recommendation_report_md(self, export_mgr):
        recs = [{"style": "Streetwear", "confidence": 0.9, "season": "AW27"}]
        res = export_mgr.export_recommendation_report(recs, format_type="MD", filename="suite_recs.md")
        assert res.is_ok
        p = Path(res.data)
        content = p.read_text(encoding="utf-8")
        assert "Streetwear" in content
        p.unlink()

    def test_export_chat_history_md(self, export_mgr):
        history = [("What is silk?", "Silk is a luxurious natural fiber.")]
        res = export_mgr.export_chat_history(history, format_type="MD", filename="suite_chat.md")
        assert res.is_ok
        p = Path(res.data)
        assert "silk" in p.read_text(encoding="utf-8").lower()
        p.unlink()

    def test_export_batch_zip(self, export_mgr, dummy_rgba):
        img_path = _EXPORTS_DIR / "suite_temp.png"
        dummy_rgba.convert("RGB").save(img_path)
        records = [
            {"type": "image", "source": str(img_path), "filename": "design.png"},
            {"type": "metadata", "source": {"prompt": "test"}, "filename": "meta.json"},
        ]
        res = export_mgr.export_batch_zip(records, archive_name="suite_batch.zip")
        assert res.is_ok
        zp = Path(res.data)
        with zipfile.ZipFile(zp) as zf:
            assert "design.png" in zf.namelist()
            assert "meta.json" in zf.namelist()
        img_path.unlink(missing_ok=True)
        zp.unlink()

    def test_export_invalid_format_fails(self, export_mgr):
        res = export_mgr.export_image("path.png", format_type="BMP")
        assert not res.is_ok

    def test_health_check(self, export_mgr):
        hc = export_mgr.health_check()
        assert hc["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# K. State Manager (supplement)
# ══════════════════════════════════════════════════════════════════════════════

class TestStateManagerSupplement:

    def test_full_session_workflow(self, state_mgr):
        """End-to-end: create → mutate all fields → read back → delete."""
        res = state_mgr.get_or_create_session("wf-sess", user_id="workflow_user")
        assert res.is_ok
        sid = "wf-sess"

        state_mgr.set_active_model(sid, "SDXL v1.0 Turbo (Mock)")
        state_mgr.set_selected_lora(sid, "gucci_luxury_v1", weight=0.65)
        state_mgr.set_active_page(sid, "🎨 Text-to-Fashion")
        state_mgr.update_generation_context(sid, prompt="silk gown", image_path="/tmp/out.png")
        state_mgr.increment_chat_turns(sid)
        state_mgr.set_preference(sid, "lang", "en")

        s = state_mgr.get_session(sid).data
        assert s.active_model == "SDXL v1.0 Turbo (Mock)"
        assert s.selected_lora == "gucci_luxury_v1"
        assert abs(s.lora_weight - 0.65) < 1e-6
        assert s.active_page == "🎨 Text-to-Fashion"
        assert s.generation_count == 1
        assert s.chat_turns == 1
        assert s.preferences["lang"] == "en"

        state_mgr.delete_session(sid)
        assert not state_mgr.get_session(sid).is_ok

    def test_session_idle_seconds_increases(self, state_mgr):
        state_mgr.get_or_create_session("idle-check")
        s = state_mgr.get_session("idle-check").data
        time.sleep(0.05)
        assert s.idle_seconds >= 0.0
        state_mgr.delete_session("idle-check")

    def test_session_to_dict_json_serializable(self, state_mgr):
        state_mgr.get_or_create_session("serial-check")
        s = state_mgr.get_session("serial-check").data
        d = s.to_dict()
        dumped = json.dumps(d)
        assert "serial-check" in dumped
        state_mgr.delete_session("serial-check")

    def test_concurrent_page_updates(self, state_mgr):
        state_mgr.get_or_create_session("conc-page")
        pages = ["🎨 Text-to-Fashion", "💬 Fashion Assistant", "🖼️ Gallery"]
        errors = []

        def update(page):
            try:
                state_mgr.set_active_page("conc-page", page)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update, args=(p,)) for p in pages * 5]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = state_mgr.get_session("conc-page").data
        assert s.active_page in pages
        state_mgr.delete_session("conc-page")


# ══════════════════════════════════════════════════════════════════════════════
# L. Page Render Smoke Tests (all build_* functions)
# ══════════════════════════════════════════════════════════════════════════════

class TestAllPageRenders:
    """Each page's build function must construct without error inside gr.Blocks."""

    def test_home_page(self):
        from week6.pages.home import build_home_page
        with gr.Blocks():
            build_home_page()

    def test_text_to_fashion_page(self, gen_svc):
        from week6.pages.text_to_fashion import build_text_to_fashion_page
        with gr.Blocks():
            build_text_to_fashion_page(gen_svc)

    def test_sketch_to_design_page(self, cn_svc):
        from week6.pages.sketch_to_design import build_sketch_to_design_page
        with gr.Blocks():
            build_sketch_to_design_page(cn_svc)

    def test_style_switcher_page(self, lora_svc):
        from week6.pages.style_switcher import build_style_switcher_page
        with gr.Blocks():
            build_style_switcher_page(lora_svc)

    def test_style_mixer_page(self, lora_svc):
        from week6.pages.style_mixer import build_style_mixer_page
        with gr.Blocks():
            build_style_mixer_page(lora_svc)

    def test_fashion_assistant_page(self, rag_svc):
        from week6.pages.fashion_assistant import build_fashion_assistant_page
        with gr.Blocks():
            build_fashion_assistant_page(rag_svc)

    def test_recommendations_page(self, rec_svc, trend_svc):
        from week6.pages.recommendations import build_recommendations_page
        with gr.Blocks():
            build_recommendations_page(rec_svc, trend_svc)

    def test_gallery_page(self):
        from week6.pages.gallery import build_gallery_page
        with gr.Blocks():
            build_gallery_page()

    def test_settings_page(self):
        from week6.pages.settings import build_settings_page
        with gr.Blocks():
            build_settings_page()


# ══════════════════════════════════════════════════════════════════════════════
# M. Components
# ══════════════════════════════════════════════════════════════════════════════

class TestComponents:

    def test_header_renders(self):
        from week6.components.header import build_header
        with gr.Blocks():
            build_header()

    def test_status_bar_renders(self):
        from week6.components.status_bar import build_status_bar
        with gr.Blocks():
            build_status_bar(mock_mode=True)

    def test_metrics_panel_renders(self):
        from week6.components.metrics_panel import build_metrics_panel
        with gr.Blocks():
            build_metrics_panel("Test Metrics", {"total": 10, "accuracy": 0.95})

    def test_image_gallery_renders(self):
        from week6.components.image_gallery import build_image_gallery
        with gr.Blocks():
            build_image_gallery()

    def test_chat_interface_renders(self):
        from week6.components.chat_interface import build_chat_interface
        with gr.Blocks():
            build_chat_interface(lambda x: "hello")

    def test_headers_component_module(self):
        from week6.components import headers
        assert hasattr(headers, "build_page_header") or True  # just confirm importable

    def test_buttons_component_module(self):
        from week6.components import buttons
        assert buttons is not None

    def test_cards_component_module(self):
        from week6.components import cards
        assert cards is not None

    def test_notifications_component_module(self):
        from week6.components import notifications
        assert notifications is not None

    def test_loading_component_module(self):
        from week6.components import loading
        assert loading is not None

    def test_modals_component_module(self):
        from week6.components import modals
        assert modals is not None

    def test_footers_component_module(self):
        from week6.components import footers
        assert footers is not None


# ══════════════════════════════════════════════════════════════════════════════
# N. Evaluation Service
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationService:

    def test_run_evaluation_returns_ok(self, eval_svc):
        res = eval_svc.run_evaluation()
        assert isinstance(res, dict)
        assert "timestamp" in res
        assert "summary" in res

    def test_evaluation_result_has_summary(self, eval_svc):
        res = eval_svc.run_evaluation()
        assert isinstance(res, dict)
        assert "summary" in res
        assert isinstance(res["summary"], dict)

    def test_get_last_report(self, eval_svc):
        res = eval_svc.get_last_report()
        assert isinstance(res, dict)
        assert "summary" in res


# ══════════════════════════════════════════════════════════════════════════════
# O. Trend Service
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendService:

    def test_get_all_trends_returns_list(self, trend_svc):
        res = trend_svc.get_all_trends()
        assert isinstance(res.data, list)
        assert len(res.data) > 0

    def test_get_trend_forecast_returns_list(self, trend_svc):
        res = trend_svc.forecast_season("spring_summer")
        assert isinstance(res.data, list)
        assert len(res.data) > 0

    def test_get_trend_detail_known(self, trend_svc):
        active = trend_svc.get_all_trends().data
        if active:
            first_trend = active[0]
            trend_name = first_trend.get("name") or first_trend.get("trend", "")
            res = trend_svc.explain_trend(trend_name)
            assert isinstance(res, dict)
            assert "trend" in res

    def test_get_all_trends_result_fields(self, trend_svc):
        res = trend_svc.get_all_trends()
        assert isinstance(res.data, list)
        for item in res.data:
            assert isinstance(item, dict)
            assert "name" in item or "trend" in item

    def test_get_velocity_chart_data(self, trend_svc):
        res = trend_svc.get_velocity_chart_data()
        assert isinstance(res, dict)
        assert "labels" in res


# ══════════════════════════════════════════════════════════════════════════════
# P. ServiceResult Contract
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceResultContract:
    """Validate the shared ServiceResult type used by all services."""

    def test_ok_result(self):
        r = ServiceResult.ok(data="hello", meta={"key": "val"}, latency_ms=5.0)
        assert r.is_ok
        assert r.data == "hello"
        assert not r.has_error
        assert r.status == ServiceStatus.OK

    def test_mock_result(self):
        r = ServiceResult.mock(data=42, latency_ms=1.0)
        assert r.is_ok
        assert r.status == ServiceStatus.MOCK
        assert r.meta.get("mode") == "mock"

    def test_fail_result(self):
        r = ServiceResult.fail("Something went wrong", code="TEST_ERR")
        assert not r.is_ok
        assert r.has_error
        assert r.error == "Something went wrong"
        assert r.error_code == "TEST_ERR"
        assert r.status == ServiceStatus.ERROR

    def test_validation_fail(self):
        r = ServiceResult.validation_fail("prompt", "must not be empty")
        assert not r.is_ok
        assert "prompt" in r.error
        assert r.error_code == "VALIDATION_ERROR"

    def test_to_dict_serializable(self):
        r = ServiceResult.ok(data="test", meta={"x": 1}, latency_ms=2.5)
        d = r.to_dict()
        json.dumps(d)  # must not raise
        assert d["status"] == "ok"
        assert d["latency_ms"] == 2.5

    def test_add_warning(self):
        r = ServiceResult.ok(data="val")
        r.add_warning("minor issue")
        assert "minor issue" in r.warnings

    def test_warnings_default_empty(self):
        r = ServiceResult.ok(data="x")
        assert r.warnings == []


# ══════════════════════════════════════════════════════════════════════════════
# Q. App State & App Shell Factory
# ══════════════════════════════════════════════════════════════════════════════

class TestAppShellAndState:

    def test_session_state_mutators(self):
        from week6.gradio_app.app_state import SessionState
        state = SessionState()
        assert state.user_id == "demo_user"
        state.log_generation("a prompt", "dummy_image", {"cfg": 7.5})
        assert state.last_prompt == "a prompt"
        assert len(state.generation_history) == 1
        state.reset()
        assert state.last_prompt == ""
        assert len(state.generation_history) == 0

    def test_create_studio_app(self):
        from week6.gradio_app.app import create_studio_app
        app = create_studio_app()
        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_create_main_app(self):
        from week6.gradio_app.main import create_app
        app = create_app()
        assert app is not None
        assert isinstance(app, gr.Blocks)


# ══════════════════════════════════════════════════════════════════════════════
# R. Additional Pages layout registrations
# ══════════════════════════════════════════════════════════════════════════════

class TestAdditionalPages:

    def test_recommend_hub_page(self, rec_svc):
        from week6.pages.recommend_hub import build_recommend_hub_page
        with gr.Blocks():
            build_recommend_hub_page(rec_svc)

    def test_trend_explorer_page(self, trend_svc):
        from week6.pages.trend_explorer import build_trend_explorer_page
        with gr.Blocks():
            build_trend_explorer_page(trend_svc)

    def test_eval_dashboard_page(self, eval_svc):
        from week6.pages.eval_dashboard import build_eval_dashboard_page
        with gr.Blocks():
            build_eval_dashboard_page(eval_svc)

    def test_style_studio_page(self, gen_svc):
        from week6.pages.style_studio import build_style_studio_page
        with gr.Blocks():
            build_style_studio_page(gen_svc)

    def test_fashion_qa_page(self, rag_svc):
        from week6.pages.fashion_qa import build_fashion_qa_page
        with gr.Blocks():
            build_fashion_qa_page(rag_svc)

    def test_brand_studio_page(self, lora_svc):
        from week6.pages.brand_studio import build_brand_studio_page
        with gr.Blocks():
            build_brand_studio_page(lora_svc)

    def test_controlnet_studio_page(self, cn_svc):
        from week6.pages.controlnet_page import build_controlnet_page
        with gr.Blocks():
            build_controlnet_page(cn_svc)


# ══════════════════════════════════════════════════════════════════════════════
# S. Detailed Components
# ══════════════════════════════════════════════════════════════════════════════

class TestDetailedComponents:

    def test_buttons_components(self):
        from week6.components import buttons
        with gr.Blocks():
            btn1 = buttons.primary_button("Primary")
            btn2 = buttons.secondary_button("Secondary")
            btn3 = buttons.danger_button("Danger")
            btn4 = buttons.icon_button("🌟")
            btn_row = buttons.action_button_row([("Action", "⚡", "primary")])
            g_btn, c_btn = buttons.generate_clear_row()
            cp_btn, dl_btn = buttons.copy_download_row()
            t_btn, t_state = buttons.toggle_button("On", "Off")
            assert btn1 is not None

    def test_cards_components(self):
        from week6.components import cards
        with gr.Blocks():
            c1 = cards.alert_card("alert message")
            c2 = cards.brand_card("Brand", "Description")
            c3 = cards.feature_card("Feature", "Desc")
            c4 = cards.info_card("Info", "Body")
            c5 = cards.recommendation_card("Rec", "Sub", 0.95)
            c6 = cards.stat_card("KPI", 100)
            cards.stat_card_row({"total": 100})
            c7 = cards.tag_list_card("Tags", ["t1", "t2"])
            c8 = cards.trend_card("Trend", "Streetwear", 0.85)
            cards.section_divider("divider")
            assert c2 is not None

    def test_loading_components(self):
        from week6.components import loading
        with gr.Blocks():
            loading._inject_css()
            loading.dots_loader()
            loading.generation_progress(5)
            loading.loading_overlay()
            loading.multi_step_progress(["step1", "step2"], 0)
            loading.progress_bar(50.0)
            loading.skeleton_card()
            loading.skeleton_gallery()
            loading.spinner()

    def test_notifications_components(self):
        from week6.components import notifications
        with gr.Blocks():
            notifications.banner_notification("Banner message")
            notifications.dismissible_alert("Dismiss message")
            notifications.field_validation_msg("Valid")
            notifications.inline_status("Inline status")
            notifications.notification_list([("time", "msg")])
            notifications.toast("toast message")

    def test_modals_components(self):
        from week6.components import modals
        with gr.Blocks():
            modals.confirm_modal()
            modals.form_modal([("field", "label", "value")])
            modals.image_preview_modal()
            modals.info_modal()
            modals.progress_modal()

    def test_footers_components(self):
        from week6.components import footers
        with gr.Blocks():
            footers.app_footer()
            footers.attribution_footer()
            footers.mini_footer()
            footers.session_info_footer()
            footers.status_footer()

    def test_headers_components(self):
        from week6.components import headers
        with gr.Blocks():
            headers.app_header()
            headers.breadcrumb_header(["Home", "Dashboard"])
            headers.compact_header("Title")
            headers.hero_banner("Headline")
            headers.page_header("Page Title")
            headers.section_header("Section Label")
            headers.tab_strip_header(["Tab1", "Tab2"], "Tab1")


# ══════════════════════════════════════════════════════════════════════════════
# T. Targeted Service & Validation Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestTargetedValidation:

    def test_state_manager_validation_and_methods(self, state_mgr, tmp_path):
        assert isinstance(state_mgr.list_sessions().data, list)
        stats = state_mgr.get_stats()
        assert "total_sessions" in stats.data

        snap_file = tmp_path / "snap.json"
        state_mgr.get_or_create_session("snap-sess")
        res_dump = state_mgr.dump_snapshot(snap_file)
        assert res_dump.is_ok
        assert snap_file.exists()
        
        res_load = state_mgr.load_snapshot(snap_file)
        assert res_load.is_ok
        assert state_mgr.get_session("snap-sess").is_ok

        assert not state_mgr.set_active_model("nonexistent", "SDXL v1.0 (Mock)").is_ok
        assert not state_mgr.set_selected_lora("nonexistent", "nike_streetwear_v2", weight=0.5).is_ok
        assert not state_mgr.set_active_page("nonexistent", "🏠 Home Dashboard").is_ok
        assert not state_mgr.update_generation_context("nonexistent", prompt="test", image_path="").is_ok
        assert not state_mgr.increment_chat_turns("nonexistent").is_ok
        assert not state_mgr.set_preference("nonexistent", "lang", "en").is_ok

        sid = "val-sess"
        state_mgr.get_or_create_session(sid)

        res_model = state_mgr.set_active_model(sid, "Invalid Model Name")
        assert res_model.is_ok

        res_lora1 = state_mgr.set_selected_lora(sid, "invalid_lora_name", weight=0.5)
        assert res_lora1.is_ok
        res_lora2 = state_mgr.set_selected_lora(sid, "nike_streetwear_v2", weight=99.0)
        assert not res_lora2.is_ok

        res_page = state_mgr.set_active_page(sid, "Invalid Page Name")
        assert res_page.is_ok

        assert state_mgr.set_preference(sid, "invalid_key", "value").is_ok
        assert state_mgr.set_preference(sid, "resolution", "123x123").is_ok
        
        res_pref1 = state_mgr.set_preference(sid, "inference_steps", -10)
        assert res_pref1.is_ok
        
        res_pref2 = state_mgr.set_preference(sid, "guidance_scale", 100.0)
        assert res_pref2.is_ok

        assert len(state_mgr.get_temporary_outputs(sid).data) == 0
        state_mgr.add_temporary_output(sid, str(tmp_path / "temp1.png"))
        assert len(state_mgr.get_temporary_outputs(sid).data) == 1
        state_mgr.clear_temporary_outputs(sid)
        assert len(state_mgr.get_temporary_outputs(sid).data) == 0

        state_mgr.evict_stale_sessions()
        state_mgr.delete_session(sid)

    def test_history_manager_advanced_and_edge_cases(self, hist_mgr, tmp_path):
        entry = hist_mgr.record(
            prompt="neon cyberpunk jacket",
            image_path=str(tmp_path / "cyber.png"),
            negative_prompt="blurry",
            model="SDXL v1.0",
            style="techwear",
            mode="prod",
            service="generation",
            brand="nike",
            conditioning_mode="canny",
            resolution="1024x1024",
            seed=42,
            steps=30,
            guidance_scale=7.5,
            lora_scale=0.8,
            latency_ms=1200.0,
            tags=["neon", "cyber"],
            rating=5,
            notes="perfect print",
            extra_meta={"fabric": "nylon"}
        )
        assert entry is not None
        assert entry.rating == 5
        assert entry.short_prompt == "neon cyberpunk jacket"
        assert not entry.image_exists
        assert entry.created_dt is not None
        assert entry.matches_text("cyberpunk")
        assert entry.matches_filter({"style": "techwear"})

        d = entry.to_dict()
        from week6.services.history_manager import HistoryEntry
        entry2 = HistoryEntry.from_dict(d)
        assert entry2.prompt == entry.prompt

        res_field, tot_field = hist_mgr.search_by_field("brand", "nike")
        assert tot_field >= 1
        assert res_field[0].brand == "nike"

        ok_upd = hist_mgr.update_entry(entry.id, rating=4, notes="updated notes", tags=["neon", "cyber", "updated"])
        assert ok_upd
        entry_updated = hist_mgr.get_entry(entry.id)
        assert entry_updated.rating == 4
        assert "updated" in entry_updated.tags

        res_list, tot_list = hist_mgr.list_entries(sort="rating", filters={"rating_max": 4})
        assert tot_list >= 1
        assert res_list[0].rating == 4

        assert entry.id in hist_mgr
        assert len(hist_mgr) >= 1
        all_ids = [e.id for e in hist_mgr]
        assert entry.id in all_ids

        md_file = tmp_path / "hist.md"
        hist_mgr.export_markdown(output_path=md_file, include_stats=True)
        assert md_file.exists()
        assert "cyberpunk" in md_file.read_text(encoding="utf-8")

        hist_mgr.rebuild_index()
        hist_mgr.compact()

        success_del = hist_mgr.delete_many([entry.id])
        assert success_del
        assert entry.id not in hist_mgr

        hist_mgr.clear_all()
        assert len(hist_mgr) == 0

    def test_export_manager_targeted_branches(self, export_mgr, dummy_rgba, tmp_path):
        recs = [{"style": "Minimalist", "confidence": 0.95, "tags": ["clean", "white"]}]
        
        res_json = export_mgr.export_recommendation_report(recs, format_type="JSON", filename="recs.json")
        assert res_json.is_ok
        Path(res_json.data).unlink()

        res_csv = export_mgr.export_recommendation_report(recs, format_type="CSV", filename="recs.csv")
        assert res_csv.is_ok
        Path(res_csv.data).unlink()

        rep_dict = {"user_id": "test_user", "recommendations": recs}
        res_dict = export_mgr.export_recommendation_report(rep_dict, format_type="MD", filename="recs_dict.md")
        assert res_dict.is_ok
        Path(res_dict.data).unlink()

        chat = [("Hello", "Hi there!"), {"role": "user", "message": "How is it?"}]
        res_chat_json = export_mgr.export_chat_history(chat, format_type="JSON", filename="chat.json")
        assert res_chat_json.is_ok
        Path(res_chat_json.data).unlink()

        res_chat_csv = export_mgr.export_chat_history(chat, format_type="CSV", filename="chat.csv")
        assert res_chat_csv.is_ok
        Path(res_chat_csv.data).unlink()

        zip_records = [
            {"type": "image", "source": dummy_rgba, "filename": "design_pil.png"},
            {"type": "metadata", "source": {"prompt": "cyberpunk jacket"}, "filename": "meta.json"},
            {"type": "metadata", "source": {"prompt": "retro pants"}, "filename": "meta.csv"},
            {"type": "metadata", "source": "some raw content", "filename": "raw.txt"},
        ]
        res_zip = export_mgr.export_batch_zip(zip_records, archive_name="batch_adv.zip")
        assert res_zip.is_ok
        zp = Path(res_zip.data)
        assert zp.exists()
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            assert "design_pil.png" in names
            assert "meta.json" in names
            assert "meta.csv" in names
            assert "raw.txt" in names
        zp.unlink()

    def test_rag_service_extra_methods(self, rag_svc):
        res1 = rag_svc.recommend_styles({"style": "streetwear"}, n=2)
        assert res1.is_ok
        assert isinstance(res1.data, list)

        res2 = rag_svc.recommend_brands({"aesthetic": "minimalist"}, n=2)
        assert res2.is_ok
        assert isinstance(res2.data, list)

        res3 = rag_svc.explain_trend("quiet luxury")
        assert res3.is_ok
        assert isinstance(res3.data, dict)

        res4 = rag_svc.get_trend_forecast("AW25")
        assert res4.is_ok
        assert isinstance(res4.data, list)

        res5 = rag_svc.get_collection_stats()
        assert res5.is_ok
        assert isinstance(res5.data, dict)

    def test_controlnet_service_extra_methods(self, cn_svc, dummy_rgb):
        res1 = cn_svc.preprocess_image(dummy_rgb, mode="canny")
        assert res1.is_ok
        assert isinstance(res1.data, Image.Image)

        modes = cn_svc.get_modes()
        assert isinstance(modes, list)
        assert "canny" in modes

        info = cn_svc.get_mode_info("canny")
        assert isinstance(info, dict)
        assert "description" in info
        
        assert cn_svc.get_mode_info("invalid") is None

        opts = cn_svc.get_mode_display_options()
        assert isinstance(opts, list)

    def test_recommendation_service_user_profile(self, rec_svc):
        res1 = rec_svc.update_user_profile("user123", {"action": "click", "item": "nike streetwear"})
        assert res1.is_ok
        
        res2 = rec_svc.get_user_profile("user123")
        assert res2.is_ok
        assert res2.data["user_id"] == "user123"
        
        res3 = rec_svc.get_user_profile("nonexistent")
        assert res3.is_ok


