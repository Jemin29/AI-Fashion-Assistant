"""
Week 6 Test Suite — Service Adapters.
Verifies mock-safe operations, presets, list functions, and generation logic.
"""
from __future__ import annotations

import pytest
from PIL import Image

from week6.services import (
    GenerationService,
    ControlNetService,
    LoRAService,
    RAGService,
    RecommendationService,
    TrendService,
    EvaluationService,
)


def test_generation_service(gen_service: GenerationService) -> None:
    """Test mock text-to-image generation and preset retrieval."""
    presets = gen_service.get_style_presets()
    assert isinstance(presets, list)
    assert len(presets) > 0
    assert "Streetwear Editorial" in presets

    # Empty prompt error case
    res = gen_service.generate("")
    assert not res.is_ok
    assert res.error is not None
    assert "prompt" in res.error

    # Valid generation
    res = gen_service.generate("A nice t-shirt")
    assert res.is_ok
    assert isinstance(res.data["image"], Image.Image)
    assert res.meta["mode"] == "mock"
    assert res.data["prompt"] == "A nice t-shirt"


def test_controlnet_service(cn_service: ControlNetService) -> None:
    """Test mock conditioned image generation and preprocessing."""
    modes = cn_service.get_modes()
    assert "canny" in modes
    assert "pose" in modes

    dummy_image = Image.new("RGB", (100, 100), color="white")

    # Empty prompt error case
    res = cn_service.generate_conditioned("", dummy_image)
    assert not res.is_ok
    assert res.error is not None

    # Missing image error case
    res = cn_service.generate_conditioned("A cool jacket", None)
    assert not res.is_ok
    assert res.error is not None

    # Valid conditioned generation
    res = cn_service.generate_conditioned("A cool jacket", dummy_image, mode="pose")
    assert res.is_ok
    assert isinstance(res.data["image"], Image.Image)
    assert res.meta["mode"] == "mock"
    assert res.meta["conditioning_mode"] == "pose"

    # Preprocessing
    pre_img = cn_service.preprocess_image(dummy_image, mode="canny")
    assert isinstance(pre_img.data, Image.Image)


def test_lora_service(lora_service: LoRAService) -> None:
    """Test mock brand personalization and style mixing."""
    brands = lora_service.get_brands()
    assert "nike" in brands
    assert "gucci" in brands

    info = lora_service.get_brand_info("nike")
    assert info["name"] == "Nike"

    # Single brand generation
    res = lora_service.generate_with_brand("A running jacket", "nike")
    assert res.is_ok
    assert isinstance(res.data["image"], Image.Image)
    assert res.meta["brand"] == "Nike"
    assert "nike_style" in res.meta.get("enriched_prompt", "")

    # Style mixing
    mix_weights = {"nike": 0.6, "gucci": 0.4}
    res = lora_service.mix_styles("A sneaker design", mix_weights)
    assert res.is_ok
    assert isinstance(res.data["image"], Image.Image)
    assert res.data["brand_weights"] == mix_weights


def test_rag_service(rag_service: RAGService) -> None:
    """Test mock conversational QA, retrieval, and forecasting."""
    # QA
    qa_res = rag_service.answer_question("Tell me about linen")
    assert qa_res.is_ok
    assert "response" in qa_res.data
    assert len(qa_res.data["source_documents"]) > 0

    # Conversational Chat Router
    chat_qa = rag_service.chat("Tell me about linen fabric")
    assert chat_qa.is_ok
    assert "response" in chat_qa.data
    assert chat_qa.data["intent"] == "fabric_explanation"
    assert len(chat_qa.data["source_documents"]) > 0

    chat_trend = rag_service.chat("Explain the quiet luxury trend")
    assert chat_trend.is_ok
    assert "response" in chat_trend.data
    assert chat_trend.data["intent"] in ("trend_forecast", "trend_explanation")

    chat_brand = rag_service.chat("Recommend brand matching luxury aesthetic")
    assert chat_brand.is_ok
    assert "response" in chat_brand.data
    assert chat_brand.data["intent"] == "brand_recommendation"

    chat_style = rag_service.chat("Recommend style for streetwear")
    assert chat_style.is_ok
    assert "response" in chat_style.data
    assert chat_style.data["intent"] == "style_recommendation"

    # Search
    search_res = rag_service.semantic_search("streetwear", n_results=2)
    assert search_res.is_ok
    assert isinstance(search_res.data, list)
    assert len(search_res.data) <= 2

    # Recommendations
    styles_rec = rag_service.recommend_styles({"style": "streetwear", "occasion": "casual"})
    assert styles_rec.is_ok
    assert isinstance(styles_rec.data, list)

    brands_rec = rag_service.recommend_brands({"styles": ["streetwear"]})
    assert brands_rec.is_ok
    assert isinstance(brands_rec.data, list)

    # Trend details
    exp = rag_service.explain_trend("Quiet Luxury")
    assert exp.is_ok
    assert exp.data["trend"] == "Quiet Luxury"
    assert "explanation" in exp.data

    # Forecast
    forecast = rag_service.get_trend_forecast("spring_summer")
    assert forecast.is_ok
    assert isinstance(forecast.data, list)
    assert len(forecast.data) > 0


def test_recommendation_service(rec_service: RecommendationService) -> None:
    """Test mock preference profiling and recommendation retrieval."""
    # Style recs
    res = rec_service.recommend_styles("women", "minimalist", "formal", "slim_fit", n=3)
    assert res.is_ok
    style_recs = res.data
    assert isinstance(style_recs, list)
    assert len(style_recs) <= 3
    assert style_recs[0]["style"] == "Quiet Luxury"

    # Brand recs
    res = rec_service.recommend_brands(["minimalist"], "Clean elegance", n=2)
    assert res.is_ok
    brand_recs = res.data
    assert isinstance(brand_recs, list)
    assert len(brand_recs) <= 2
    assert brand_recs[0]["brand"] == "Toteme"

    # Profile
    res = rec_service.get_user_profile("test_user")
    assert res.is_ok
    profile = res.data
    assert profile["user_id"] == "test_user"
    assert "top_styles" in profile


def test_trend_service(trend_service: TrendService) -> None:
    """Test mock trend listings, velocity calculations, and seasons."""
    seasons = trend_service.get_seasons()
    assert "spring_summer" in seasons

    all_trends = trend_service.get_all_trends()
    assert isinstance(all_trends.data, list)
    assert all_trends.data[0]["name"] == "Quiet Luxury"

    explain = trend_service.explain_trend("Quiet Luxury")
    assert explain["trend"] == "Quiet Luxury"
    assert "velocity" in explain

    forecast = trend_service.forecast_season("spring_summer")
    assert isinstance(forecast.data, list)

    chart_data = trend_service.get_velocity_chart_data()
    assert "labels" in chart_data
    assert "velocities" in chart_data


def test_evaluation_service(eval_service: EvaluationService) -> None:
    """Test loading and running evaluation reports."""
    report = eval_service.get_last_report()
    assert "summary" in report
    assert "test_cases" in report

    new_report = eval_service.run_evaluation()
    assert "summary" in new_report
    assert "test_cases" in new_report
