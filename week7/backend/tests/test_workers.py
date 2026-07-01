from __future__ import annotations

import base64
import io
import pytest
from PIL import Image

from week7.backend.workers.generation_worker import generate_image_task
from week7.backend.workers.controlnet_worker import (
    generate_from_sketch_task,
    generate_from_pose_task,
    generate_from_depth_task,
)
from week7.backend.workers.lora_worker import (
    generate_lora_style_task,
    mix_styles_task,
)
from week7.backend.workers.rag_worker import (
    answer_question_task,
    recommend_styles_task,
    semantic_search_task,
    explain_trend_task,
)


@pytest.fixture
def dummy_image_b64() -> str:
    """Fixture supplying a valid base64-encoded dummy PNG image."""
    img = Image.new("RGB", (64, 64), color="red")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_generation_background_task():
    """Verify background image generation runs successfully via Celery eager mode."""
    res = generate_image_task.delay(
        prompt="A chic summer dress",
        negative_prompt="blurry",
        seed=42,
        resolution="512x512"
    )
    data = res.result
    assert isinstance(data, dict)
    assert "image" in data
    assert "metadata" in data
    assert data["metadata"]["prompt"] == "A chic summer dress"


def test_controlnet_background_tasks(dummy_image_b64):
    """Verify background sketch, pose, and depth conditioned generations."""
    # 1. Sketch task
    res_sketch = generate_from_sketch_task.delay(
        prompt="A silk dress",
        sketch_image_b64=dummy_image_b64,
        seed=123
    )
    assert "image" in res_sketch.result

    # 2. Pose task
    res_pose = generate_from_pose_task.delay(
        prompt="Activewear sports bra",
        pose_image_b64=dummy_image_b64,
        seed=123
    )
    assert "image" in res_pose.result

    # 3. Depth task
    res_depth = generate_from_depth_task.delay(
        prompt="Leather jacket",
        depth_image_b64=dummy_image_b64,
        seed=123
    )
    assert "image" in res_depth.result


def test_lora_background_tasks():
    """Verify background LoRA styling and style mixing."""
    res_lora = generate_lora_style_task.delay(
        prompt="Neon running shoes",
        brand="nike",
        lora_scale=0.8,
        seed=100
    )
    assert "image" in res_lora.result

    res_mix = mix_styles_task.delay(
        prompt="Luxury sportswear jacket",
        brand_weights={"nike": 0.6, "gucci": 0.4},
        seed=100
    )
    assert "image" in res_mix.result


def test_rag_background_tasks():
    """Verify background RAG retrieval, recommendations, and trend forecasts."""
    # 1. Q&A
    res_qa = answer_question_task.delay("What fabrics are trending for summer?")
    assert "response" in res_qa.result

    # 2. Recommendations
    res_rec = recommend_styles_task.delay({"style": "casual", "gender": "unisex"}, n=3)
    assert len(res_rec.result) > 0

    # 3. Semantic Search
    res_search = semantic_search_task.delay("linen shirts", n_results=2)
    assert len(res_search.result) > 0

    # 4. Trend Explanation
    res_trend = explain_trend_task.delay("Eco-cotton leisurewear")
    assert "explanation" in res_trend.result
