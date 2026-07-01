from __future__ import annotations

import base64
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from week7.backend.main import app
from week7.backend.services.watermark import WatermarkService

client = TestClient(app)


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Fixture supplying a simple solid RGB image."""
    return Image.new("RGB", (200, 100), color="blue")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Fixture supplying a simple solid RGBA image."""
    return Image.new("RGBA", (200, 100), color=(0, 0, 255, 255))


def test_watermark_rgb(sample_rgb_image):
    """Verify watermark application on RGB (JPEG-style) image keeps RGB mode and dimensions."""
    svc = WatermarkService()
    result = svc.apply_watermark(sample_rgb_image)
    assert result.mode == "RGB"
    assert result.size == (200, 100)


def test_watermark_rgba(sample_rgba_image):
    """Verify watermark application on RGBA (PNG-style) image keeps RGBA mode and dimensions."""
    svc = WatermarkService()
    result = svc.apply_watermark(sample_rgba_image)
    assert result.mode == "RGBA"
    assert result.size == (200, 100)


def test_watermark_base64_conversion(sample_rgb_image):
    """Verify watermark service properly decodes, watermarks, and re-encodes base64 strings."""
    # Convert image to base64
    buffered = io.BytesIO()
    sample_rgb_image.save(buffered, format="JPEG")
    orig_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    svc = WatermarkService()
    watermarked_b64 = svc.apply_watermark_to_b64(orig_b64)

    # Decode and verify it is a valid image
    img_data = base64.b64decode(watermarked_b64)
    img_result = Image.open(io.BytesIO(img_data))
    assert img_result.size == (200, 100)


def test_end_to_end_generation_contains_watermark():
    """Verify generated images from the REST/Celery tasks have valid image structures after watermarking."""
    payload = {
        "prompt": "A stylish denim jacket",
        "seed": 99
    }
    response = client.post("/task/start", json={
        "task_type": "generation",
        "payload": payload
    })
    assert response.status_code == 201
    task_id = response.json()["task_id"]

    get_resp = client.get(f"/task/{task_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["status"] == "SUCCESS"
    assert "image" in data["result"]

    # The result image must be a valid base64 encoded PNG image
    img_b64 = data["result"]["image"]
    img_data = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_data))
    assert img.size == (1024, 1024)
