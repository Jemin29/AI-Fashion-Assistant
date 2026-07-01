from __future__ import annotations

import base64
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from week7.backend.main import app
from week7.backend.middleware.nsfw_filter import NSFWFilter

client = TestClient(app)


@pytest.fixture
def normal_image_b64() -> str:
    """Fixture supplying a normal dummy base64 image."""
    img = Image.new("RGB", (64, 64), color="blue")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@pytest.fixture
def unsafe_image_b64() -> str:
    """Fixture supplying a mock unsafe base64 image."""
    return "unsafe_mock_base64_nudity_content_detected"


def test_nsfw_filter_unit(normal_image_b64, unsafe_image_b64):
    """Test the NSFWFilter class in mock mode."""
    filt = NSFWFilter(mock_mode=True)
    # 1. Normal image should pass
    assert filt.scan_image(normal_image_b64) is True
    # 2. Unsafe image should be rejected
    assert filt.scan_image(unsafe_image_b64) is False
    # 3. Empty image is safe
    assert filt.scan_image("") is True


def test_nsfw_middleware_intercepts_unsafe():
    """Verify that the middleware intercept blocks unsafe generated image outputs."""
    # When prompt is "unsafe", our mock generator produces an image string prefixed with "unsafe_mock_content_"
    payload = {
        "prompt": "unsafe content nudity",
        "seed": 42
    }
    response = client.post("/task/start", json={
        "task_type": "generation",
        "payload": payload
    })

    assert response.status_code == 201
    task_id = response.json()["task_id"]

    # Retrieve status. The response json contains "image" in the result, containing "unsafe_mock_content_".
    # The middleware should intercept the GET response and return a 400 Bad Request.
    get_resp = client.get(f"/task/{task_id}")
    assert get_resp.status_code == 400
    data = get_resp.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSAFE_CONTENT_DETECTED"
    assert "flagged by our safety pipeline" in data["error"]["message"]


def test_nsfw_middleware_allows_safe():
    """Verify that safe generated images pass the middleware filter successfully."""
    payload = {
        "prompt": "A completely safe summer dress",
        "seed": 10
    }
    response = client.post("/task/start", json={
        "task_type": "generation",
        "payload": payload
    })
    assert response.status_code == 201
    task_id = response.json()["task_id"]

    get_resp = client.get(f"/task/{task_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "SUCCESS"
