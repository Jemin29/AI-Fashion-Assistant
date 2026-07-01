from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from week7.backend.main import app
from week7.backend.configs.rate_limit import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter_fixture():
    """Fixture to reset the rate limiter state before and after each test."""
    limiter.reset()
    yield
    limiter.reset()


def test_rate_limiting_generation():
    """Verify that calling the /generation/generate endpoint exceeding 10 times/minute triggers 429."""
    payload = {
        "prompt": "A trendy modern t-shirt",
        "negative_prompt": "",
        "style_preset": None,
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg": 7.5,
        "seed": 42,
        "session_id": "test-session"
    }

    # Call it 10 times (which is the limit)
    for _ in range(10):
        resp = client.post("/api/v1/generation/generate", json=payload)
        # Verify it does not hit 429
        assert resp.status_code != 429

    # The 11th call must trigger 429
    resp_blocked = client.post("/api/v1/generation/generate", json=payload)
    assert resp_blocked.status_code == 429
    
    data = resp_blocked.json()
    assert data["success"] is False
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Rate limit exceeded" in data["error"]["message"]


def test_rate_limiting_auth_register():
    """Verify rate limit of 15/minute on auth registration endpoint."""
    # Register 15 dummy users
    for i in range(15):
        payload = {
            "username": f"rate_limit_user_{i}",
            "password": "password123"
        }
        resp = client.post("/auth/register", json=payload)
        assert resp.status_code != 429

    # The 16th registration must trigger 429
    payload_blocked = {
        "username": "rate_limit_user_blocked",
        "password": "password123"
    }
    resp_blocked = client.post("/auth/register", json=payload_blocked)
    assert resp_blocked.status_code == 429
    assert resp_blocked.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
