from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from week7.backend.main import app

client = TestClient(app)


def test_health_check():
    """Verify that the diagnostic health check endpoint works and returns standard schemas."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    body = response.json()
    assert body["success"] is True
    assert "data" in body
    assert body["data"]["status"] == "healthy"
    assert "environment_mode" in body["data"]


def test_404_error_handler():
    """Verify that requesting a nonexistent endpoint returns our custom HTTP_ERROR_404 JSON response."""
    response = client.get("/api/v1/nonexistent-route-for-testing")
    assert response.status_code == 404
    
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    assert body["error"]["code"] == "HTTP_ERROR_404"
    assert "Not Found" in body["error"]["message"]


def test_x_process_time_header():
    """Verify that the request logger middleware successfully appends latency tracking headers."""
    response = client.get("/api/v1/health")
    assert "X-Process-Time-Ms" in response.headers
    
    # Value should be parseable as float
    latency = float(response.headers["X-Process-Time-Ms"])
    assert latency >= 0.0


def test_week2_generate():
    """Verify that the Week 2 SDXL generation endpoint accepts parameters and returns the base64 image."""
    payload = {
        "prompt": "A stylish red silk blazer, studio photography",
        "negative_prompt": "blurry, dark",
        "seed": 42,
        "cfg": 8.0,
        "resolution": "512x512"
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    
    body = response.json()
    assert body["success"] is True
    assert "image" in body
    assert len(body["image"]) > 0
    assert "metadata" in body
    assert body["metadata"]["seed"] == 42
    assert body["metadata"]["guidance_scale"] == 8.0
    assert body["metadata"]["width"] == 512
    assert "generation_time" in body


def test_week3_controlnet():
    """Verify that the Week 3 ControlNet endpoints accept file uploads and return base64 images."""
    from PIL import Image
    import io
    
    # 1. Prepare dummy image in memory
    img = Image.new("RGB", (256, 256), color="white")
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()
    
    # 2. Test /sketch endpoint
    resp = client.post(
        "/sketch",
        files={"file": ("sketch.png", img_bytes, "image/png")},
        data={
            "prompt": "Minimalist linen shirt",
            "control_strength": "0.8",
            "negative_prompt": "blurry",
            "seed": "42"
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "image" in body
    assert len(body["image"]) > 0
    assert "metadata" in body
    assert body["metadata"]["generation"]["controlnet_type"] == "sketch"
    
    # 3. Test /pose endpoint
    resp = client.post(
        "/pose",
        files={"file": ("pose.png", img_bytes, "image/png")},
        data={
            "prompt": "Athleisure running outfit",
            "control_strength": "0.9",
            "seed": "42"
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "image" in body
    assert body["metadata"]["generation"]["controlnet_type"] == "pose"
    
    # 4. Test /depth endpoint
    resp = client.post(
        "/depth",
        files={"file": ("depth.png", img_bytes, "image/png")},
        data={
            "prompt": "Haute couture gown",
            "control_strength": "0.75",
            "seed": "42"
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "image" in body
    assert body["metadata"]["generation"]["controlnet_type"] == "depth"


def test_week4_lora():
    """Verify that the Week 4 LoRA endpoints accept parameters and return base64 images."""
    # 1. Test /lora endpoint
    payload = {
        "prompt": "Streetwear cargo pants",
        "brand": "nike",
        "lora_scale": 0.85,
        "seed": 42
    }
    resp = client.post("/lora", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "image" in body
    assert len(body["image"]) > 0
    assert body["metadata"]["brand"] == "nike"
    assert body["metadata"]["scale"] == 0.85

    # 2. Test /style-switch endpoint
    payload_switch = {
        "prompt": "Velvet luxury suit",
        "brand": "gucci",
        "lora_scale": 1.1,
        "seed": 42
    }
    resp = client.post("/style-switch", json=payload_switch)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["metadata"]["brand"] == "gucci"

    # 3. Test /style-mix endpoint
    payload_mix = {
        "prompt": "Contemporary oversized sweater",
        "brand_weights": {
            "nike": 0.6,
            "gucci": 0.4
        },
        "seed": 42
    }
    resp = client.post("/style-mix", json=payload_mix)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "image" in body
    assert body["metadata"]["mixed_weights"]["nike"] == 0.6
    assert body["metadata"]["mixed_weights"]["gucci"] == 0.4


def test_week5_assistant():
    """Verify that the Week 5 Assistant endpoints accept queries and return RAG fields."""
    # 1. Test /ask
    resp = client.post("/ask", json={"question": "What fabrics are most breathable?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "answer" in body
    assert isinstance(body["recommendations"], list)
    assert "confidence_score" in body
    assert "confidence score" in body
    assert isinstance(body["citations"], list)
    
    # 2. Test /recommend
    resp = client.post("/recommend", json={"style": "casual", "occasion": "work", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "recommendations" in body
    assert len(body["recommendations"]) <= 3
    
    # 3. Test /search
    resp = client.post("/search", json={"query": "Linen material properties", "limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["citations"], list)
    
    # 4. Test /trend
    resp = client.post("/trend", json={"trend_name": "Quiet Luxury"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "answer" in body
    assert "confidence_score" in body


def test_v1_controlnet_endpoints():
    """Verify version 1 ControlNet endpoints for modes, preprocessing, and generation."""
    # 1. Test modes
    resp = client.get("/api/v1/controlnet/modes")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0

    # 2. Test preprocess
    from PIL import Image
    import io
    img = Image.new("RGB", (256, 256), color="white")
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    resp = client.post(
        "/api/v1/controlnet/preprocess",
        files={"file": ("sketch.png", img_bytes, "image/png")},
        data={"mode": "canny"}
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # 3. Test generate
    resp = client.post(
        "/api/v1/controlnet/generate",
        files={"file": ("sketch.png", img_bytes, "image/png")},
        data={
            "prompt": "Linen summer trousers",
            "mode": "canny",
            "conditioning_scale": "0.7",
            "session_id": "test-session"
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "image" in resp.json()["data"]


def test_v1_lora_endpoints():
    """Verify version 1 LoRA endpoints for listing adapters, generating, and mixing styles."""
    # 1. Test adapters list
    resp = client.get("/api/v1/lora/adapters")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0

    # 2. Test generate
    resp = client.post(
        "/api/v1/lora/generate",
        json={
            "prompt": "Technical running shorts",
            "brand": "nike",
            "lora_scale": 0.9,
            "session_id": "test-session"
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "saved_path" in resp.json()["data"]

    # 3. Test mix
    resp = client.post(
        "/api/v1/lora/mix",
        json={
            "prompt": "Hybrid couture jacket",
            "brand_weights": {"nike": 0.5, "gucci": 0.5},
            "session_id": "test-session"
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "saved_path" in resp.json()["data"]


def test_v1_rag_endpoints():
    """Verify version 1 RAG assistant chat, semantic search, and collection stats."""
    # 1. Test chat
    resp = client.post(
        "/api/v1/rag/chat",
        json={
            "message": "Tell me about linen fabric characteristics.",
            "session_id": "test-session"
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "answer" in resp.json()["data"]

    # 2. Test search
    resp = client.post(
        "/api/v1/rag/search",
        json={
            "query": "breathable summer materials",
            "limit": 2
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]["citations"]) > 0

    # 3. Test stats
    resp = client.get("/api/v1/rag/stats")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_v1_recommendations_endpoints():
    """Verify version 1 style, brand, active trends, and forecast recommendations."""
    # 1. Test style recommendations
    resp = client.post(
        "/api/v1/recommendations/styles",
        json={
            "gender": "unisex",
            "style": "streetwear",
            "occasion": "casual",
            "fit": "oversized",
            "limit": 3
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0

    # 2. Test brand recommendations
    resp = client.post(
        "/api/v1/recommendations/brands",
        json={
            "styles": ["minimalist", "clean"],
            "aesthetic": "minimalist",
            "limit": 2
        }
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0

    # 3. Test trends list
    resp = client.get("/api/v1/recommendations/trends")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0

    # 4. Test seasonal forecast
    resp = client.get("/api/v1/recommendations/trends/forecast?season=spring_summer")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(resp.json()["data"]) > 0





