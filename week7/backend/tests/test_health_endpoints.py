from __future__ import annotations

from fastapi.testclient import TestClient
from week7.backend.main import app

client = TestClient(app)

def test_liveness_endpoint() -> None:
    """Verify that the liveness diagnostic endpoint returns HTTP 200."""
    response = client.get("/liveness")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "alive"

def test_readiness_endpoint() -> None:
    """Verify that the readiness dependency checker returns an appropriate status code."""
    response = client.get("/readiness")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "dependencies" in data

def test_aggregated_health_endpoint() -> None:
    """Verify that the aggregated health check returns resource and service metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "system_resources" in data["data"]
