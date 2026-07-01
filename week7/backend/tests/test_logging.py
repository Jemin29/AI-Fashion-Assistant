from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from loguru import logger

from week7.backend.main import app
from week7.backend.logging_config import (
    log_error,
    log_warning,
    log_api_request,
    log_generation_failure,
    log_redis_failure,
    log_celery_failure,
)

client = TestClient(app)

# Define a temporary testing router to trigger unhandled exceptions
logging_test_router = APIRouter(prefix="/test-logging", tags=["Testing"])


@logging_test_router.get("/trigger-error")
def trigger_unhandled_exception():
    raise RuntimeError("Simulated crash for logging middleware validation.")


# Include the test router temporarily
app.include_router(logging_test_router)


@pytest.fixture
def capture_logs():
    """Fixture to capture loguru log messages during a test execution."""
    captured = []
    # Add a custom lambda sink to capture messages
    sink_id = logger.add(lambda msg: captured.append(msg.record), level="DEBUG")
    yield captured
    logger.remove(sink_id)


def test_centralized_helpers(capture_logs):
    """Verify that centralized logging helper methods structure and output log messages correctly."""
    # Test error logging
    log_error("Test critical failure", exc=ValueError("Dummy value error"))
    assert any(
        "Test critical failure" in rec["message"] and rec["level"].name == "ERROR"
        for rec in capture_logs
    )

    # Test warning logging
    log_warning("Test warning indicator", extra={"context": "test_configs"})
    assert any(
        "Test warning indicator" in rec["message"] and rec["level"].name == "WARNING"
        for rec in capture_logs
    )

    # Test api request logging
    log_api_request("GET", "/api/dummy", status_code=200, latency_ms=12.34, client_ip="127.0.0.1")
    assert any(
        "API Request: GET /api/dummy" in rec["message"] and rec["level"].name == "INFO"
        for rec in capture_logs
    )

    # Test generation failure logging
    log_generation_failure(prompt="A gold jacket", error_msg="GPU out of memory", session_id="session123")
    assert any(
        "Generation Failure" in rec["message"] and rec["level"].name == "ERROR"
        for rec in capture_logs
    )

    # Test Redis failure logging
    log_redis_failure("cache_set", "Connection refused")
    assert any(
        "Redis Connection Failure" in rec["message"] and rec["level"].name == "ERROR"
        for rec in capture_logs
    )

    # Test Celery failure logging
    log_celery_failure("tasks.generate_image_task", "task-uuid-abc", "Celery queue timeout")
    assert any(
        "Celery Task Failure" in rec["message"] and rec["level"].name == "ERROR"
        for rec in capture_logs
    )


def test_error_handler_middleware(capture_logs):
    """Verify that ErrorHandlerMiddleware intercepts unhandled exceptions, logs tracebacks, and returns standard 500 JSON envelopes."""
    resp = client.get("/test-logging/trigger-error")
    assert resp.status_code == 500
    
    data = resp.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "unhandled server error" in data["error"]["message"]

    # Verify that the unhandled exception was captured in loguru
    assert any(
        "Unhandled system exception" in rec["message"] and rec["level"].name == "ERROR"
        for rec in capture_logs
    )
