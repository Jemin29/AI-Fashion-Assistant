from __future__ import annotations

import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from week7.backend.main import app
from week7.backend.configs.config import get_settings
from week7.backend.middleware.security import (
    validate_production_environment,
    SecurityHeadersMiddleware
)
from week7.backend.configs.rate_limit import get_client_ip

client = TestClient(app)
settings = get_settings()

def test_security_headers_middleware():
    """Verify that secure HTTP headers are correctly set by the middleware."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    # Assert security headers
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "default-src 'self'" in response.headers.get("Content-Security-Policy", "")
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"


def test_get_client_ip_resolver():
    """Verify proxy IP resolver parses X-Forwarded-For and X-Real-IP headers."""
    # Mock FastAPI request object
    class MockRequest:
        def __init__(self, headers, client_host=None):
            self.headers = headers
            self.client = type("Client", (), {"host": client_host}) if client_host else None

    # Test X-Forwarded-For header
    req = MockRequest(headers={"X-Forwarded-For": "203.0.113.195, 70.41.3.18"})
    assert get_client_ip(req) == "203.0.113.195"

    # Test X-Real-IP header
    req_real = MockRequest(headers={"X-Real-IP": "198.51.100.22"})
    assert get_client_ip(req_real) == "198.51.100.22"

    # Test fallback to connection host
    req_client = MockRequest(headers={}, client_host="192.0.2.1")
    assert get_client_ip(req_client) == "192.0.2.1"

    # Test fallback to local
    req_local = MockRequest(headers={})
    assert get_client_ip(req_local) == "127.0.0.1"


def test_production_environment_validation(monkeypatch):
    """Verify that insecure production settings raise validation errors."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    # Save original values
    orig_debug = settings.server.debug
    orig_reload = settings.server.reload
    orig_secret = settings.security.jwt_secret_key

    # Initialize to safe values to isolate test conditions
    settings.server.debug = False
    settings.server.reload = False
    settings.security.jwt_secret_key = "safe-unique-secret-key-32-characters-long"

    try:
        # 1. Test debug flag fails
        settings.server.debug = True
        with pytest.raises(ValueError, match="SERVER__DEBUG cannot be set to True"):
            validate_production_environment()
        settings.server.debug = False

        # 2. Test reload flag fails
        settings.server.reload = True
        with pytest.raises(ValueError, match="SERVER__RELOAD must be disabled"):
            validate_production_environment()
        settings.server.reload = False

        # 3. Test default secret fails
        settings.security.jwt_secret_key = "production-secret-signing-key-placeholder-32-chars"
        with pytest.raises(ValueError, match="SECURITY__JWT_SECRET_KEY is using a default"):
            validate_production_environment()

        # 4. Test short secret fails
        settings.security.jwt_secret_key = "short-key"
        with pytest.raises(ValueError, match="must be at least 32 characters long"):
            validate_production_environment()

    finally:
        # Restore original settings
        settings.server.debug = orig_debug
        settings.server.reload = orig_reload
        settings.security.jwt_secret_key = orig_secret
