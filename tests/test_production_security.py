from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from week7.backend.configs.config import get_settings
from week7.backend.middleware.security import validate_production_environment

def test_validate_dev_env():
    """In development/development mode, no validation errors are raised."""
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        settings = get_settings()
        with patch.object(settings.server, "debug", True), \
             patch.object(settings.server, "allowed_origins", ["*"]):
            validate_production_environment()  # Should pass without raising

def test_validate_prod_env_wildcard_origins():
    """In production mode, wildcard allowed_origins should raise ValueError."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        settings = get_settings()
        with patch.object(settings.server, "debug", False), \
             patch.object(settings.server, "reload", False), \
             patch.object(settings.security, "jwt_secret_key", "a-secure-secret-key-32-chars-long-minimum-xyz"), \
             patch.object(settings.server, "allowed_origins", ["*"]):
            with pytest.raises(ValueError) as excinfo:
                validate_production_environment()
            assert "SERVER__ALLOWED_ORIGINS cannot contain wildcard" in str(excinfo.value)

def test_validate_prod_env_explicit_origins_pass():
    """In production mode, explicit allowed_origins should pass without allowed_origins errors."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        settings = get_settings()
        with patch.object(settings.server, "debug", False), \
             patch.object(settings.server, "reload", False), \
             patch.object(settings.security, "jwt_secret_key", "a-secure-secret-key-32-chars-long-minimum-xyz"), \
             patch.object(settings.server, "allowed_origins", ["https://fashion-studio.example.com"]):
            try:
                validate_production_environment()
            except ValueError as e:
                assert "SERVER__ALLOWED_ORIGINS" not in str(e)
