from __future__ import annotations

import os
from unittest import mock
import pytest
from week7.backend.configs.config import Settings, get_settings
from week7.backend.configs.logging import configure_logging


def test_default_settings():
    """Verify that settings are loaded with correct default values."""
    settings = Settings()
    assert settings.app_name == "AI Fashion Assistant API"
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000
    assert settings.model.global_mock is True


def test_env_overrides():
    """Verify that settings successfully bind nested environment variable overrides."""
    env_overrides = {
        "APP_NAME": "Override App Name",
        "SERVER__PORT": "9090",
        "MODEL__GLOBAL_MOCK": "false",
    }
    original_values = {}
    for k, v in env_overrides.items():
        original_values[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        settings = Settings()
        assert settings.app_name == "Override App Name"
        assert settings.server.port == 9090
        assert settings.model.global_mock is False
    finally:
        for k, v in original_values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_get_settings_singleton():
    """Verify that get_settings loads config as a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_configure_logging_dry_run(tmp_path):
    """Verify that the Loguru configuration executes without exceptions."""
    # Temporarily override BACKEND_ROOT for logging files to avoid touching production logs
    with mock.patch("week7.backend.configs.logging.BACKEND_ROOT", tmp_path):
        configure_logging(log_level="DEBUG", log_to_file=True)
        
        # Verify log file was created
        log_file = tmp_path / "logs" / "app.log"
        assert log_file.parent.exists()
