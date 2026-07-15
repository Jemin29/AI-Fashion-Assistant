from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from week6.gradio_app.config import get_config, PathsConfig
from week6.gradio_app.main import create_app
from week6.gradio_app.app import create_studio_app


def test_paths_config_assets_dir_resolves_absolutely():
    """Verify that PathsConfig.assets_dir is resolved to an absolute path."""
    cfg = get_config()
    assets_dir_path = Path(cfg.paths.assets_dir)
    assert assets_dir_path.is_absolute()
    assert assets_dir_path.name == "assets"
    assert assets_dir_path.parent.name == "week6"


def test_paths_config_resolves_relative_to_package():
    """Verify that validation resolves custom relative paths to assets_dir relative to package."""
    # Test setting a relative path starting with week6
    config = PathsConfig(assets_dir="week6/custom_assets")
    assert Path(config.assets_dir).is_absolute()
    assert Path(config.assets_dir).name == "custom_assets"
    
    # Test setting a relative path not starting with week6
    config2 = PathsConfig(assets_dir="custom_assets")
    assert Path(config2.assets_dir).is_absolute()
    assert Path(config2.assets_dir).name == "custom_assets"


def test_create_app_raises_on_missing_css_in_non_mock_mode():
    """Verify that create_app raises FileNotFoundError when css file is missing and mock is False."""
    with patch("week6.gradio_app.main.get_config") as mock_get_config:
        mock_cfg = MagicMock()
        mock_cfg.mock.global_mock = False
        # Set a non-existent assets dir so css_path doesn't exist
        mock_cfg.paths.assets_dir = "/nonexistent_assets_path_123"
        mock_cfg.name = "Test Studio"
        mock_get_config.return_value = mock_cfg
        
        with pytest.raises(FileNotFoundError):
            create_app()


def test_create_studio_app_raises_on_missing_css_in_non_mock_mode():
    """Verify that create_studio_app raises FileNotFoundError when css file is missing and mock is False."""
    with patch("week6.gradio_app.app.get_config") as mock_get_config:
        mock_cfg = MagicMock()
        mock_cfg.mock.global_mock = False
        # Set a non-existent assets dir so css_path doesn't exist
        mock_cfg.paths.assets_dir = "/nonexistent_assets_path_123"
        mock_cfg.name = "Test Studio"
        mock_get_config.return_value = mock_cfg
        
        with pytest.raises(FileNotFoundError):
            create_studio_app()
