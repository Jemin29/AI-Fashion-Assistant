from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of the Backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class ServerSettings(BaseModel):
    """Web Server settings."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    allowed_origins: List[str] = ["*"]


class SecuritySettings(BaseModel):
    """Authentication and cryptography settings."""
    jwt_secret_key: str = "production-secret-signing-key-placeholder-32-chars"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    bcrypt_rounds: int = 12


class DatabaseSettings(BaseModel):
    """Database persistence settings."""
    sqlite_url: str = "sqlite:///./fashion_assistant.db"
    chroma_db_dir: str = "./outputs/chroma_db"
    history_file_path: str = "./outputs/history.jsonl"


class ModelSettings(BaseModel):
    """Machine learning model execution settings."""
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_cache_dir: str = "./outputs/loras"
    global_mock: bool = True


class RedisSettings(BaseModel):
    """Redis cache, session, and queue settings."""
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    enabled: bool = True
    mock_fallback: bool = True  # Use in-memory mock if Redis is unavailable or in mock mode


class Settings(BaseSettings):
    """Unified system configuration object."""
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / "configs" / ".env"),
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "AI Fashion Assistant API"
    app_version: str = "1.0.0"
    
    server: ServerSettings = ServerSettings()
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    model: ModelSettings = ModelSettings()
    redis: RedisSettings = RedisSettings()


# Global Singleton for App-wide Configuration
_settings: Settings | None = None


def get_settings() -> Settings:
    """Load and return the system configuration singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        if _settings.model.global_mock:
            from loguru import logger
            banner = (
                "\n" + "═" * 60 + "\n"
                "⚠️  RUNNING IN MOCK MODE (GLOBAL_MOCK=True)  ⚠️\n"
                "This backend run is in Mock Mode. No GPU or real weights will be loaded.\n"
                "To run in Real GPU Mode, set model__global_mock=False in backend config/.env.\n"
                + "═" * 60
            )
            logger.warning(banner)
    return _settings
