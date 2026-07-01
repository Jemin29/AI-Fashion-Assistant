"""
Week 6 Test Suite — Conftest.
Defines mock fixtures for testing pages, components, and services.
"""
from __future__ import annotations
import pytest
from typing import Generator

from week6.gradio_app.config import get_config, AppConfig
from week6.services import (
    GenerationService,
    ControlNetService,
    LoRAService,
    RAGService,
    RecommendationService,
    TrendService,
    EvaluationService,
)


@pytest.fixture(scope="session", autouse=True)
def force_mock_mode() -> None:
    """Ensure mock mode is globally enabled during test execution."""
    cfg = get_config()
    cfg.mock.global_mock = True


@pytest.fixture
def app_config() -> AppConfig:
    """Return the active application configuration."""
    return get_config()


@pytest.fixture
def gen_service() -> GenerationService:
    """Return mock GenerationService."""
    return GenerationService(mock_mode=True)


@pytest.fixture
def cn_service() -> ControlNetService:
    """Return mock ControlNetService."""
    return ControlNetService(mock_mode=True)


@pytest.fixture
def lora_service() -> LoRAService:
    """Return mock LoRAService."""
    return LoRAService(mock_mode=True)


@pytest.fixture
def rag_service() -> RAGService:
    """Return mock RAGService."""
    return RAGService(mock_mode=True)


@pytest.fixture
def rec_service() -> RecommendationService:
    """Return mock RecommendationService."""
    return RecommendationService(mock_mode=True)


@pytest.fixture
def trend_service() -> TrendService:
    """Return mock TrendService."""
    return TrendService(mock_mode=True)


@pytest.fixture
def eval_service() -> EvaluationService:
    """Return mock EvaluationService."""
    return EvaluationService(mock_mode=True)
