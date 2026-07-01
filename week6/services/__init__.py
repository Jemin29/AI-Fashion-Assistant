"""
Week 6 — Services Package.

Exports all service classes used by the Gradio UI layer to communicate
with the Week 4–5 backend modules.

Shared contract
---------------
Every service method that performs a computation returns a result that
includes at minimum:

  - The primary output  (image, text, list, …)
  - A ``meta`` dict     with keys: mode, model, latency_ms, timestamp, …
  - An optional ``error`` string (None on success)

Import the ``ServiceResult`` dataclass and ``ServiceError`` exception for
typed error handling in event callbacks.
"""
from __future__ import annotations

from week6.services.base import ServiceResult, ServiceError, ServiceStatus
from week6.services.generation_service import GenerationService
from week6.services.controlnet_service import ControlNetService
from week6.services.lora_service import LoRAService
from week6.services.rag_service import RAGService
from week6.services.recommendation_service import RecommendationService
from week6.services.trend_service import TrendService
from week6.services.eval_service import EvaluationService
from week6.services.export_manager import ExportManager
from week6.services.state_manager import StateManager

__all__ = [
    # Shared contract types
    "ServiceResult",
    "ServiceError",
    "ServiceStatus",
    # Service classes
    "GenerationService",
    "ControlNetService",
    "LoRAService",
    "RAGService",
    "RecommendationService",
    "TrendService",
    "EvaluationService",
    "ExportManager",
    "StateManager",
]

