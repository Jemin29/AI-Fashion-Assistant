from __future__ import annotations

from typing import TYPE_CHECKING
from week7.backend.configs.config import get_settings

if TYPE_CHECKING:
    from week7.backend.services.generation_service import GenerationService
    from week7.backend.services.controlnet_service import ControlNetService
    from week7.backend.services.lora_service import LoraService
    from week7.backend.services.rag_service import RAGService
    from week7.backend.services.recommendation_service import RecommendationService
    from week7.backend.services.history_service import HistoryService
    from week7.backend.services.redis_manager import RedisManager
    from week7.backend.services.task_tracker import TaskTracker
    from week7.backend.middleware.nsfw_filter import NSFWFilter
    from week7.backend.services.watermark import WatermarkService
    from week6.services.trend_service import TrendService
    from week6.services.state_manager import StateManager

# Singleton holders for lazy loading
_generation_service = None
_controlnet_service = None
_lora_service = None
_rag_service = None
_recommendation_service = None
_history_service = None
_trend_service = None
_state_manager = None
_redis_manager = None
_task_tracker = None
_nsfw_filter = None
_watermark_service = None




def get_generation_service() -> GenerationService:
    """Lazily load and return the GenerationService singleton."""
    global _generation_service
    if _generation_service is None:
        from week7.backend.services.generation_service import GenerationService
        _generation_service = GenerationService()
    return _generation_service


def get_controlnet_service() -> ControlNetService:
    """Lazily load and return the ControlNetService singleton."""
    global _controlnet_service
    if _controlnet_service is None:
        from week7.backend.services.controlnet_service import ControlNetService
        _controlnet_service = ControlNetService()
    return _controlnet_service


def get_lora_service() -> LoraService:
    """Lazily load and return the LoraService singleton."""
    global _lora_service
    if _lora_service is None:
        from week7.backend.services.lora_service import LoraService
        _lora_service = LoraService()
    return _lora_service


def get_rag_service() -> RAGService:
    """Lazily load and return the RAGService singleton."""
    global _rag_service
    if _rag_service is None:
        from week7.backend.services.rag_service import RAGService
        settings = get_settings()
        _rag_service = RAGService(mock_mode=settings.model.global_mock)
    return _rag_service


def get_recommendation_service() -> RecommendationService:
    """Lazily load and return the RecommendationService singleton."""
    global _recommendation_service
    if _recommendation_service is None:
        from week7.backend.services.recommendation_service import RecommendationService
        settings = get_settings()
        _recommendation_service = RecommendationService(mock_mode=settings.model.global_mock)
    return _recommendation_service


def get_history_service() -> HistoryService:
    """Lazily load and return the HistoryService singleton."""
    global _history_service
    if _history_service is None:
        from week7.backend.services.history_service import HistoryService
        _history_service = HistoryService()
    return _history_service


def get_history_manager() -> HistoryService:
    """Legacy alias. Returns the new HistoryService singleton."""
    return get_history_service()


def get_trend_service() -> TrendService:
    """Lazily load and return the TrendService singleton."""
    global _trend_service
    if _trend_service is None:
        from week6.services.trend_service import TrendService
        settings = get_settings()
        _trend_service = TrendService(mock_mode=settings.model.global_mock)
    return _trend_service


def get_state_manager() -> StateManager:
    """Lazily load and return the StateManager singleton."""
    global _state_manager
    if _state_manager is None:
        from week6.services.state_manager import StateManager
        settings = get_settings()
        _state_manager = StateManager(mock_mode=settings.model.global_mock)
    return _state_manager


# Week 2 Generator singleton holders
_sdxl_generator = None


def get_sdxl_generator():
    """Lazily load and return the Week 2 FashionSDXLGenerator singleton."""
    global _sdxl_generator
    if _sdxl_generator is None:
        from src.generation.generator.sdxl_generator import FashionSDXLGenerator
        settings = get_settings()
        _sdxl_generator = FashionSDXLGenerator(
            model_id=settings.model.sdxl_model_id,
            device="cpu" if settings.model.global_mock else "auto",
        )
    return _sdxl_generator


# Week 3 ControlNet singleton holders
_controlnet_engine = None


def get_controlnet_engine():
    """Lazily load and return the Week 3 FashionControlNetEngine singleton."""
    global _controlnet_engine
    if _controlnet_engine is None:
        from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine
        settings = get_settings()
        _controlnet_engine = FashionControlNetEngine(mock=settings.model.global_mock)
    return _controlnet_engine


# Week 4 LoRA singleton holders
_lora_inference_system = None


def get_lora_inference_system():
    """Lazily load and return the Week 4 LoraInferenceSystem singleton."""
    global _lora_inference_system
    if _lora_inference_system is None:
        from src.lora.inference.lora_inference import LoraInferenceSystem
        from week7.backend.configs.config import get_settings
        settings = get_settings()
        _lora_inference_system = LoraInferenceSystem(
            device="cpu" if settings.model.global_mock else "cuda",
            dry_run=settings.model.global_mock
        )
        # Pre-populate registry with dummy adapter weights if empty
        if not _lora_inference_system.registry.models:
            import tempfile
            from pathlib import Path
            temp_dir = Path(tempfile.gettempdir()) / "mock_loras"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for brand in ["nike", "gucci", "zara", "h&m"]:
                mock_file = temp_dir / f"{brand}_mock.safetensors"
                if not mock_file.exists():
                    mock_file.write_bytes(b"mock safetensors content")
                _lora_inference_system.registry.register_model(
                    brand=brand,
                    model_path=mock_file,
                    metadata={"style_description": f"Mock adapter for {brand}"}
                )
    return _lora_inference_system


# Redis singleton holders
_redis_manager = None


def get_redis_manager() -> RedisManager:
    """Lazily load and return the RedisManager singleton."""
    global _redis_manager
    if _redis_manager is None:
        from week7.backend.services.redis_manager import RedisManager
        _redis_manager = RedisManager()
        _redis_manager.connect()
    return _redis_manager


def get_task_tracker() -> TaskTracker:
    """Lazily load and return the TaskTracker singleton."""
    global _task_tracker
    if _task_tracker is None:
        from week7.backend.services.task_tracker import TaskTracker
        _task_tracker = TaskTracker()
    return _task_tracker


def get_nsfw_filter() -> NSFWFilter:
    """Lazily load and return the NSFWFilter singleton."""
    global _nsfw_filter
    if _nsfw_filter is None:
        from week7.backend.middleware.nsfw_filter import NSFWFilter
        settings = get_settings()
        _nsfw_filter = NSFWFilter(mock_mode=settings.model.global_mock)
    return _nsfw_filter


def get_watermark_service() -> WatermarkService:
    """Lazily load and return the WatermarkService singleton."""
    global _watermark_service
    if _watermark_service is None:
        from week7.backend.services.watermark import WatermarkService
        _watermark_service = WatermarkService()
    return _watermark_service




