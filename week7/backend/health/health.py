from __future__ import annotations

from fastapi import APIRouter, Depends
from week7.backend.api.dependencies import (
    get_generation_service,
    get_controlnet_service,
    get_lora_service,
    get_rag_service,
    get_state_manager,
    get_redis_manager,
)
from week6.services.generation_service import GenerationService
from week6.services.controlnet_service import ControlNetService
from week6.services.lora_service import LoRAService
from week6.services.rag_service import RAGService
from week6.services.state_manager import StateManager
from week7.backend.configs.config import get_settings

from week7.backend.monitoring.metrics import SystemMetricsCollector, ApplicationMetricsCollector
from week7.backend.monitoring.health_checks import HealthChecker

settings = get_settings()
router = APIRouter(prefix="/health", tags=["Diagnostics"])

@router.get("")
async def get_system_health(
    gen_svc: GenerationService = Depends(get_generation_service),
    cn_svc: ControlNetService = Depends(get_controlnet_service),
    lora_svc: LoRAService = Depends(get_lora_service),
    rag_svc: RAGService = Depends(get_rag_service),
    state_mgr: StateManager = Depends(get_state_manager),
    redis_mgr = Depends(get_redis_manager),
):
    """Aggregate health status and metrics reports across services and system resources."""
    redis_client = getattr(redis_mgr, "client", None)
    checker = HealthChecker(redis_client=redis_client, db_dir=settings.database.chroma_db_dir)
    app_collector = ApplicationMetricsCollector(redis_client=redis_client)
    
    redis_health = checker.check_redis()
    db_health = checker.check_database()
    sys_health = checker.check_system()
    
    system_status = "healthy"
    if (redis_health.get("status") == "unhealthy" or 
        db_health.get("status") == "unhealthy" or 
        sys_health.get("status") == "unhealthy"):
        system_status = "unhealthy"

    return {
        "success": True,
        "data": {
            "status": system_status,
            "app_name": settings.app_name,
            "version": settings.app_version,
            "mock_mode": settings.model.global_mock,
            "environment_mode": "mock" if settings.model.global_mock else "production",
            "system_resources": {
                "cpu_usage_percent": SystemMetricsCollector.get_cpu_usage(),
                "memory_usage": SystemMetricsCollector.get_memory_usage(),
                "api_process_memory_mb": SystemMetricsCollector.get_process_memory(),
            },
            "queues": {
                "celery_queue_length": app_collector.get_queue_length(),
            },
            "dependencies": {
                "redis": redis_health,
                "database": db_health,
                "system_headroom": sys_health,
            },
            "services": {
                "generation": gen_svc.health_check(),
                "controlnet": cn_svc.health_check(),
                "lora": lora_svc.health_check(),
                "rag": rag_svc.health_check(),
                "state_manager": state_mgr.health_check(),
            }
        }
    }
