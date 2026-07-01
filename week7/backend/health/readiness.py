from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from week7.backend.api.dependencies import get_redis_manager
from week7.backend.configs.config import get_settings
from week7.backend.monitoring.health_checks import HealthChecker

settings = get_settings()
router = APIRouter(prefix="/readiness", tags=["Diagnostics"])

@router.get("")
async def get_readiness(redis_mgr=Depends(get_redis_manager)):
    """Verifies that all external dependencies are ready to accept traffic."""
    redis_client = getattr(redis_mgr, "client", None)
    checker = HealthChecker(redis_client=redis_client, db_dir=settings.database.chroma_db_dir)
    
    redis_health = checker.check_redis()
    db_health = checker.check_database()
    
    ready = redis_health.get("connected", False) and db_health.get("writable", False)
    
    payload = {
        "success": True,
        "status": "ready" if ready else "not_ready",
        "dependencies": {
            "redis": redis_health,
            "database": db_health
        }
    }
    
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload
        )
    return payload
