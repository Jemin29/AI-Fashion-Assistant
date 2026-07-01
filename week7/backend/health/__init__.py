from __future__ import annotations

from fastapi import APIRouter
from week7.backend.health.health import router as health_router
from week7.backend.health.readiness import router as readiness_router
from week7.backend.health.liveness import router as liveness_router

router = APIRouter()
router.include_router(health_router)
router.include_router(readiness_router)
router.include_router(liveness_router)
