from __future__ import annotations

import logging
from celery import Celery
from week7.backend.configs.config import get_settings

logger = logging.getLogger("celery_app")

settings = get_settings()

# Construct the Redis URL broker & backend
password_part = f":{settings.redis.password}@" if settings.redis.password else ""
redis_url = f"redis://{password_part}{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"

celery_app = Celery(
    "fashion_ai_workers",
    broker=redis_url,
    backend=redis_url,
)

# Celery Configuration Settings
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # Results expire after 1 hour
)

# Eager mode for testing/fallback if Redis is disabled or global mock is enabled
if settings.model.global_mock or not settings.redis.enabled:
    logger.info("Celery configured in Eager (synchronous) execution mode.")
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )

# List of task modules to import
celery_app.conf.imports = [
    "week7.backend.workers.generation_worker",
    "week7.backend.workers.controlnet_worker",
    "week7.backend.workers.lora_worker",
    "week7.backend.workers.rag_worker",
]
