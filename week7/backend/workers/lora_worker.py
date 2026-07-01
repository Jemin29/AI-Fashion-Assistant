from __future__ import annotations

from typing import Any, Dict, Optional
from week7.backend.workers.celery_app import celery_app
from week7.backend.services.lora_service import LoraService
from week7.backend.api.dependencies import get_task_tracker


@celery_app.task(bind=True, name="tasks.generate_lora_style_task")
def generate_lora_style_task(
    self,
    prompt: str,
    brand: str,
    lora_scale: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Background task for single brand style LoRA generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = LoraService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.generate_lora_style(
            prompt=prompt,
            brand=brand,
            lora_scale=lora_scale,
            seed=seed
        )
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.generate_lora_style_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.mix_styles_task")
def mix_styles_task(
    self,
    prompt: str,
    brand_weights: Dict[str, float],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Background task for multi-brand style mixing LoRA generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = LoraService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.mix_styles(
            prompt=prompt,
            brand_weights=brand_weights,
            seed=seed
        )
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.mix_styles_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc
