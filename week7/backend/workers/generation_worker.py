from __future__ import annotations

from typing import Any, Dict
from week7.backend.workers.celery_app import celery_app
from week7.backend.services.generation_service import GenerationService


from week7.backend.api.dependencies import get_task_tracker

@celery_app.task(bind=True, name="tasks.generate_image_task")
def generate_image_task(
    self,
    prompt: str,
    negative_prompt: str = "",
    seed: int = -1,
    cfg: float = 7.5,
    resolution: str = "1024x1024"
) -> Dict[str, Any]:
    """Background task for text-to-image fashion design generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = GenerationService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            cfg=cfg,
            resolution=resolution
        )
        tracker.complete_task(task_id, res.data)
        return res.data
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.generate_image_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc
