from __future__ import annotations

import base64
import io
from typing import Any, Dict
from PIL import Image
from week7.backend.workers.celery_app import celery_app
from week7.backend.services.controlnet_service import ControlNetService
from week7.backend.api.dependencies import get_task_tracker


def _b64_to_pil(b64_str: str) -> Image.Image:
    """Helper to convert base64 image data back to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


@celery_app.task(bind=True, name="tasks.generate_from_sketch_task")
def generate_from_sketch_task(
    self,
    prompt: str,
    sketch_image_b64: str,
    negative_prompt: str = "",
    control_strength: float = 0.7,
    seed: int = -1
) -> Dict[str, Any]:
    """Background task for sketch-conditioned design generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        img = _b64_to_pil(sketch_image_b64)
        svc = ControlNetService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.generate_from_sketch(
            prompt=prompt,
            sketch_image=img,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.generate_from_sketch_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.generate_from_pose_task")
def generate_from_pose_task(
    self,
    prompt: str,
    pose_image_b64: str,
    negative_prompt: str = "",
    control_strength: float = 0.7,
    seed: int = -1
) -> Dict[str, Any]:
    """Background task for pose-conditioned design generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        img = _b64_to_pil(pose_image_b64)
        svc = ControlNetService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.generate_from_pose(
            prompt=prompt,
            pose_image=img,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.generate_from_pose_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.generate_from_depth_task")
def generate_from_depth_task(
    self,
    prompt: str,
    depth_image_b64: str,
    negative_prompt: str = "",
    control_strength: float = 0.7,
    seed: int = -1
) -> Dict[str, Any]:
    """Background task for depth-conditioned design generation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        img = _b64_to_pil(depth_image_b64)
        svc = ControlNetService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.generate_from_depth(
            prompt=prompt,
            depth_image=img,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.generate_from_depth_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc
