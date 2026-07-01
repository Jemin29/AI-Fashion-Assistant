from __future__ import annotations

import uuid
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_task_tracker
from week7.backend.services.task_tracker import TaskTracker
from week7.backend.workers.celery_app import celery_app

# Import Celery task wrappers
from week7.backend.workers.generation_worker import generate_image_task
from week7.backend.workers.controlnet_worker import (
    generate_from_sketch_task,
    generate_from_pose_task,
    generate_from_depth_task,
)
from week7.backend.workers.lora_worker import (
    generate_lora_style_task,
    mix_styles_task,
)
from week7.backend.workers.rag_worker import (
    answer_question_task,
    recommend_styles_task,
    semantic_search_task,
    explain_trend_task,
)

router = APIRouter(prefix="/task", tags=["Task Manager"])


class StartTaskRequest(BaseModel):
    task_type: str = Field(..., description="Type of task: generation, sketch, pose, depth, lora, mix, ask, recommend, search, trend")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the task")


@router.post("/start", status_code=status.HTTP_201_CREATED)
def start_task(request: StartTaskRequest, tracker: TaskTracker = Depends(get_task_tracker)):
    """Dispatch a background task asynchronously and initialize metadata tracking."""
    task_type = request.task_type.lower()
    payload = request.payload

    # Pre-generate task_id so we can initialize tracking before dispatching Celery tasks.
    # This prevents eager mode (which runs synchronously) from overwriting completed states.
    task_id = str(uuid.uuid4())
    tracker.init_task(task_id, task_type)

    try:
        if task_type == "generation":
            generate_image_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "negative_prompt": payload.get("negative_prompt", ""),
                    "seed": payload.get("seed", -1),
                    "cfg": payload.get("cfg", 7.5),
                    "resolution": payload.get("resolution", "1024x1024"),
                },
                task_id=task_id
            )
        elif task_type == "sketch":
            generate_from_sketch_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "sketch_image_b64": payload.get("sketch_image_b64", ""),
                    "negative_prompt": payload.get("negative_prompt", ""),
                    "control_strength": payload.get("control_strength", 0.7),
                    "seed": payload.get("seed", -1),
                },
                task_id=task_id
            )
        elif task_type == "pose":
            generate_from_pose_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "pose_image_b64": payload.get("pose_image_b64", ""),
                    "negative_prompt": payload.get("negative_prompt", ""),
                    "control_strength": payload.get("control_strength", 0.7),
                    "seed": payload.get("seed", -1),
                },
                task_id=task_id
            )
        elif task_type == "depth":
            generate_from_depth_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "depth_image_b64": payload.get("depth_image_b64", ""),
                    "negative_prompt": payload.get("negative_prompt", ""),
                    "control_strength": payload.get("control_strength", 0.7),
                    "seed": payload.get("seed", -1),
                },
                task_id=task_id
            )
        elif task_type == "lora":
            generate_lora_style_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "brand": payload.get("brand", ""),
                    "lora_scale": payload.get("lora_scale", 1.0),
                    "seed": payload.get("seed", None),
                },
                task_id=task_id
            )
        elif task_type == "mix":
            mix_styles_task.apply_async(
                kwargs={
                    "prompt": payload.get("prompt", ""),
                    "brand_weights": payload.get("brand_weights", {}),
                    "seed": payload.get("seed", None),
                },
                task_id=task_id
            )
        elif task_type == "ask":
            answer_question_task.apply_async(
                kwargs={
                    "question": payload.get("question", "")
                },
                task_id=task_id
            )
        elif task_type == "recommend":
            recommend_styles_task.apply_async(
                kwargs={
                    "preferences": payload.get("preferences", {}),
                    "n": payload.get("n", 5),
                },
                task_id=task_id
            )
        elif task_type == "search":
            semantic_search_task.apply_async(
                kwargs={
                    "query": payload.get("query", ""),
                    "n_results": payload.get("n_results", 5),
                },
                task_id=task_id
            )
        elif task_type == "trend":
            explain_trend_task.apply_async(
                kwargs={
                    "trend_name": payload.get("trend_name", "")
                },
                task_id=task_id
            )
        else:
            tracker.delete_task(task_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported task type: {task_type}"
            )
    except HTTPException:
        raise
    except Exception as exc:
        tracker.delete_task(task_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue celery task: {str(exc)}"
        )

    # Return PENDING initially; the synchronous execution in Eager mode
    # will have already completed this task when queried next by the user.
    return {
        "task_id": task_id,
        "task_type": task_type,
        "status": "PENDING"
    }


@router.get("/{id}")
def get_task_status(id: str, tracker: TaskTracker = Depends(get_task_tracker)):
    """Retrieve the tracking metadata, status, progress, and execution time of a task."""
    meta = tracker.get_task(id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {id} not found."
        )
    return meta


@router.delete("/{id}")
def cancel_task(id: str, tracker: TaskTracker = Depends(get_task_tracker)):
    """Revoke/terminate an active background task and clear its tracking metadata."""
    # Revoke in Celery
    try:
        celery_app.control.revoke(id, terminate=True)
    except Exception:
        # Ignore if celery is offline/not running (e.g. mock mode)
        pass

    # Update state to revoked in tracker, then remove or return success
    meta = tracker.get_task(id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {id} not found."
        )

    tracker.delete_task(id)
    return {
        "task_id": id,
        "status": "REVOKED",
        "detail": "Task revoked and tracking data cleared."
    }
