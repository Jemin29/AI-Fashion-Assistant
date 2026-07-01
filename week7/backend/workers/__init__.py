from __future__ import annotations

from week7.backend.workers.celery_app import celery_app
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

__all__ = [
    "celery_app",
    "generate_image_task",
    "generate_from_sketch_task",
    "generate_from_pose_task",
    "generate_from_depth_task",
    "generate_lora_style_task",
    "mix_styles_task",
    "answer_question_task",
    "recommend_styles_task",
    "semantic_search_task",
    "explain_trend_task",
]
