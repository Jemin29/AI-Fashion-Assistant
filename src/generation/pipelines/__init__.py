"""week2/pipelines/__init__.py — Pipelines package."""

from src.generation.pipelines.base_pipeline import BasePipeline, PipelineRunResult
from src.generation.pipelines.text2image_pipeline import Text2ImagePipeline
from src.generation.pipelines.batch_pipeline import BatchPipeline, BatchItem, BatchResult
from src.generation.pipelines.fashion_generation_pipeline import (
    FashionGenerationPipeline,
    PipelineResult,
    PipelineConfig,
    ItemResult,
    StageResult,
)

__all__ = [
    # ── Existing pipelines ────────────────────────────────────────────────
    "BasePipeline",
    "PipelineRunResult",
    "Text2ImagePipeline",
    "BatchPipeline",
    "BatchItem",
    "BatchResult",
    # ── Master fashion pipeline ───────────────────────────────────────────
    "FashionGenerationPipeline",
    "PipelineResult",
    "PipelineConfig",
    "ItemResult",
    "StageResult",
]
