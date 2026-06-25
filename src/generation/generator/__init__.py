"""week2/generator/__init__.py — Generator package."""

from src.generation.generator.sdxl_generator import (
    FashionSDXLGenerator,
    GenerationOutput,
    SDXLGenerator,
    GenerationResult,          # backward-compat alias
    SIZE_PRESETS,
    SCHEDULER_MAP,
)
from src.generation.generator.model_manager import ModelManager, LoadedModels
from src.generation.generator.scheduler_factory import SchedulerFactory
from src.generation.generator.image_processor import (
    generate_image_id,
    build_metadata,
    save_image,
    save_metadata_sidecar,
    resize_image,
    add_watermark,
    images_to_grid,
)
from src.generation.generator.batch_generator import (
    BatchGenerator,
    BatchItem,
    ItemResult,
    BatchReport,
    BatchGenerationConfig,
    run_batch_from_csv,
    run_batch_from_list,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MAX_RETRIES,
    CSV_COLUMN_ALIASES,
)

__all__ = [
    # ── Primary public API ────────────────────────────────────────────────
    "FashionSDXLGenerator",
    "GenerationOutput",
    # ── Batch generation ──────────────────────────────────────────────────
    "BatchGenerator",
    "BatchItem",
    "ItemResult",
    "BatchReport",
    "BatchGenerationConfig",
    "run_batch_from_csv",
    "run_batch_from_list",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MAX_RETRIES",
    "CSV_COLUMN_ALIASES",
    # ── Legacy / internal ─────────────────────────────────────────────────
    "SDXLGenerator",
    "GenerationResult",
    # ── Constants ─────────────────────────────────────────────────────────
    "SIZE_PRESETS",
    "SCHEDULER_MAP",
    # ── Supporting classes ────────────────────────────────────────────────
    "ModelManager",
    "LoadedModels",
    "SchedulerFactory",
    # ── Image processor utilities ─────────────────────────────────────────
    "generate_image_id",
    "build_metadata",
    "save_image",
    "save_metadata_sidecar",
    "resize_image",
    "add_watermark",
    "images_to_grid",
]
