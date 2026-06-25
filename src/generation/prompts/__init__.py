"""week2/prompts/__init__.py — Prompts package."""

from src.generation.prompts.prompt_builder import PromptBuilder, BuiltPrompt
from src.generation.prompts.style_presets import (
    StylePreset,
    get_preset,
    list_presets,
    get_all_presets,
    register_preset,
)
from src.generation.prompts.negative_prompts import (
    get_base_negative,
    get_fashion_negative,
    get_full_negative,
    get_style_negative,
    format_negative,
    QUALITY_NEGATIVES,
    FASHION_NEGATIVES,
)
from src.generation.prompts.prompt_validator import PromptValidator, ValidationResult
from src.generation.prompts.prompt_templates import (
    # ── Core functions ────────────────────────────────────────────────────
    generate_prompt,
    generate_negative_prompt,
    prompt_enhancer,
    # ── Batch helpers ─────────────────────────────────────────────────────
    generate_prompt_pair,
    generate_batch_prompts,
    enhance_batch,
    # ── Inspection ────────────────────────────────────────────────────────
    explain_prompt,
    # ── Registry ──────────────────────────────────────────────────────────
    FashionTemplate,
    register_template,
    get_template,
    list_templates,
    list_template_details,
    # ── Constants ─────────────────────────────────────────────────────────
)

__all__ = [
    # ── PromptBuilder (structured assembly) ──────────────────────────────
    "PromptBuilder",
    "BuiltPrompt",
    # ── Style presets ─────────────────────────────────────────────────────
    "StylePreset",
    "get_preset",
    "list_presets",
    "get_all_presets",
    "register_preset",
    # ── Negative prompts ──────────────────────────────────────────────────
    "get_base_negative",
    "get_fashion_negative",
    "get_full_negative",
    "get_style_negative",
    "format_negative",
    "QUALITY_NEGATIVES",
    "FASHION_NEGATIVES",
    # ── Validator ─────────────────────────────────────────────────────────
    "PromptValidator",
    "ValidationResult",
    # ── Template engine (generate_prompt / generate_negative_prompt / prompt_enhancer)
    "generate_prompt",
    "generate_negative_prompt",
    "prompt_enhancer",
    "generate_prompt_pair",
    "generate_batch_prompts",
    "enhance_batch",
    "explain_prompt",
    "FashionTemplate",
    "register_template",
    "get_template",
    "list_templates",
    "list_template_details",
]
