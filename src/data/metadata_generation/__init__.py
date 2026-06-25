"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/metadata_generation/__init__.py — Metadata Sub-Package
=============================================================================
Public API for the automated fashion metadata generation engine.

Primary exports:

    from src.data.metadata_generation import MetadataGeneratorEngine
    from src.data.metadata_generation import MetadataResult, ExtractionResult

    # Quick usage:
    engine = MetadataGeneratorEngine()
    result = engine.generate("Black oversized hoodie with neon graphics")
    print(result.to_dict())

Engine layers:
  1. RuleBasedExtractor  — keyword/regex extraction (zero dependencies)
  2. NLPExtractor        — spaCy noun-chunk + POS analysis (optional)
  3. FallbackResolver    — cross-inference & taxonomy-locked defaults

Attributes extracted:
  style | category | season | gender | occasion | fit | pattern | color
=============================================================================
"""

from src.data.metadata_generation.metadata_generator import (
    # ── Orchestrator ──────────────────────────────────────────────────────────
    MetadataGeneratorEngine,
    # ── Data models ───────────────────────────────────────────────────────────
    MetadataResult,
    ExtractionResult,
    # ── Individual extraction layers (exposed for testing / composition) ──────
    RuleBasedExtractor,
    NLPExtractor,
    FallbackResolver,
    # ── Confidence tier constants ─────────────────────────────────────────────
    _CONF_EXACT,
    _CONF_PHRASE,
    _CONF_NLP,
    _CONF_CROSS_INFER,
    _CONF_DEFAULT,
    # ── Lexicons (exposed for extension / custom overrides) ───────────────────
    _CATEGORY_KEYWORDS,
    _COLOR_KEYWORDS,
    _STYLE_KEYWORDS,
    _SEASON_KEYWORDS,
    _FIT_KEYWORDS,
    _PATTERN_KEYWORDS,
    _GENDER_KEYWORDS,
    _OCCASION_KEYWORDS,
)

__all__ = [
    # Orchestrator
    "MetadataGeneratorEngine",
    # Data models
    "MetadataResult",
    "ExtractionResult",
    # Individual layers
    "RuleBasedExtractor",
    "NLPExtractor",
    "FallbackResolver",
    # Confidence tiers
    "_CONF_EXACT",
    "_CONF_PHRASE",
    "_CONF_NLP",
    "_CONF_CROSS_INFER",
    "_CONF_DEFAULT",
    # Lexicons
    "_CATEGORY_KEYWORDS",
    "_COLOR_KEYWORDS",
    "_STYLE_KEYWORDS",
    "_SEASON_KEYWORDS",
    "_FIT_KEYWORDS",
    "_PATTERN_KEYWORDS",
    "_GENDER_KEYWORDS",
    "_OCCASION_KEYWORDS",
]
