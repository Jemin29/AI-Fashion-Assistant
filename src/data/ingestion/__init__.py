"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/ingestion/__init__.py — Ingestion Sub-Package
=============================================================================
Exports all dataset ingesters and both full ingestion pipelines.

Usage:
    # FashionGen full pipeline (HDF5):
    from src.data.ingestion import FashionGenLoader
    result = FashionGenLoader().run(max_records=500)

    # DeepFashion full pipeline (TXT annotations + image folder):
    from src.data.ingestion import DeepFashionLoader
    result = DeepFashionLoader().run(split="train", max_records=500)

    # Individual FashionGen layers:
    from src.data.ingestion import (
        FashionGenExtractor, FashionGenTransformer,
        FashionGenValidator, FashionGenWriter,
    )

    # Individual DeepFashion layers:
    from src.data.ingestion import (
        DeepFashionAnnotationParser, DeepFashionExtractor,
        DeepFashionTransformer, DeepFashionValidator, DeepFashionWriter,
    )

    # Data models:
    from src.data.ingestion import (
        FashionGenRecord, PipelineStats,
        DeepFashionRecord, RawDeepFashionRecord, DFPipelineStats,
    )
=============================================================================
"""

# ── Legacy low-level streamers ─────────────────────────────────────────────────
from src.data.ingestion.fashiongen_ingester  import FashionGenIngester
from src.data.ingestion.deepfashion_ingester import DeepFashionIngester

# ── FashionGen production pipeline ─────────────────────────────────────────────
from src.data.ingestion.fashiongen_loader import (
    FashionGenLoader,
    FashionGenExtractor,
    FashionGenTransformer,
    FashionGenValidator,
    FashionGenWriter,
    FashionGenRecord,
    RawFashionGenRecord,
    PipelineStats,
)

# ── DeepFashion production pipeline ────────────────────────────────────────────
from src.data.ingestion.deepfashion_loader import (
    DeepFashionLoader,
    DeepFashionAnnotationParser,
    DeepFashionExtractor,
    DeepFashionTransformer,
    DeepFashionValidator,
    DeepFashionWriter,
    DeepFashionRecord,
    RawDeepFashionRecord,
    DFPipelineStats,
)

__all__ = [
    # ── Legacy streamers ──────────────────────────────────────────────────────
    "FashionGenIngester",
    "DeepFashionIngester",
    # ── FashionGen pipeline ───────────────────────────────────────────────────
    "FashionGenLoader",
    "FashionGenExtractor",
    "FashionGenTransformer",
    "FashionGenValidator",
    "FashionGenWriter",
    "FashionGenRecord",
    "RawFashionGenRecord",
    "PipelineStats",
    # ── DeepFashion pipeline ──────────────────────────────────────────────────
    "DeepFashionLoader",
    "DeepFashionAnnotationParser",
    "DeepFashionExtractor",
    "DeepFashionTransformer",
    "DeepFashionValidator",
    "DeepFashionWriter",
    "DeepFashionRecord",
    "RawDeepFashionRecord",
    "DFPipelineStats",
]
