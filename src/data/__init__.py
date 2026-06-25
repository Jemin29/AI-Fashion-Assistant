"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/__init__.py — Pipeline Package Root
=============================================================================
Exposes the top-level pipeline modules so they can be imported cleanly:

    from src.data.ingestion  import FashionGenLoader, DeepFashionLoader
    from src.data.schema     import UnifiedFashionItem, FashionDatasetBatch
    from src.data.knowledge_base import FashionDomainResearch

Week 1 scope: ingestion → schema unification → preprocessing →
              validation → metadata generation.

Sub-packages:
  ingestion/       FashionGen + DeepFashion ingestion pipelines
  schema/          Unified Pydantic v2 schema (UnifiedFashionItem)
  knowledge_base/  Fashion taxonomy knowledge base & domain research
  preprocessing/   Image and metadata preprocessing (Week 1 stubs)
  validation/      Dataset quality validation (Week 1 stubs)
  metadata_generation/ Automated metadata generation (Week 1 stubs)
=============================================================================
"""

# ── Package metadata ───────────────────────────────────────────────────────────
__version__     = "1.0.0"
__week__        = 1
__description__ = "Fashion Domain Research & Dataset Curation Pipeline"
__author__      = "Fashion AI Team"

# ── Schema sub-package — available immediately ─────────────────────────────────
from src.data.schema import (
    UnifiedFashionItem,
    FashionDatasetBatch,
    LandmarkPoint,
    BoundingBox,
    DatasetSource,
    GenderEnum,
    CategoryEnum,
    StyleEnum,
    FitEnum,
    SeasonEnum,
    OccasionEnum,
)

# ── Ingestion loaders — imported lazily (require h5py / dataset files) ─────────
# from src.data.ingestion import FashionGenLoader, DeepFashionLoader

# ── Knowledge base ─────────────────────────────────────────────────────────────
# from src.data.knowledge_base import FashionDomainResearch

# ── Future sub-packages (Week 2+) ─────────────────────────────────────────────
# from src.data.preprocessing.image_preprocessor import FashionPreprocessor
# from src.data.validation.data_validator import DataValidator
# from src.data.metadata_generation.metadata_generator import MetadataGenerator

__all__ = [
    "UnifiedFashionItem",
    "FashionDatasetBatch",
    "LandmarkPoint",
    "BoundingBox",
    "DatasetSource",
    "GenderEnum",
    "CategoryEnum",
    "StyleEnum",
    "FitEnum",
    "SeasonEnum",
    "OccasionEnum",
]
