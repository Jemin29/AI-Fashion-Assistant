"""
week5/knowledge_base
====================
Fashion knowledge base, metadata tagging, and article parsing models.
"""
from __future__ import annotations

from src.data.knowledge_base.fashion_knowledge_base import (
    FashionKnowledgeBase,
    KnowledgeItem,
    VALID_CATEGORIES,
)
from src.data.knowledge_base.trend_dataset_builder import (
    TrendDatasetBuilder,
    TrendItem,
)
from src.data.knowledge_base.fashion_qa_dataset import (
    FashionQADatasetBuilder,
    FashionQARecord,
)

__all__ = [
    "FashionKnowledgeBase",
    "KnowledgeItem",
    "VALID_CATEGORIES",
    "TrendDatasetBuilder",
    "TrendItem",
    "FashionQADatasetBuilder",
    "FashionQARecord",
]

