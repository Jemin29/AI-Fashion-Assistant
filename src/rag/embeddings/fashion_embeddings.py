"""
week5/embeddings/fashion_embeddings.py
======================================
Fashion Embedding Engine for Week 5.
Generates, saves, and loads dense representations for styles, trends, brands,
and recommendations using the BAAI/bge-small-en-v1.5 pre-trained model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from src.utils.config_manager import EmbeddingConfig, get_default_config
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase, KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder, TrendItem


class FashionEmbeddingEngine:
    """
    Pipeline engine coordinating dense representation modeling for fashion styles,
    trends, brands, and recommendation models using the BAAI/bge-small-en-v1.5 model.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        kb: Optional[FashionKnowledgeBase] = None,
        trend_builder: Optional[TrendDatasetBuilder] = None,
        force_mock: bool = False
    ) -> None:
        """
        Initialize the Fashion Embedding Engine.

        Parameters
        ----------
        config : EmbeddingConfig, optional
        kb : FashionKnowledgeBase, optional
        trend_builder : TrendDatasetBuilder, optional
        force_mock : bool
        """
        if config is None:
            self.config = get_default_config().embeddings
        else:
            self.config = config

        # Explicitly configure the required BAAI/bge-small-en-v1.5 model
        self.config.model_name = "BAAI/bge-small-en-v1.5"

        self.kb = kb or FashionKnowledgeBase()
        self.trend_builder = trend_builder or TrendDatasetBuilder()
        self.embedder = EmbeddingsGenerator(config=self.config, force_mock=force_mock)
        
        # In-memory mapping of item ID to numpy embedding vector
        self.embeddings_dict: Dict[str, np.ndarray] = {}

    def generate_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Compile and embed fashion styles, trends, brands, and recommendations.

        Returns
        -------
        dict mapping item ID (str) to dense vector (np.ndarray)
        """
        logger.info("Gathering fashion objects for embedding generation...")

        # 1. Gather styles and brands from knowledge base
        styles = self.kb.list_items(category="fashion_styles")
        brands = self.kb.list_items(category="brand_profiles")

        # 2. Gather trends and forecasts (recommendations) from trend dataset
        trends = [
            t for t in self.trend_builder.list_trends()
            if t.category in {"streetwear", "luxury", "seasonal", "color", "fabric"}
        ]
        recommendations = self.trend_builder.list_trends(category="forecast")

        all_items: List[Union[KnowledgeItem, TrendItem]] = []
        all_items.extend(styles)
        all_items.extend(brands)
        all_items.extend(trends)
        all_items.extend(recommendations)

        if not all_items:
            logger.warning("No items found to generate embeddings for.")
            self.embeddings_dict = {}
            return {}

        logger.info(
            f"Generating embeddings using '{self.config.model_name}' for StyleCount: {len(styles)}, "
            f"BrandCount: {len(brands)}, TrendCount: {len(trends)}, RecCount: {len(recommendations)}"
        )

        vectors = self.embedder.embed_items(all_items)

        # Store in memory dict
        self.embeddings_dict = {}
        for item, vector in zip(all_items, vectors):
            self.embeddings_dict[item.id] = vector

        logger.success(f"Generated {len(self.embeddings_dict)} fashion embeddings.")
        return self.embeddings_dict

    def save_embeddings(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Serialize and save generated embeddings to disk in compressed .npz format.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to write the .npz archive. Defaults to outputs/embeddings/fashion_embeddings.npz.
        """
        if not self.embeddings_dict:
            logger.warning("Embeddings cache is empty. Run generate_embeddings() first.")
            return

        if output_path:
            path = Path(output_path).resolve()
        else:
            path = Path("outputs/embeddings/fashion_embeddings.npz").resolve()

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save numpy arrays as a compressed zip archive
            np.savez_compressed(str(path), **self.embeddings_dict)
            logger.success(f"Saved fashion embeddings archive to: {path}")
        except Exception as err:
            logger.error(f"Failed to serialize fashion embeddings to {path}: {err}")
            raise

    def load_embeddings(self, input_path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
        """
        Load and deserialize fashion embeddings from disk.

        Parameters
        ----------
        input_path : str or Path, optional
            Path to read the .npz archive. Defaults to outputs/embeddings/fashion_embeddings.npz.

        Returns
        -------
        dict containing loaded embeddings mapping
        """
        if input_path:
            path = Path(input_path).resolve()
        else:
            path = Path("outputs/embeddings/fashion_embeddings.npz").resolve()

        if not path.exists():
            logger.warning(f"Embedding file not found at: {path}. Returning empty dictionary.")
            self.embeddings_dict = {}
            return {}

        try:
            with np.load(str(path)) as data:
                # np.load returns a lazy dict-like object; copy keys out to memory dict
                self.embeddings_dict = {key: data[key].copy() for key in data.files}
            logger.success(f"Loaded {len(self.embeddings_dict)} fashion embeddings from: {path}")
            return self.embeddings_dict
        except Exception as err:
            logger.error(f"Failed to parse fashion embeddings archive {path}: {err}")
            self.embeddings_dict = {}
            raise
