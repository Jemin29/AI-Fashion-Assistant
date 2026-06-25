"""
week5/embeddings/embeddings_generator.py
========================================
Dense vector embeddings generator for Week 5 RAG system.
Wraps SentenceTransformers with an offline fallback mechanism using
deterministic hash-seeded pseudo-embeddings.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Union

import numpy as np
from loguru import logger

from src.utils.config_manager import EmbeddingConfig, get_default_config
from src.data.knowledge_base.fashion_knowledge_base import KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendItem

# Dynamic sentence-transformers import to support flexible environment setups
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embedding generator will run in fallback mock mode.")


class EmbeddingsGenerator:
    """
    Generates dense embeddings for text queries and Fashion database items.
    Supports SentenceTransformers and offline fallback modes.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        force_mock: bool = False
    ) -> None:
        """
        Initialize the Embeddings Generator.

        Parameters
        ----------
        config : EmbeddingConfig, optional
            Config parameters. If omitted, uses default config.
        force_mock : bool
            If True, bypasses SentenceTransformer loading and uses the mock embedder.
        """
        if config is None:
            self.config = get_default_config().embeddings
        else:
            self.config = config

        self.force_mock = force_mock
        self.model: Optional[SentenceTransformer] = None
        self.is_mock_mode = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the transformer model or configure mock mode."""
        if self.force_mock or not _SENTENCE_TRANSFORMERS_AVAILABLE:
            self._configure_mock_mode("Forced mock or library missing.")
            return

        try:
            logger.info(f"Loading SentenceTransformer model '{self.config.model_name}' on device '{self.config.device}'...")
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=self.config.cache_folder
            )
            self.is_mock_mode = False
            logger.success(f"Successfully loaded model '{self.config.model_name}'")
        except Exception as err:
            logger.warning(
                f"Failed to load SentenceTransformer model '{self.config.model_name}': {err}. "
                "Switching to deterministic mock embedding fallback."
            )
            self._configure_mock_mode(str(err))

    def _configure_mock_mode(self, reason: str) -> None:
        """Set up parameters for mock embedder."""
        self.is_mock_mode = True
        self.model = None
        logger.info(f"Configured deterministic mock embeddings (Dimension: {self.config.dimension}) | Reason: {reason}")

    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate deterministic mock embeddings based on text hashes.
        Uses SHA-256 to seed numpy RNG and outputs normalized unit vectors.
        """
        embeddings = []
        for text in texts:
            # Deterministic seed from text content
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            seed = int(sha, 16) % (2**32)
            
            # Construct deterministic normal distribution vector
            rng = np.random.default_rng(seed)
            vector = rng.normal(0.0, 1.0, self.config.dimension)
            
            # Normalize vector to unit L2 length
            norm = np.linalg.norm(vector)
            if norm > 1e-9:
                vector = vector / norm
                
            embeddings.append(vector)
            
        return np.array(embeddings, dtype=np.float32)

    # ── Generation APIs ──────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a dense embedding vector for a single text string.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray (1D array of float32)
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embedding vectors for a list of text strings.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray (2D array of shape [num_texts, dimension] and type float32)
        """
        if not texts:
            return np.zeros((0, self.config.dimension), dtype=np.float32)

        if self.is_mock_mode or self.model is None:
            return self._generate_mock_embeddings(texts)

        try:
            # Generate real embeddings via SentenceTransformers
            # convert_to_numpy=True ensures np.ndarray returns
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.config.device
            )
            # Ensure return is float32
            return np.array(embeddings, dtype=np.float32)
        except Exception as err:
            logger.error(f"Error during SentenceTransformer encoding: {err}. Falling back to mock generator.")
            return self._generate_mock_embeddings(texts)

    def embed_items(self, items: List[Union[KnowledgeItem, TrendItem]]) -> np.ndarray:
        """
        Generate embeddings for a list of structured knowledge base or trend items.

        Parameters
        ----------
        items : list of KnowledgeItem or TrendItem

        Returns
        -------
        np.ndarray (2D array of float32 embeddings)
        """
        texts = []
        for item in items:
            if hasattr(item, "category") and hasattr(item, "name") and hasattr(item, "content"):
                # KnowledgeItem structure
                tags_str = ", ".join(item.tags) if hasattr(item, "tags") else ""
                text = f"Name: {item.name}. Category: {item.category}. Content: {item.content}. Tags: {tags_str}."
            elif hasattr(item, "category") and hasattr(item, "name") and hasattr(item, "description"):
                # TrendItem structure
                text = f"Name: {item.name}. Category: {item.category}. Description: {item.description}."
            else:
                text = str(item)
            texts.append(text)
            
        return self.embed_batch(texts)
