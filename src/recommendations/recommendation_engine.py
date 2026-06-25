"""
week5/recommendations/recommendation_engine.py
=============================================
Fashion Design Recommendation Engine for Week 5.
Implements diversity-penalized recommendations utilizing Maximal Marginal Relevance (MMR)
against user style preferences and database items.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from src.utils.config_manager import RecommendationConfig, get_default_config
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendItem
from src.rag.retrieval.hybrid_retriever import HybridRetriever


class RecommendationEngine:
    """
    Computes personalized recommendations based on style, fabric, and color profiles.
    Balances similarity against catalog diversity using a customizable diversity bias penalty.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        embedder: EmbeddingsGenerator,
        config: Optional[RecommendationConfig] = None
    ) -> None:
        """
        Initialize the Recommendation Engine.

        Parameters
        ----------
        retriever : HybridRetriever
        embedder : EmbeddingsGenerator
        config : RecommendationConfig, optional
        """
        self.retriever = retriever
        self.embedder = embedder

        if config is None:
            self.config = get_default_config().recommendations
        else:
            self.config = config

    def _build_preference_query(self, user_preferences: Dict[str, Any]) -> str:
        """Flatten dictionary of preferences into a retrieval query string."""
        parts = []
        for key, val in user_preferences.items():
            if isinstance(val, list):
                parts.append(" ".join(str(v) for v in val))
            elif isinstance(val, (str, int, float)):
                parts.append(str(val))
        return " ".join(parts)

    def recommend(
        self,
        user_preferences: Dict[str, Any],
        top_n: Optional[int] = None
    ) -> List[Tuple[Union[KnowledgeItem, TrendItem], float]]:
        """
        Recommend fashion items matching user preferences with diversity-based ranking (MMR).

        Parameters
        ----------
        user_preferences : dict
            Attributes such as styles, colors, fabrics, or raw query text.
        top_n : int, optional
            Number of recommendations to return. Defaults to config settings.

        Returns
        -------
        list of tuple of (item, blended_score)
        """
        max_rec = top_n or self.config.max_recommendations

        # 1. Compile preference query and retrieve candidates
        pref_query = self._build_preference_query(user_preferences)
        if not pref_query:
            return []

        # Retrieve up to 40 candidates to apply diversity selection on
        candidates_raw = self.retriever.retrieve(pref_query, top_k=40)
        
        # 2. Filter candidates based on similarity threshold
        candidates = [
            (item, score) for item, score in candidates_raw 
            if score >= self.config.similarity_threshold
        ]

        if not candidates:
            logger.info("No recommendation candidates passed similarity threshold criteria.")
            return []

        # 3. Generate embeddings for candidate items
        items = [c[0] for c in candidates]
        embeddings = self.embedder.embed_items(items)

        # 4. Maximal Marginal Relevance (MMR) Selection Loop
        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidates)))

        # Select first item (highest relevance)
        first_idx = remaining_indices[0]
        for idx in remaining_indices:
            if candidates[idx][1] > candidates[first_idx][1]:
                first_idx = idx

        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select subsequent items with diversity penalty
        diversity_bias = self.config.diversity_bias

        while len(selected_indices) < max_rec and remaining_indices:
            best_penalized_score = -999.0
            best_idx = -1

            for idx in remaining_indices:
                c_item, relevance = candidates[idx]
                c_emb = embeddings[idx]

                # Compute maximum similarity to already selected items
                max_sim = -1.0
                for sel_idx in selected_indices:
                    sel_emb = embeddings[sel_idx]
                    # Cosine similarity (since mock/real vectors are normalized)
                    sim = float(np.dot(c_emb, sel_emb))
                    if sim > max_sim:
                        max_sim = sim

                # Apply penalty: score = relevance - bias * max_sim
                penalized_score = relevance - (diversity_bias * max_sim)

                if penalized_score > best_penalized_score:
                    best_penalized_score = penalized_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        # Construct final recommendations list
        recommendations = [(candidates[idx][0], candidates[idx][1]) for idx in selected_indices]
        logger.info(f"Generated {len(recommendations)} diversified recommendations for user preferences.")
        return recommendations
