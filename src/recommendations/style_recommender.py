"""
week5/recommendations/style_recommender.py
=========================================
Fashion Style Recommendation System.
Combines rule-based styling logic, semantic similarity lookups,
and unified personalized matching algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.vector_db.chromadb_manager import ChromaDbManager


class StyleRecommender:
    """
    Suggests fashion styles based on user preferences (favorite style, favorite color)
    using rule-based matches, semantic similarity lookups, and personalized blending.
    """

    # Expert-defined style pairing rules
    STYLE_RULES = {
        ("streetwear", "black"): ["Techwear", "Urban Minimal", "Oversized Casual"],
        ("streetwear", "white"): ["Minimalist Street", "Skater Casual"],
        ("athleisure", "grey"): ["Sporty Minimal", "Active Lounge"],
        ("vintage", "brown"): ["Retro Americana", "Bohemian Indie"],
        ("luxury", "gold"): ["Baroque Couture", "Glamour Elegance"]
    }

    # Fallback recommendations by style category
    STYLE_FALLBACKS = {
        "streetwear": ["Oversized Casual", "Urban Minimal", "Skater Casual"],
        "athleisure": ["Sporty Minimal", "Active Lounge"],
        "vintage": ["Retro Americana", "Bohemian Indie"],
        "luxury": ["Baroque Couture", "Opulent Classic"]
    }

    # Absolute fallback list
    DEFAULT_RECOMMENDATIONS = ["Classic Casual", "Smart Chic", "Minimalist Modern"]

    def __init__(
        self,
        retriever: FashionRetriever,
        db_manager: ChromaDbManager
    ) -> None:
        """
        Initialize the Style Recommender.

        Parameters
        ----------
        retriever : FashionRetriever
            Relevance retrieval engine.
        db_manager : ChromaDbManager
            ChromaDB interface manager.
        """
        self.retriever = retriever
        self.db_manager = db_manager
        logger.info("StyleRecommender successfully initialized.")

    def recommend_by_rules(self, preferences: Dict[str, str]) -> List[str]:
        """
        Recommend styles using predefined rule-based pairings.

        Parameters
        ----------
        preferences : Dict[str, str]
            Dict containing "favorite_style" and/or "favorite_color".

        Returns
        -------
        List[str]
            List of rule-based recommended styles.
        """
        fav_style = preferences.get("favorite_style", "").strip().lower()
        fav_color = preferences.get("favorite_color", "").strip().lower()

        # 1. Exact Match Rule
        rule_key = (fav_style, fav_color)
        if rule_key in self.STYLE_RULES:
            logger.info(f"Rule-based match found for style '{fav_style}' and color '{fav_color}'.")
            return self.STYLE_RULES[rule_key]

        # 2. Style-only Fallback Rule
        if fav_style in self.STYLE_FALLBACKS:
            logger.info(f"Fallback style match found for category '{fav_style}'.")
            return self.STYLE_FALLBACKS[fav_style]

        # 3. Default Fallback
        logger.info("No style rules matched. Yielding default recommendations.")
        return self.DEFAULT_RECOMMENDATIONS

    def recommend_by_similarity(
        self,
        preferences: Dict[str, str],
        n_results: int = 5
    ) -> List[str]:
        """
        Recommend styles using semantic vector similarity search over DB style items.

        Parameters
        ----------
        preferences : Dict[str, str]
        n_results : int

        Returns
        -------
        List[str]
            List of matching styles retrieved from the database.
        """
        # Formulate query text from preferences keys
        query_parts = []
        for k, v in preferences.items():
            if v:
                query_parts.append(f"{v}")
        query = " ".join(query_parts)

        if not query:
            return self.DEFAULT_RECOMMENDATIONS[:n_results]

        logger.info(f"Similarity-based style lookup for query: '{query}'")

        # Query the database 'fashion_styles' collection
        results = self.retriever.retrieve(
            query=query,
            search_type="style",
            collection_name="fashion_styles",
            n_results=n_results
        )

        styles = []
        for item in results:
            meta = item.get("metadata", {})
            # Extract style metadata name
            style_name = meta.get("style", meta.get("name", meta.get("category", "")))
            if not style_name:
                doc = item.get("document", "")
                # Parse style name from document text
                style_name = doc.split(".")[0].strip()
            
            if style_name and style_name not in styles:
                styles.append(style_name)

        # Fallback to defaults if database is empty/returns no styles
        if not styles:
            return self.DEFAULT_RECOMMENDATIONS[:n_results]

        return styles[:n_results]

    def recommend_personalized(
        self,
        preferences: Dict[str, str],
        n_results: int = 5
    ) -> List[str]:
        """
        Generate personalized style recommendations blending rules and similarity lookups.

        Parameters
        ----------
        preferences : Dict[str, str]
        n_results : int

        Returns
        -------
        List[str]
            Blended style recommendation list.
        """
        logger.info(f"Generating personalized recommendations for preferences: {preferences}")

        # 1. Gather candidates from rule-based engine
        rules_recs = self.recommend_by_rules(preferences)

        # 2. Gather candidates from similarity database search
        similarity_recs = self.recommend_by_similarity(preferences, n_results=n_results)

        # 3. Merge lists preserving order and removing duplicates
        blended_recs = []
        for style in rules_recs:
            if style not in blended_recs:
                blended_recs.append(style)

        for style in similarity_recs:
            if style not in blended_recs:
                blended_recs.append(style)

        return blended_recs[:n_results]
