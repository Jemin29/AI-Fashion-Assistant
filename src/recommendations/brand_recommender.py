"""
week5/recommendations/brand_recommender.py
=========================================
Fashion Brand Recommendation Engine.
Provides brand recommendations matching user style profiles using rule-based mappings,
semantic vector similarity searches over brand knowledge, and personalized blending.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.vector_db.chromadb_manager import ChromaDbManager


class BrandRecommender:
    """
    Suggests brands based on user style profiles (e.g. favorite style, color, fit)
    using rule-based matches, semantic similarity lookups, and personalized blending.
    """

    # Expert-defined style to brand mappings
    BRAND_RULES = {
        "streetwear": ["Nike", "Supreme", "Stussy"],
        "luxury": ["Gucci", "Prada", "Louis Vuitton"],
        "athleisure": ["Nike", "Adidas", "Puma"],
        "minimalist": ["Zara", "Uniqlo", "COS"],
        "vintage": ["Levi's", "Champion", "Carhartt"],
        "casual": ["H&M", "Zara", "Gap"]
    }

    # Absolute fallback list
    DEFAULT_RECOMMENDATIONS = ["Nike", "Zara", "Gucci", "H&M"]

    def __init__(
        self,
        retriever: FashionRetriever,
        db_manager: ChromaDbManager
    ) -> None:
        """
        Initialize the Brand Recommender.

        Parameters
        ----------
        retriever : FashionRetriever
            Relevance retrieval engine.
        db_manager : ChromaDbManager
            ChromaDB interface manager.
        """
        self.retriever = retriever
        self.db_manager = db_manager
        logger.info("BrandRecommender successfully initialized.")

    def recommend_by_rules(self, profile: Dict[str, Any]) -> List[str]:
        """
        Recommend brands using predefined rule-based pairings.

        Parameters
        ----------
        profile : Dict[str, Any]
            Dict containing style preferences (e.g., "favorite_style", "style").

        Returns
        -------
        List[str]
            List of rule-based recommended brands.
        """
        # Support both 'favorite_style' and 'style' keys
        style = profile.get("favorite_style", profile.get("style", "")).strip().lower()

        if style in self.BRAND_RULES:
            logger.info(f"Rule-based brand match found for style '{style}'.")
            return self.BRAND_RULES[style]

        # Check sub-strings or partial matches in the rules keys
        for key, brands in self.BRAND_RULES.items():
            if key in style or style in key:
                logger.info(f"Partial rule-based brand match found for style '{style}' -> '{key}'.")
                return brands

        logger.info("No brand rules matched. Yielding default recommendations.")
        return self.DEFAULT_RECOMMENDATIONS

    def recommend_by_similarity(
        self,
        profile: Dict[str, Any],
        n_results: int = 5
    ) -> List[str]:
        """
        Recommend brands using semantic vector similarity search over brand knowledge.

        Parameters
        ----------
        profile : Dict[str, Any]
            User style profile dictionary.
        n_results : int
            Number of results to return.

        Returns
        -------
        List[str]
            List of matching brand names retrieved from the database.
        """
        # Formulate query text from profile key-value values
        query_parts = []
        for k, v in profile.items():
            if v and isinstance(v, (str, int, float)):
                query_parts.append(str(v))
            elif v and isinstance(v, list):
                query_parts.extend([str(item) for item in v])
        
        query = " ".join(query_parts).strip()

        if not query:
            return self.DEFAULT_RECOMMENDATIONS[:n_results]

        logger.info(f"Similarity-based brand lookup for query: '{query}'")

        # Query the database 'brand_knowledge' collection
        results = self.retriever.retrieve(
            query=query,
            search_type="brand",
            collection_name="brand_knowledge",
            n_results=n_results
        )

        brands = []
        for item in results:
            meta = item.get("metadata", {})
            # Extract brand name from metadata keys
            brand_name = meta.get("brand", meta.get("name", meta.get("category", "")))
            if not brand_name:
                doc = item.get("document", "")
                # Parse brand name from document text (take the first word or before period)
                brand_name = doc.split(".")[0].strip()
                # If still empty or too long, split by spaces and take the first token
                if " " in brand_name and len(brand_name) > 30:
                    brand_name = brand_name.split(" ")[0].strip()

            if brand_name and brand_name not in brands:
                brands.append(brand_name)

        # Fallback if database query returned no brands
        if not brands:
            return self.DEFAULT_RECOMMENDATIONS[:n_results]

        return brands[:n_results]

    def recommend_personalized(
        self,
        profile: Dict[str, Any],
        n_results: int = 5
    ) -> List[str]:
        """
        Generate personalized brand recommendations blending rules and similarity lookups.

        Parameters
        ----------
        profile : Dict[str, Any]
        n_results : int

        Returns
        -------
        List[str]
            Blended, de-duplicated brand recommendation list.
        """
        logger.info(f"Generating personalized brand recommendations for profile: {profile}")

        # 1. Gather candidates from rules
        rules_recs = self.recommend_by_rules(profile)

        # 2. Gather candidates from similarity database search
        similarity_recs = self.recommend_by_similarity(profile, n_results=n_results)

        # 3. Blend lists, rules first, similarity second, removing duplicates
        blended_recs = []
        for brand in rules_recs:
            if brand not in blended_recs:
                blended_recs.append(brand)

        for brand in similarity_recs:
            if brand not in blended_recs:
                blended_recs.append(brand)

        return blended_recs[:n_results]
