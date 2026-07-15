"""
week5/rag/fashion_assistant.py
==============================
Context-Aware Fashion Assistant.
Combines RAG queries, style/brand recommendations, trend forecasting, and
fabric/style explanations into a unified conversational coordinator.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.utils.config_manager import Week5Config, get_default_config
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.vector_db.chromadb_manager import ChromaDbManager
from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.recommendations.style_recommender import StyleRecommender
from src.recommendations.brand_recommender import BrandRecommender
from src.recommendations.user_profile_manager import UserProfileManager, UserProfile
from src.trends.trend_forecaster import TrendForecaster
from src.rag.fashion_rag import FashionRAG


class FashionAssistant:
    """
    Context-aware AI Fashion Assistant coordinating RAG queries,
    style and brand recommendations, trend forecasting, and fabric/style explanations.
    """

    def __init__(
        self,
        config: Optional[Week5Config] = None,
        kb_path: Optional[str] = None,
        trend_db_path: Optional[str] = None,
        user_profile_db_path: Optional[str] = None,
        force_mock_embeddings: bool = False
    ) -> None:
        """
        Initialize the Fashion Assistant.

        Parameters
        ----------
        config : Week5Config, optional
        kb_path : str, optional
            Custom path to knowledge base JSON database.
        trend_db_path : str, optional
            Custom path to trend dataset JSON database.
        user_profile_db_path : str, optional
            Custom path to user profile database.
        force_mock_embeddings : bool
            Force embedder and databases to run in mock mode.
        """
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config

        # 1. Initialize RAG coordinator
        self.rag_coordinator = FashionRAG(
            config=self.config,
            kb_path=kb_path,
            trend_db_path=trend_db_path,
            force_mock_embeddings=force_mock_embeddings
        )

        # 2. Initialize ChromaDB Manager
        chroma_dir = None
        if hasattr(self.config.vector_db, "storage_path"):
            chroma_dir = str(Path(self.config.vector_db.storage_path).parent / "chromadb")
        
        self.db_manager = ChromaDbManager(
            persist_directory=chroma_dir,
            force_mock=force_mock_embeddings
        )

        # 3. Initialize Embeddings and Chroma-based Retriever
        self.embeddings_generator = EmbeddingsGenerator(
            config=self.config.embeddings,
            force_mock=force_mock_embeddings
        )
        self.chroma_retriever = FashionRetriever(
            embedder=self.embeddings_generator,
            db_manager=self.db_manager
        )

        # 4. Initialize Style and Brand Recommenders
        self.style_recommender = StyleRecommender(
            retriever=self.chroma_retriever,
            db_manager=self.db_manager
        )
        self.brand_recommender = BrandRecommender(
            retriever=self.chroma_retriever,
            db_manager=self.db_manager
        )

        # 5. Initialize User Profile Manager
        self.user_profile_manager = UserProfileManager(db_path=user_profile_db_path)

        # 6. Initialize Trend Forecaster
        self.trend_forecaster = TrendForecaster(
            analyzer=self.rag_coordinator.trend_analyzer,
            retriever=self.chroma_retriever
        )

        # 7. Seed ChromaDB from knowledge base to keep them synced
        self._seed_chromadb_from_kb()

        logger.success("FashionAssistant successfully initialized.")

    def _seed_chromadb_from_kb(self) -> None:
        """Seed the ChromaDB database collections with knowledge base & trend items."""
        logger.info("Synchronizing ChromaDB collections with JSON databases...")

        def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            import json
            sanitized = {}
            for k, v in meta.items():
                if isinstance(v, (list, tuple)):
                    sanitized[k] = ", ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    sanitized[k] = json.dumps(v)
                elif isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif v is None:
                    sanitized[k] = ""
                else:
                    sanitized[k] = str(v)
            return sanitized

        # Seed Styles
        styles = self.rag_coordinator.kb.list_items(category="fashion_styles")
        if styles:
            self.db_manager.insert_documents(
                collection_name="fashion_styles",
                ids=[item.id for item in styles],
                documents=[item.content for item in styles],
                metadatas=[_sanitize_metadata({
                    "name": item.name,
                    "category": item.category,
                    "tags": ", ".join(item.tags),
                    **item.metadata
                }) for item in styles]
            )

        # Seed Brands
        brands = self.rag_coordinator.kb.list_items(category="brand_profiles")
        if brands:
            self.db_manager.insert_documents(
                collection_name="brand_knowledge",
                ids=[item.id for item in brands],
                documents=[item.content for item in brands],
                metadatas=[_sanitize_metadata({
                    "brand": item.name,
                    "category": item.category,
                    "tags": ", ".join(item.tags),
                    **item.metadata
                }) for item in brands]
            )

        # Seed Trends
        trends = self.rag_coordinator.trend_dataset.list_trends()
        if trends:
            self.db_manager.insert_documents(
                collection_name="trends",
                ids=[item.id for item in trends],
                documents=[item.description for item in trends],
                metadatas=[_sanitize_metadata({
                    "name": item.name,
                    "category": item.category,
                    "popularity_score": item.popularity_score,
                    "growth_rate": item.growth_rate,
                    **item.metadata
                }) for item in trends]
            )
        logger.info("ChromaDB collections synchronized successfully.")

    def answer_question(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer general fashion questions using the RAG pipeline.

        Parameters
        ----------
        query : str
            User query text.
        user_id : str, optional

        Returns
        -------
        Dict[str, Any]
        """
        logger.info(f"FashionAssistant: answering question | '{query}'")
        if user_id:
            self.user_profile_manager.record_search(user_id, query)

        return self.rag_coordinator.query(query)

    def recommend_styles(
        self,
        preferences: Dict[str, str],
        user_id: Optional[str] = None,
        n_results: int = 3
    ) -> List[str]:
        """
        Recommend styles using predefined rules and similarity checks.

        Parameters
        ----------
        preferences : Dict[str, str]
        user_id : str, optional
        n_results : int

        Returns
        -------
        List[str]
        """
        logger.info(f"FashionAssistant: recommending styles | preferences={preferences}")
        recs = self.style_recommender.recommend_personalized(preferences, n_results=n_results)

        if user_id:
            pref_str = " ".join(f"{k}:{v}" for k, v in preferences.items())
            self.user_profile_manager.record_search(user_id, f"style_recommendation {pref_str}")
            self.user_profile_manager.record_recommendations(user_id, recs)

        return recs

    def recommend_brands(
        self,
        profile: Dict[str, Any],
        user_id: Optional[str] = None,
        n_results: int = 3
    ) -> List[str]:
        """
        Recommend brands matching the user's style preferences.

        Parameters
        ----------
        profile : Dict[str, Any]
        user_id : str, optional
        n_results : int

        Returns
        -------
        List[str]
        """
        logger.info(f"FashionAssistant: recommending brands | profile={profile}")
        recs = self.brand_recommender.recommend_personalized(profile, n_results=n_results)

        if user_id:
            pref_str = " ".join(f"{k}:{v}" for k, v in profile.items() if isinstance(v, (str, int, float)))
            self.user_profile_manager.record_search(user_id, f"brand_recommendation {pref_str}")
            self.user_profile_manager.record_recommendations(user_id, recs)

        return recs

    def explain_trend(self, trend_name: str) -> Dict[str, Any]:
        """
        Provide analysis and details for a fashion trend.

        Parameters
        ----------
        trend_name : str

        Returns
        -------
        Dict[str, Any]
        """
        logger.info(f"FashionAssistant: explaining trend | '{trend_name}'")
        
        # 1. Check Trend Dataset Builder JSON first
        trends = self.rag_coordinator.trend_dataset.list_trends()
        matched_trend = None
        for t in trends:
            if trend_name.lower() in t.name.lower() or t.name.lower() in trend_name.lower():
                matched_trend = t
                break

        desc = ""
        metadata = {}
        if matched_trend:
            desc = matched_trend.description
            metadata = {
                "name": matched_trend.name,
                "category": matched_trend.category,
                "popularity_score": matched_trend.popularity_score,
                "growth_rate": matched_trend.growth_rate,
                **matched_trend.metadata
            }
        else:
            # 2. Try querying ChromaDB trends collection
            matched = self.chroma_retriever.retrieve(query=trend_name, search_type="trend", n_results=1)
            if matched:
                temp_desc = matched[0].get("document", "")
                temp_meta = matched[0].get("metadata", {})
                temp_name = temp_meta.get("name", temp_meta.get("trend", ""))
                
                # Only use search results if there is naming or keyword overlap
                if (trend_name.lower() in temp_name.lower() or temp_name.lower() in trend_name.lower()
                    or any(w in temp_desc.lower() for w in trend_name.lower().split())):
                    desc = temp_desc
                    metadata = temp_meta

        # Find forecast details
        confidence = 0.5
        reasoning = "General industry interest."
        for template in self.trend_forecaster.FORECAST_TEMPLATES:
            if template["trend"].lower() in trend_name.lower() or trend_name.lower() in template["trend"].lower():
                confidence = template["base_confidence"]
                reasoning = template["reasoning"]
                break

        if not desc:
            kb_trends = (
                self.rag_coordinator.kb.list_items(category="streetwear_trends") +
                self.rag_coordinator.kb.list_items(category="luxury_trends")
            )
            for item in kb_trends:
                if trend_name.lower() in item.name.lower() or item.name.lower() in trend_name.lower():
                    desc = item.content
                    metadata = item.metadata
                    break

        if not desc:
            desc = f"A popular modern fashion trend highlighting '{trend_name}' aesthetics."

        return {
            "trend": trend_name,
            "explanation": desc,
            "confidence": float(metadata.get("popularity_score", metadata.get("growth_rate", confidence))),
            "reasoning": reasoning,
            "metadata": metadata
        }

    def get_trend_forecast(self, season: str, n_predictions: int = 3) -> List[Dict[str, Any]]:
        """
        Provide future trend forecasting for a targeted season.

        Parameters
        ----------
        season : str
        n_predictions : int

        Returns
        -------
        List[Dict[str, Any]]
        """
        return self.trend_forecaster.forecast_trends(current_season=season, n_predictions=n_predictions)

    def explain_fabric(self, fabric_name: str) -> Dict[str, Any]:
        """
        Retrieve properties and design characteristics of a fabric type.

        Parameters
        ----------
        fabric_name : str

        Returns
        -------
        Dict[str, Any]
        """
        logger.info(f"FashionAssistant: explaining fabric | '{fabric_name}'")
        fabrics = self.rag_coordinator.kb.list_items(category="fabric_types")
        matched = None
        for item in fabrics:
            if fabric_name.lower() in item.name.lower() or item.name.lower() in fabric_name.lower():
                matched = item
                break

        if matched:
            return {
                "fabric": matched.name,
                "explanation": matched.content,
                "tags": matched.tags,
                "durability": matched.metadata.get("durability", "unknown"),
                "breathability": matched.metadata.get("breathability", "unknown"),
                "weight": matched.metadata.get("weight", "unknown"),
                "metadata": matched.metadata
            }

        return {
            "fabric": fabric_name,
            "explanation": f"No specific fabric profile found for '{fabric_name}'. Commonly used in garment separates.",
            "tags": [],
            "durability": "unknown",
            "breathability": "unknown",
            "weight": "unknown",
            "metadata": {}
        }

    def explain_style(self, style_name: str) -> Dict[str, Any]:
        """
        Describe aesthetic principles, key items, and cuts of a style.

        Parameters
        ----------
        style_name : str

        Returns
        -------
        Dict[str, Any]
        """
        logger.info(f"FashionAssistant: explaining style | '{style_name}'")
        styles = self.rag_coordinator.kb.list_items(category="fashion_styles")
        matched = None
        for item in styles:
            if style_name.lower() in item.name.lower() or item.name.lower() in style_name.lower():
                matched = item
                break

        if matched:
            return {
                "style": matched.name,
                "explanation": matched.content,
                "tags": matched.tags,
                "fit": matched.metadata.get("fit", "unknown"),
                "key_items": matched.metadata.get("key_items", []),
                "metadata": matched.metadata
            }

        return {
            "style": style_name,
            "explanation": f"Style aesthetic '{style_name}' represents modern design values.",
            "tags": [],
            "fit": "unknown",
            "key_items": [],
            "metadata": {}
        }

    def chat(self, message: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Unified conversational router that detects intent and maps to the appropriate capabilities.

        Parameters
        ----------
        message : str
        user_id : str

        Returns
        -------
        Dict[str, Any]
        """
        logger.info(f"FashionAssistant chat received | user='{user_id}' | message='{message}'")
        msg_clean = message.lower().strip()

        # Log query to search history
        self.user_profile_manager.record_search(user_id, message)

        profile = self.user_profile_manager.get_profile(user_id)
        if not profile:
            profile = self.user_profile_manager.create_profile(user_id)

        # 1. Fabric Explanation
        if any(w in msg_clean for w in ["fabric", "material", "linen", "denim", "silk", "cotton", "wool", "cashmere"]):
            fabric_name = ""
            for w in ["linen", "denim", "silk", "cotton", "wool", "cashmere"]:
                if w in msg_clean:
                    fabric_name = w
                    break
            if not fabric_name:
                fabric_name = message.split()[-1].strip("?.!")

            res = self.explain_fabric(fabric_name)
            response_text = (
                f"### Fabric Explanation: {res['fabric'].capitalize()}\n"
                f"{res['explanation']}\n\n"
                f"- **Durability**: {res['durability']}\n"
                f"- **Breathability**: {res['breathability']}\n"
                f"- **Weight**: {res['weight']}"
            )
            return {
                "intent": "fabric_explanation",
                "fabric": res["fabric"],
                "response": response_text,
                "data": res
            }

        # 2. Style Explanation
        elif "explain style" in msg_clean or "describe style" in msg_clean or any(
            f"{w} style" in msg_clean for w in ["streetwear", "luxury", "minimalist", "athleisure"]
        ):
            style_name = ""
            for w in ["streetwear", "luxury", "minimalist", "athleisure", "vintage"]:
                if w in msg_clean:
                    style_name = w
                    break
            if not style_name:
                style_name = message.split()[-1].strip("?.!")

            res = self.explain_style(style_name)
            key_items_str = ", ".join(res["key_items"]) if res["key_items"] else "None listed"
            response_text = (
                f"### Style Aesthetic: {res['style'].capitalize()}\n"
                f"{res['explanation']}\n\n"
                f"- **Fit**: {res['fit']}\n"
                f"- **Key Items**: {key_items_str}"
            )
            return {
                "intent": "style_explanation",
                "style": res["style"],
                "response": response_text,
                "data": res
            }

        # 3. Brand Suggestion
        elif any(w in msg_clean for w in ["recommend brand", "suggest brand", "brand recommendation", "which brand"]):
            style_pref = profile.favorite_styles[0] if profile.favorite_styles else "streetwear"
            for w in ["streetwear", "luxury", "athleisure", "minimalist", "vintage", "casual"]:
                if w in msg_clean:
                    style_pref = w
                    break

            brands = self.recommend_brands({"favorite_style": style_pref}, user_id=user_id)
            brands_str = ", ".join(brands)
            response_text = f"Based on the **{style_pref}** style profile, I recommend the following brands: **{brands_str}**."
            return {
                "intent": "brand_recommendation",
                "brands": brands,
                "response": response_text,
                "data": {"style_queried": style_pref, "brands": brands}
            }

        # 4. Style Recommendation
        elif any(w in msg_clean for w in ["recommend style", "style recommendation", "what style"]) or ("recommend" in msg_clean and "style" in msg_clean):
            fav_style = profile.favorite_styles[0] if profile.favorite_styles else "streetwear"
            fav_color = profile.favorite_colors[0] if profile.favorite_colors else "black"

            for w in ["streetwear", "athleisure", "vintage", "luxury"]:
                if w in msg_clean:
                    fav_style = w
                    break
            for c in ["black", "white", "grey", "brown", "gold"]:
                if c in msg_clean:
                    fav_color = c
                    break

            preferences = {"favorite_style": fav_style, "favorite_color": fav_color}
            recs = self.recommend_styles(preferences, user_id=user_id)
            recs_str = ", ".join(recs)
            response_text = f"Based on your preferences ({fav_style} in {fav_color}), here are my style recommendations: **{recs_str}**."
            return {
                "intent": "style_recommendation",
                "recommendations": recs,
                "response": response_text,
                "data": {"preferences": preferences, "styles": recs}
            }

        # 5. Trend Spotlights & Forecasts
        elif "forecast" in msg_clean or "trends" in msg_clean or "trend" in msg_clean:
            season = "summer"
            for s in ["summer", "winter", "spring", "autumn"]:
                if s in msg_clean:
                    season = s
                    break

            if "forecast" in msg_clean:
                forecasts = self.get_trend_forecast(season)
                lines = [f"### Future Trend Forecasts for {season.capitalize()}"]
                for f in forecasts:
                    lines.append(f"- **{f['trend']}** (Confidence: {f['confidence']:.2f})")
                    lines.append(f"  *Reasoning*: {f['reasoning']}")
                response_text = "\n".join(lines)
                return {
                    "intent": "trend_forecast",
                    "season": season,
                    "response": response_text,
                    "data": forecasts
                }
            else:
                trend_name = "Cyber Utility Wear"
                for t in ["cyber utility wear", "patterned silk minimalism", "organic linen loungewear", "active loungewear tech"]:
                    if t in msg_clean:
                        trend_name = t
                        break

                res = self.explain_trend(trend_name)
                response_text = (
                    f"### Trend Spotlight: {res['trend'].title()}\n"
                    f"{res['explanation']}\n\n"
                    f"- **Growth Confidence**: {res['confidence']:.2f}\n"
                    f"- **Drivers**: {res['reasoning']}"
                )
                return {
                    "intent": "trend_explanation",
                    "trend": res["trend"],
                    "response": response_text,
                    "data": res
                }

        # 6. General Q&A / Fallback
        else:
            res = self.answer_question(message, user_id=user_id)
            citations = res.get("citations", [item["id"] for item in res.get("retrieved_items", [])])
            sources = res.get("sources", list(set([item.get("category", "unknown") for item in res.get("retrieved_items", [])])))
            return {
                "intent": "general_qa",
                "response": res["response"],
                "citations": citations,
                "sources": sources,
                "data": res
            }
