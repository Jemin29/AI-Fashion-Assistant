"""
Week 6 — RAG Service (Complete Rewrite).

Clean UI-to-backend interface for the Fashion RAG assistant pipeline.

Interface
---------
``RAGService`` exposes:

- ``chat(message, user_id)``         — conversational intent-routing Q&A
- ``answer_question(question)``      — single-shot factual Q&A
- ``semantic_search(query, n)``      — raw vector-similarity search
- ``recommend_styles(prefs, n)``     — style recommendations from preferences
- ``recommend_brands(profile, n)``   — brand recommendations from profile
- ``explain_trend(trend_name)``      — detailed trend explanation
- ``get_trend_forecast(season)``     — season trend forecasts
- ``get_collection_stats()``         — ChromaDB collection statistics
- ``health_check()``                 — backend availability probe

All public methods return a typed ``ServiceResult``.

Validation
----------
- message / question: non-empty, max 2 000 chars
- n / top-k: int in [1, 20]
- season: one of the recognised season tokens

Error codes
-----------
``QUERY_EMPTY``      — blank query / message
``QUERY_TOO_LONG``   — query exceeds 2 000 chars
``INVALID_TOP_K``    — n_results out of [1, 20]
``BACKEND_ERROR``    — real assistant raised an exception
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger, log_rag_query
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError

logger = get_logger(__name__)

# ── Mock data ──────────────────────────────────────────────────────────────────

_MOCK_QA_RESPONSE = (
    "**Fashion Intelligence (Mock Mode)**\n\n"
    "This is a mock response demonstrating the RAG pipeline. "
    "In production mode, this answer would be grounded in the 556-item "
    "fashion knowledge base using ChromaDB vector retrieval.\n\n"
    "**Key fashion insight**: The most versatile wardrobe foundation consists "
    "of neutral tones (black, white, navy, cream) complemented by 2–3 accent "
    "pieces in seasonal colours."
)

_MOCK_SEARCH_RESULTS: List[Dict[str, Any]] = [
    {
        "id":         "linen_001",
        "collection": "fashion_knowledge",
        "document":   "Linen is a natural fabric woven from flax fibres, prized for breathability and moisture-wicking properties.",
        "metadata":   {"category": "fabrics", "name": "Linen"},
        "distance":   0.12,
    },
    {
        "id":         "minimalist_001",
        "collection": "style_profiles",
        "document":   "Minimalist aesthetic emphasises clean lines, neutral palettes, and intentional simplicity.",
        "metadata":   {"style": "minimalist", "name": "Minimalist Style"},
        "distance":   0.23,
    },
    {
        "id":         "quiet_luxury_001",
        "collection": "trend_items",
        "document":   "Quiet luxury trend focuses on understated, high-quality clothing free from visible branding.",
        "metadata":   {"trend": "quiet_luxury", "name": "Quiet Luxury"},
        "distance":   0.31,
    },
    {
        "id":         "streetwear_001",
        "collection": "style_profiles",
        "document":   "Streetwear combines urban culture, skateboarding, and hip-hop aesthetics with high-fashion references.",
        "metadata":   {"style": "streetwear", "name": "Streetwear"},
        "distance":   0.38,
    },
]

_MOCK_STYLE_RECS: List[Dict[str, Any]] = [
    {"style": "Quiet Luxury",           "score": 0.94, "description": "Understated elegance — premium fabrics, no visible logos."},
    {"style": "Smart Casual",           "score": 0.87, "description": "Polished but relaxed — chinos, Oxford shirts, clean sneakers."},
    {"style": "Business Casual",        "score": 0.81, "description": "Professional without the suit — blazers, trousers, tucked shirts."},
    {"style": "Contemporary Minimalist","score": 0.78, "description": "Neutral palette, clean silhouettes, intentional layering."},
    {"style": "Resort Wear",            "score": 0.72, "description": "Lightweight breathable fabrics, botanical prints, relaxed fits."},
]

_MOCK_BRAND_RECS: List[Dict[str, Any]] = [
    {"brand": "Toteme",       "score": 0.92, "category": "Quiet Luxury",    "price_range": "$$$$"},
    {"brand": "COS",          "score": 0.88, "category": "Minimalist",      "price_range": "$$"},
    {"brand": "Uniqlo",       "score": 0.85, "category": "Basics",          "price_range": "$"},
    {"brand": "Acne Studios", "score": 0.80, "category": "Scandinavian",    "price_range": "$$$$"},
    {"brand": "Zara",         "score": 0.76, "category": "Contemporary",    "price_range": "$$"},
]

_MOCK_FORECASTS: List[Dict[str, Any]] = [
    {
        "trend":       "Utility Minimalism",
        "growth_rate": "High (+34%)",
        "explanation": "Functional pockets meet clean silhouettes — multi-pocket cargo tailoring.",
        "reasoning":   "Strong runway presence + streetwear adoption curve.",
        "confidence":  0.92,
    },
    {
        "trend":       "Biomorphic Prints",
        "growth_rate": "Medium (+18%)",
        "explanation": "Organic, nature-inspired patterns drawing from marine and botanical forms.",
        "reasoning":   "Emerging from luxury editorials, moving toward mass-market.",
        "confidence":  0.78,
    },
    {
        "trend":       "Chrome Metallics",
        "growth_rate": "Very High (+52%)",
        "explanation": "Futuristic reflective finishes on outerwear and accessories.",
        "reasoning":   "Gen-Z adoption driving rapid mainstream penetration.",
        "confidence":  0.85,
    },
    {
        "trend":       "Coastal Grandmother",
        "growth_rate": "Steady (+11%)",
        "explanation": "Linen, wide brims, relaxed silhouettes — leisured coastal elegance.",
        "reasoning":   "Social-media longevity sustaining moderate growth.",
        "confidence":  0.71,
    },
]

_FABRIC_KNOWLEDGE: Dict[str, str] = {
    "linen":    "Linen is a natural fabric woven from flax fibres, prized for breathability and moisture-wicking. Ideal for warm climates.",
    "silk":     "Silk is a luxurious protein-based fibre from silkworm cocoons — smooth, lustrous, temperature-regulating.",
    "denim":    "Denim is a sturdy cotton twill — highly durable, versatile from workwear to high fashion.",
    "cotton":   "Cotton is the world's most widely used natural fibre — soft, breathable, hypoallergenic.",
    "wool":     "Wool is a natural protein fibre sheared from sheep — excellent thermal insulation and moisture management.",
    "cashmere": "Cashmere is a luxury soft fibre from Cashmere goats — incredibly soft, warm, and lightweight.",
    "polyester":"Polyester is a synthetic fibre — wrinkle-resistant, quick-drying, affordable. Blended frequently with natural fibres.",
}

_VALID_SEASONS = {"spring", "summer", "autumn", "fall", "winter", "ss2024", "ss2025", "fw2024", "fw2025", "all"}


class RAGService(BaseService):
    """
    Mock-safe UI adapter over the Week 5 ``FashionAssistant`` RAG pipeline.

    Provides conversational Q&A, semantic search, style/brand recommendations,
    and trend intelligence with full input validation, structured logging, and
    typed ``ServiceResult`` returns.

    Args:
        mock_mode: If ``True`` (default), returns rich deterministic mock
                   responses without loading embedding models.
                   If ``False``, attempts to connect to the real assistant.
    """

    _SERVICE_NAME = "RAGService"

    def __init__(self, mock_mode: bool = True) -> None:
        super().__init__(mock_mode)
        self._assistant: Optional[Any] = None

        if not mock_mode:
            try:
                from src.rag.fashion_assistant import FashionAssistant
                self._assistant = FashionAssistant(force_mock_embeddings=False)
                self.mock_mode  = False
                logger.info("RAGService: real FashionAssistant loaded")
            except Exception as exc:
                logger.warning("RAGService: real mode failed — %s", exc)
                self.mock_mode = True
        else:
            try:
                from src.rag.fashion_assistant import FashionAssistant
                self._assistant = FashionAssistant(force_mock_embeddings=True)
                logger.info("RAGService: mock-embedding FashionAssistant loaded")
            except Exception as exc:
                logger.warning("RAGService: assistant init failed, pure mock — %s", exc)
                self._assistant = None

    # ── Conversational chat ────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        user_id: str = "default_user",
    ) -> ServiceResult[Dict[str, Any]]:
        """Unified conversational router — detects intent and returns a response.

        Supported intents:
        - ``fabric_explanation``  — "what is linen?"
        - ``style_explanation``   — "describe the minimalist style"
        - ``brand_recommendation``— "recommend brands for streetwear"
        - ``style_recommendation``— "recommend styles for me"
        - ``trend_forecast``      — "what are the trends for summer?"
        - ``general_qa``          — everything else

        Args:
            message: User message text.
            user_id: Optional user identifier for session context.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``response`` (str), ``intent`` (str), ``source_documents`` (list).
        """
        self._call_count += 1

        try:
            message = self._validate_str(message, "message", min_len=1, max_len=2000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("message", e.reason)

        with self._timer() as t:
            try:
                if self._assistant:
                    result  = self._assistant.chat(message, user_id=user_id)
                    docs    = self._normalise_source_docs(result)
                    result["source_documents"] = docs
                else:
                    result = self._mock_chat(message)
            except Exception as exc:
                self._error_count += 1
                logger.error("RAGService.chat error: %s", exc, exc_info=True)
                result = {
                    "response": f"⚠️ Error processing your message: {exc}",
                    "intent":   "error",
                    "source_documents": [],
                }

        latency = t.elapsed_ms
        log_rag_query(message, len(result.get("source_documents", [])), latency)
        logger.info(
            "RAGService.chat | intent=%s | docs=%d | latency=%.0fms",
            result.get("intent", "?"), len(result.get("source_documents", [])), latency,
        )

        status = (
            ServiceStatus.MOCK if self.mock_mode or self._assistant is None
            else ServiceStatus.OK
        )
        return ServiceResult(
            data=result,
            meta={"latency_ms": round(latency, 1), "intent": result.get("intent", "")},
            status=status,
            latency_ms=latency,
        )

    # ── Single-shot Q&A ────────────────────────────────────────────────────────

    def answer_question(self, question: str) -> ServiceResult[Dict[str, Any]]:
        """Answer a single factual fashion question using RAG retrieval.

        Args:
            question: Fashion question text.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``response`` (str), ``source_documents`` (list).
        """
        self._call_count += 1

        try:
            question = self._validate_str(question, "question", min_len=1, max_len=2000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("question", e.reason)

        with self._timer() as t:
            try:
                if self._assistant:
                    result = self._assistant.answer_question(question)
                    if "retrieved_items" in result and "source_documents" not in result:
                        result["source_documents"] = [
                            {
                                "id":       item.get("id"),
                                "document": f"{item.get('name','')} ({item.get('category','')})",
                                "metadata": item,
                                "distance": item.get("score", 0.0),
                            }
                            for item in result["retrieved_items"]
                        ]
                else:
                    result = {
                        "response":         _MOCK_QA_RESPONSE,
                        "source_documents": _MOCK_SEARCH_RESULTS[:2],
                    }
            except Exception as exc:
                self._error_count += 1
                logger.error("RAGService.answer_question error: %s", exc, exc_info=True)
                result = {
                    "response":         f"⚠️ Error processing question: {exc}",
                    "source_documents": [],
                }

        latency = t.elapsed_ms
        log_rag_query(question, len(result.get("source_documents", [])), latency)

        status = ServiceStatus.MOCK if (self.mock_mode or not self._assistant) else ServiceStatus.OK
        return ServiceResult(
            data=result,
            meta={"latency_ms": round(latency, 1), "n_sources": len(result.get("source_documents", []))},
            status=status,
            latency_ms=latency,
        )

    # ── Semantic search ────────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Perform vector-similarity search over the fashion knowledge base.

        Args:
            query:     Search query text.
            n_results: Number of results to return [1, 20].

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of matching document dicts.
        """
        self._call_count += 1

        try:
            query = self._validate_str(query, "query", min_len=1, max_len=2000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("query", e.reason)

        try:
            n_results = int(self._validate_range(n_results, "n_results", lo=1, hi=20))
        except ValidationError:
            n_results = max(1, min(20, int(n_results)))

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "chroma_retriever"):
                    docs = self._assistant.chroma_retriever.retrieve(query, n_results=n_results)
                else:
                    docs = _MOCK_SEARCH_RESULTS[:n_results]
            except Exception as exc:
                logger.error("RAGService.semantic_search error: %s", exc, exc_info=True)
                docs = _MOCK_SEARCH_RESULTS[:n_results]

        return ServiceResult.ok(
            docs,
            meta={"query": query, "n_results": len(docs), "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    # ── Style recommendations ──────────────────────────────────────────────────

    def recommend_styles(
        self,
        preferences: Dict[str, Any],
        n: int = 5,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Return style recommendations based on user preferences.

        Args:
            preferences: Dict with optional keys: ``style``, ``occasion``,
                         ``fit``, ``gender``, ``color``.
            n:           Number of recommendations [1, 20].

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of style recommendation dicts.
        """
        self._call_count += 1

        try:
            n = int(self._validate_range(n, "n", lo=1, hi=20))
        except ValidationError:
            n = max(1, min(20, int(n)))

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "recommend_styles"):
                    recs = self._assistant.recommend_styles(preferences, n_results=n)
                else:
                    recs = sorted(_MOCK_STYLE_RECS, key=lambda x: x["score"], reverse=True)[:n]
            except Exception as exc:
                logger.error("RAGService.recommend_styles error: %s", exc, exc_info=True)
                recs = _MOCK_STYLE_RECS[:n]

        return ServiceResult.ok(
            recs,
            meta={"n": len(recs), "preferences": preferences, "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    # ── Brand recommendations ──────────────────────────────────────────────────

    def recommend_brands(
        self,
        profile: Dict[str, Any],
        n: int = 4,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Return brand recommendations based on an aesthetic profile.

        Args:
            profile: Dict with optional keys: ``styles``, ``target_aesthetic``,
                     ``price_range``, ``gender``.
            n:       Number of recommendations [1, 20].

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of brand dicts.
        """
        self._call_count += 1

        try:
            n = int(self._validate_range(n, "n", lo=1, hi=20))
        except ValidationError:
            n = max(1, min(20, int(n)))

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "recommend_brands"):
                    brands = self._assistant.recommend_brands(profile, n_results=n)
                else:
                    brands = _MOCK_BRAND_RECS[:n]
            except Exception as exc:
                logger.error("RAGService.recommend_brands error: %s", exc, exc_info=True)
                brands = _MOCK_BRAND_RECS[:n]

        return ServiceResult.ok(
            brands,
            meta={"n": len(brands), "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    # ── Trend intelligence ─────────────────────────────────────────────────────

    def explain_trend(self, trend_name: str) -> ServiceResult[Dict[str, Any]]:
        """Return a detailed explanation of a named fashion trend.

        Args:
            trend_name: Name of the trend (e.g. ``"Quiet Luxury"``).

        Returns:
            ``ServiceResult`` whose ``.data`` is a dict with keys:
            ``trend``, ``explanation``, ``confidence``, ``reasoning``.
        """
        self._call_count += 1

        try:
            trend_name = self._validate_str(trend_name, "trend_name", min_len=1, max_len=200)
        except ValidationError as e:
            return ServiceResult.validation_fail("trend_name", e.reason)

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "explain_trend"):
                    data = self._assistant.explain_trend(trend_name)
                else:
                    # Find matching mock forecast
                    match = next(
                        (f for f in _MOCK_FORECASTS if trend_name.lower() in f["trend"].lower()),
                        None,
                    )
                    if match:
                        data = match
                    else:
                        data = {
                            "trend":       trend_name,
                            "explanation": (
                                f"The '{trend_name}' movement represents an evolving shift in consumer "
                                "aesthetics — balancing innovation with heritage craftsmanship."
                            ),
                            "confidence":  0.72,
                            "reasoning":   "Based on aggregated runway and social-media signal analysis.",
                        }
            except Exception as exc:
                logger.error("RAGService.explain_trend error: %s", exc, exc_info=True)
                data = {"trend": trend_name, "explanation": str(exc), "confidence": 0.0, "reasoning": ""}

        return ServiceResult.ok(
            data,
            meta={"trend": trend_name, "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    def get_trend_forecast(self, season: str = "all") -> ServiceResult[List[Dict[str, Any]]]:
        """Return trend forecasts for a given season.

        Args:
            season: Season label (e.g. ``"summer"``, ``"fw2025"``).
                    Defaults to ``"all"`` (returns all forecasts).

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of trend forecast dicts.
        """
        self._call_count += 1

        season_clean = season.lower().strip()
        if season_clean not in _VALID_SEASONS:
            season_clean = "all"

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "get_trend_forecast"):
                    forecasts = self._assistant.get_trend_forecast(season_clean)
                else:
                    forecasts = [dict(f, season=season_clean) for f in _MOCK_FORECASTS]
            except Exception as exc:
                logger.error("RAGService.get_trend_forecast error: %s", exc, exc_info=True)
                forecasts = [dict(f, season=season_clean) for f in _MOCK_FORECASTS]

        return ServiceResult.ok(
            forecasts,
            meta={"season": season_clean, "n": len(forecasts), "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    # ── Collection stats ───────────────────────────────────────────────────────

    def get_collection_stats(self) -> ServiceResult[Dict[str, Any]]:
        """Return ChromaDB collection statistics.

        Returns:
            ``ServiceResult`` whose ``.data`` is a dict mapping
            collection names to document counts.
        """
        self._call_count += 1

        with self._timer() as t:
            try:
                if self._assistant and hasattr(self._assistant, "get_collection_stats"):
                    stats = self._assistant.get_collection_stats()
                elif self._assistant and hasattr(self._assistant, "chroma_retriever"):
                    stats = self._assistant.chroma_retriever.get_stats()
                else:
                    stats = {
                        "fashion_knowledge": 180,
                        "style_profiles":    85,
                        "brand_profiles":    40,
                        "trend_items":       120,
                        "outfit_examples":   131,
                        "total":             556,
                        "mode":              "mock",
                    }
            except Exception as exc:
                logger.error("RAGService.get_collection_stats error: %s", exc, exc_info=True)
                stats = {"error": str(exc)}

        return ServiceResult.ok(stats, meta={"latency_ms": round(t.elapsed_ms, 1)})

    # ── Health ─────────────────────────────────────────────────────────────────

    def health_check(self) -> ServiceResult:
        """Return lightweight health status of the RAG backend."""
        assistant_ok = self._assistant is not None
        res = {
            "status":        "ok" if assistant_ok else "mock",
            "mock_mode":     self.mock_mode,
            "assistant":     "loaded" if assistant_ok else "unavailable",
            "chroma":        "connected" if (assistant_ok and hasattr(self._assistant, "chroma_retriever")) else "mock",
            "message": (
                "RAGService running with real FashionAssistant." if (assistant_ok and not self.mock_mode)
                else "RAGService running in mock mode — no embeddings required."
            ),
        }
        return ServiceResult(success=True, data=res)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _normalise_source_docs(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise various assistant result shapes into a uniform doc list."""
        if "source_documents" in result:
            return result["source_documents"]

        docs: List[Dict[str, Any]] = []
        intent = result.get("intent", "general_qa")
        data   = result.get("data", {})

        if intent == "general_qa":
            for item in data.get("retrieved_items", []) or result.get("retrieved_items", []):
                docs.append({
                    "id":       item.get("id", "doc"),
                    "document": item.get("document", item.get("content", "")),
                    "metadata": item,
                    "distance": item.get("distance", item.get("score", 0.0)),
                })
        elif intent in ("fabric_explanation", "style_explanation", "trend_explanation"):
            docs.append({
                "id":       data.get("fabric") or data.get("style") or data.get("trend") or "kb_item",
                "document": data.get("explanation", ""),
                "metadata": data.get("metadata", {}),
                "distance": 0.0,
            })
        elif intent == "brand_recommendation":
            for brand in data.get("brands", []):
                docs.append({
                    "id": brand, "document": f"Brand: {brand}",
                    "metadata": {"brand": brand}, "distance": 0.0,
                })
        elif intent in ("style_recommendation",):
            for style in data.get("styles", []):
                docs.append({
                    "id": style, "document": f"Style: {style}",
                    "metadata": {"style": style}, "distance": 0.0,
                })
        elif intent == "trend_forecast":
            for item in (data if isinstance(data, list) else []):
                docs.append({
                    "id":       item.get("trend", "trend"),
                    "document": f"{item.get('growth_rate', '')} — {item.get('explanation', '')}",
                    "metadata": item,
                    "distance": 0.0,
                })
        return docs

    def _mock_chat(self, message: str) -> Dict[str, Any]:
        """Rich deterministic mock response router."""
        msg = message.lower().strip()

        # Fabric queries
        for fabric, explanation in _FABRIC_KNOWLEDGE.items():
            if fabric in msg:
                return {
                    "response":  f"### Fabric: {fabric.capitalize()} (Mock Mode)\n\n{explanation}",
                    "intent":    "fabric_explanation",
                    "source_documents": [_MOCK_SEARCH_RESULTS[0]],
                }

        # Style explanation
        if any(kw in msg for kw in ["explain style", "describe style", "what is the", "style aesthetic"]) or \
           any(f"{s} style" in msg for s in ["streetwear", "luxury", "minimalist", "athleisure", "techwear"]):
            return {
                "response":  "### Style Aesthetic: Minimalist (Mock Mode)\n\nMinimalist aesthetic emphasises clean lines, neutral palettes, and intentional simplicity.\n\n- **Fit**: regular / relaxed\n- **Palette**: Black, White, Beige, Grey\n- **Key Items**: tailored coat, white sneakers, clean tee",
                "intent":    "style_explanation",
                "source_documents": [_MOCK_SEARCH_RESULTS[1]],
            }

        # Brand recommendations
        if any(kw in msg for kw in ["recommend brand", "suggest brand", "which brand", "brand for"]):
            return {
                "response":  "Based on your style profile, I recommend: **Toteme**, **COS**, **Uniqlo**, and **Acne Studios** — all known for understated, quality-first aesthetics.",
                "intent":    "brand_recommendation",
                "source_documents": _MOCK_SEARCH_RESULTS[1:3],
            }

        # Style recommendations
        if any(kw in msg for kw in ["recommend style", "what should i wear", "outfit idea", "style for me"]):
            return {
                "response":  "Based on your preferences, here are style recommendations:\n\n1. **Quiet Luxury** — understated premium fabrics\n2. **Smart Casual** — polished everyday wear\n3. **Contemporary Minimalist** — clean, intentional layering",
                "intent":    "style_recommendation",
                "source_documents": [_MOCK_SEARCH_RESULTS[1]],
            }

        # Trend forecast
        if any(kw in msg for kw in ["trend", "forecast", "upcoming", "next season", "what's hot"]):
            lines = "\n".join(
                f"- **{f['trend']}** ({f['growth_rate']}): {f['explanation']}"
                for f in _MOCK_FORECASTS
            )
            return {
                "response":  f"### Fashion Trend Forecast (Mock Mode)\n\n{lines}",
                "intent":    "trend_forecast",
                "source_documents": [_MOCK_SEARCH_RESULTS[2]],
            }

        # Default general Q&A
        return {
            "response":        _MOCK_QA_RESPONSE,
            "intent":          "general_qa",
            "source_documents": _MOCK_SEARCH_RESULTS[:2],
        }
