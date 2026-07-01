"""
Week 6 — Recommendation Service (Complete Rewrite).

Clean UI-to-backend interface for personalised style and brand recommendations.

Interface
---------
``RecommendationService`` exposes:

- ``recommend_styles(...)``        — top-N style recommendations by preference
- ``recommend_brands(...)``        — top-N brand recommendations by aesthetic
- ``recommend_outfits(...)``       — complete outfit recommendations
- ``get_style_compatibility(...)`` — pairwise style compatibility score
- ``update_user_profile(...)``     — persist user interaction signal
- ``get_user_profile(...)``        — retrieve preference summary
- ``get_available_filters()``      — metadata for UI filter widgets
- ``health_check()``               — lightweight backend status probe

All public methods return a typed ``ServiceResult``.

Validation
----------
- gender:    one of {"men", "women", "unisex", "all"}
- style:     non-empty string, max 100 chars
- occasion:  non-empty string, max 100 chars
- fit:       one of {"regular", "slim", "oversized", "relaxed", "tailored"}
- n:         int in [1, 20]

Error codes
-----------
``INVALID_GENDER``    — unrecognised gender value
``INVALID_FIT``       — unrecognised fit value
``INVALID_N``         — n out of [1, 20]
``PROFILE_NOT_FOUND`` — user_id has no stored profile
``BACKEND_ERROR``     — real recommender raised an exception
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError

logger = get_logger(__name__)

# ── Mock data ──────────────────────────────────────────────────────────────────

_MOCK_STYLE_RECS: List[Dict[str, Any]] = [
    {
        "style":       "Quiet Luxury",
        "score":       0.94,
        "description": "Understated elegance with premium fabrics — cashmere, silk, fine wool. No visible logos.",
        "occasions":   ["business", "formal", "evening"],
        "key_items":   ["tailored blazer", "cashmere sweater", "slim trousers", "leather loafers"],
        "color_palette": ["Beige", "Ivory", "Camel", "Black"],
    },
    {
        "style":       "Smart Casual",
        "score":       0.87,
        "description": "Polished but relaxed — chinos, Oxford shirts, clean sneakers or loafers.",
        "occasions":   ["business_casual", "casual", "social"],
        "key_items":   ["chinos", "Oxford shirt", "clean sneakers", "light blazer"],
        "color_palette": ["Navy", "White", "Grey", "Light Blue"],
    },
    {
        "style":       "Business Casual",
        "score":       0.81,
        "description": "Professional without the suit — blazers over trousers, tucked shirts.",
        "occasions":   ["business_casual", "office", "presentation"],
        "key_items":   ["blazer", "trousers", "tucked shirt", "oxford shoes"],
        "color_palette": ["Charcoal", "Navy", "White", "Light Grey"],
    },
    {
        "style":       "Contemporary Minimalist",
        "score":       0.78,
        "description": "Neutral palette, clean silhouettes, intentional layering.",
        "occasions":   ["casual", "weekend", "social"],
        "key_items":   ["monochrome tee", "wide-leg trousers", "minimal sneakers"],
        "color_palette": ["Black", "White", "Stone", "Ecru"],
    },
    {
        "style":       "Streetwear",
        "score":       0.74,
        "description": "Urban culture meets high fashion — oversized fits, bold graphics, premium sneakers.",
        "occasions":   ["casual", "weekend", "social", "concert"],
        "key_items":   ["graphic hoodie", "cargo pants", "high-top sneakers", "bucket hat"],
        "color_palette": ["Black", "White", "Neon Green", "Safety Orange"],
    },
    {
        "style":       "Athleisure",
        "score":       0.69,
        "description": "Performance wear adapted for everyday — technical fabrics, sleek silhouettes.",
        "occasions":   ["casual", "sport", "weekend"],
        "key_items":   ["performance joggers", "fitted tee", "running shoes", "windbreaker"],
        "color_palette": ["Black", "Grey", "Electric Blue", "White"],
    },
    {
        "style":       "Resort Wear",
        "score":       0.65,
        "description": "Lightweight, breathable fabrics with botanical prints and relaxed fits.",
        "occasions":   ["holiday", "beach", "summer", "travel"],
        "key_items":   ["linen shirt", "relaxed shorts", "sandals", "straw hat"],
        "color_palette": ["Turquoise", "Sand", "Coral", "White"],
    },
]

_MOCK_BRAND_RECS: List[Dict[str, Any]] = [
    {"brand": "Toteme",        "score": 0.92, "category": "Quiet Luxury",    "price_range": "$$$$", "origin": "Sweden"},
    {"brand": "COS",           "score": 0.88, "category": "Minimalist",      "price_range": "$$",   "origin": "Sweden"},
    {"brand": "Uniqlo",        "score": 0.85, "category": "Basics",          "price_range": "$",    "origin": "Japan"},
    {"brand": "Acne Studios",  "score": 0.80, "category": "Scandinavian",    "price_range": "$$$$", "origin": "Sweden"},
    {"brand": "Zara",          "score": 0.76, "category": "Contemporary",    "price_range": "$$",   "origin": "Spain"},
    {"brand": "Arket",         "score": 0.72, "category": "Modern Basics",   "price_range": "$$$",  "origin": "Sweden"},
    {"brand": "& Other Stories","score": 0.68,"category": "European Chic",   "price_range": "$$",   "origin": "Sweden"},
    {"brand": "Massimo Dutti", "score": 0.64, "category": "Smart Casual",    "price_range": "$$$",  "origin": "Spain"},
]

_MOCK_OUTFIT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name":      "Monochrome Power Look",
        "style":     "Quiet Luxury",
        "items": {
            "top":        "Ivory cashmere crewneck",
            "bottom":     "Tailored camel trousers",
            "outerwear":  "Camel wrap coat",
            "footwear":   "Pointed-toe leather loafers",
            "accessories":["Minimal gold chain", "Leather tote"],
        },
        "occasions": ["business", "evening", "formal"],
        "score":     0.94,
    },
    {
        "name":      "Smart Weekend Casual",
        "style":     "Smart Casual",
        "items": {
            "top":        "Classic white Oxford shirt (untucked)",
            "bottom":     "Slim navy chinos",
            "outerwear":  "Unstructured navy blazer",
            "footwear":   "White leather sneakers",
            "accessories":["Minimal watch", "Canvas tote"],
        },
        "occasions": ["weekend", "casual", "social"],
        "score":     0.88,
    },
    {
        "name":      "Urban Streetwear Edit",
        "style":     "Streetwear",
        "items": {
            "top":        "Oversized graphic hoodie",
            "bottom":     "Tapered cargo pants",
            "outerwear":  None,
            "footwear":   "High-top premium sneakers",
            "accessories":["Bucket hat", "Mini crossbody bag"],
        },
        "occasions": ["casual", "weekend", "social"],
        "score":     0.80,
    },
]

_STYLE_COMPATIBILITY: Dict[str, Dict[str, float]] = {
    "quiet_luxury":   {"smart_casual": 0.88, "business_casual": 0.92, "minimalist": 0.90, "streetwear": 0.40},
    "streetwear":     {"athleisure": 0.88,   "techwear": 0.82,       "quiet_luxury": 0.38,"business_casual": 0.35},
    "minimalist":     {"quiet_luxury": 0.90, "smart_casual": 0.85,   "streetwear": 0.55,  "resort": 0.65},
    "athleisure":     {"streetwear": 0.85,   "resort": 0.72,         "business_casual": 0.40},
    "business_casual":{"smart_casual": 0.92, "quiet_luxury": 0.88,   "streetwear": 0.30},
}

_VALID_GENDERS  = {"men", "women", "unisex", "all"}
_VALID_FITS     = {"regular", "slim", "oversized", "relaxed", "tailored", "wide", "fitted"}
_VALID_OCCASIONS = {
    "casual", "business", "business_casual", "formal", "evening",
    "weekend", "sport", "beach", "holiday", "travel", "social",
    "concert", "date", "office", "all",
}


class RecommendationService(BaseService):
    """
    Mock-safe UI adapter over ``StyleRecommender``, ``BrandRecommender``,
    and ``UserProfileManager``.

    Provides personalised style, brand, and outfit recommendations with
    full input validation, structured logging, and typed ``ServiceResult`` returns.

    Args:
        mock_mode: If ``False``, attempts to load the real recommenders.
                   If ``True`` (default when real mode fails), uses rich mock data.
    """

    _SERVICE_NAME = "RecommendationService"

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode)
        self._style_rec:  Optional[Any] = None
        self._brand_rec:  Optional[Any] = None
        self._profile_mgr:Optional[Any] = None

        if not mock_mode:
            try:
                from src.recommendations.style_recommender import StyleRecommender
                from src.recommendations.brand_recommender import BrandRecommender
                from src.recommendations.user_profile_manager import UserProfileManager
                self._style_rec   = StyleRecommender()
                self._brand_rec   = BrandRecommender()
                self._profile_mgr = UserProfileManager()
                self.mock_mode    = False
                logger.info("RecommendationService: real recommenders loaded")
            except Exception as exc:
                logger.warning("RecommendationService: real mode unavailable — %s", exc)
                self.mock_mode = True
        else:
            logger.info("RecommendationService: started in mock mode")

    # ── Style recommendations ──────────────────────────────────────────────────

    def recommend_styles(
        self,
        gender: str = "all",
        style: str = "",
        occasion: str = "casual",
        fit: str = "regular",
        *,
        n: int = 5,
        color: str = "",
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Return top-N personalised style recommendations.

        Args:
            gender:   Target gender: ``"men"`` / ``"women"`` / ``"unisex"`` / ``"all"``.
            style:    Preferred base style (e.g. ``"minimalist"``).
            occasion: Target occasion (e.g. ``"casual"``).
            fit:      Preferred fit type.
            n:        Number of results to return [1, 20].
            color:    Optional preferred colour hint.

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of style recommendation dicts.
        """
        self._call_count += 1
        warnings: List[str] = []

        # Validation
        gender  = gender.lower().strip()
        if gender not in _VALID_GENDERS:
            warnings.append(f"Unknown gender '{gender}' — using 'all'.")
            gender = "all"

        fit_clean = fit.lower().strip()
        if fit_clean not in _VALID_FITS:
            warnings.append(f"Unknown fit '{fit}' — using 'regular'.")
            fit_clean = "regular"

        try:
            n = int(self._validate_range(n, "n", lo=1, hi=20))
        except ValidationError:
            n = max(1, min(20, int(n)))

        prefs = {"gender": gender, "style": style, "occasion": occasion, "fit": fit_clean, "color": color}

        with self._timer() as t:
            try:
                if not self.mock_mode and self._style_rec:
                    recs = self._style_rec.recommend(prefs, n=n)
                else:
                    recs = sorted(_MOCK_STYLE_RECS, key=lambda x: x["score"], reverse=True)[:n]
            except Exception as exc:
                self._error_count += 1
                logger.error("RecommendationService.recommend_styles error: %s", exc, exc_info=True)
                recs = _MOCK_STYLE_RECS[:n]
                warnings.append(f"Recommender error — returning mock results. Detail: {exc}")

        logger.info(
            "RecommendationService.recommend_styles | n=%d | occasion=%s | latency=%.0fms",
            len(recs), occasion, t.elapsed_ms,
        )
        status = ServiceStatus.MOCK if self.mock_mode else ServiceStatus.OK
        return ServiceResult(
            data=recs,
            meta={"n": len(recs), "preferences": prefs, "latency_ms": round(t.elapsed_ms, 1)},
            status=status,
            warnings=warnings,
            latency_ms=t.elapsed_ms,
        )

    # ── Brand recommendations ──────────────────────────────────────────────────

    def recommend_brands(
        self,
        styles: List[str],
        aesthetic: str = "",
        *,
        price_range: str = "",
        n: int = 4,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Return top-N brand recommendations for a given aesthetic profile.

        Args:
            styles:      List of preferred style names.
            aesthetic:   Target aesthetic description.
            price_range: Optional filter (``"$"`` / ``"$$"`` / ``"$$$"`` / ``"$$$$"``).
            n:           Number of results [1, 20].

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of brand recommendation dicts.
        """
        self._call_count += 1
        warnings: List[str] = []

        try:
            n = int(self._validate_range(n, "n", lo=1, hi=20))
        except ValidationError:
            n = max(1, min(20, int(n)))

        profile = {"styles": styles, "target_aesthetic": aesthetic, "price_range": price_range}

        with self._timer() as t:
            try:
                if not self.mock_mode and self._brand_rec:
                    brands = self._brand_rec.recommend(profile, n=n)
                else:
                    brands = _MOCK_BRAND_RECS[:n]
                    if price_range:
                        brands = [b for b in _MOCK_BRAND_RECS if b.get("price_range") == price_range][:n] or _MOCK_BRAND_RECS[:n]
            except Exception as exc:
                self._error_count += 1
                logger.error("RecommendationService.recommend_brands error: %s", exc, exc_info=True)
                brands = _MOCK_BRAND_RECS[:n]
                warnings.append(f"Recommender error — returning mock results. Detail: {exc}")

        logger.info(
            "RecommendationService.recommend_brands | n=%d | latency=%.0fms", len(brands), t.elapsed_ms
        )
        status = ServiceStatus.MOCK if self.mock_mode else ServiceStatus.OK
        return ServiceResult(
            data=brands,
            meta={"n": len(brands), "latency_ms": round(t.elapsed_ms, 1)},
            status=status,
            warnings=warnings,
            latency_ms=t.elapsed_ms,
        )

    # ── Outfit recommendations ─────────────────────────────────────────────────

    def recommend_outfits(
        self,
        style: str,
        occasion: str = "casual",
        *,
        n: int = 3,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Return complete outfit recommendations for a style and occasion.

        Args:
            style:    Target style name.
            occasion: Target occasion.
            n:        Number of outfit suggestions [1, 10].

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of outfit dicts,
            each with ``name``, ``style``, ``items``, ``occasions``, ``score``.
        """
        self._call_count += 1

        try:
            n = int(self._validate_range(n, "n", lo=1, hi=10))
        except ValidationError:
            n = max(1, min(10, int(n)))

        with self._timer() as t:
            style_lower = style.lower().strip()
            matching = [
                o for o in _MOCK_OUTFIT_TEMPLATES
                if style_lower in o["style"].lower() or occasion in o.get("occasions", [])
            ]
            outfits = (matching or _MOCK_OUTFIT_TEMPLATES)[:n]

        return ServiceResult.ok(
            outfits,
            meta={"n": len(outfits), "style": style, "occasion": occasion, "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    # ── Style compatibility ────────────────────────────────────────────────────

    def get_style_compatibility(
        self,
        style_a: str,
        style_b: str,
    ) -> ServiceResult[Dict[str, Any]]:
        """Return a pairwise style compatibility score.

        Args:
            style_a: First style name.
            style_b: Second style name.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``style_a``, ``style_b``, ``score`` (0–1), ``verdict`` (str).
        """
        self._call_count += 1
        a = style_a.lower().strip().replace(" ", "_")
        b = style_b.lower().strip().replace(" ", "_")

        score = (
            _STYLE_COMPATIBILITY.get(a, {}).get(b)
            or _STYLE_COMPATIBILITY.get(b, {}).get(a)
            or 0.5
        )
        verdict = (
            "Highly compatible ✓" if score >= 0.8 else
            "Moderately compatible" if score >= 0.6 else
            "Low compatibility — bold choice"
        )
        return ServiceResult.ok(
            {"style_a": style_a, "style_b": style_b, "score": score, "verdict": verdict},
            meta={"latency_ms": 0.0},
        )

    # ── User profile management ────────────────────────────────────────────────

    def update_user_profile(
        self,
        user_id: str,
        interaction: Dict[str, Any],
    ) -> ServiceResult[bool]:
        """Record a user interaction to update the preference model.

        Args:
            user_id:     Unique user identifier.
            interaction: Dict describing the interaction (style, brand, action, etc.).

        Returns:
            ``ServiceResult`` whose ``.data`` is ``True`` on success.
        """
        self._call_count += 1
        try:
            user_id = self._validate_str(user_id, "user_id", min_len=1, max_len=200)
        except ValidationError as e:
            return ServiceResult.validation_fail("user_id", e.reason)

        with self._timer() as t:
            try:
                if not self.mock_mode and self._profile_mgr:
                    self._profile_mgr.add_interaction(user_id, interaction)
                    success = True
                else:
                    success = True  # mock silently accepts
            except Exception as exc:
                self._error_count += 1
                logger.error("RecommendationService.update_user_profile error: %s", exc, exc_info=True)
                success = False

        logger.debug(
            "RecommendationService.update_user_profile | user=%s | latency=%.0fms", user_id, t.elapsed_ms
        )
        return ServiceResult.ok(success, meta={"user_id": user_id, "latency_ms": round(t.elapsed_ms, 1)})

    def get_user_profile(self, user_id: str) -> ServiceResult[Dict[str, Any]]:
        """Return the preference profile summary for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            ``ServiceResult`` whose ``.data`` is a profile dict.
        """
        self._call_count += 1

        with self._timer() as t:
            try:
                if not self.mock_mode and self._profile_mgr:
                    profile = self._profile_mgr.get_profile(user_id)
                else:
                    profile = {
                        "user_id":            user_id,
                        "top_styles":         ["minimalist", "smart_casual", "quiet_luxury"],
                        "favorite_colors":    ["black", "white", "navy", "camel"],
                        "favorite_occasions": ["casual", "business_casual"],
                        "favorite_brands":    ["Uniqlo", "COS", "Zara"],
                        "interaction_count":  0,
                        "mode":               "mock",
                    }
            except Exception as exc:
                logger.error("RecommendationService.get_user_profile error: %s", exc, exc_info=True)
                profile = {"user_id": user_id, "error": str(exc)}

        return ServiceResult.ok(profile, meta={"user_id": user_id, "latency_ms": round(t.elapsed_ms, 1)})

    # ── Filter metadata ────────────────────────────────────────────────────────

    def get_available_filters(self) -> Dict[str, List[str]]:
        """Return available filter options for UI dropdowns.

        Returns:
            Dict mapping filter name to sorted list of values.
        """
        return {
            "genders":   sorted(_VALID_GENDERS),
            "fits":      sorted(_VALID_FITS),
            "occasions": sorted(_VALID_OCCASIONS),
            "styles":    [r["style"] for r in _MOCK_STYLE_RECS],
            "brands":    [b["brand"] for b in _MOCK_BRAND_RECS],
        }

    # ── Health ─────────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Return lightweight health status of the recommendation backend."""
        return {
            "status":         "ok",
            "mock_mode":      self.mock_mode,
            "style_rec":      "loaded" if self._style_rec else "mock",
            "brand_rec":      "loaded" if self._brand_rec else "mock",
            "profile_mgr":    "loaded" if self._profile_mgr else "mock",
            "style_count":    len(_MOCK_STYLE_RECS),
            "brand_count":    len(_MOCK_BRAND_RECS),
            "outfit_count":   len(_MOCK_OUTFIT_TEMPLATES),
            "message": (
                "RecommendationService loaded with real recommenders."
                if not self.mock_mode
                else "RecommendationService running in mock mode."
            ),
        }
