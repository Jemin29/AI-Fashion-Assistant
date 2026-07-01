from __future__ import annotations

from typing import Any, Dict, List, Optional
from week6.services.recommendation_service import RecommendationService as Week6RecommendationService


class RecommendationService:
    """Business logic for style and brand recommendations."""

    def __init__(self, mock_mode: bool = True) -> None:
        self._svc = Week6RecommendationService(mock_mode=mock_mode)

    def recommend_styles(
        self,
        gender: str = "all",
        style: str = "streetwear",
        occasion: str = "casual",
        fit: str = "regular",
        n: int = 4,
        color: str = "",
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if preferences is not None:
            gender = preferences.get("gender", gender)
            style = preferences.get("style", style)
            occasion = preferences.get("occasion", occasion)
            fit = preferences.get("fit", fit)
            color = preferences.get("color", color)

        res = self._svc.recommend_styles(
            gender=gender,
            style=style,
            occasion=occasion,
            fit=fit,
            n=n,
            color=color
        )

        if not res.is_ok:
            raise ValueError(res.error or "Style recommendations failed.")
        return res.data

    def recommend_brands(
        self,
        styles: List[str] = None,
        aesthetic: str = "",
        price_range: str = "",
        n: int = 4,
        profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if profile is not None:
            styles = profile.get("styles", styles)
            aesthetic = profile.get("aesthetic", aesthetic)
            price_range = profile.get("price_range", price_range)

        res = self._svc.recommend_brands(
            styles=styles or [],
            aesthetic=aesthetic,
            price_range=price_range,
            n=n
        )

        if not res.is_ok:
            raise ValueError(res.error or "Brand recommendations failed.")
        return res.data

    def health_check(self) -> Dict[str, Any]:
        res = self._svc.health_check()
        if hasattr(res, "data"):
            return res.data
        return res
