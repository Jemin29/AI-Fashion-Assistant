from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_recommendation_service, get_trend_service
from week7.backend.configs.rate_limit import limiter
from week7.backend.services.recommendation_service import RecommendationService
from week6.services.trend_service import TrendService

router = APIRouter(prefix="/recommendations", tags=["Recommendations & Trend Explorer"])


class StyleRecommendationRequest(BaseModel):
    """Validation schema for style recommendations."""
    gender: str = Field("all", description="Target gender group (men, women, unisex, all).")
    style: str = Field("streetwear", description="Preferred base style category.")
    occasion: str = Field("casual", description="Target occasion category.")
    fit: str = Field("regular", description="Preferred clothing fit (regular, slim, oversized, relaxed, tailored).")
    limit: int = Field(4, ge=1, le=20, description="Maximum suggestions to return.")
    color: Optional[str] = Field("", description="Optional accent color preference.")


class BrandRecommendationRequest(BaseModel):
    """Validation schema for brand recommendations."""
    styles: List[str] = Field(..., description="List of preferred style terms.")
    aesthetic: Optional[str] = Field("", description="Preferred brand aesthetic (e.g. minimalist, contemporary).")
    price_range: Optional[str] = Field("", description="Target price category ($, $$, $$$, $$$$).")
    limit: int = Field(4, ge=1, le=20, description="Maximum suggestions to return.")


@router.post("/styles")
@limiter.limit("20/minute")
async def recommend_styles(
    request: Request,
    payload: StyleRecommendationRequest,
    rec_svc: RecommendationService = Depends(get_recommendation_service)
):
    """Generate style suggestions matching user occasion and fit parameters."""
    try:
        res = rec_svc.recommend_styles(
            gender=payload.gender,
            style=payload.style,
            occasion=payload.occasion,
            fit=payload.fit,
            n=payload.limit,
            color=payload.color or ""
        )
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Failed to recommend styles."
            )
        return {
            "success": True,
            "data": res.data,
            "meta": res.metadata
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.post("/brands")
@limiter.limit("20/minute")
async def recommend_brands(
    request: Request,
    payload: BrandRecommendationRequest,
    rec_svc: RecommendationService = Depends(get_recommendation_service)
):
    """Generate brand suggestions matching designer style collections."""
    try:
        res = rec_svc.recommend_brands(
            styles=payload.styles,
            aesthetic=payload.aesthetic or "",
            price_range=payload.price_range or "",
            n=payload.limit
        )
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Failed to recommend brands."
            )
        return {
            "success": True,
            "data": res.data,
            "meta": res.metadata
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/trends")
async def list_active_trends(
    trend_svc: TrendService = Depends(get_trend_service)
):
    """Retrieve all logged fashion trends ranked by velocity."""
    res = trend_svc.get_all_trends()
    if not res.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Failed to retrieve trends."
        )
    return {
        "success": True,
        "data": res.data
    }


@router.get("/trends/forecast")
async def get_seasonal_forecast(
    season: str = Query("spring_summer", description="Target season (spring_summer, autumn_winter)."),
    trend_svc: TrendService = Depends(get_trend_service)
):
    """Retrieve top forecasted trend cards for a given season."""
    res = trend_svc.forecast_season(season.strip().lower())
    if not res.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Failed to retrieve forecast."
        )
    return {
        "success": True,
        "data": res.data
    }
