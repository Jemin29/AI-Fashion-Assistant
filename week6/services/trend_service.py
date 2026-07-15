"""
Week 6 — Trend Service
Adapter over TrendForecaster and TrendAnalyzer from Week 5.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger
from week6.services.base import ServiceResult

logger = get_logger(__name__)

_MOCK_TRENDS = [
    {"name": "Quiet Luxury", "velocity": 0.87, "season": "autumn_winter", "growth": "+41%", "confidence": 0.91},
    {"name": "Chrome Metallics", "velocity": 0.82, "season": "spring_summer", "growth": "+52%", "confidence": 0.88},
    {"name": "Utility Minimalism", "velocity": 0.76, "season": "autumn_winter", "growth": "+34%", "confidence": 0.84},
    {"name": "Biomorphic Prints", "velocity": 0.63, "season": "spring_summer", "growth": "+18%", "confidence": 0.72},
    {"name": "Asymmetric Tailoring", "velocity": 0.58, "season": "autumn_winter", "growth": "+23%", "confidence": 0.69},
    {"name": "Sheer Layering", "velocity": 0.54, "season": "spring_summer", "growth": "+29%", "confidence": 0.67},
    {"name": "Varsity Revival", "velocity": 0.48, "season": "autumn_winter", "growth": "+15%", "confidence": 0.61},
    {"name": "Micro-Florals", "velocity": 0.42, "season": "spring_summer", "growth": "+12%", "confidence": 0.55},
]


class TrendService:
    """Adapter over TrendForecaster and TrendAnalyzer from Week 5."""

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self._forecaster: Optional[Any] = None
        self._analyzer: Optional[Any] = None

        if not mock_mode:
            try:
                from src.rag.fashion_assistant import FashionAssistant
                self._assistant = FashionAssistant(force_mock_embeddings=False)
                self._forecaster = self._assistant.trend_forecaster
                self._analyzer = self._assistant.rag_coordinator.trend_analyzer
                logger.info("TrendService initialized with real forecaster")
            except Exception as exc:
                logger.warning(f"TrendService: real mode failed — {exc}")
                self.mock_mode = True

    def get_all_trends(self) -> ServiceResult[List[Dict[str, Any]]]:
        """Return all known trends sorted by velocity."""
        try:
            if not self.mock_mode and self._analyzer:
                res = self._analyzer.get_all_trends()
                return ServiceResult(success=True, data=res)
        except Exception as exc:
            logger.error(f"TrendService.get_all_trends error: {exc}")
        
        res = sorted(_MOCK_TRENDS, key=lambda x: x["velocity"], reverse=True)
        return ServiceResult(success=True, data=res)

    def explain_trend(self, trend_name: str) -> ServiceResult[Dict[str, Any]]:
        """Return detailed analysis for a specific trend."""
        try:
            if not self.mock_mode and self._analyzer:
                res = self._analyzer.explain(trend_name)
                return ServiceResult(success=True, data=res)
        except Exception as exc:
            logger.error(f"TrendService.explain_trend error: {exc}")

        # Find in mock trends
        for t in _MOCK_TRENDS:
            if trend_name.lower() in t["name"].lower():
                res = {
                    "trend": t["name"],
                    "velocity": t["velocity"],
                    "confidence": t["confidence"],
                    "growth_rate": t["growth"],
                    "target_season": t["season"],
                    "explanation": f"{t['name']} is gaining strong momentum in the fashion industry. "
                                   f"Velocity score: {t['velocity']:.2f}. Growth rate: {t['growth']}.",
                    "key_influences": ["Runway shows", "Social media adoption", "Celebrity endorsement"],
                    "target_demographics": ["25-34", "Urban millennials", "Style-conscious professionals"],
                }
                return ServiceResult(success=True, data=res)
        # Generic fallback
        res = {
            "trend": trend_name or "Unknown Trend",
            "velocity": 0.5,
            "confidence": 0.5,
            "growth_rate": "+15%",
            "target_season": "spring_summer",
            "explanation": f"Analysis for '{trend_name}' — trend data is being analyzed from the fashion knowledge base.",
            "key_influences": ["Fashion week", "Social media"],
            "target_demographics": ["General audience"],
        }
        return ServiceResult(success=True, data=res)

    def forecast_season(self, season: str) -> ServiceResult[List[Dict[str, Any]]]:
        """Forecast top trends for a given season."""
        try:
            if not self.mock_mode and self._forecaster:
                res = self._forecaster.forecast_trends(current_season=season)
                return ServiceResult(success=True, data=res)
        except Exception as exc:
            logger.error(f"TrendService.forecast_season error: {exc}")
        
        res = [t for t in _MOCK_TRENDS if t["season"] == season] or _MOCK_TRENDS[:3]
        return ServiceResult(success=True, data=res)

    def get_velocity_chart_data(self) -> ServiceResult[Dict[str, Any]]:
        """Return data suitable for a velocity bar/radar chart."""
        trends_res = self.get_all_trends()
        trends = trends_res.data or []
        res = {
            "labels": [t["name"] for t in trends],
            "velocities": [t["velocity"] for t in trends],
            "confidence": [t["confidence"] for t in trends],
            "growth": [float(t["growth"].replace("%", "").replace("+", "")) for t in trends],
        }
        return ServiceResult(success=True, data=res)

    def health_check(self) -> ServiceResult:
        res = {
            "status": "ok",
            "name": "TrendService",
            "mode": "mock" if self.mock_mode else "production"
        }
        return ServiceResult(success=True, data=res)

    @staticmethod
    def get_seasons() -> List[str]:
        return ["spring_summer", "autumn_winter"]
