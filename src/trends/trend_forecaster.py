"""
week5/trends/trend_forecaster.py
================================
Fashion Trend Forecasting Engine.
Predicts future fashion trends by analyzing growth velocity (TrendAnalyzer),
style catalog popularity (FashionRetriever), and seasonal preferences.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.trends.trend_analyzer import TrendAnalyzer


class TrendForecaster:
    """
    Correlates real-time trend velocity, catalog popularities, and seasonal suitability
    to forecast future fashion trends with confidence ratings and qualitative reasoning.
    """

    # Structured database of predictive trend candidates
    FORECAST_TEMPLATES = [
        {
            "trend": "Cyber Utility Wear",
            "category": "streetwear",
            "base_confidence": 0.82,
            "seasonal_alignment": ["winter", "autumn_winter", "autumn", "cold"],
            "keywords": ["utility", "cargo", "pockets", "windbreaker", "layering", "techwear"],
            "reasoning": "High demand for utility cargo pockets and windbreaker layering coupled with cold-weather insulation needs."
        },
        {
            "trend": "Patterned Silk Minimalism",
            "category": "luxury",
            "base_confidence": 0.76,
            "seasonal_alignment": ["summer", "spring_summer", "spring", "warm"],
            "keywords": ["silk", "blazer", "pattern", "minimalist", "embroidery"],
            "reasoning": "Strong luxury trend towards printed silk blazers and custom embroidery in warm-weather separates."
        },
        {
            "trend": "Organic Linen Loungewear",
            "category": "minimalist",
            "base_confidence": 0.74,
            "seasonal_alignment": ["summer", "spring_summer", "spring", "warm"],
            "keywords": ["linen", "breathable", "minimalist", "essential", "cotton"],
            "reasoning": "Minimalist wardrobe shift prioritizing lightweight flax fabrics and breathable lounge cuts."
        },
        {
            "trend": "Active Loungewear Tech",
            "category": "athleisure",
            "base_confidence": 0.72,
            "seasonal_alignment": ["winter", "autumn_winter", "autumn", "cold"],
            "keywords": ["athletic", "fleece", "lounge", "sportswear", "grey"],
            "reasoning": "Expansion of active athleisure styles incorporating premium grey fleece lining for temperature regulation."
        }
    ]

    def __init__(
        self,
        analyzer: TrendAnalyzer,
        retriever: FashionRetriever
    ) -> None:
        """
        Initialize the Trend Forecaster.

        Parameters
        ----------
        analyzer : TrendAnalyzer
            Ingested trend velocity analyzer.
        retriever : FashionRetriever
            Semantic catalog retriever.
        """
        self.analyzer = analyzer
        self.retriever = retriever
        logger.info("TrendForecaster successfully initialized.")

    def forecast_trends(
        self,
        current_season: Optional[str] = None,
        n_predictions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate future trend predictions matching analyzer stats, database catalogs, and seasons.

        Parameters
        ----------
        current_season : str, optional
            The current or upcoming season filter.
        n_predictions : int
            Number of predictions to return.

        Returns
        -------
        List[Dict[str, Any]]
            Sorted list of future trend predictions.
        """
        season_clean = (current_season or "").strip().lower()
        logger.info(f"Generating trend forecasts for season: '{season_clean or 'unspecified'}'")

        # 1. Fetch active/rising trends from the analyzer
        forecast_data = self.analyzer.get_trend_forecast()
        rising_trends = {t["element"].lower(): t for t in forecast_data.get("rising_trends", [])}
        stable_trends = {t["element"].lower(): t for t in forecast_data.get("stable_trends", [])}

        predictions = []

        # 2. Iterate through candidate templates and compute suitability scores
        for template in self.FORECAST_TEMPLATES:
            confidence = template["base_confidence"]
            boost_reasons = []

            # A. Check seasonal alignment
            if season_clean:
                if season_clean in template["seasonal_alignment"] or any(s in season_clean for s in template["seasonal_alignment"]):
                    confidence += 0.07
                    boost_reasons.append("aligned with target season")
                else:
                    confidence -= 0.05
                    boost_reasons.append("out-of-season penalty applied")

            # B. Correlate keywords with rising/stable trends from the analyzer
            keyword_matches = 0
            for kw in template["keywords"]:
                if kw in rising_trends:
                    growth_val = rising_trends[kw]["growth_rate"]
                    confidence += min(0.05, growth_val * 0.05)  # Scale boost with growth velocity
                    keyword_matches += 1
                elif kw in stable_trends:
                    confidence += 0.02
                    keyword_matches += 1

            if keyword_matches > 0:
                boost_reasons.append(f"correlated with {keyword_matches} active trend keywords")

            # Clamp confidence to a realistic range [0.1, 0.99]
            confidence = round(max(0.1, min(0.99, confidence)), 2)

            # Build reasoning summary
            reason_text = template["reasoning"]
            if boost_reasons:
                reason_text += f" (Forecast confidence adjusted based on: {', '.join(boost_reasons)})."

            predictions.append({
                "trend": template["trend"],
                "confidence": confidence,
                "category": template["category"],
                "reasoning": reason_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            })

        # 3. Sort predictions descending by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:n_predictions]
