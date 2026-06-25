"""
week5/trends/trend_analyzer.py
==============================
Fashion Trend Tracker and Forecasting Engine for Week 5.
Ingests keyword mentions and analyzes growth velocity over sliding time windows.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from src.utils.config_manager import TrendConfig, get_default_config


class TrendAnalyzer:
    """
    Ingests and tracks occurrences of fashion keywords (styles, colors, fabrics)
    and classifies them as active trends based on frequency and growth velocity.
    """

    def __init__(self, config: Optional[TrendConfig] = None) -> None:
        """
        Initialize the Trend Analyzer.

        Parameters
        ----------
        config : TrendConfig, optional
            Config parameters. If omitted, uses default config.
        """
        if config is None:
            self.config = get_default_config().trends
        else:
            self.config = config

        # List of dicts: [{"element": str, "timestamp": float}] (epoch time)
        self.mentions: List[Dict[str, Any]] = []

    def _parse_timestamp(self, timestamp: Optional[str]) -> float:
        """Convert string timestamp in %Y-%m-%d %H:%M:%S format to epoch float."""
        if not timestamp:
            return time.time()
        try:
            # Try standard format
            return time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            try:
                # Fallback to date only
                return time.mktime(time.strptime(timestamp, "%Y-%m-%d"))
            except ValueError:
                logger.warning(f"Could not parse timestamp '{timestamp}'. Defaulting to current time.")
                return time.time()

    def add_mention(self, element: str, timestamp: Optional[str] = None) -> None:
        """
        Log a single mention of a fashion element.

        Parameters
        ----------
        element : str
            The visual element, color, fabric, or style keyword.
        timestamp : str, optional
            Timestamp of the mention. Defaults to current time.
        """
        clean_elem = element.lower().strip()
        epoch_time = self._parse_timestamp(timestamp)
        self.mentions.append({"element": clean_elem, "timestamp": epoch_time})
        logger.debug(f"Logged mention | element={clean_elem} | time={epoch_time}")

    def add_mentions(self, elements: List[str], timestamp: Optional[str] = None) -> None:
        """
        Log a list of fashion element mentions in batch.

        Parameters
        ----------
        elements : list of str
        timestamp : str, optional
        """
        for elem in elements:
            self.add_mention(elem, timestamp)

    def get_active_trends(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter and rank active trends within the configured lookback window.
        Calculates growth velocity by comparing older and recent halves of the window.

        Parameters
        ----------
        category : str, optional
            Optional tag category filter.

        Returns
        -------
        list of dict of trend stats
        """
        now = time.time()
        window_seconds = self.config.time_window_days * 86400
        cutoff_time = now - window_seconds
        midpoint_time = now - (window_seconds / 2)

        # Filter mentions in lookback window
        active_mentions = [m for m in self.mentions if m["timestamp"] >= cutoff_time]

        # Aggregate counts: total, old half, recent half
        stats: Dict[str, Dict[str, int]] = {}
        for m in active_mentions:
            elem = m["element"]
            if elem not in stats:
                stats[elem] = {"total": 0, "old": 0, "new": 0}
            
            stats[elem]["total"] += 1
            if m["timestamp"] < midpoint_time:
                stats[elem]["old"] += 1
            else:
                stats[elem]["new"] += 1

        active_trends = []
        for elem, counts in stats.items():
            total = counts["total"]
            old_count = counts["old"]
            new_count = counts["new"]

            # Calculate growth velocity: (Recent - Older) / max(1, Older)
            growth = (new_count - old_count) / max(1, old_count)

            # Check criteria
            if total >= self.config.min_mention_count and growth >= self.config.growth_threshold:
                active_trends.append({
                    "element": elem,
                    "mentions_count": total,
                    "growth_rate": round(growth, 4),
                    "is_active": True
                })

        # Sort descending by growth rate then total count
        active_trends.sort(key=lambda x: (x["growth_rate"], x["mentions_count"]), reverse=True)
        return active_trends

    def get_trend_forecast(self) -> Dict[str, Any]:
        """
        Forecast rising and declining trends across all ingested mentions.

        Returns
        -------
        dict containing trend groups
        """
        now = time.time()
        # Lookback is time_window_days
        window_seconds = self.config.time_window_days * 86400
        cutoff_time = now - window_seconds

        active_mentions = [m for m in self.mentions if m["timestamp"] >= cutoff_time]
        
        # Aggregate counts
        counts: Dict[str, int] = {}
        for m in active_mentions:
            counts[m["element"]] = counts.get(m["element"], 0) + 1

        # Classify based on active trends criteria
        active_list = self.get_active_trends()
        active_names = {t["element"] for t in active_list}

        rising = []
        stable = []
        declining = []

        for elem, total in counts.items():
            # Find in active list for growth velocity
            active_info = next((t for t in active_list if t["element"] == elem), None)
            if active_info:
                rising.append({
                    "element": elem,
                    "count": total,
                    "growth_rate": active_info["growth_rate"]
                })
            elif total >= self.config.min_mention_count:
                stable.append({
                    "element": elem,
                    "count": total,
                    "growth_rate": 0.0
                })
            else:
                declining.append({
                    "element": elem,
                    "count": total
                })

        # Sort forecasts
        rising.sort(key=lambda x: x["growth_rate"], reverse=True)
        stable.sort(key=lambda x: x["count"], reverse=True)

        return {
            "rising_trends": rising,
            "stable_trends": stable,
            "emerging_keywords_count": len(declining),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
