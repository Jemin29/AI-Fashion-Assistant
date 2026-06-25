"""
week5/knowledge_base/trend_dataset_builder.py
============================================
Fashion Trend Dataset Builder for Week 5.
Collects, structures, validates, and stores trend records across streetwear,
luxury, seasonal, color, fabric, and forecast categories.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ── Trend Item Model
# =============================================================================

class TrendItem(BaseModel):
    """Data model representing a structured fashion trend entry."""
    id: str = Field(description="Unique identifier for the trend.")
    category: str = Field(description="Trend category classification.")
    name: str = Field(description="Visual name of the trend.")
    description: str = Field(description="Detailed narrative explaining the trend.")
    popularity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Popularity metric score between 0.0 and 1.0."
    )
    growth_rate: float = Field(
        default=0.0,
        description="Mentions growth rate coefficient (velocity metric)."
    )
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Timestamp of trend registration/update."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom attributes (e.g. key_colors, textures, subcultures)."
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, val: str) -> str:
        cat = val.lower().strip()
        valid = {"streetwear", "luxury", "seasonal", "color", "fabric", "forecast"}
        if cat not in valid:
            raise ValueError(f"Category '{val}' not supported. Choose from: {valid}")
        return cat


# =============================================================================
# ── Helpers
# =============================================================================

def slugify(text: str) -> str:
    """Convert text into a safe lowercase alphanumeric identifier with underscores."""
    s = text.lower().strip()
    s = re.sub(r"[\s\-\/]+", "_", s)
    s = re.sub(r"[^\w\_]+", "", s)
    return s


# =============================================================================
# ── Trend Dataset Builder Class
# =============================================================================

class TrendDatasetBuilder:
    """
    Builder and persistent database manager for the Fashion Trend Dataset.
    Generates and updates 'trend_dataset.json' file with metadata and schema checks.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the Trend Dataset Builder.

        Parameters
        ----------
        db_path : Path or str, optional
            Path to the JSON database. Defaults to outputs/knowledge_base/trend_dataset.json.
        """
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path("outputs/knowledge_base/trend_dataset.json").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trends: Dict[str, TrendItem] = {}
        self._load_dataset()

        # Seed defaults if dataset is empty
        if not self.trends:
            logger.info("Fashion Trend Dataset is empty. Seeding high-fidelity trend records...")
            self.seed_default_trends()

    def _load_dataset(self) -> None:
        """Load trend items from the JSON database file."""
        if not self.db_path.exists():
            self.trends = {}
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.trends = {}
            for trend_id, trend_dict in data.items():
                try:
                    self.trends[trend_id] = TrendItem(**trend_dict)
                except Exception as err:
                    logger.error(f"Failed to parse trend item '{trend_id}': {err}")
            logger.info(f"Loaded {len(self.trends)} records from Trend Dataset: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to read dataset file {self.db_path}: {err}. Starting empty.")
            self.trends = {}

    def save_dataset(self) -> None:
        """Serialize and save all trends to disk in JSON format."""
        try:
            data = {trend_id: item.model_dump() for trend_id, item in self.trends.items()}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.info(f"Saved {len(self.trends)} records to Trend Dataset file: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to save Trend Dataset file {self.db_path}: {err}")

    # ── CRUD Operations ──────────────────────────────────────────────────────

    def add_trend(
        self,
        category: str,
        name: str,
        description: str,
        popularity_score: float = 0.5,
        growth_rate: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrendItem:
        """
        Add a new trend item to the dataset.

        Parameters
        ----------
        category : str
            One of: streetwear, luxury, seasonal, color, fabric, forecast.
        name : str
        description : str
        popularity_score : float
        growth_rate : float
        metadata : dict, optional

        Returns
        -------
        TrendItem
        """
        clean_cat = category.lower().strip()
        # Trigger validation via dummy instantiation to fail fast
        # (this throws ValidationError if invalid)
        dummy = TrendItem(
            id="temp",
            category=clean_cat,
            name=name,
            description=description,
            popularity_score=popularity_score,
            growth_rate=growth_rate
        )

        trend_id = f"trend_{clean_cat}_{slugify(name)}"
        if trend_id in self.trends:
            raise ValueError(f"Trend '{name}' already exists in category '{clean_cat}' (ID: {trend_id}).")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        item = TrendItem(
            id=trend_id,
            category=clean_cat,
            name=name,
            description=description,
            popularity_score=popularity_score,
            growth_rate=growth_rate,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        self.trends[trend_id] = item
        self.save_dataset()
        logger.success(f"Added trend record | ID={trend_id} | category={clean_cat}")
        return item

    def get_trend(self, trend_id: str) -> Optional[TrendItem]:
        """Retrieve a trend by ID."""
        return self.trends.get(trend_id)

    def update_trend(
        self,
        trend_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        popularity_score: Optional[float] = None,
        growth_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrendItem:
        """Update an existing trend item."""
        if trend_id not in self.trends:
            raise KeyError(f"Trend item with ID '{trend_id}' not found.")

        item = self.trends[trend_id]
        if name is not None:
            item.name = name
        if description is not None:
            item.description = description
        if popularity_score is not None:
            item.popularity_score = popularity_score
        if growth_rate is not None:
            item.growth_rate = growth_rate
        if metadata is not None:
            item.metadata = {**item.metadata, **metadata}

        item.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.trends[trend_id] = item
        self.save_dataset()
        logger.success(f"Updated trend record | ID={trend_id}")
        return item

    def delete_trend(self, trend_id: str) -> bool:
        """Delete a trend item from the dataset."""
        if trend_id not in self.trends:
            logger.warning(f"Delete target trend ID '{trend_id}' not found.")
            return False

        del self.trends[trend_id]
        self.save_dataset()
        logger.success(f"Deleted trend record | ID={trend_id}")
        return True

    def list_trends(self, category: Optional[str] = None) -> List[TrendItem]:
        """List trends, optionally filtered by category."""
        results = list(self.trends.values())
        if category:
            clean_cat = category.lower().strip()
            results = [item for item in results if item.category == clean_cat]
        return results

    # ── Seeding Default Dataset ──────────────────────────────────────────────

    def seed_default_trends(self) -> None:
        """Seed high-fidelity default records across all 6 trend categories."""
        # 1. Streetwear Trends
        self.add_trend(
            category="streetwear",
            name="Oversized Utility Cargoes",
            description="Relaxed-fit cargo pants featuring oversized modular multi-pockets, heavy strap details, and water-repellent nylon construction. Highly popularized by Japanese street culture.",
            popularity_score=0.88,
            growth_rate=0.15,
            metadata={"aesthetic": "gorpcore", "fabrics": ["ripstop_nylon", "canvas"], "colors": ["olive_drab", "matte_black"]}
        )
        self.add_trend(
            category="streetwear",
            name="Distressed Graphic Hoodies",
            description="French terry cotton hoodies characterized by faded vintage washes, frayed edge distressing, boxy/cropped silhouettes, and large puff-printed typography.",
            popularity_score=0.85,
            growth_rate=0.08,
            metadata={"aesthetic": "grunge", "fit": "boxy_cropped", "key_elements": ["puff_print", "vintage_wash"]}
        )

        # 2. Luxury Fashion Trends
        self.add_trend(
            category="luxury",
            name="Quiet Luxury Deconstructivism",
            description="Tailored garments made from ultra-premium cashmere and fine wools that feature subverted tailoring, asymmetric buttoning, raw hems, and lack visible branding.",
            popularity_score=0.92,
            growth_rate=0.22,
            metadata={"aesthetic": "quiet_luxury", "colors": ["camel", "oatmeal", "navy"], "price_tier": "haute_couture"}
        )
        self.add_trend(
            category="luxury",
            name="Metallic Liquid Lamé Gowns",
            description="Avant-garde evening wear using high-shine metallic threads woven into silks to create a liquid draping effect resembling molten silver and gold.",
            popularity_score=0.75,
            growth_rate=0.12,
            metadata={"aesthetic": "high_glam", "key_fabrics": ["metallic_lamé", "silk_satin"]}
        )

        # 3. Seasonal Trends
        self.add_trend(
            category="seasonal",
            name="Summer Linen Separates",
            description="Ultra-lightweight, highly breathable co-ord sets containing loose-fit open collar shirts and elasticated drawstring trousers. Ideal for high heat conditions.",
            popularity_score=0.90,
            growth_rate=0.25,
            metadata={"season": "spring_summer", "fabrics": ["pure_linen"], "colors": ["ecru", "sand", "soft_blue"]}
        )
        self.add_trend(
            category="seasonal",
            name="Winter Shearling Aviator Jackets",
            description="Heavyweight leather aviator jackets lined with thick insulating sheepskin shearling. Features leather buckles and double collar straps.",
            popularity_score=0.78,
            growth_rate=0.05,
            metadata={"season": "autumn_winter", "fabrics": ["sheepskin_shearling", "cowhide_leather"]}
        )

        # 4. Color Trends
        self.add_trend(
            category="color",
            name="Digital Neo-Mint",
            description="An optimistic, gender-neutral tech pastel hue that bridges the gap between science and nature. Dominated by youth-focused activewear and streetwear brands.",
            popularity_score=0.82,
            growth_rate=0.18,
            metadata={"hex": "#98FF98", "pantone_ref": "15-6437_TCX", "target_demographic": "gen_z"}
        )
        self.add_trend(
            category="color",
            name="Earthy Terracotta Rust",
            description="Warm, rich clay-based orange-brown hues that evoke nature and grounding. Extremely popular in cozy knitwear and heavy winter outerwear palettes.",
            popularity_score=0.80,
            growth_rate=0.11,
            metadata={"hex": "#C2B280", "associated_seasons": ["autumn"]}
        )

        # 5. Fabric Trends
        self.add_trend(
            category="fabric",
            name="Recycled Ocean Polyester",
            description="Sustainable performance technical textiles synthesized from upcycled ocean plastics and discarded nets. Features high durability, moisture wicking, and eco certifications.",
            popularity_score=0.86,
            growth_rate=0.30,
            metadata={"certificates": ["GRS", "OEKO-TEX"], "durability": "extreme"}
        )
        self.add_trend(
            category="fabric",
            name="Organic Slub Cotton Jersey",
            description="Eco-friendly cotton jersey characterized by slight irregularities in thread thickness, giving a rich textured surface and highly breathable qualities.",
            popularity_score=0.84,
            growth_rate=0.07,
            metadata={"weight": "light-to-medium", "feel": "soft_textured"}
        )

        # 6. Fashion Forecasts
        self.add_trend(
            category="forecast",
            name="Biophilic Responsive Textiles",
            description="Future fabric concept infused with micro-algae or responsive biopolymers that actively react to carbon dioxide and humidity, changing texture and color patterns in real-time.",
            popularity_score=0.45,
            growth_rate=0.40,
            metadata={"horizon_years": "2028-2030", "rd_focus": "biotechnology"}
        )
        self.add_trend(
            category="forecast",
            name="Augmented Wearable Projections",
            description="Integration of clothes with AR-tags allowing wearers to project dynamic digital textures, colors, and 3D graphics onto physical white garments via smart lenses.",
            popularity_score=0.55,
            growth_rate=0.50,
            metadata={"horizon_years": "2027-2029", "tech_stack": ["ar_tags", "smart_lenses"]}
        )
