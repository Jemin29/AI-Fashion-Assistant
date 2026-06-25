"""
week5/knowledge_base/fashion_knowledge_base.py
==============================================
Fashion Knowledge Base Module for RAG system.
Stores and manages domain information across styles, color theory, seasons,
fabrics, categories, brand profiles, and seasonal trends.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator

VALID_CATEGORIES = {
    "fashion_styles",
    "color_theory",
    "fashion_seasons",
    "fabric_types",
    "fashion_categories",
    "brand_profiles",
    "streetwear_trends",
    "luxury_trends",
    "fashion_terminology"
}


def slugify(text: str) -> str:
    """Convert text into a safe lowercase alphanumeric identifier with underscores."""
    s = text.lower().strip()
    s = re.sub(r"[\s\-\/]+", "_", s)
    s = re.sub(r"[^\w\_]+", "", s)
    return s


class KnowledgeItem(BaseModel):
    """Data model representing a single structured knowledge base entry."""
    id: str = Field(description="Unique alphanumeric identifier for the item.")
    category: str = Field(description="Domain category.")
    name: str = Field(description="Name/label of the fashion concept.")
    content: str = Field(description="Textual description or definition of the concept.")
    tags: List[str] = Field(default_factory=list, description="List of search tags.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes.")
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    @field_validator("category")
    @classmethod
    def validate_category(cls, val: str) -> str:
        cat = val.lower().strip()
        if cat not in VALID_CATEGORIES:
            raise ValueError(f"Category '{val}' not supported. Choose from: {VALID_CATEGORIES}")
        return cat


class FashionKnowledgeBase:
    """
    Local JSON-backed database containing fashion intelligence domain knowledge.
    Provides schema-validated CRUD operations, tag searches, and category filter operations.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the Fashion Knowledge Base.

        Parameters
        ----------
        db_path : Path or str, optional
            Path to the JSON database. Defaults to outputs/knowledge_base/fashion_knowledge_base.json.
        """
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path("outputs/knowledge_base/fashion_knowledge_base.json").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.items: Dict[str, KnowledgeItem] = {}
        self._load_db()

        # Seed defaults if database is empty
        if not self.items:
            logger.info("Fashion Knowledge Base database empty. Seeding default domain records...")
            self.seed_default_knowledge()

    def _load_db(self) -> None:
        """Load items from the JSON database file."""
        if not self.db_path.exists():
            self.items = {}
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.items = {}
            for item_id, item_dict in data.items():
                try:
                    self.items[item_id] = KnowledgeItem(**item_dict)
                except Exception as err:
                    logger.error(f"Failed to parse database item '{item_id}': {err}")
            logger.info(f"Loaded {len(self.items)} records from Fashion Knowledge Base database: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to read database file {self.db_path}: {err}. Starting empty.")
            self.items = {}

    def _save_db(self) -> None:
        """Serialize and save database items to disk."""
        try:
            data = {item_id: item.model_dump() for item_id, item in self.items.items()}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.debug(f"Saved {len(self.items)} records to database file: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to save database file {self.db_path}: {err}")

    # ── CRUD Operations ──────────────────────────────────────────────────────

    def create_item(
        self,
        category: str,
        name: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeItem:
        """
        Create and insert a new knowledge item into the database.

        Parameters
        ----------
        category : str
        name : str
        content : str
        tags : list of str, optional
        metadata : dict, optional

        Returns
        -------
        KnowledgeItem
        """
        clean_cat = category.lower().strip()
        if clean_cat not in VALID_CATEGORIES:
            raise ValueError(f"Category '{category}' is invalid. Supported: {VALID_CATEGORIES}")

        item_id = f"kb_{clean_cat}_{slugify(name)}"
        if item_id in self.items:
            raise ValueError(f"An item with name '{name}' already exists in category '{clean_cat}' (ID: {item_id}).")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        item = KnowledgeItem(
            id=item_id,
            category=clean_cat,
            name=name,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
            created_at=timestamp,
            updated_at=timestamp
        )

        self.items[item_id] = item
        self._save_db()
        logger.success(f"Created knowledge item | ID={item_id} | category={clean_cat}")
        return item

    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """
        Retrieve a knowledge item by ID.

        Parameters
        ----------
        item_id : str

        Returns
        -------
        KnowledgeItem or None
        """
        return self.items.get(item_id)

    def update_item(
        self,
        item_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeItem:
        """
        Update an existing knowledge item.

        Parameters
        ----------
        item_id : str
        name : str, optional
        content : str, optional
        tags : list of str, optional
        metadata : dict, optional

        Returns
        -------
        KnowledgeItem
        """
        if item_id not in self.items:
            raise KeyError(f"Item with ID '{item_id}' not found in database.")

        item = self.items[item_id]
        
        if name is not None:
            item.name = name
        if content is not None:
            item.content = content
        if tags is not None:
            item.tags = tags
        if metadata is not None:
            # Merge or overwrite metadata dict
            item.metadata = {**item.metadata, **metadata}

        item.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.items[item_id] = item
        self._save_db()
        logger.success(f"Updated knowledge item | ID={item_id}")
        return item

    def delete_item(self, item_id: str) -> bool:
        """
        Delete a knowledge item from the database.

        Parameters
        ----------
        item_id : str

        Returns
        -------
        bool
            True if item was deleted, False if item_id not found.
        """
        if item_id not in self.items:
            logger.warning(f"Delete target ID '{item_id}' not found in database.")
            return False

        del self.items[item_id]
        self._save_db()
        logger.success(f"Deleted knowledge item | ID={item_id}")
        return True

    def list_items(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[KnowledgeItem]:
        """
        List indexed items matching category and/or tag query filters.

        Parameters
        ----------
        category : str, optional
        tags : list of str, optional

        Returns
        -------
        list of KnowledgeItem
        """
        results = list(self.items.values())

        if category:
            clean_cat = category.lower().strip()
            results = [item for item in results if item.category == clean_cat]

        if tags:
            clean_tags = [t.lower().strip() for t in tags]
            results = [
                item for item in results
                if any(t in [it.lower().strip() for it in item.tags] for t in clean_tags)
            ]

        return results

    # ── Database Seeding ─────────────────────────────────────────────────────

    def seed_default_knowledge(self) -> None:
        """Seed the database with comprehensive default fashion intelligence records."""
        # 1. Fashion Styles
        self.create_item(
            category="fashion_styles",
            name="Streetwear",
            content="A style of casual clothing which became global in the 1990s. Influenced by hip-hop, skate culture, sportswear, and contemporary haute couture. Characters are oversized cuts, graphical screen prints, drawstring hoodies, sneaker matching, and utility elements.",
            tags=["casual", "skate", "sportswear", "comfort"],
            metadata={"fit": "oversized", "key_items": ["hoodie", "sneakers", "graphic_tee"]}
        )
        self.create_item(
            category="fashion_styles",
            name="Luxury Haute-Couture",
            content="Opulent, high-end designer garments tailor-made by hand. Characterized by premium fabrics (silk, velvet), gold embroideries, sharp structured lines, patterned blazers, tailored fits, and high-fashion aesthetics.",
            tags=["formal", "designer", "tailoring", "embroidery"],
            metadata={"fit": "tailored", "key_items": ["gown", "blazer", "suit"]}
        )
        self.create_item(
            category="fashion_styles",
            name="Minimalist",
            content="Focuses on simple silhouettes, clean lines, organic textures, and neutral palettes. Avoids excessive ornamentations, printing, or branding, relying instead on high-quality fabrics and wardrobe essentials.",
            tags=["basic", "minimal", "essential", "neutral"],
            metadata={"fit": "regular", "key_items": ["cotton_tshirt", "linen_pant"]}
        )

        # 2. Color Theory
        self.create_item(
            category="color_theory",
            name="Monochromatic",
            content="A color scheme consisting of variations of a single hue, adjusted by shade, tone, and tint. Yields cohesive, elongating, and sleek fashion silhouettes.",
            tags=["color", "harmony", "hue", "monochrome"],
            metadata={"contrasts": "low", "recommended_palette": ["all_black", "all_grey", "all_beige"]}
        )
        self.create_item(
            category="color_theory",
            name="Analogous Color Scheme",
            content="Colors adjacent to each other on the color wheel, such as blue, blue-green, and green. Creates a harmonious, visually calming palette suited for coordinated separates.",
            tags=["color", "harmony", "adjacent", "blended"],
            metadata={"contrasts": "medium", "color_wheel_span": 3}
        )
        self.create_item(
            category="color_theory",
            name="Complementary Colors",
            content="Colors directly opposite each other on the color wheel (e.g. red/green, blue/orange). High contrast palettes that make design details pop dramatically.",
            tags=["color", "contrast", "pop", "complementary"],
            metadata={"contrasts": "high", "vibrancy": "extreme"}
        )

        # 3. Fashion Seasons
        self.create_item(
            category="fashion_seasons",
            name="Spring Summer",
            content="Warm-weather collections prioritizing breathable, lightweight fabrics like linen, cotton, and silk. Palettes feature bright colors, pastel shades, and cream/beige tones, with loose cuts and activewear fits.",
            tags=["warm", "summer", "breathable", "lightweight"],
            metadata={"months": ["march", "august"], "average_temp": "25C+"}
        )
        self.create_item(
            category="fashion_seasons",
            name="Autumn Winter",
            content="Cold-weather collections focusing on heavy fabrics, insulation, and layering. Wool, cashmere, fleece, denim, and leather dominate. Colors lean toward dark, earthy, and rich neutral tones, with structured coats and jackets.",
            tags=["cold", "winter", "layering", "insulation"],
            metadata={"months": ["september", "february"], "average_temp": "under 15C"}
        )

        # 4. Fabric Types
        self.create_item(
            category="fabric_types",
            name="Denim",
            content="A sturdy cotton warp-faced twill textile in which the weft passes under two or more warp threads. Rugged, durable, commonly dyed with indigo to yield jeans and structured casual jackets.",
            tags=["cotton", "twill", "rugged", "casual"],
            metadata={"durability": "high", "breathability": "medium", "weight": "heavy"}
        )
        self.create_item(
            category="fabric_types",
            name="Linen",
            content="A woven textile made from the fibers of the flax plant. Exceptional breathability, quick-drying, and crisp textured weave. Ideal for warm weather, despite being prone to wrinkling.",
            tags=["flax", "breathable", "crisp", "summer"],
            metadata={"durability": "medium", "breathability": "extreme", "weight": "light"}
        )
        self.create_item(
            category="fabric_types",
            name="Silk",
            content="A luxury natural protein fiber produced by silkworms. Possesses high sheen, exceptionally soft texture, and fluid drape. Ideal for high-end luxury gowns and blouses.",
            tags=["natural", "sheen", "soft", "luxury"],
            metadata={"durability": "low", "breathability": "high", "weight": "light"}
        )

        # 5. Fashion Categories
        self.create_item(
            category="fashion_categories",
            name="Hoodies",
            content="A casual sweatshirt top featuring an attached drawstring hood. Usually made from knitted fleece or French terry cotton. Associated with sportswear and streetwear silhouettes.",
            tags=["sweatshirt", "hood", "sportswear", "fleece"],
            metadata={"has_hood": True, "casual_factor": 10}
        )
        self.create_item(
            category="fashion_categories",
            name="Blazers",
            content="A structured, collared jacket similar to a suit jacket but cut more casually. Often features double or single-breasted button plackets and structured shoulders.",
            tags=["jacket", "suit", "formal", "structured"],
            metadata={"has_hood": False, "formal_factor": 8}
        )

        # 6. Brand Profiles
        self.create_item(
            category="brand_profiles",
            name="Nike",
            content="Global athletic brand specializing in performance activewear. Design language blends functional tech fabrics, athletic fits, swoosh branding, and a sleek black/grey/white color palette.",
            tags=["brand", "athletic", "streetwear", "performance"],
            metadata={"colors": ["black", "grey", "white"], "fit": "athletic"}
        )
        self.create_item(
            category="brand_profiles",
            name="Gucci",
            content="Italian luxury fashion house specializing in high-end leather goods, tailoring, and avant-garde designs. Renowned for gold embroidery, velvet materials, and eccentric red/green palettes.",
            tags=["brand", "luxury", "tailoring", "embroidery"],
            metadata={"colors": ["brown", "red", "green", "gold"], "fit": "tailored"}
        )
        self.create_item(
            category="brand_profiles",
            name="Zara",
            content="Fast-fashion retail leader specializing in contemporary, trend-driven designs. Aesthetics emphasize minimalist cuts, neutral earth tones (beige, cream), and casual tailored garments.",
            tags=["brand", "casual", "contemporary", "minimalist"],
            metadata={"colors": ["beige", "cream", "black", "grey"], "fit": "regular"}
        )
        self.create_item(
            category="brand_profiles",
            name="H&M",
            content="Swedish multinational clothing retailer focusing on fast-fashion basics. Key offerings center on essential organic cotton t-shirts and casual staple trousers in a range of simple colors.",
            tags=["brand", "basics", "organic", "essential"],
            metadata={"colors": ["grey", "white", "red", "blue"], "fit": "regular"}
        )

        # 7. Streetwear Trends
        self.create_item(
            category="streetwear_trends",
            name="Hypebeast Cargo Joggers",
            content="Streetwear trend incorporating utility pocket cargo styling onto relaxed jogger silhouettes. Often paired with technical sneaker releases.",
            tags=["trend", "cargo", "pockets", "streetwear"],
            metadata={"popularity": "high", "season": "all_season"}
        )
        self.create_item(
            category="streetwear_trends",
            name="Technical windbreaker layering",
            content="Active layering trend using water-resistant tech windbreakers and drawstring hoodies to create structural panels.",
            tags=["trend", "windbreaker", "techwear", "layering"],
            metadata={"popularity": "high", "season": "autumn_winter"}
        )

        # 8. Luxury Trends
        self.create_item(
            category="luxury_trends",
            name="Velvet Gown Opulence",
            content="Opulent luxury trend featuring deep velvet fabrics decorated with gold thread embroidery and metallic chains for haute-couture gowns.",
            tags=["trend", "velvet", "gold", "gown"],
            metadata={"popularity": "medium", "season": "winter"}
        )
        self.create_item(
            category="luxury_trends",
            name="Patterned Silk Blazer",
            content="Contemporary luxury styling utilizing printed silk fabrics for slim-fit blazers with bold patterns.",
            tags=["trend", "silk", "blazer", "pattern"],
            metadata={"popularity": "medium", "season": "spring_summer"}
        )

        # 9. Fashion Terminology
        self.create_item(
            category="fashion_terminology",
            name="Silhouette",
            content="The overall outline, shape, or contour of a garment when worn on the body (e.g. A-line, oversized, hour-glass, boxy).",
            tags=["term", "shape", "contour", "outline"],
            metadata={"importance": "high"}
        )
        self.create_item(
            category="fashion_terminology",
            name="Drape",
            content="The fluid way a textile hangs, falls, or flows under its own weight on a body or mannequin. Soft fabrics (silk, rayon) have high drape, while stiff fabrics (denim, canvas) have low drape.",
            tags=["term", "fabric", "fall", "flow"],
            metadata={"importance": "high"}
        )
        self.create_item(
            category="fashion_terminology",
            name="Placket",
            content="An opening or slit in a garment (usually on a shirt neck or sleeve) that allows it to be put on easily, reinforced with a double-layered strip of fabric often hosting buttons and button holes.",
            tags=["term", "shirt", "button", "opening"],
            metadata={"importance": "medium"}
        )
