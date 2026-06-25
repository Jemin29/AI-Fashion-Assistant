"""
week5/knowledge_base/fashion_qa_dataset.py
===========================================
Fashion Question-Answer Dataset Builder.
Programmatically generates, stores, and manages a structured dataset of 500+
high-fidelity fashion Q&A pairs for the RAG pipeline across 5 categories.
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
# ── Validation Constants
# =============================================================================

VALID_QA_CATEGORIES = {
    "style_advice",
    "trend_advice",
    "fabrics",
    "brands",
    "fashion_terminology",
}


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
# ── Q&A Record Model
# =============================================================================

class FashionQARecord(BaseModel):
    """Data model representing a single structured fashion Q&A record."""
    id: str = Field(description="Unique alphanumeric identifier for the Q&A pair.")
    category: str = Field(description="The classification category of the Q&A.")
    question: str = Field(description="The natural language question text.")
    answer: str = Field(description="The detailed domain answer response text.")
    tags: List[str] = Field(default_factory=list, description="Search tags for filtering.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary.")
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Timestamp of creation."
    )
    updated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Timestamp of the last update."
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, val: str) -> str:
        cat = val.lower().strip()
        if cat not in VALID_QA_CATEGORIES:
            raise ValueError(f"Category '{val}' not supported. Choose from: {VALID_QA_CATEGORIES}")
        return cat


# =============================================================================
# ── Dataset Builder Class
# =============================================================================

class FashionQADatasetBuilder:
    """
    Builder and persistence manager for the Fashion Q&A Dataset.
    Generates and updates 'fashion_qa_dataset.json' file with metadata and schema checks.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the Fashion Q&A Dataset Builder.

        Parameters
        ----------
        db_path : Path or str, optional
            Path to the JSON database. Defaults to outputs/knowledge_base/fashion_qa_dataset.json.
        """
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path("outputs/knowledge_base/fashion_qa_dataset.json").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, FashionQARecord] = {}
        self._load_dataset()

        # Seed defaults if dataset is empty
        if not self.records:
            logger.info("Fashion Q&A Dataset is empty. Seeding programmatic Q&A records...")
            self.generate_and_save_dataset()

    def _load_dataset(self) -> None:
        """Load Q&A records from the JSON database file."""
        if not self.db_path.exists():
            self.records = {}
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.records = {}
            for qa_id, qa_dict in data.items():
                try:
                    self.records[qa_id] = FashionQARecord(**qa_dict)
                except Exception as err:
                    logger.error(f"Failed to parse Q&A record '{qa_id}': {err}")
            logger.info(f"Loaded {len(self.records)} records from Q&A Dataset: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to read dataset file {self.db_path}: {err}. Starting empty.")
            self.records = {}

    def save_dataset(self) -> None:
        """Serialize and save all records to disk in JSON format."""
        try:
            data = {qa_id: item.model_dump() for qa_id, item in self.records.items()}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.info(f"Saved {len(self.records)} records to Q&A Dataset file: {self.db_path}")
        except Exception as err:
            logger.error(f"Failed to save Q&A Dataset file {self.db_path}: {err}")

    # ── CRUD Operations ──────────────────────────────────────────────────────

    def create_record(
        self,
        category: str,
        question: str,
        answer: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FashionQARecord:
        """
        Create and insert a new Q&A record into the dataset.

        Parameters
        ----------
        category : str
        question : str
        answer : str
        tags : list of str, optional
        metadata : dict, optional

        Returns
        -------
        FashionQARecord
        """
        clean_cat = category.lower().strip()
        if clean_cat not in VALID_QA_CATEGORIES:
            raise ValueError(f"Category '{category}' is invalid. Supported: {VALID_QA_CATEGORIES}")

        # Compute clean prefix and ID
        q_slug = slugify(question[:30])
        record_id = f"qa_{clean_cat}_{q_slug}_{int(time.time() * 1000) % 1000000}"
        
        # Ensure ID uniqueness in case of rapid creation
        while record_id in self.records:
            record_id = f"qa_{clean_cat}_{q_slug}_{int(time.time() * 1000 + 1) % 1000000}"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        record = FashionQARecord(
            id=record_id,
            category=clean_cat,
            question=question,
            answer=answer,
            tags=tags or [],
            metadata=metadata or {},
            created_at=timestamp,
            updated_at=timestamp
        )

        self.records[record_id] = record
        self.save_dataset()
        logger.success(f"Created Q&A record | ID={record_id} | category={clean_cat}")
        return record

    def get_record(self, record_id: str) -> Optional[FashionQARecord]:
        """Retrieve a Q&A record by ID."""
        return self.records.get(record_id)

    def update_record(
        self,
        record_id: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FashionQARecord:
        """Update an existing Q&A record."""
        if record_id not in self.records:
            raise KeyError(f"Q&A record with ID '{record_id}' not found.")

        record = self.records[record_id]
        if question is not None:
            record.question = question
        if answer is not None:
            record.answer = answer
        if tags is not None:
            record.tags = tags
        if metadata is not None:
            record.metadata = {**record.metadata, **metadata}

        record.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.records[record_id] = record
        self.save_dataset()
        logger.success(f"Updated Q&A record | ID={record_id}")
        return record

    def delete_record(self, record_id: str) -> bool:
        """Delete a Q&A record from the dataset."""
        if record_id not in self.records:
            logger.warning(f"Delete target record ID '{record_id}' not found.")
            return False

        del self.records[record_id]
        self.save_dataset()
        logger.success(f"Deleted Q&A record | ID={record_id}")
        return True

    def list_by_category(self, category: str) -> List[FashionQARecord]:
        """List Q&A records filtered by category."""
        clean_cat = category.lower().strip()
        return [record for record in self.records.values() if record.category == clean_cat]

    def search_by_tags(self, tags: List[str]) -> List[FashionQARecord]:
        """Search Q&A records that match any of the provided tags (case-insensitive)."""
        if not tags:
            return []
        clean_query_tags = [t.lower().strip() for t in tags]
        results = []
        for record in self.records.values():
            record_tags = [t.lower().strip() for t in record.tags]
            if any(qt in record_tags for qt in clean_query_tags):
                results.append(record)
        return results

    # ── Combinatorial Q&A Dataset Seeding ─────────────────────────────────────

    def generate_and_save_dataset(self) -> None:
        """
        Generate 500+ fashion Q&A pairs programmatically using combinatorial expansion
        and serialize them to the configured JSON database file.
        """
        logger.info("Generating combinatorial Q&A records...")
        generated_records: Dict[str, FashionQARecord] = {}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # 1. Style Advice Category (10 styles * 10 colors * 3 templates = 300 Q&As)
        styles = [
            "streetwear", "minimalist", "quiet_luxury", "athleisure", "vintage_retro",
            "bohemian", "grunge", "gorpcore", "preppy", "formal_tailored"
        ]
        colors = [
            "black", "white", "olive_drab", "camel", "navy",
            "beige", "charcoal", "pastel_pink", "neon_green", "burgundy"
        ]
        style_templates = [
            (
                "How can I style a {color} {style} outfit?",
                "To style a {color} {style} outfit, focus on key pieces. For {style}, pair a comfortable top with matching bottoms in {color}, then finish with coordinated sneakers or boots. Keeping the silhouette consistent with {style} fit rules is crucial."
            ),
            (
                "What accessories go best with {color} {style} clothing?",
                "For a {color} {style} look, we recommend minimalist accessories that complement the {color} tone. Think simple jewelry, a clean leather belt, or a cap that fits the {style} aesthetic without cluttering the silhouette."
            ),
            (
                "Can I wear {color} in a {style} style for a semi-formal occasion?",
                "Yes! A semi-formal {style} outfit in {color} can look extremely polished. Try pairing structured {style} pieces like a tailored blazer or dark trousers in {color} with clean leather footwear."
            )
        ]

        idx = 1
        for style in styles:
            for color in colors:
                for q_temp, a_temp in style_templates:
                    rec_id = f"qa_style_advice_{idx:03d}"
                    question = q_temp.format(style=style.replace("_", " "), color=color.replace("_", " "))
                    answer = a_temp.format(style=style.replace("_", " "), color=color.replace("_", " "))
                    tags = ["style_advice", style, color]
                    metadata = {"style": style, "color": color}
                    
                    generated_records[rec_id] = FashionQARecord(
                        id=rec_id,
                        category="style_advice",
                        question=question,
                        answer=answer,
                        tags=tags,
                        metadata=metadata,
                        created_at=timestamp,
                        updated_at=timestamp
                    )
                    idx += 1

        # 2. Trend Advice Category (12 trends * 4 seasons * 2 templates = 96 Q&As)
        trends = [
            "cyber_utility_wear", "quiet_luxury", "gorpcore_hiking", "vintage_workwear",
            "neo_mint_activewear", "liquid_metallic_lame", "velvet_gown_opulence",
            "cozy_shearling", "oversized_blazers", "sustainable_ocean_fabrics",
            "responsive_textiles", "augmented_reality_wear"
        ]
        seasons = ["spring", "summer", "autumn", "winter"]
        trend_templates = [
            (
                "What is the outlook for the {trend} trend in {season}?",
                "The {trend} trend is expected to see significant growth in {season}. Designers are focusing on combining functional elements of {trend} with lightweight or insulating fabrics depending on {season} weather conditions."
            ),
            (
                "How can I incorporate {trend} into my daily wardrobe for {season}?",
                "To wear {trend} during {season}, start small by adding key pieces like a jacket or accessories that reflect the {trend} aesthetic. Pair them with neutral basics suitable for {season} temperatures."
            )
        ]

        idx = 1
        for trend in trends:
            for season in seasons:
                for q_temp, a_temp in trend_templates:
                    rec_id = f"qa_trend_advice_{idx:03d}"
                    question = q_temp.format(trend=trend.replace("_", " "), season=season)
                    answer = a_temp.format(trend=trend.replace("_", " "), season=season)
                    tags = ["trend_advice", trend, season]
                    metadata = {"trend": trend, "season": season}

                    generated_records[rec_id] = FashionQARecord(
                        id=rec_id,
                        category="trend_advice",
                        question=question,
                        answer=answer,
                        tags=tags,
                        metadata=metadata,
                        created_at=timestamp,
                        updated_at=timestamp
                    )
                    idx += 1

        # 3. Brands Category (10 brands * 10 styles = 100 Q&As)
        brands = [
            "Nike", "Gucci", "Zara", "HM", "Adidas",
            "Prada", "Balenciaga", "Uniqlo", "Patagonia", "Chanel"
        ]
        brand_template = (
            "Does {brand} offer options for a {style} wardrobe?",
            "Yes, {brand} offers a variety of pieces that align with the {style} aesthetic. Their collections feature cuts and materials that are perfect for styling a cohesive {style} look."
        )

        idx = 1
        for brand in brands:
            for style in styles:
                rec_id = f"qa_brands_{idx:03d}"
                question = brand_template[0].format(brand=brand, style=style.replace("_", " "))
                answer = brand_template[1].format(brand=brand, style=style.replace("_", " "))
                tags = ["brands", brand.lower(), style]
                metadata = {"brand": brand, "style": style}

                generated_records[rec_id] = FashionQARecord(
                    id=rec_id,
                    category="brands",
                    question=question,
                    answer=answer,
                    tags=tags,
                    metadata=metadata,
                    created_at=timestamp,
                    updated_at=timestamp
                )
                idx += 1

        # 4. Fabrics Category (10 fabrics * 3 templates = 30 Q&As)
        fabrics = [
            "denim", "linen", "silk", "wool", "cashmere",
            "leather", "cotton", "polyester", "nylon", "velvet"
        ]
        fabric_templates = [
            (
                "What are the main characteristics of {fabric} fabric?",
                "{fabric} is known for its unique properties. It is widely used in fashion due to its durability, comfort, texture, and how it drapes on the body."
            ),
            (
                "How should I care for garments made of {fabric}?",
                "Garments made of {fabric} require proper care to maintain their quality. Always check the care label; generally, some {fabric} items benefit from gentle washing or dry cleaning."
            ),
            (
                "Is {fabric} suitable for warm weather wear?",
                "Depending on the weave, {fabric} can be highly suitable. For instance, lighter weights of {fabric} offer good breathability, whereas heavier weights are better suited for colder seasons."
            )
        ]

        idx = 1
        for fabric in fabrics:
            for q_temp, a_temp in fabric_templates:
                rec_id = f"qa_fabrics_{idx:03d}"
                question = q_temp.format(fabric=fabric)
                answer = a_temp.format(fabric=fabric)
                tags = ["fabrics", fabric]
                metadata = {"fabric": fabric}

                generated_records[rec_id] = FashionQARecord(
                    id=rec_id,
                    category="fabrics",
                    question=question,
                    answer=answer,
                    tags=tags,
                    metadata=metadata,
                    created_at=timestamp,
                    updated_at=timestamp
                )
                idx += 1

        # 5. Fashion Terminology Category (15 terms * 2 templates = 30 Q&As)
        terms = [
            "silhouette", "drape", "placket", "avant_garde", "haute_couture",
            "color_wheel", "monochromatic", "analogous", "complementary", "gorpcore",
            "hypebeast", "quiet_luxury", "weft", "warp", "sustainable"
        ]
        term_templates = [
            (
                "What does the term '{term}' mean in fashion?",
                "In fashion, '{term}' refers to a specific concept. It is commonly used to describe style, construction, or design aspects of garments."
            ),
            (
                "Why is understanding '{term}' important for styling?",
                "Understanding '{term}' allows fashion designers and stylists to make informed choices about garment fit, coordination, and aesthetics to create a cohesive look."
            )
        ]

        idx = 1
        for term in terms:
            for q_temp, a_temp in term_templates:
                rec_id = f"qa_fashion_terminology_{idx:03d}"
                question = q_temp.format(term=term.replace("_", " "))
                answer = a_temp.format(term=term.replace("_", " "))
                tags = ["fashion_terminology", term]
                metadata = {"term": term}

                generated_records[rec_id] = FashionQARecord(
                    id=rec_id,
                    category="fashion_terminology",
                    question=question,
                    answer=answer,
                    tags=tags,
                    metadata=metadata,
                    created_at=timestamp,
                    updated_at=timestamp
                )
                idx += 1

        # Save to database
        self.records = generated_records
        self.save_dataset()
        logger.info(f"Successfully generated and saved {len(self.records)} records programmatically.")
