"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/knowledge_base/fashion_domain_research.py
=============================================================================
MODULE PURPOSE
--------------
This module is the single source of truth for all fashion domain knowledge
used throughout the AI Fashion Design Assistant pipeline. It:

  1. Loads and validates the fashion_knowledge.json knowledge base.
  2. Provides the FashionDomainResearch class — a rich query API over the KB.
  3. Exposes standalone normalizer functions for attribute canonicalization.
  4. Exposes lookup helpers for cross-reference queries (style → category, etc.)
  5. Provides a FashionRecord validator with multi-layer rule checking.
  6. Defines typed data models (FashionRecord, ValidationResult, etc.)
  7. Generates derived JSON artifacts (category_mapping.json, style_profiles.json)
     and saves them to datasets/metadata/.

DESIGN PRINCIPLES
-----------------
  - Single Responsibility: This file owns all fashion taxonomy logic.
  - Immutability: The loaded knowledge base is never mutated after loading.
  - Graceful Degradation: All lookups return safe defaults (None, [], {}).
  - Type Safety: All public functions are fully type-annotated.
  - Comprehensive Logging: Every major operation is logged at appropriate level.
  - Testability: No side effects in constructors; all I/O is explicit.

USAGE EXAMPLES
--------------
    # 1. Basic knowledge base query
    from data_pipeline.knowledge_base import FashionDomainResearch

    kb = FashionDomainResearch()
    print(kb.get_category("t_shirts"))
    print(kb.get_style("streetwear"))

    # 2. Attribute normalization
    from data_pipeline.knowledge_base import normalize_color, normalize_style
    canon = normalize_color("navy blue")   # → "Navy"
    style = normalize_style("hype beast")  # → None (unknown)

    # 3. Cross-reference lookups
    from data_pipeline.knowledge_base import get_categories_for_gender
    cats = get_categories_for_gender("women")

    # 4. Record validation
    from data_pipeline.knowledge_base import validate_fashion_record
    result = validate_fashion_record({"category": "dresses", "gender": "men"})
    print(result.errors)  # ["Dresses are only valid for gender=women"]

    # 5. Generate derived artifacts
    kb.generate_all_artifacts()

=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

# ─── Third-party ──────────────────────────────────────────────────────────────
from loguru import logger


# =============================================================================
# ── Constants & Paths
# =============================================================================

# Absolute path to fashion_knowledge.json (relative to this file's location)
_MODULE_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT    = _MODULE_DIR.parent.parent          # fashion-ai-assistant/
_KNOWLEDGE_JSON  = _PROJECT_ROOT / "datasets" / "metadata" / "fashion_knowledge.json"
_METADATA_OUTPUT = _PROJECT_ROOT / "datasets" / "metadata"

# Valid genders, categories, styles — used throughout for guard clauses
VALID_GENDERS     = frozenset({"men", "women", "unisex"})
VALID_CATEGORIES  = frozenset({
    "t_shirts", "shirts", "hoodies", "jackets", "pants",
    "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
})
VALID_STYLES      = frozenset({
    "streetwear", "luxury", "formal", "business_casual",
    "techwear", "minimalist", "vintage", "athleisure"
})
VALID_FITS        = frozenset({
    "slim_fit", "regular_fit", "relaxed_fit", "oversized",
    "cropped", "skinny", "straight", "athletic_fit"
})
VALID_SEASONS     = frozenset({"spring", "summer", "autumn", "winter", "all_season"})
VALID_OCCASIONS   = frozenset({
    "casual", "business_casual", "formal", "party",
    "sport", "outdoor", "beach", "wedding_festive", "lounge"
})


# =============================================================================
# ── Typed Data Models
# =============================================================================

@dataclass
class FashionRecord:
    """
    Represents one annotated fashion item in the pipeline.

    Attributes map directly to the attribute taxonomy in fashion_knowledge.json.
    All fields except image_id, category, gender, and dataset_source are optional
    to accommodate partially annotated records from different datasets.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    image_id: str
    dataset_source: str                    # "fashiongen" | "deepfashion"

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    category: str                          # e.g. "t_shirts"
    gender: str                            # "men" | "women" | "unisex"
    subcategory: Optional[str] = None      # e.g. "graphic_tee"

    # ── Attributes ────────────────────────────────────────────────────────────
    colors: List[str]         = field(default_factory=list)   # ["Navy", "White"]
    fabrics: List[str]        = field(default_factory=list)   # ["Cotton"]
    styles: List[str]         = field(default_factory=list)   # ["streetwear"]
    fit: Optional[str]        = None                          # "slim_fit"
    patterns: List[str]       = field(default_factory=list)   # ["stripes"]
    seasons: List[str]        = field(default_factory=list)   # ["spring"]
    occasions: List[str]      = field(default_factory=list)   # ["casual"]

    # ── Derived / Pipeline Fields ─────────────────────────────────────────────
    description: Optional[str]     = None
    is_valid: bool                 = False
    validation_errors: List[str]   = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    created_at: str                = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "image_id"            : self.image_id,
            "dataset_source"      : self.dataset_source,
            "category"            : self.category,
            "gender"              : self.gender,
            "subcategory"         : self.subcategory,
            "colors"              : self.colors,
            "fabrics"             : self.fabrics,
            "styles"              : self.styles,
            "fit"                 : self.fit,
            "patterns"            : self.patterns,
            "seasons"             : self.seasons,
            "occasions"           : self.occasions,
            "description"         : self.description,
            "is_valid"            : self.is_valid,
            "validation_errors"   : self.validation_errors,
            "validation_warnings" : self.validation_warnings,
            "created_at"          : self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FashionRecord":
        """Deserialise from a plain dictionary."""
        return cls(
            image_id          = data.get("image_id", ""),
            dataset_source    = data.get("dataset_source", ""),
            category          = data.get("category", ""),
            gender            = data.get("gender", ""),
            subcategory       = data.get("subcategory"),
            colors            = data.get("colors", []),
            fabrics           = data.get("fabrics", []),
            styles            = data.get("styles", []),
            fit               = data.get("fit"),
            patterns          = data.get("patterns", []),
            seasons           = data.get("seasons", []),
            occasions         = data.get("occasions", []),
            description       = data.get("description"),
            is_valid          = data.get("is_valid", False),
            validation_errors = data.get("validation_errors", []),
            validation_warnings = data.get("validation_warnings", []),
        )


@dataclass
class ValidationResult:
    """
    Result of validating a FashionRecord against the knowledge base rules.

    Attributes:
        is_valid  : True only if no errors exist.
        errors    : Hard failures — record should be rejected.
        warnings  : Soft issues — record is kept but flagged.
        info      : Informational suggestions (e.g., recommended fabrics).
        record_id : The image_id of the validated record.
    """
    is_valid: bool               = True
    errors:   List[str]          = field(default_factory=list)
    warnings: List[str]          = field(default_factory=list)
    info:     List[str]          = field(default_factory=list)
    record_id: str               = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status} | id={self.record_id} | "
            f"errors={len(self.errors)} | warnings={len(self.warnings)}"
        )


@dataclass
class CategoryMapping:
    """
    Pre-computed cross-reference mapping for a single category.

    Captures which genders, styles, occasions, fabrics, and attributes
    are associated with this category according to the knowledge base.
    """
    category_code : str
    category_label: str
    genders       : List[str]
    styles        : List[str]
    occasions     : List[str]
    recommended_fabrics: List[str]
    attributes    : List[str]
    subcategories : List[str]
    aliases       : List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_code"      : self.category_code,
            "category_label"     : self.category_label,
            "genders"            : self.genders,
            "styles"             : self.styles,
            "occasions"          : self.occasions,
            "recommended_fabrics": self.recommended_fabrics,
            "attributes"         : self.attributes,
            "subcategories"      : self.subcategories,
            "aliases"            : self.aliases,
        }


@dataclass
class StyleProfile:
    """
    Enriched profile for a fashion style, combining KB data with derived metrics.

    Used to build style-aware search and recommendation systems.
    """
    style_key      : str
    label          : str
    code           : str
    tier           : int
    description    : str
    color_palette  : List[str]
    key_categories : List[str]
    key_fabrics    : List[str]
    key_occasions  : List[str]
    aesthetic_tags : List[str]
    brand_archetypes: List[str]
    compatible_styles   : List[str]
    incompatible_styles : List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style_key"           : self.style_key,
            "label"               : self.label,
            "code"                : self.code,
            "tier"                : self.tier,
            "description"         : self.description,
            "color_palette"       : self.color_palette,
            "key_categories"      : self.key_categories,
            "key_fabrics"         : self.key_fabrics,
            "key_occasions"       : self.key_occasions,
            "aesthetic_tags"      : self.aesthetic_tags,
            "brand_archetypes"    : self.brand_archetypes,
            "compatible_styles"   : self.compatible_styles,
            "incompatible_styles" : self.incompatible_styles,
        }


# =============================================================================
# ── FashionDomainResearch — Main Knowledge Base Class
# =============================================================================

class FashionDomainResearch:
    """
    Central knowledge base for the AI Fashion Design Assistant.

    Loads fashion_knowledge.json once and exposes a rich query API:
      - Direct lookups (get_category, get_style, get_attribute)
      - Cross-reference queries (categories for a style, styles for an occasion)
      - Attribute normalization (raw string → canonical taxonomy key)
      - Record validation against taxonomy rules
      - Artifact generation (category_mapping.json, style_profiles.json)
      - Statistics and analytics (coverage reports, taxonomy trees)

    Thread-safe: all state is read-only after initialization.
    """

    def __init__(
        self,
        knowledge_json_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load and index the fashion knowledge base.

        Args:
            knowledge_json_path : Override path to fashion_knowledge.json.
            output_dir          : Override output directory for artifacts.
        """
        self._kb_path   = Path(knowledge_json_path) if knowledge_json_path else _KNOWLEDGE_JSON
        self._output_dir = Path(output_dir) if output_dir else _METADATA_OUTPUT
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load knowledge base ───────────────────────────────────────────────
        self._kb: Dict[str, Any] = self._load_knowledge_base()

        # ── Build in-memory indexes for O(1) lookups ──────────────────────────
        self._alias_to_color:    Dict[str, str] = {}  # alias → canonical name
        self._alias_to_fabric:   Dict[str, str] = {}  # alias → canonical name
        self._alias_to_gender:   Dict[str, str] = {}  # alias → gender key
        self._alias_to_category: Dict[str, str] = {}  # alias → category key
        self._alias_to_style:    Dict[str, str] = {}  # alias → style key

        self._build_indexes()

        logger.success(
            f"FashionDomainResearch loaded | "
            f"categories={len(self._kb.get('categories', {}))} | "
            f"styles={len(self._kb.get('style_hierarchy', {}))} | "
            f"kb_path={self._kb_path.name}"
        )

    # =========================================================================
    # ── 1. Direct Lookup API
    # =========================================================================

    def get_category(self, category_key: str) -> Optional[Dict[str, Any]]:
        """
        Return the full category definition dict for a given category key.

        Args:
            category_key : e.g. "t_shirts", "dresses", "footwear"

        Returns:
            Category dict, or None if not found.

        Example:
            >>> kb.get_category("t_shirts")
            {"label": "T-Shirts", "code": "TSH", "genders": [...], ...}
        """
        result = self._kb.get("categories", {}).get(category_key)
        if result is None:
            logger.debug(f"Category not found: '{category_key}'")
        return result

    def get_all_categories(self) -> Dict[str, Dict[str, Any]]:
        """Return all category definitions as a dict keyed by category_key."""
        return dict(self._kb.get("categories", {}))

    def get_style(self, style_key: str) -> Optional[Dict[str, Any]]:
        """
        Return the full style definition for a given style key.

        Args:
            style_key : e.g. "streetwear", "luxury", "minimalist"

        Returns:
            Style dict, or None if not found.
        """
        result = self._kb.get("style_hierarchy", {}).get(style_key)
        if result is None:
            logger.debug(f"Style not found: '{style_key}'")
        return result

    def get_all_styles(self) -> Dict[str, Dict[str, Any]]:
        """Return all style definitions as a dict keyed by style_key."""
        return dict(self._kb.get("style_hierarchy", {}))

    def get_attribute(self, attribute_key: str) -> Optional[Dict[str, Any]]:
        """
        Return the full attribute definition (e.g., all color values).

        Args:
            attribute_key : "color" | "fabric" | "style" | "fit" |
                            "pattern" | "season" | "occasion"

        Returns:
            Attribute dict with label, description, type, and values.
        """
        result = self._kb.get("attributes", {}).get(attribute_key)
        if result is None:
            logger.debug(f"Attribute not found: '{attribute_key}'")
        return result

    def get_gender(self, gender_key: str) -> Optional[Dict[str, Any]]:
        """Return gender definition for 'men', 'women', or 'unisex'."""
        return self._kb.get("genders", {}).get(gender_key)

    def get_subcategories(self, category_key: str) -> Dict[str, Dict[str, Any]]:
        """
        Return all subcategory definitions for a given category.

        Args:
            category_key : e.g. "jeans"

        Returns:
            Dict of {subcategory_key: {"label": ..., "description": ...}}
        """
        cat = self.get_category(category_key)
        if cat is None:
            return {}
        return cat.get("subcategories", {})

    def get_style_tiers(self) -> Dict[int, List[str]]:
        """
        Return styles grouped by their tier (1=foundation, 2=derived, 3=specific).

        Returns:
            Dict mapping tier int → list of style keys.
        """
        tiers: Dict[int, List[str]] = defaultdict(list)
        for key, style in self._kb.get("style_hierarchy", {}).items():
            tier = style.get("tier", 1)
            tiers[tier].append(key)
        return dict(tiers)

    # =========================================================================
    # ── 2. Cross-Reference Lookup Helpers
    # =========================================================================

    def get_categories_for_gender(self, gender: str) -> List[str]:
        """
        Return the list of valid category keys for a given gender.

        Args:
            gender : "men" | "women" | "unisex"

        Returns:
            List of category keys. Empty list if gender not found.

        Example:
            >>> kb.get_categories_for_gender("women")
            ["t_shirts", "shirts", ..., "dresses", ...]
        """
        gender = self._normalize_gender_key(gender)
        return list(
            self._kb.get("category_mappings", {})
            .get("gender_to_categories", {})
            .get(gender, [])
        )

    def get_categories_for_style(self, style_key: str) -> List[str]:
        """
        Return recommended category keys for a given style.

        Args:
            style_key : e.g. "streetwear"

        Returns:
            List of category keys.
        """
        style_key = style_key.lower().strip().replace(" ", "_").replace("-", "_")
        return list(
            self._kb.get("category_mappings", {})
            .get("style_to_categories", {})
            .get(style_key, [])
        )

    def get_styles_for_occasion(self, occasion_key: str) -> List[str]:
        """
        Return recommended style keys for a given occasion.

        Args:
            occasion_key : e.g. "casual", "formal", "sport"

        Returns:
            List of style keys.
        """
        occasion_key = occasion_key.lower().strip().replace(" ", "_")
        return list(
            self._kb.get("category_mappings", {})
            .get("occasion_to_styles", {})
            .get(occasion_key, [])
        )

    def get_fabrics_for_season(self, season_key: str) -> List[str]:
        """
        Return recommended fabric names for a given season.

        Args:
            season_key : "spring" | "summer" | "autumn" | "winter" | "all_season"

        Returns:
            List of fabric name strings.
        """
        season_key = season_key.lower().strip().replace(" ", "_")
        return list(
            self._kb.get("category_mappings", {})
            .get("season_to_fabrics", {})
            .get(season_key, [])
        )

    def get_attributes_for_category(self, category_key: str) -> List[str]:
        """
        Return which attribute keys apply to a given category.

        Args:
            category_key : e.g. "footwear"

        Returns:
            List of attribute keys (subset of color, fabric, fit, etc.)
        """
        return list(
            self._kb.get("category_mappings", {})
            .get("category_to_attributes", {})
            .get(category_key, [])
        )

    def get_compatible_styles(self, style_key: str) -> List[str]:
        """
        Return styles that are compatible/complementary to the given style.

        Compatibility is derived from shared occasion and parent/child relationships.

        Args:
            style_key : e.g. "minimalist"

        Returns:
            List of compatible style keys.
        """
        style = self.get_style(style_key)
        if not style:
            return []
        # Parent + child styles are considered compatible
        compatible = (
            style.get("parent_styles", []) +
            style.get("child_styles", [])
        )
        return [s for s in compatible if s]

    def get_incompatible_styles(self, style_key: str) -> List[str]:
        """
        Return styles that conflict with the given style.

        Args:
            style_key : e.g. "streetwear"

        Returns:
            List of incompatible style keys.
        """
        style = self.get_style(style_key)
        if not style:
            return []
        return list(style.get("incompatible_with", []))

    def get_color_hex(self, color_name: str) -> Optional[str]:
        """
        Look up the hex code for a named color.

        Args:
            color_name : Canonical or alias color name.

        Returns:
            Hex string (e.g. "#FFFFFF") or None if not found / not applicable.
        """
        canonical = normalize_color(color_name, kb=self)
        if not canonical:
            return None
        color_attr = self._kb.get("attributes", {}).get("color", {})
        for group in color_attr.get("values", {}).values():
            for color_entry in group.get("colors", []):
                if color_entry["name"].lower() == canonical.lower():
                    return color_entry.get("hex")
        return None

    def get_fabric_properties(self, fabric_name: str) -> Dict[str, Any]:
        """
        Return properties and care instructions for a given fabric.

        Args:
            fabric_name : Canonical or alias fabric name.

        Returns:
            Dict with "properties" and "care" keys, or empty dict.
        """
        canonical = normalize_fabric(fabric_name, kb=self)
        if not canonical:
            return {}
        fabric_attr = self._kb.get("attributes", {}).get("fabric", {})
        for group in fabric_attr.get("values", {}).values():
            for fabric in group.get("fabrics", []):
                if fabric["name"].lower() == canonical.lower():
                    return {
                        "name"      : fabric["name"],
                        "properties": fabric.get("properties", []),
                        "care"      : fabric.get("care", []),
                    }
        return {}

    # =========================================================================
    # ── 3. Search API
    # =========================================================================

    def search_by_tags(
        self,
        gender: Optional[str] = None,
        category: Optional[str] = None,
        style: Optional[str] = None,
        occasion: Optional[str] = None,
        season: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Return recommended attribute combinations for the given filter set.

        This is the core recommendation engine for the knowledge base.
        Given any combination of gender / category / style / occasion / season,
        it returns the intersection of recommended values for each attribute.

        Args:
            gender   : "men" | "women" | "unisex"
            category : e.g. "t_shirts"
            style    : e.g. "streetwear"
            occasion : e.g. "casual"
            season   : e.g. "summer"

        Returns:
            Dict with keys: categories, styles, fabrics, occasions, colors, attributes.

        Example:
            >>> kb.search_by_tags(gender="women", style="athleisure", season="summer")
            {
                "categories": ["t_shirts", "hoodies", ...],
                "fabrics": ["Performance Mesh", "Cotton", ...],
                "colors": ["Black", "Navy", "Neon"],
                ...
            }
        """
        result: Dict[str, List[str]] = {}

        # ── Recommended categories ─────────────────────────────────────────────
        candidate_cats: Set[str] = set(VALID_CATEGORIES)
        if gender:
            gender_norm = self._normalize_gender_key(gender)
            gender_cats = set(self.get_categories_for_gender(gender_norm))
            candidate_cats &= gender_cats
        if style:
            style_norm = style.lower().strip().replace(" ", "_").replace("-", "_")
            style_cats = set(self.get_categories_for_style(style_norm))
            if style_cats:
                candidate_cats &= style_cats
        if category:
            cat_norm = category.lower().strip().replace(" ", "_").replace("-", "_")
            candidate_cats = {cat_norm} if cat_norm in candidate_cats else set()
        result["categories"] = sorted(candidate_cats)

        # ── Recommended styles ─────────────────────────────────────────────────
        if occasion:
            result["styles"] = self.get_styles_for_occasion(occasion)
        elif style:
            result["styles"] = [style.lower().strip().replace(" ", "_").replace("-", "_")]
        else:
            result["styles"] = list(VALID_STYLES)

        # ── Recommended fabrics ────────────────────────────────────────────────
        if season:
            result["fabrics"] = self.get_fabrics_for_season(season)
        elif style:
            style_def = self.get_style(style.lower().strip().replace(" ", "_"))
            result["fabrics"] = (
                style_def.get("key_attributes", {}).get("fabrics", [])
                if style_def else []
            )
        else:
            result["fabrics"] = []

        # ── Recommended occasions ──────────────────────────────────────────────
        if style:
            style_def = self.get_style(style.lower().strip().replace(" ", "_"))
            result["occasions"] = (
                style_def.get("key_attributes", {}).get("occasions", [])
                if style_def else []
            )
        else:
            result["occasions"] = list(VALID_OCCASIONS)

        # ── Recommended colors ─────────────────────────────────────────────────
        if style:
            style_def = self.get_style(style.lower().strip().replace(" ", "_"))
            result["colors"] = (
                style_def.get("color_palette", []) if style_def else []
            )
        else:
            result["colors"] = []

        # ── Applicable attributes ──────────────────────────────────────────────
        if category:
            result["attributes"] = self.get_attributes_for_category(
                category.lower().strip().replace(" ", "_")
            )
        else:
            result["attributes"] = ["color", "fabric", "fit", "pattern", "style", "occasion", "season"]

        return result

    def get_taxonomy_tree(self) -> Dict[str, Any]:
        """
        Return the complete taxonomy as a nested tree structure.

        The tree is:
            root
            └── gender
                └── category
                    └── subcategory

        Returns:
            Nested dict representing the full taxonomy.
        """
        tree: Dict[str, Any] = {}
        gender_to_cats = self._kb.get("category_mappings", {}).get("gender_to_categories", {})

        for gender, cat_keys in gender_to_cats.items():
            tree[gender] = {}
            for cat_key in cat_keys:
                cat = self.get_category(cat_key)
                if cat is None:
                    continue
                subcats = {
                    sub_key: sub_val.get("label", sub_key)
                    for sub_key, sub_val in cat.get("subcategories", {}).items()
                }
                tree[gender][cat_key] = {
                    "label": cat.get("label", cat_key),
                    "subcategories": subcats,
                }
        return tree

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics about the knowledge base contents.

        Useful for pipeline health checks and documentation generation.

        Returns:
            Dict with counts for categories, styles, attributes, etc.
        """
        categories   = self._kb.get("categories", {})
        styles       = self._kb.get("style_hierarchy", {})
        attributes   = self._kb.get("attributes", {})

        total_subcats = sum(
            len(cat.get("subcategories", {})) for cat in categories.values()
        )
        total_colors = sum(
            len(group.get("colors", []))
            for group in attributes.get("color", {}).get("values", {}).values()
        )
        total_fabrics = sum(
            len(group.get("fabrics", []))
            for group in attributes.get("fabric", {}).get("values", {}).values()
        )

        return {
            "knowledge_base_version"  : self._kb.get("_meta", {}).get("version"),
            "total_categories"        : len(categories),
            "total_subcategories"     : total_subcats,
            "total_styles"            : len(styles),
            "total_genders"           : len(self._kb.get("genders", {})),
            "total_attribute_types"   : len(attributes),
            "total_color_entries"     : total_colors,
            "total_fabric_entries"    : total_fabrics,
            "total_alias_color_index" : len(self._alias_to_color),
            "total_alias_fabric_index": len(self._alias_to_fabric),
            "total_alias_category_index": len(self._alias_to_category),
        }

    # =========================================================================
    # ── 4. Artifact Generation
    # =========================================================================

    def generate_category_mapping(self) -> Path:
        """
        Generate and save category_mapping.json to the metadata output directory.

        The file contains one CategoryMapping per category — a pre-computed
        cross-reference combining gender, style, occasion, fabric, and attribute
        associations from the full knowledge base.

        Returns:
            Path to the saved JSON file.
        """
        mappings: Dict[str, Any] = {}

        for cat_key, cat_def in self._kb.get("categories", {}).items():
            mapping = build_category_mapping(cat_key, kb=self)
            mappings[cat_key] = mapping.to_dict()

        output = {
            "_meta": {
                "generated_at"  : datetime.now(timezone.utc).isoformat(),
                "source"        : self._kb_path.name,
                "description"   : "Pre-computed category cross-reference mappings",
                "total_categories": len(mappings),
            },
            "mappings": mappings,
        }

        out_path = self._output_dir / "category_mapping.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.success(
            f"category_mapping.json → {out_path} "
            f"({len(mappings)} categories)"
        )
        return out_path

    def generate_style_profiles(self) -> Path:
        """
        Generate and save style_profiles.json to the metadata output directory.

        Each StyleProfile combines the raw KB style definition with derived
        compatibility mappings and enriched metadata for downstream use.

        Returns:
            Path to the saved JSON file.
        """
        profiles: Dict[str, Any] = {}

        for style_key in self._kb.get("style_hierarchy", {}):
            profile = build_style_profile(style_key, kb=self)
            profiles[style_key] = profile.to_dict()

        output = {
            "_meta": {
                "generated_at" : datetime.now(timezone.utc).isoformat(),
                "source"       : self._kb_path.name,
                "description"  : "Enriched style profiles with compatibility matrices",
                "total_styles" : len(profiles),
            },
            "profiles": profiles,
        }

        out_path = self._output_dir / "style_profiles.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.success(
            f"style_profiles.json → {out_path} "
            f"({len(profiles)} styles)"
        )
        return out_path

    def generate_taxonomy_tree(self) -> Path:
        """
        Generate and save taxonomy_tree.json with the full nested taxonomy.

        Returns:
            Path to the saved JSON file.
        """
        tree = self.get_taxonomy_tree()
        stats = self.get_knowledge_base_stats()

        output = {
            "_meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source"      : self._kb_path.name,
                "description" : "Full gender → category → subcategory taxonomy tree",
                "stats"       : stats,
            },
            "tree": tree,
        }

        out_path = self._output_dir / "taxonomy_tree.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.success(f"taxonomy_tree.json → {out_path}")
        return out_path

    def generate_alias_index(self) -> Path:
        """
        Generate and save alias_index.json — maps all known synonyms to canonical values.

        This file is consumed by normalizer functions and search pre-processors.

        Returns:
            Path to the saved JSON file.
        """
        output = {
            "_meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "description" : "Alias → canonical value index for all attribute types",
            },
            "color_aliases"   : dict(sorted(self._alias_to_color.items())),
            "fabric_aliases"  : dict(sorted(self._alias_to_fabric.items())),
            "gender_aliases"  : dict(sorted(self._alias_to_gender.items())),
            "category_aliases": dict(sorted(self._alias_to_category.items())),
            "style_aliases"   : dict(sorted(self._alias_to_style.items())),
        }

        out_path = self._output_dir / "alias_index.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.success(
            f"alias_index.json → {out_path} | "
            f"colors={len(self._alias_to_color)} | "
            f"fabrics={len(self._alias_to_fabric)}"
        )
        return out_path

    def generate_all_artifacts(self) -> Dict[str, Path]:
        """
        Generate all four derived JSON artifacts in one call.

        Artifacts written to datasets/metadata/:
          - category_mapping.json
          - style_profiles.json
          - taxonomy_tree.json
          - alias_index.json

        Returns:
            Dict mapping artifact name → saved Path.
        """
        logger.info("Generating all knowledge base artifacts…")
        return {
            "category_mapping" : self.generate_category_mapping(),
            "style_profiles"   : self.generate_style_profiles(),
            "taxonomy_tree"    : self.generate_taxonomy_tree(),
            "alias_index"      : self.generate_alias_index(),
        }

    # =========================================================================
    # ── 5. Validation Entry Point (delegates to module-level function)
    # =========================================================================

    def validate(self, record_dict: Dict[str, Any]) -> ValidationResult:
        """
        Validate a raw record dict against the knowledge base taxonomy rules.

        This is a convenience wrapper around the module-level
        `validate_fashion_record()` function, passing self as the kb reference.

        Args:
            record_dict : Dict with fashion item attributes.

        Returns:
            ValidationResult with errors, warnings, and info messages.
        """
        return validate_fashion_record(record_dict, kb=self)

    # =========================================================================
    # ── Private: Knowledge Base Loading
    # =========================================================================

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load and parse fashion_knowledge.json.

        Raises:
            FileNotFoundError : If the JSON file does not exist.
            json.JSONDecodeError : If the file is not valid JSON.
        """
        if not self._kb_path.exists():
            raise FileNotFoundError(
                f"fashion_knowledge.json not found at: {self._kb_path}\n"
                f"Expected path: {self._kb_path.resolve()}"
            )

        logger.info(f"Loading knowledge base: {self._kb_path.name}")
        with open(self._kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        # Minimal integrity check
        required_top_keys = ["genders", "categories", "attributes", "style_hierarchy", "category_mappings"]
        missing = [k for k in required_top_keys if k not in kb]
        if missing:
            raise ValueError(
                f"fashion_knowledge.json is missing required top-level keys: {missing}"
            )

        return kb

    # =========================================================================
    # ── Private: Index Building
    # =========================================================================

    def _build_indexes(self) -> None:
        """
        Pre-build all alias → canonical lookup dictionaries.

        Called once at __init__ time. All lookups thereafter are O(1).
        """
        self._build_color_alias_index()
        self._build_fabric_alias_index()
        self._build_gender_alias_index()
        self._build_category_alias_index()
        self._build_style_alias_index()
        logger.debug(
            f"Alias indexes built | "
            f"colors={len(self._alias_to_color)} | "
            f"fabrics={len(self._alias_to_fabric)} | "
            f"categories={len(self._alias_to_category)}"
        )

    def _build_color_alias_index(self) -> None:
        """Index all color aliases → canonical color name."""
        color_attr = self._kb.get("attributes", {}).get("color", {})
        for group in color_attr.get("values", {}).values():
            for color_entry in group.get("colors", []):
                canonical = color_entry["name"]
                # Register canonical name itself
                self._alias_to_color[canonical.lower()] = canonical
                # Register all aliases
                for alias in color_entry.get("aliases", []):
                    self._alias_to_color[alias.lower()] = canonical

    def _build_fabric_alias_index(self) -> None:
        """Index all fabric aliases → canonical fabric name."""
        fabric_attr = self._kb.get("attributes", {}).get("fabric", {})
        for group in fabric_attr.get("values", {}).values():
            for fabric in group.get("fabrics", []):
                canonical = fabric["name"]
                self._alias_to_fabric[canonical.lower()] = canonical
                for alias in fabric.get("aliases", []):
                    self._alias_to_fabric[alias.lower()] = canonical

    def _build_gender_alias_index(self) -> None:
        """Index all gender aliases → gender key."""
        for gender_key, gender_def in self._kb.get("genders", {}).items():
            self._alias_to_gender[gender_key.lower()] = gender_key
            self._alias_to_gender[gender_def.get("label", "").lower()] = gender_key
            for alias in gender_def.get("aliases", []):
                self._alias_to_gender[alias.lower()] = gender_key

    def _build_category_alias_index(self) -> None:
        """Index all category aliases → category key."""
        for cat_key, cat_def in self._kb.get("categories", {}).items():
            self._alias_to_category[cat_key.lower()]                = cat_key
            self._alias_to_category[cat_def.get("label", "").lower()] = cat_key
            for alias in cat_def.get("aliases", []):
                self._alias_to_category[alias.lower()] = cat_key

    def _build_style_alias_index(self) -> None:
        """Index style labels → style key."""
        for style_key, style_def in self._kb.get("style_hierarchy", {}).items():
            self._alias_to_style[style_key.lower()]                  = style_key
            self._alias_to_style[style_def.get("label", "").lower()] = style_key

    @staticmethod
    def _normalize_gender_key(gender: str) -> str:
        """Lowercase and strip a gender string."""
        return gender.lower().strip()


# =============================================================================
# ── Module-Level Normalizer Functions
# =============================================================================
# These can be imported and used without instantiating FashionDomainResearch.
# They accept an optional `kb` argument to use a pre-loaded instance;
# if not provided, they load a fresh instance from disk.
# =============================================================================

def _get_kb(kb: Optional[FashionDomainResearch]) -> FashionDomainResearch:
    """Return the provided kb or create a fresh singleton."""
    if kb is not None:
        return kb
    return FashionDomainResearch()


def _slug(text: str) -> str:
    """
    Normalize a string to lowercase, strip accents, and replace spaces/hyphens with underscores.

    Used for robust matching (e.g. "T-Shirt" → "t_shirt").
    """
    # Decompose accented characters (é → e + combining accent), then drop non-ASCII
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", ascii_only.lower().strip())


def normalize_color(
    raw_color: str,
    kb: Optional[FashionDomainResearch] = None,
) -> Optional[str]:
    """
    Convert a raw color string to its canonical taxonomy name.

    Performs:
      1. Lowercase + strip
      2. Alias lookup in pre-built index
      3. Partial match fallback (for strings like "dark navy blue")

    Args:
        raw_color : Any color string (e.g. "navy blue", "off-white", "écru").
        kb        : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        Canonical color name (e.g. "Navy") or None if no match found.

    Examples:
        >>> normalize_color("off white")   → "White"
        >>> normalize_color("cognac")      → "Brown"
        >>> normalize_color("cobalt blue") → "Royal Blue"
        >>> normalize_color("xyz999")      → None
    """
    if not raw_color or not isinstance(raw_color, str):
        return None

    _kb = _get_kb(kb)
    normalized = raw_color.lower().strip()

    # Direct alias lookup
    if normalized in _kb._alias_to_color:
        return _kb._alias_to_color[normalized]

    # Partial / substring match (covers "dark navy blue" → "Navy")
    for alias, canonical in _kb._alias_to_color.items():
        if alias in normalized or normalized in alias:
            return canonical

    logger.debug(f"Color not recognized: '{raw_color}'")
    return None


def normalize_fabric(
    raw_fabric: str,
    kb: Optional[FashionDomainResearch] = None,
) -> Optional[str]:
    """
    Convert a raw fabric string to its canonical taxonomy name.

    Args:
        raw_fabric : Any fabric string (e.g. "polycotton", "100% merino wool").
        kb         : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        Canonical fabric name or None.

    Examples:
        >>> normalize_fabric("polycotton")     → "Cotton-Polyester Blend"
        >>> normalize_fabric("merino wool")    → "Wool"
        >>> normalize_fabric("elastane")       → "Spandex"
    """
    if not raw_fabric or not isinstance(raw_fabric, str):
        return None

    _kb = _get_kb(kb)
    normalized = raw_fabric.lower().strip()

    if normalized in _kb._alias_to_fabric:
        return _kb._alias_to_fabric[normalized]

    for alias, canonical in _kb._alias_to_fabric.items():
        if alias in normalized or normalized in alias:
            return canonical

    logger.debug(f"Fabric not recognized: '{raw_fabric}'")
    return None


def normalize_style(
    raw_style: str,
    kb: Optional[FashionDomainResearch] = None,
) -> Optional[str]:
    """
    Convert a raw style string to its canonical style key.

    Args:
        raw_style : Any style string (e.g. "Business Casual", "street wear").
        kb        : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        Canonical style key (e.g. "business_casual") or None.

    Examples:
        >>> normalize_style("Business Casual") → "business_casual"
        >>> normalize_style("street wear")     → "streetwear"
        >>> normalize_style("hype beast")      → None
    """
    if not raw_style or not isinstance(raw_style, str):
        return None

    _kb = _get_kb(kb)
    slugged = _slug(raw_style)

    # Direct match on key or label
    if slugged in _kb._alias_to_style:
        return _kb._alias_to_style[slugged]

    # Remove underscores and try again (handles "street_wear" → "streetwear")
    collapsed = slugged.replace("_", "")
    for alias, style_key in _kb._alias_to_style.items():
        if alias.replace("_", "") == collapsed:
            return style_key

    logger.debug(f"Style not recognized: '{raw_style}'")
    return None


def normalize_fit(raw_fit: str) -> Optional[str]:
    """
    Convert a raw fit string to its canonical fit key.

    Does not require a KB instance — uses the static VALID_FITS set.

    Args:
        raw_fit : Any fit string (e.g. "slim", "regular fit", "extra oversized").

    Returns:
        Canonical fit key (e.g. "slim_fit") or None.

    Examples:
        >>> normalize_fit("fitted")         → "slim_fit"
        >>> normalize_fit("baggy")          → "oversized"
        >>> normalize_fit("regular fit")    → "regular_fit"
    """
    if not raw_fit or not isinstance(raw_fit, str):
        return None

    slugged = _slug(raw_fit)

    # Direct match
    if slugged in VALID_FITS:
        return slugged

    # Alias mapping for common informal terms
    _FIT_ALIASES: Dict[str, str] = {
        "slim"         : "slim_fit",
        "fitted"       : "slim_fit",
        "tailored"     : "slim_fit",
        "close_fit"    : "slim_fit",
        "regular"      : "regular_fit",
        "classic_fit"  : "regular_fit",
        "standard_fit" : "regular_fit",
        "normal_fit"   : "regular_fit",
        "relaxed"      : "relaxed_fit",
        "easy_fit"     : "relaxed_fit",
        "comfort_fit"  : "relaxed_fit",
        "loose"        : "relaxed_fit",
        "loose_fit"    : "relaxed_fit",
        "boxy"         : "oversized",
        "slouchy"      : "oversized",
        "baggy"        : "oversized",
        "oversize"     : "oversized",
        "crop"         : "cropped",
        "cut_off"      : "cropped",
        "super_slim"   : "skinny",
        "ultra_slim"   : "skinny",
        "straight_cut" : "straight",
        "box_cut"      : "straight",
        "muscle_fit"   : "athletic_fit",
        "sport_fit"    : "athletic_fit",
    }

    if slugged in _FIT_ALIASES:
        return _FIT_ALIASES[slugged]

    # Substring fallback
    for alias, canonical in _FIT_ALIASES.items():
        if alias in slugged or slugged in alias:
            return canonical

    logger.debug(f"Fit not recognized: '{raw_fit}'")
    return None


def normalize_pattern(raw_pattern: str) -> Optional[str]:
    """
    Convert a raw pattern string to its canonical pattern key.

    Args:
        raw_pattern : Any pattern description string.

    Returns:
        Canonical pattern key or None.

    Examples:
        >>> normalize_pattern("leopard")       → "animal_print"
        >>> normalize_pattern("dotted")        → "polka_dot"
        >>> normalize_pattern("plain")         → "solid"
    """
    if not raw_pattern or not isinstance(raw_pattern, str):
        return None

    slugged = _slug(raw_pattern)

    _PATTERN_ALIASES: Dict[str, str] = {
        "plain"          : "solid",
        "solid_color"    : "solid",
        "monochrome"     : "solid",
        "no_print"       : "solid",
        "striped"        : "stripes",
        "stripy"         : "stripes",
        "pinstripe"      : "stripes",
        "plaid"          : "checks",
        "checkered"      : "checks",
        "checked"        : "checks",
        "tartan"         : "checks",
        "gingham"        : "checks",
        "houndstooth"    : "checks",
        "flowers"        : "floral",
        "botanical"      : "floral",
        "floral_print"   : "floral",
        "shapes"         : "geometric",
        "geo_print"      : "geometric",
        "chevron"        : "geometric",
        "leopard"        : "animal_print",
        "zebra"          : "animal_print",
        "snake_print"    : "animal_print",
        "cheetah"        : "animal_print",
        "animal_pattern" : "animal_print",
        "camo"           : "camouflage",
        "military_print" : "camouflage",
        "tie_dye"        : "tie_dye",
        "dip_dye"        : "tie_dye",
        "ombre"          : "tie_dye",
        "logo_print"     : "graphic",
        "text_print"     : "graphic",
        "screen_print"   : "graphic",
        "dots"           : "polka_dot",
        "spotted"        : "polka_dot",
        "dotted"         : "polka_dot",
        "painterly"      : "abstract",
        "artistic"       : "abstract",
        "boteh"          : "paisley",
        "teardrop_pattern": "paisley",
    }

    if slugged in _PATTERN_ALIASES:
        return _PATTERN_ALIASES[slugged]

    # Check if slugged IS a valid pattern key already
    _VALID_PATTERNS = frozenset({
        "solid", "stripes", "checks", "floral", "geometric",
        "animal_print", "camouflage", "tie_dye", "paisley",
        "graphic", "abstract", "polka_dot"
    })
    if slugged in _VALID_PATTERNS:
        return slugged

    for alias, canonical in _PATTERN_ALIASES.items():
        if alias in slugged or slugged in alias:
            return canonical

    logger.debug(f"Pattern not recognized: '{raw_pattern}'")
    return None


def normalize_season(raw_season: str) -> Optional[str]:
    """
    Convert a raw season string to its canonical season key.

    Args:
        raw_season : Any season description.

    Returns:
        Canonical season key or None.

    Examples:
        >>> normalize_season("SS")         → "spring"
        >>> normalize_season("AW")         → "autumn"
        >>> normalize_season("year-round") → "all_season"
    """
    if not raw_season or not isinstance(raw_season, str):
        return None

    slugged = _slug(raw_season).upper()

    _SEASON_ALIASES: Dict[str, str] = {
        "SPRING"      : "spring",
        "SPRING/SUMMER": "spring",
        "SS"          : "spring",
        "SUMMER"      : "summer",
        "HOT_WEATHER" : "summer",
        "WARM_WEATHER": "summer",
        "AUTUMN"      : "autumn",
        "FALL"        : "autumn",
        "AUTUMN/WINTER": "autumn",
        "AW"          : "autumn",
        "FW"          : "autumn",
        "WINTER"      : "winter",
        "COLD_WEATHER": "winter",
        "ALL_SEASON"  : "all_season",
        "YEAR_ROUND"  : "all_season",
        "YEAR-ROUND"  : "all_season",
        "MULTI_SEASON": "all_season",
        "TRANS_SEASON": "all_season",
    }

    if slugged in _SEASON_ALIASES:
        return _SEASON_ALIASES[slugged]

    lower = slugged.lower()
    for alias, canonical in _SEASON_ALIASES.items():
        if alias.lower() in lower or lower in alias.lower():
            return canonical

    logger.debug(f"Season not recognized: '{raw_season}'")
    return None


def normalize_occasion(raw_occasion: str) -> Optional[str]:
    """
    Convert a raw occasion string to its canonical occasion key.

    Args:
        raw_occasion : Any occasion description.

    Returns:
        Canonical occasion key or None.

    Examples:
        >>> normalize_occasion("gym")       → "sport"
        >>> normalize_occasion("black tie") → "formal"
        >>> normalize_occasion("hiking")    → "outdoor"
    """
    if not raw_occasion or not isinstance(raw_occasion, str):
        return None

    slugged = _slug(raw_occasion)

    _OCCASION_ALIASES: Dict[str, str] = {
        "everyday"        : "casual",
        "day_to_day"      : "casual",
        "weekend"         : "casual",
        "leisure"         : "casual",
        "smart_casual"    : "business_casual",
        "office_casual"   : "business_casual",
        "business"        : "business_casual",
        "office"          : "business_casual",
        "black_tie"       : "formal",
        "gala"            : "formal",
        "formal_event"    : "formal",
        "evening"         : "formal",
        "night_out"       : "party",
        "club"            : "party",
        "social"          : "party",
        "celebration"     : "party",
        "gym"             : "sport",
        "workout"         : "sport",
        "athletic"        : "sport",
        "training"        : "sport",
        "running"         : "sport",
        "yoga"            : "sport",
        "hiking"          : "outdoor",
        "camping"         : "outdoor",
        "travel"          : "outdoor",
        "adventure"       : "outdoor",
        "pool"            : "beach",
        "resort"          : "beach",
        "vacation"        : "beach",
        "holiday"         : "beach",
        "swim"            : "beach",
        "wedding"         : "wedding_festive",
        "festival"        : "wedding_festive",
        "ceremony"        : "wedding_festive",
        "festive"         : "wedding_festive",
        "homewear"        : "lounge",
        "loungewear"      : "lounge",
        "sleepwear"       : "lounge",
        "pyjamas"         : "lounge",
    }

    if slugged in _OCCASION_ALIASES:
        return _OCCASION_ALIASES[slugged]

    if slugged in VALID_OCCASIONS:
        return slugged

    for alias, canonical in _OCCASION_ALIASES.items():
        if alias in slugged or slugged in alias:
            return canonical

    logger.debug(f"Occasion not recognized: '{raw_occasion}'")
    return None


# =============================================================================
# ── Module-Level Cross-Reference Lookup Functions
# =============================================================================

def get_categories_for_gender(
    gender: str,
    kb: Optional[FashionDomainResearch] = None,
) -> List[str]:
    """Return valid category keys for a gender string (handles aliases)."""
    _kb = _get_kb(kb)
    gender_key = _kb._alias_to_gender.get(gender.lower().strip(), gender.lower().strip())
    return _kb.get_categories_for_gender(gender_key)


def get_categories_for_style(
    style: str,
    kb: Optional[FashionDomainResearch] = None,
) -> List[str]:
    """Return recommended category keys for a style string."""
    _kb = _get_kb(kb)
    style_key = normalize_style(style, kb=_kb) or style
    return _kb.get_categories_for_style(style_key)


def get_styles_for_occasion(
    occasion: str,
    kb: Optional[FashionDomainResearch] = None,
) -> List[str]:
    """Return recommended style keys for an occasion string."""
    _kb = _get_kb(kb)
    occasion_key = normalize_occasion(occasion) or occasion
    return _kb.get_styles_for_occasion(occasion_key)


def get_fabrics_for_season(
    season: str,
    kb: Optional[FashionDomainResearch] = None,
) -> List[str]:
    """Return recommended fabric names for a season string."""
    _kb = _get_kb(kb)
    season_key = normalize_season(season) or season
    return _kb.get_fabrics_for_season(season_key)


def get_attributes_for_category(
    category: str,
    kb: Optional[FashionDomainResearch] = None,
) -> List[str]:
    """Return applicable attribute keys for a category string."""
    _kb = _get_kb(kb)
    cat_key = _kb._alias_to_category.get(category.lower().strip(), category.lower().strip())
    return _kb.get_attributes_for_category(cat_key)


# =============================================================================
# ── Validation Function
# =============================================================================

def validate_fashion_record(
    record: Dict[str, Any],
    kb: Optional[FashionDomainResearch] = None,
) -> ValidationResult:
    """
    Validate a raw record dictionary against the knowledge base taxonomy rules.

    VALIDATION LAYERS (in order):
        1. Required fields check (image_id, category, gender, dataset_source)
        2. Gender validity check (must be in VALID_GENDERS)
        3. Category validity check (must be in VALID_CATEGORIES)
        4. Gender-category compatibility (e.g., dresses only for women)
        5. Attribute validity (fit, season, occasion, style)
        6. Conditional rules from the knowledge base (CR001–CR005)

    Args:
        record : Dict with fashion record fields.
        kb     : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        ValidationResult with is_valid flag, errors, warnings, and info.

    Example:
        >>> validate_fashion_record({"category": "dresses", "gender": "men"})
        ValidationResult(is_valid=False, errors=["Dresses are only valid for gender=women"])
    """
    _kb = _get_kb(kb)
    record_id = str(record.get("image_id", "unknown"))
    result = ValidationResult(record_id=record_id)

    # ── Layer 1: Required fields ───────────────────────────────────────────────
    required = ["image_id", "category", "gender", "dataset_source"]
    for field_name in required:
        if field_name not in record or record[field_name] is None:
            result.add_error(f"Missing required field: '{field_name}'")
        elif isinstance(record[field_name], str) and not record[field_name].strip():
            result.add_error(f"Field '{field_name}' is empty")

    # If required fields are missing, abort early (later checks would fail)
    if not result.is_valid:
        return result

    category       = record.get("category", "").lower().strip().replace(" ", "_")
    gender         = record.get("gender", "").lower().strip()
    dataset_source = record.get("dataset_source", "")

    # ── Layer 2: Gender validity ───────────────────────────────────────────────
    gender_key = _kb._alias_to_gender.get(gender, gender)
    if gender_key not in VALID_GENDERS:
        result.add_error(
            f"Invalid gender '{gender}'. "
            f"Must be one of: {sorted(VALID_GENDERS)}"
        )
    else:
        gender = gender_key  # use canonical key

    # ── Layer 3: Category validity ─────────────────────────────────────────────
    cat_key = _kb._alias_to_category.get(category, category)
    if cat_key not in VALID_CATEGORIES:
        result.add_error(
            f"Invalid category '{category}'. "
            f"Must be one of: {sorted(VALID_CATEGORIES)}"
        )
    else:
        category = cat_key  # use canonical key

    # ── Layer 4: Gender-Category compatibility ─────────────────────────────────
    if result.is_valid:  # Only if previous layers passed
        valid_cats_for_gender = _kb.get_categories_for_gender(gender)
        if valid_cats_for_gender and category not in valid_cats_for_gender:
            result.add_error(
                f"Category '{category}' is not valid for gender '{gender}'. "
                f"Valid categories: {valid_cats_for_gender}"
            )

    # ── Layer 5: Optional attribute validity ───────────────────────────────────
    # Fit
    fit = record.get("fit")
    if fit:
        fit_norm = normalize_fit(str(fit))
        if fit_norm is None:
            result.add_warning(
                f"Fit '{fit}' not recognized in taxonomy. "
                f"Valid fits: {sorted(VALID_FITS)}"
            )
        else:
            # Fit applicability
            if category in ("footwear", "accessories") and fit:
                result.add_warning(
                    f"Fit attribute '{fit}' is unusual for category '{category}'"
                )

    # Season
    for season in record.get("seasons", []):
        if normalize_season(str(season)) is None:
            result.add_warning(f"Season '{season}' not recognized in taxonomy")

    # Occasion
    for occasion in record.get("occasions", []):
        if normalize_occasion(str(occasion)) is None:
            result.add_warning(f"Occasion '{occasion}' not recognized in taxonomy")

    # Style
    for style in record.get("styles", []):
        if normalize_style(str(style), kb=_kb) is None:
            result.add_warning(f"Style '{style}' not recognized in taxonomy")

    # ── Layer 6: Conditional rules from KB ────────────────────────────────────
    cond_rules = (
        _kb._kb.get("validation_rules", {})
        .get("conditional_rules", [])
    )
    for rule in cond_rules:
        rule_id = rule.get("rule_id", "UNKNOWN")
        severity = rule.get("severity", "warning")
        condition = rule.get("condition", "")

        # Parse simple "field == value" conditions
        triggered = _eval_rule_condition(condition, record, category, gender)
        if not triggered:
            continue

        # Check "required" constraint
        required_expr = rule.get("required", "")
        if required_expr and not _eval_required_constraint(
            required_expr, record, category, gender
        ):
            msg = f"[{rule_id}] {rule.get('description', 'Rule violated')}"
            if severity == "error":
                result.add_error(msg)
            elif severity == "warning":
                result.add_warning(msg)
            else:
                result.add_info(msg)

        # Check "recommended_fabrics" (info-level suggestion)
        if "recommended_fabrics" in rule:
            record_fabrics = record.get("fabrics", [])
            rec_fabrics    = rule["recommended_fabrics"]
            if record_fabrics and not any(
                f in rec_fabrics for f in record_fabrics
            ):
                result.add_info(
                    f"[{rule_id}] For style '{record.get('styles', [''])[0]}', "
                    f"recommended fabrics are: {rec_fabrics}"
                )

    return result


def _eval_rule_condition(
    condition: str,
    record: Dict[str, Any],
    category: str,
    gender: str,
) -> bool:
    """
    Evaluate a simple rule condition string like "category == 'dresses'".

    Only supports the patterns used in our validation_rules:
      - "category == 'value'"
      - "style == 'value'"

    Returns True if the condition is satisfied by the record.
    """
    try:
        if "category ==" in condition:
            target = condition.split("==")[1].strip().strip("'\"")
            return category == target
        if "style ==" in condition:
            target = condition.split("==")[1].strip().strip("'\"")
            return target in record.get("styles", [])
    except Exception:
        pass
    return False


def _eval_required_constraint(
    required_expr: str,
    record: Dict[str, Any],
    category: str,
    gender: str,
) -> bool:
    """
    Evaluate a simple required constraint expression.

    Supported patterns:
      - "gender in ['women']"
      - "occasion is not null"
      - "occasion in ['formal', ...]"
      - "category in ['t_shirts', ...]"
    """
    try:
        if "gender in" in required_expr:
            allowed = re.findall(r"'([^']+)'", required_expr)
            return gender in allowed

        if "occasion is not null" in required_expr:
            return bool(record.get("occasions"))

        if "occasion in" in required_expr:
            allowed = re.findall(r"'([^']+)'", required_expr)
            return any(occ in allowed for occ in record.get("occasions", []))

        if "category in" in required_expr:
            allowed = re.findall(r"'([^']+)'", required_expr)
            return category in allowed

    except Exception:
        pass
    return True  # Constraint not evaluable → don't reject


# =============================================================================
# ── Builder Functions
# =============================================================================

def build_category_mapping(
    category_key: str,
    kb: Optional[FashionDomainResearch] = None,
) -> CategoryMapping:
    """
    Build a CategoryMapping for one category by aggregating all KB cross-references.

    Args:
        category_key : e.g. "t_shirts"
        kb           : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        CategoryMapping dataclass.
    """
    _kb = _get_kb(kb)
    cat_def = _kb.get_category(category_key) or {}

    # Genders that include this category
    genders = [
        g for g in VALID_GENDERS
        if category_key in _kb.get_categories_for_gender(g)
    ]

    # Styles that include this category
    styles = [
        s for s in VALID_STYLES
        if category_key in _kb.get_categories_for_style(s)
    ]

    # Occasions from those styles
    occasion_set: Set[str] = set()
    for style in styles:
        style_def = _kb.get_style(style) or {}
        for occ in style_def.get("key_attributes", {}).get("occasions", []):
            occasion_set.add(occ)

    # Recommended fabrics across all seasons
    all_fabrics: Set[str] = set()
    for style in styles:
        style_def = _kb.get_style(style) or {}
        for fab in style_def.get("key_attributes", {}).get("fabrics", []):
            all_fabrics.add(fab)

    return CategoryMapping(
        category_code      = cat_def.get("code", ""),
        category_label     = cat_def.get("label", category_key),
        genders            = sorted(genders),
        styles             = sorted(styles),
        occasions          = sorted(occasion_set),
        recommended_fabrics= sorted(all_fabrics),
        attributes         = _kb.get_attributes_for_category(category_key),
        subcategories      = list(cat_def.get("subcategories", {}).keys()),
        aliases            = cat_def.get("aliases", []),
    )


def build_style_profile(
    style_key: str,
    kb: Optional[FashionDomainResearch] = None,
) -> StyleProfile:
    """
    Build an enriched StyleProfile for one style.

    Args:
        style_key : e.g. "streetwear"
        kb        : Optional pre-loaded FashionDomainResearch instance.

    Returns:
        StyleProfile dataclass.
    """
    _kb = _get_kb(kb)
    style_def = _kb.get_style(style_key) or {}
    key_attrs = style_def.get("key_attributes", {})

    # Compatible styles = parent + child styles
    compatible = (
        style_def.get("parent_styles", []) +
        style_def.get("child_styles", [])
    )

    return StyleProfile(
        style_key           = style_key,
        label               = style_def.get("label", style_key),
        code                = style_def.get("code", ""),
        tier                = style_def.get("tier", 1),
        description         = style_def.get("description", ""),
        color_palette       = style_def.get("color_palette", []),
        key_categories      = key_attrs.get("categories", []),
        key_fabrics         = key_attrs.get("fabrics", []),
        key_occasions       = key_attrs.get("occasions", []),
        aesthetic_tags      = style_def.get("aesthetic_tags", []),
        brand_archetypes    = style_def.get("brand_archetypes", []),
        compatible_styles   = [s for s in compatible if s],
        incompatible_styles = style_def.get("incompatible_with", []),
    )


# =============================================================================
# ── CLI Entry Point
# =============================================================================

def _run_cli() -> None:
    """
    Command-line interface for the Fashion Knowledge Base module.

    Usage:
        python fashion_domain_research.py                   # generate all artifacts
        python fashion_domain_research.py --stats           # print KB stats
        python fashion_domain_research.py --lookup dresses  # look up a category
        python fashion_domain_research.py --search women casual summer
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fashion Knowledge Base — Domain Research Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate all derived JSON artifacts (default action)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print knowledge base statistics"
    )
    parser.add_argument(
        "--lookup", metavar="KEY",
        help="Look up a category or style (e.g. 't_shirts', 'streetwear')"
    )
    parser.add_argument(
        "--normalize-color", metavar="COLOR",
        help="Normalize a color string to its canonical name"
    )
    parser.add_argument(
        "--normalize-fabric", metavar="FABRIC",
        help="Normalize a fabric string to its canonical name"
    )
    parser.add_argument(
        "--search", nargs="+", metavar="FILTER",
        help="Search with filters: gender style occasion season (e.g. women streetwear casual summer)"
    )
    parser.add_argument(
        "--validate", metavar="JSON",
        help='Validate a record JSON string (e.g. \'{"image_id":"x","category":"dresses","gender":"men","dataset_source":"test"}\')'
    )

    args = parser.parse_args()

    # If no flags, default to --generate
    if not any(vars(args).values()):
        args.generate = True

    kb = FashionDomainResearch()

    if args.stats:
        stats = kb.get_knowledge_base_stats()
        print("\n📊 Fashion Knowledge Base Statistics")
        print("=" * 50)
        for k, v in stats.items():
            print(f"  {k:<36} : {v}")

    if args.lookup:
        key = args.lookup
        cat = kb.get_category(key)
        if cat:
            print(f"\n📦 Category: {cat.get('label')}")
            print(json.dumps(cat, indent=2))
        else:
            style = kb.get_style(key)
            if style:
                print(f"\n🎨 Style: {style.get('label')}")
                print(json.dumps(style, indent=2))
            else:
                print(f"❌ '{key}' not found as a category or style.")

    if args.normalize_color:
        result = normalize_color(args.normalize_color, kb=kb)
        print(f"\n🎨 normalize_color('{args.normalize_color}') → {result!r}")

    if args.normalize_fabric:
        result = normalize_fabric(args.normalize_fabric, kb=kb)
        print(f"\n🧵 normalize_fabric('{args.normalize_fabric}') → {result!r}")

    if args.search:
        # Parse up to 4 positional search args as: gender style occasion season
        search_kwargs = {}
        for token in args.search:
            t = token.lower()
            if t in VALID_GENDERS or t in {"male", "female"}:
                search_kwargs["gender"] = t
            elif normalize_style(t, kb=kb):
                search_kwargs["style"] = t
            elif normalize_occasion(t):
                search_kwargs["occasion"] = t
            elif normalize_season(t):
                search_kwargs["season"] = t
        results = kb.search_by_tags(**search_kwargs)
        print(f"\n🔍 Search results for {search_kwargs}:")
        print(json.dumps(results, indent=2))

    if args.validate:
        try:
            record_dict = json.loads(args.validate)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            sys.exit(1)
        vr = kb.validate(record_dict)
        print(f"\n{vr.summary()}")
        if vr.errors:
            print("  Errors:")
            for e in vr.errors:
                print(f"    ❌ {e}")
        if vr.warnings:
            print("  Warnings:")
            for w in vr.warnings:
                print(f"    ⚠️  {w}")

    if args.generate:
        print("\n🏗  Generating all knowledge base artifacts…\n")
        artifacts = kb.generate_all_artifacts()
        print("\n✅ Artifacts saved:")
        for name, path in artifacts.items():
            size_kb = path.stat().st_size / 1024
            print(f"   {name:<25} → {path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    _run_cli()