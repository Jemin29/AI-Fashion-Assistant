"""
=============================================================================
AI-Powered Fashion Design Assistant
data_pipeline/metadata_generation/metadata_generator.py
=============================================================================
MODULE  : Automated Fashion Metadata Generation Engine
WEEK    : 1 — Fashion Domain Research & Dataset Curation
AUTHOR  : Fashion AI Team

PURPOSE
-------
Automatically extract rich, structured fashion metadata from raw text
descriptions using a three-layer pipeline:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Layer 1 — Rule-Based Extraction                                    │
  │    Keyword dictionaries, regex patterns, priority-ordered matching  │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Layer 2 — NLP-Based Extraction                                     │
  │    spaCy token/dependency analysis, noun-chunk scanning             │
  │    (graceful fallback if spaCy is not installed)                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Layer 3 — Fallback Logic                                           │
  │    KB-aligned defaults, cross-attribute inference, conflict        │
  │    resolution (e.g. "beach" occasion → "summer" season override)   │
  └─────────────────────────────────────────────────────────────────────┘

EXTRACTED ATTRIBUTES
--------------------
  style     : streetwear | luxury | formal | business_casual |
              techwear | minimalist | vintage | athleisure
  category  : t_shirts | shirts | hoodies | jackets | pants | jeans |
              shorts | dresses | ethnic_wear | footwear | accessories
  season    : spring | summer | autumn | winter | all_season
  gender    : men | women | unisex
  occasion  : casual | business_casual | formal | party | sport |
              outdoor | beach | wedding_festive | lounge
  fit       : slim_fit | regular_fit | relaxed_fit | oversized |
              cropped | skinny | straight | athletic_fit
  pattern   : solid | stripes | checks | floral | geometric |
              animal_print | camouflage | tie_dye | paisley |
              graphic | abstract | polka_dot
  color     : 30+ normalised color names (Black, White, Navy, …)

ARCHITECTURE
------------
  ExtractionResult     — Dataclass: one extracted attribute + confidence
  MetadataResult       — Dataclass: all 8 attributes + per-field details
  RuleBasedExtractor   — Layer 1: keyword/regex matching
  NLPExtractor         — Layer 2: spaCy-based extraction
  FallbackResolver     — Layer 3: cross-inference & defaults
  MetadataGeneratorEngine — Orchestrator: runs all 3 layers, logs, scores

CONFIDENCE SCORING
------------------
  Each extracted value carries a confidence in [0.0, 1.0]:
    1.0  — exact keyword match in rule-based layer
    0.85 — multi-word phrase match
    0.7  — NLP-derived (noun chunk / dependency)
    0.5  — cross-attribute inference (fallback Layer 3)
    0.3  — default fallback value (no evidence found)

INPUT  : A plain text description string
OUTPUT : MetadataResult dataclass (dict-serializable)

EXAMPLE
-------
  engine = MetadataGeneratorEngine()
  result = engine.generate("Black oversized hoodie with neon graphics")
  print(result.to_dict())
  # {
  #   "style":    "streetwear",  confidence: 0.85
  #   "category": "hoodies",    confidence: 1.0
  #   "season":   "winter",     confidence: 0.5
  #   "gender":   "unisex",     confidence: 0.3
  #   "occasion": "casual",     confidence: 0.5
  #   "fit":      "oversized",  confidence: 1.0
  #   "pattern":  "graphic",    confidence: 0.85
  #   "color":    "Black",      confidence: 1.0
  # }

USAGE
-----
  from src.data.metadata_generation import MetadataGeneratorEngine

  engine = MetadataGeneratorEngine()
  result = engine.generate("slim fit navy chinos for office wear")
  print(result.to_dict())
  print(result.to_summary())

  # Batch mode
  results = engine.generate_batch([
      "Floral print summer dress",
      "Black leather biker jacket",
  ])
=============================================================================
"""

from __future__ import annotations

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Third-party ──────────────────────────────────────────────────────────────
from loguru import logger

# ─── Optional: spaCy NLP ──────────────────────────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed — NLP extraction disabled. "
        "Install: pip install spacy && python -m spacy download en_core_web_sm"
    )

# ─── Resolve project root ──────────────────────────────────────────────────────
_FILE_DIR    = Path(__file__).resolve().parent         # metadata_generation/
_PROJECT_ROOT = _FILE_DIR.parent.parent.parent                # fashion-ai-assistant/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Knowledge-base import (graceful degradation) ────────────────────────────
try:
    from src.data.knowledge_base.fashion_domain_research import (
        VALID_CATEGORIES,
        VALID_STYLES,
        VALID_FITS,
        VALID_SEASONS,
        VALID_OCCASIONS,
        VALID_GENDERS,
    )
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    # Inline fallbacks — mirrors fashion_domain_research.py constants exactly
    VALID_CATEGORIES = frozenset({
        "t_shirts", "shirts", "hoodies", "jackets", "pants",
        "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
    })
    VALID_STYLES = frozenset({
        "streetwear", "luxury", "formal", "business_casual",
        "techwear", "minimalist", "vintage", "athleisure"
    })
    VALID_FITS = frozenset({
        "slim_fit", "regular_fit", "relaxed_fit", "oversized",
        "cropped", "skinny", "straight", "athletic_fit"
    })
    VALID_SEASONS   = frozenset({"spring", "summer", "autumn", "winter", "all_season"})
    VALID_OCCASIONS = frozenset({
        "casual", "business_casual", "formal", "party",
        "sport", "outdoor", "beach", "wedding_festive", "lounge"
    })
    VALID_GENDERS = frozenset({"men", "women", "unisex"})
    logger.warning("fashion_domain_research not importable — using inline KB constants.")


# =============================================================================
# ── 1. Data Models
# =============================================================================

@dataclass
class ExtractionResult:
    """
    Result for a single extracted attribute field.

    Attributes:
        value      : The extracted canonical value (e.g. "streetwear").
        confidence : Confidence score in [0.0, 1.0].
        method     : Which extraction layer produced this value:
                     "rule" | "nlp" | "fallback" | "default".
        evidence   : The text span(s) that triggered this extraction.
    """
    value      : Optional[str]
    confidence : float = 0.0
    method     : str   = "default"
    evidence   : str   = ""

    def is_extracted(self) -> bool:
        """Return True if a real value was found (not a default)."""
        return self.value is not None and self.method != "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value"     : self.value,
            "confidence": round(self.confidence, 4),
            "method"    : self.method,
            "evidence"  : self.evidence,
        }


@dataclass
class MetadataResult:
    """
    Complete metadata result for one fashion item description.

    Attributes:
        description       : Original input text.
        style             : Extracted style key.
        category          : Extracted category key.
        season            : Extracted season key.
        gender            : Extracted gender key.
        occasion          : Extracted occasion key.
        fit               : Extracted fit key.
        pattern           : Extracted pattern key.
        color             : Extracted color name.
        details           : Per-field ExtractionResult objects.
        overall_confidence: Mean confidence across all 8 fields.
        processing_time_ms: Time taken to generate this result.
        generated_at      : ISO-8601 UTC timestamp.
    """
    description        : str
    style              : Optional[str] = None
    category           : Optional[str] = None
    season             : Optional[str] = None
    gender             : Optional[str] = None
    occasion           : Optional[str] = None
    fit                : Optional[str] = None
    pattern            : Optional[str] = None
    color              : Optional[str] = None
    details            : Dict[str, ExtractionResult] = field(default_factory=dict)
    overall_confidence : float = 0.0
    processing_time_ms : float = 0.0
    generated_at       : str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        """
        Serialise to a plain dictionary.

        Args:
            include_details : If True, include per-field confidence/method/evidence.

        Returns:
            JSON-safe dict.
        """
        base: Dict[str, Any] = {
            "style"    : self.style,
            "category" : self.category,
            "season"   : self.season,
            "gender"   : self.gender,
            "occasion" : self.occasion,
            "fit"      : self.fit,
            "pattern"  : self.pattern,
            "color"    : self.color,
        }
        if include_details:
            base["_details"] = {
                k: v.to_dict() for k, v in self.details.items()
            }
            base["_meta"] = {
                "description"       : self.description,
                "overall_confidence": round(self.overall_confidence, 4),
                "processing_time_ms": round(self.processing_time_ms, 2),
                "generated_at"      : self.generated_at,
            }
        return base

    def to_json(self, indent: int = 2, include_details: bool = False) -> str:
        """Return formatted JSON string."""
        return json.dumps(
            self.to_dict(include_details=include_details),
            indent=indent,
            ensure_ascii=False,
        )

    def to_summary(self) -> str:
        """Return a compact human-readable one-line summary."""
        return (
            f"[{self.overall_confidence:.0%}] "
            f"cat={self.category} style={self.style} "
            f"gender={self.gender} fit={self.fit} "
            f"color={self.color} season={self.season} "
            f"occasion={self.occasion} pattern={self.pattern} "
            f"({self.processing_time_ms:.1f}ms)"
        )


# =============================================================================
# ── 2. Extraction Lexicons (Rule-Based Layer)
# =============================================================================

# ── Category keyword map: keyword → canonical category key ────────────────────
# Ordered from most specific to most generic within each group.
_CATEGORY_KEYWORDS: Dict[str, str] = {
    # Ethnic wear (check before hoodies/shirts to avoid mis-matches)
    "kurta"       : "ethnic_wear",   "salwar"       : "ethnic_wear",
    "saree"       : "ethnic_wear",   "sari"         : "ethnic_wear",
    "lehenga"     : "ethnic_wear",   "sherwani"     : "ethnic_wear",
    "dhoti"       : "ethnic_wear",   "anarkali"     : "ethnic_wear",
    "churidar"    : "ethnic_wear",   "dupatta"      : "ethnic_wear",
    # Footwear
    "sneaker"     : "footwear",      "sneakers"     : "footwear",
    "shoe"        : "footwear",      "shoes"        : "footwear",
    "boot"        : "footwear",      "boots"        : "footwear",
    "sandal"      : "footwear",      "sandals"      : "footwear",
    "heel"        : "footwear",      "heels"        : "footwear",
    "loafer"      : "footwear",      "loafers"      : "footwear",
    "flat"        : "footwear",      "flats"        : "footwear",
    "oxford"      : "footwear",      "oxfords"      : "footwear",
    "mule"        : "footwear",      "mules"        : "footwear",
    "pump"        : "footwear",      "pumps"        : "footwear",
    "slipper"     : "footwear",      "slippers"     : "footwear",
    "trainers"    : "footwear",

    # Accessories
    "bag"         : "accessories",   "watch"        : "accessories",
    "belt"        : "accessories",   "sunglasses"   : "accessories",
    "hat"         : "accessories",   "scarf"        : "accessories",
    "jewelry"     : "accessories",   "necklace"     : "accessories",
    "bracelet"    : "accessories",   "wallet"       : "accessories",
    "cap"         : "accessories",   "beanie"       : "accessories",
    "backpack"    : "accessories",   "handbag"      : "accessories",
    "tote"        : "accessories",   "purse"        : "accessories",
    # Dresses / skirts
    "dress"       : "dresses",       "skirt"        : "dresses",
    "gown"        : "dresses",       "jumpsuit"     : "dresses",
    "romper"      : "dresses",       "playsuit"     : "dresses",
    "maxi"        : "dresses",       "midi"         : "dresses",
    "mini dress"  : "dresses",       "sundress"     : "dresses",
    "bodycon"     : "dresses",       "wrap dress"   : "dresses",
    # Jackets / outerwear
    "jacket"      : "jackets",       "coat"         : "jackets",
    "blazer"      : "jackets",       "windbreaker"  : "jackets",
    "parka"       : "jackets",       "trench"       : "jackets",
    "bomber"      : "jackets",       "puffer"       : "jackets",
    "anorak"      : "jackets",       "peacoat"      : "jackets",
    "overcoat"    : "jackets",       "raincoat"     : "jackets",
    "vest"        : "jackets",       "gilet"        : "jackets",
    # Hoodies / sweatshirts
    "hoodie"      : "hoodies",       "sweatshirt"   : "hoodies",
    "pullover"    : "hoodies",       "sweater"      : "hoodies",
    "jumper"      : "hoodies",       "crewneck"     : "hoodies",
    "fleece"      : "hoodies",       "cardigan"     : "hoodies",
    "knit"        : "hoodies",       "knitwear"     : "hoodies",
    # Jeans
    "jeans"       : "jeans",         "denim"        : "jeans",
    "denims"      : "jeans",
    # Shorts
    "shorts"      : "shorts",        "bermuda"      : "shorts",
    "cutoffs"     : "shorts",        "cargo shorts" : "shorts",
    "boardshorts" : "shorts",
    # Pants / trousers
    "pants"       : "pants",         "trousers"     : "pants",
    "chinos"      : "pants",         "slacks"       : "pants",
    "joggers"     : "pants",         "leggings"     : "pants",
    "culottes"    : "pants",         "palazzos"     : "pants",
    "sweatpants"  : "pants",         "trackpants"   : "pants",
    # Shirts
    "shirt"       : "shirts",        "dress shirt"  : "shirts",
    "button down" : "shirts",        "button-down"  : "shirts",
    "blouse"      : "shirts",        "oxford shirt" : "shirts",
    "flannel"     : "shirts",        "chambray"     : "shirts",
    # T-shirts / tops (least specific — check last)
    "t-shirt"     : "t_shirts",      "tshirt"       : "t_shirts",
    "tee"         : "t_shirts",      "tank top"     : "t_shirts",
    "tank"        : "t_shirts",      "top"          : "t_shirts",
    "polo"        : "t_shirts",      "henley"       : "t_shirts",
    "crop top"    : "t_shirts",      "sleeveless"   : "t_shirts",
    "graphic tee" : "t_shirts",      "v-neck"       : "t_shirts",
}

# ── Color keyword map: keyword → canonical color name ────────────────────────
_COLOR_KEYWORDS: Dict[str, str] = {
    "black"        : "Black",      "white"      : "White",
    "grey"         : "Grey",       "gray"       : "Grey",
    "charcoal"     : "Charcoal",   "navy"       : "Navy",
    "navy blue"    : "Navy",       "cobalt"     : "Cobalt Blue",
    "royal blue"   : "Royal Blue", "blue"       : "Blue",
    "sky blue"     : "Sky Blue",   "teal"       : "Teal",
    "turquoise"    : "Turquoise",  "red"        : "Red",
    "crimson"      : "Crimson",    "burgundy"   : "Burgundy",
    "maroon"       : "Maroon",     "wine"       : "Burgundy",
    "pink"         : "Pink",       "hot pink"   : "Hot Pink",
    "blush"        : "Blush",      "rose"       : "Rose",
    "coral"        : "Coral",      "salmon"     : "Salmon",
    "orange"       : "Orange",     "rust"       : "Rust",
    "yellow"       : "Yellow",     "mustard"    : "Mustard",
    "gold"         : "Gold",       "green"      : "Green",
    "olive"        : "Olive",      "khaki"      : "Khaki",
    "sage"         : "Sage",       "mint"       : "Mint",
    "forest"       : "Forest Green","emerald"   : "Emerald",
    "purple"       : "Purple",     "lavender"   : "Lavender",
    "lilac"        : "Lilac",      "violet"     : "Violet",
    "indigo"       : "Indigo",     "plum"       : "Plum",
    "brown"        : "Brown",      "camel"      : "Camel",
    "tan"          : "Tan",        "beige"      : "Beige",
    "cream"        : "Cream",      "ivory"      : "Ivory",
    "off white"    : "Off-White",  "off-white"  : "Off-White",
    "ecru"         : "Ecru",       "nude"       : "Nude",
    "silver"       : "Silver",     "neon"       : "Neon",
    "multicolor"   : "Multicolor", "multi"      : "Multicolor",
    "tie-dye"      : "Multicolor", "rainbow"    : "Multicolor",
    "print"        : None,  # generic "print" is not a color
}

# ── Style keyword map: keyword/phrase → style key ─────────────────────────────
_STYLE_KEYWORDS: Dict[str, str] = {
    # Streetwear
    "streetwear"     : "streetwear", "street style"   : "streetwear",
    "urban"          : "streetwear", "graphic"        : "streetwear",
    "hype"           : "streetwear", "graffiti"       : "streetwear",
    "skate"          : "streetwear", "hip hop"        : "streetwear",
    "hip-hop"        : "streetwear", "logo print"     : "streetwear",
    "neon"           : "streetwear", "oversized hoodie": "streetwear",
    # Luxury
    "luxury"         : "luxury",     "premium"        : "luxury",
    "designer"       : "luxury",     "couture"        : "luxury",
    "high-end"       : "luxury",     "high end"       : "luxury",
    "exclusive"      : "luxury",     "cashmere"       : "luxury",
    "silk"           : "luxury",     "bespoke"        : "luxury",
    "couture"        : "luxury",     "satin"          : "luxury",
    # Formal
    "formal"         : "formal",     "office"         : "formal",
    "professional"   : "formal",     "suit"           : "formal",
    "tailored"       : "formal",     "corporate"      : "formal",
    "dress shirt"    : "formal",     "blazer"         : "formal",
    "tuxedo"         : "formal",     "gala"           : "formal",
    "black tie"      : "formal",     "black-tie"      : "formal",
    # Business casual
    "business casual": "business_casual", "smart casual": "business_casual",
    "office casual"  : "business_casual", "chinos"      : "business_casual",
    "polo"           : "business_casual", "loafers"     : "business_casual",
    # Techwear
    "techwear"       : "techwear",   "technical"      : "techwear",
    "waterproof"     : "techwear",   "gore-tex"       : "techwear",
    "utility"        : "techwear",   "tactical"       : "techwear",
    "modular"        : "techwear",   "functional"     : "techwear",
    # Minimalist
    "minimalist"     : "minimalist", "minimal"        : "minimalist",
    "clean lines"    : "minimalist", "monochrome"     : "minimalist",
    "understated"    : "minimalist", "simple"         : "minimalist",
    "neutral tones"  : "minimalist", "capsule"        : "minimalist",
    # Vintage
    "vintage"        : "vintage",    "retro"          : "vintage",
    "throwback"      : "vintage",    "old school"     : "vintage",
    "classic"        : "vintage",    "70s"            : "vintage",
    "80s"            : "vintage",    "90s"            : "vintage",
    "distressed"     : "vintage",    "washed"         : "vintage",
    # Athleisure
    "athleisure"     : "athleisure", "sport"          : "athleisure",
    "athletic"       : "athleisure", "gym"            : "athleisure",
    "workout"        : "athleisure", "yoga"           : "athleisure",
    "running"        : "athleisure", "performance"    : "athleisure",
    "activewear"     : "athleisure", "training"       : "athleisure",
    "fitness"        : "athleisure", "sportswear"     : "athleisure",
}

# ── Season keyword map ────────────────────────────────────────────────────────
_SEASON_KEYWORDS: Dict[str, str] = {
    "summer"     : "summer",    "beach"      : "summer",
    "tropical"   : "summer",    "linen"      : "summer",
    "breathable" : "summer",    "lightweight": "summer",
    "sundress"   : "summer",    "tank top"   : "summer",
    "shorts"     : "summer",    "sleeveless" : "summer",
    "swim"       : "summer",    "heat"       : "summer",
    "winter"     : "winter",    "wool"       : "winter",
    "fleece"     : "winter",    "insulated"  : "winter",
    "warm"       : "winter",    "puffer"     : "winter",
    "down jacket": "winter",    "heavyweight": "winter",
    "thermal"    : "winter",    "cozy"       : "winter",
    "sweater"    : "winter",    "hoodie"     : "winter",
    "coat"       : "winter",    "scarf"      : "winter",
    "beanie"     : "winter",    "cold"       : "winter",
    "spring"     : "spring",    "floral"     : "spring",
    "pastel"     : "spring",    "blossom"    : "spring",
    "refresh"    : "spring",    "light jacket": "spring",
    "rain jacket": "spring",    "trench"     : "spring",
    "autumn"     : "autumn",    "fall"       : "autumn",
    "earthy"     : "autumn",    "rust"       : "autumn",
    "mustard"    : "autumn",    "corduroy"   : "autumn",
    "layering"   : "autumn",    "harvest"    : "autumn",
    "knit"       : "autumn",    "plaid"      : "autumn",
}

# ── Fit keyword map ───────────────────────────────────────────────────────────
_FIT_KEYWORDS: Dict[str, str] = {
    "slim fit"    : "slim_fit",    "slim-fit"     : "slim_fit",
    "slim"        : "slim_fit",    "fitted"       : "slim_fit",
    "tailored"    : "slim_fit",    "close fit"    : "slim_fit",
    "regular fit" : "regular_fit", "regular-fit"  : "regular_fit",
    "regular"     : "regular_fit", "classic fit"  : "regular_fit",
    "standard"    : "regular_fit", "straight cut" : "regular_fit",
    "relaxed fit" : "relaxed_fit", "relaxed-fit"  : "relaxed_fit",
    "relaxed"     : "relaxed_fit", "comfort fit"  : "relaxed_fit",
    "easy fit"    : "relaxed_fit", "loose"        : "relaxed_fit",
    "loose fit"   : "relaxed_fit",
    "oversized"   : "oversized",   "oversize"     : "oversized",
    "boxy"        : "oversized",   "baggy"        : "oversized",
    "slouchy"     : "oversized",   "drop shoulder": "oversized",
    "cropped"     : "cropped",     "crop"         : "cropped",
    "cutoff"      : "cropped",     "cut off"      : "cropped",
    "belly"       : "cropped",
    "skinny"      : "skinny",      "ultra slim"   : "skinny",
    "super slim"  : "skinny",      "skin tight"   : "skinny",
    "straight"    : "straight",    "straight leg" : "straight",
    "straight-leg": "straight",    "box cut"      : "straight",
    "athletic fit": "athletic_fit","muscle fit"   : "athletic_fit",
    "athletic"    : "athletic_fit","sport fit"    : "athletic_fit",
}

# ── Pattern keyword map ───────────────────────────────────────────────────────
_PATTERN_KEYWORDS: Dict[str, str] = {
    "solid"        : "solid",       "plain"       : "solid",
    "no print"     : "solid",       "monochrome"  : "solid",
    "one color"    : "solid",       "single color": "solid",
    "stripes"      : "stripes",     "striped"     : "stripes",
    "pinstripe"    : "stripes",     "stripy"      : "stripes",
    "checks"       : "checks",      "checked"     : "checks",
    "plaid"        : "checks",      "tartan"      : "checks",
    "gingham"      : "checks",      "houndstooth" : "checks",
    "buffalo check": "checks",      "windowpane"  : "checks",
    "floral"       : "floral",      "flower"      : "floral",
    "botanical"    : "floral",      "rose print"  : "floral",
    "daisy"        : "floral",
    "geometric"    : "geometric",   "geo"         : "geometric",
    "chevron"      : "geometric",   "diamond"     : "geometric",
    "hexagon"      : "geometric",
    "animal print" : "animal_print","leopard"     : "animal_print",
    "zebra"        : "animal_print","snake print" : "animal_print",
    "cheetah"      : "animal_print","tiger"       : "animal_print",
    "camouflage"   : "camouflage",  "camo"        : "camouflage",
    "military print": "camouflage",
    "tie dye"      : "tie_dye",     "tie-dye"     : "tie_dye",
    "dip dye"      : "tie_dye",     "ombre"       : "tie_dye",
    "paisley"      : "paisley",     "boteh"       : "paisley",
    "graphic"      : "graphic",     "logo"        : "graphic",
    "text print"   : "graphic",     "slogan"      : "graphic",
    "screen print" : "graphic",     "illustration": "graphic",
    "neon graphic" : "graphic",     "neon print"  : "graphic",
    "abstract"     : "abstract",    "painterly"   : "abstract",
    "artistic"     : "abstract",    "watercolor"  : "abstract",
    "polka dot"    : "polka_dot",   "polka-dot"   : "polka_dot",
    "dots"         : "polka_dot",   "spotted"     : "polka_dot",
    "dotted"       : "polka_dot",
}

# ── Gender keyword map ────────────────────────────────────────────────────────
_GENDER_KEYWORDS: Dict[str, str] = {
    "men"           : "men",      "man"         : "men",
    "male"          : "men",      "men's"       : "men",
    "mens"          : "men",      "boys"        : "men",
    "boyfriend"     : "men",      "masculine"   : "men",
    "women"         : "women",    "woman"       : "women",
    "female"        : "women",    "women's"     : "women",
    "womens"        : "women",    "ladies"      : "women",
    "girls"         : "women",    "feminine"    : "women",
    "unisex"        : "unisex",   "gender neutral": "unisex",
    "gender-neutral": "unisex",   "everyone"    : "unisex",
    "all genders"   : "unisex",
}

# ── Occasion keyword map ──────────────────────────────────────────────────────
_OCCASION_KEYWORDS: Dict[str, str] = {
    "casual"         : "casual",          "everyday"     : "casual",
    "day to day"     : "casual",          "weekend"      : "casual",
    "leisure"        : "casual",          "relaxed"      : "casual",
    "business casual": "business_casual", "smart casual" : "business_casual",
    "office casual"  : "business_casual",
    "formal"         : "formal",          "office"       : "formal",
    "business"       : "formal",          "work"         : "formal",
    "professional"   : "formal",          "corporate"    : "formal",
    "black tie"      : "formal",          "gala"         : "formal",
    "party"          : "party",           "night out"    : "party",
    "club"           : "party",           "celebration"  : "party",
    "social"         : "party",           "cocktail"     : "party",
    "sport"          : "sport",           "gym"          : "sport",
    "workout"        : "sport",           "running"      : "sport",
    "yoga"           : "sport",           "training"     : "sport",
    "athletic"       : "sport",           "fitness"      : "sport",
    "outdoor"        : "outdoor",         "hiking"       : "outdoor",
    "camping"        : "outdoor",         "adventure"    : "outdoor",
    "travel"         : "outdoor",
    "beach"          : "beach",           "pool"         : "beach",
    "resort"         : "beach",           "vacation"     : "beach",
    "swim"           : "beach",           "surf"         : "beach",
    "wedding"        : "wedding_festive", "bridal"       : "wedding_festive",
    "festival"       : "wedding_festive", "ceremony"     : "wedding_festive",
    "festive"        : "wedding_festive", "diwali"       : "wedding_festive",
    "prom"           : "wedding_festive",
    "lounge"         : "lounge",          "homewear"     : "lounge",
    "pajama"         : "lounge",          "pyjama"       : "lounge",
    "sleepwear"      : "lounge",          "comfortable"  : "lounge",
}

# ── Confidence tiers ──────────────────────────────────────────────────────────
_CONF_EXACT         = 1.00   # single exact keyword match
_CONF_PHRASE        = 0.85   # multi-word phrase match
_CONF_NLP           = 0.70   # spaCy-derived
_CONF_CROSS_INFER   = 0.50   # inferred from another attribute
_CONF_DEFAULT       = 0.30   # pure default (no evidence)


# =============================================================================
# ── 3. Layer 1 — Rule-Based Extractor
# =============================================================================

class RuleBasedExtractor:
    """
    Extracts fashion attributes from text using keyword dictionaries and
    regular expression patterns.

    Strategy:
      1. Text is lowercased and normalised (collapse spaces, strip punctuation).
      2. Multi-word phrases are checked first (longer matches take priority).
      3. Single keyword matches are checked next.
      4. Regex patterns are used for colours (e.g. "neon pink", "dark red").
      5. Confidence is set to _CONF_PHRASE for multi-word and _CONF_EXACT for
         single-word matches.

    This extractor operates entirely without ML models or network calls.
    """

    # ── Regex patterns for colour modifiers ───────────────────────────────────
    _COLOR_MOD_RE = re.compile(
        r"\b(?:dark|light|bright|deep|pale|soft|muted|rich|bold|neon|pastel)\s+"
        r"(\w+)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        # Pre-sort all dictionaries: longest phrases first to prefer specificity
        self._sorted_category = self._sort_by_len(_CATEGORY_KEYWORDS)
        self._sorted_color     = self._sort_by_len(_COLOR_KEYWORDS)
        self._sorted_style     = self._sort_by_len(_STYLE_KEYWORDS)
        self._sorted_season    = self._sort_by_len(_SEASON_KEYWORDS)
        self._sorted_fit       = self._sort_by_len(_FIT_KEYWORDS)
        self._sorted_pattern   = self._sort_by_len(_PATTERN_KEYWORDS)
        self._sorted_gender    = self._sort_by_len(_GENDER_KEYWORDS)
        self._sorted_occasion  = self._sort_by_len(_OCCASION_KEYWORDS)

    @staticmethod
    def _sort_by_len(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Sort keyword dict items by descending key length (longest first)."""
        return sorted(d.items(), key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise text for matching:
          1. Lowercase.
          2. Collapse multiple whitespace to single space.
          3. Keep hyphens (they're part of terms like "slim-fit").
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _match_keyword(
        self,
        text_norm  : str,
        sorted_dict: List[Tuple[str, str]],
    ) -> ExtractionResult:
        """
        Scan sorted_dict for the first keyword found in text_norm.
        Longer (multi-word) phrases are checked first.

        Args:
            text_norm   : Normalised (lowercased, collapsed) input text.
            sorted_dict : Pre-sorted list of (keyword, value) tuples.

        Returns:
            ExtractionResult with value/confidence/evidence set,
            or ExtractionResult(value=None) if no match found.
        """
        for keyword, value in sorted_dict:
            if not value:          # skip None-value entries (e.g. "print" in colors)
                continue
            # Use word-boundary-aware search for single words, substring for phrases
            if " " in keyword or "-" in keyword:
                if keyword in text_norm:
                    conf = _CONF_PHRASE
                    return ExtractionResult(value, conf, "rule", keyword)
            else:
                pattern = r"\b" + re.escape(keyword) + r"\b"
                m = re.search(pattern, text_norm)
                if m:
                    conf = _CONF_EXACT
                    return ExtractionResult(value, conf, "rule", m.group())
        return ExtractionResult(None)

    def extract_all(self, text: str) -> Dict[str, ExtractionResult]:
        """
        Run rule-based extraction on all 8 attribute fields.

        Args:
            text : Raw description string.

        Returns:
            Dict mapping field name → ExtractionResult.
        """
        norm = self._normalise(text)

        results = {
            "category" : self._match_keyword(norm, self._sorted_category),
            "color"    : self._extract_color(norm),
            "style"    : self._match_keyword(norm, self._sorted_style),
            "season"   : self._match_keyword(norm, self._sorted_season),
            "fit"      : self._match_keyword(norm, self._sorted_fit),
            "pattern"  : self._match_keyword(norm, self._sorted_pattern),
            "gender"   : self._match_keyword(norm, self._sorted_gender),
            "occasion" : self._match_keyword(norm, self._sorted_occasion),
        }

        logger.debug(
            "Rule-based: "
            + ", ".join(f"{k}={v.value}({v.confidence:.2f})"
                        for k, v in results.items() if v.value)
        )
        return results

    def _extract_color(self, text_norm: str) -> ExtractionResult:
        """
        Extract color with modifier-aware matching.

        Checks modifier+color patterns first (e.g. "dark navy"), then
        falls back to plain keyword matching.

        Args:
            text_norm : Normalised input text.

        Returns:
            ExtractionResult for the color field.
        """
        # 1. Modifier+colour regex (e.g. "neon pink", "dark teal")
        m = self._COLOR_MOD_RE.search(text_norm)
        if m:
            base = m.group(1).lower()
            if base in _COLOR_KEYWORDS and _COLOR_KEYWORDS[base]:
                canonical = _COLOR_KEYWORDS[base]
                return ExtractionResult(canonical, _CONF_PHRASE, "rule", m.group())

        # 2. Standard keyword match (sorted longest first)
        return self._match_keyword(text_norm, self._sorted_color)


# =============================================================================
# ── 4. Layer 2 — NLP-Based Extractor
# =============================================================================

class NLPExtractor:
    """
    Extracts fashion attributes using spaCy linguistic analysis.

    When spaCy is available this layer:
      - Scans noun chunks for category terms.
      - Uses dependency parsing to find adjectival modifiers of garments.
      - Reads ADJ/NOUN POS tags to pick up attribute words missed by rules.

    When spaCy is unavailable the layer is a no-op — all results come back
    with value=None, which the orchestrator handles gracefully.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp = None
        if _SPACY_AVAILABLE:
            self._nlp = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Optional[Any]:
        """Load a spaCy model by name. Silently return None on failure."""
        try:
            nlp = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded for NLP extraction.")
            return nlp
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Install: python -m spacy download {model_name}"
            )
            return None
        except Exception as exc:
            logger.warning(f"Failed to load spaCy model: {exc}")
            return None

    @property
    def available(self) -> bool:
        """True if spaCy is installed and a model was loaded successfully."""
        return self._nlp is not None

    def extract_all(self, text: str) -> Dict[str, ExtractionResult]:
        """
        Run NLP extraction on all 8 fields.

        For each field, tries:
          1. Noun chunk scanning against keyword dicts.
          2. Token-level adjective/noun scanning.

        Args:
            text : Raw description string.

        Returns:
            Dict mapping field name → ExtractionResult (value=None if unextracted).
        """
        empty: Dict[str, ExtractionResult] = {
            k: ExtractionResult(None)
            for k in ("style", "category", "season", "gender",
                      "occasion", "fit", "pattern", "color")
        }

        if not self.available:
            return empty

        try:
            doc = self._nlp(text)
        except Exception as exc:
            logger.warning(f"spaCy processing failed: {exc}")
            return empty

        # Collect noun chunks and individual tokens for scanning
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        adj_tokens  = [
            t.text.lower()
            for t in doc
            if t.pos_ in ("ADJ", "NOUN", "PROPN")
        ]
        all_tokens  = noun_chunks + adj_tokens

        results: Dict[str, ExtractionResult] = {}
        for field, sorted_kw in [
            ("category", _CATEGORY_KEYWORDS),
            ("style",    _STYLE_KEYWORDS),
            ("season",   _SEASON_KEYWORDS),
            ("fit",      _FIT_KEYWORDS),
            ("pattern",  _PATTERN_KEYWORDS),
            ("gender",   _GENDER_KEYWORDS),
            ("occasion", _OCCASION_KEYWORDS),
            ("color",    _COLOR_KEYWORDS),
        ]:
            results[field] = self._scan_tokens(all_tokens, sorted_kw)

        logger.debug(
            "NLP: "
            + ", ".join(f"{k}={v.value}({v.confidence:.2f})"
                        for k, v in results.items() if v.value)
        )
        return results

    @staticmethod
    def _scan_tokens(
        tokens    : List[str],
        keyword_map: Dict[str, str],
    ) -> ExtractionResult:
        """
        Scan a list of token strings for the first match in keyword_map.

        Args:
            tokens      : Lowercased noun chunk and token strings.
            keyword_map : Keyword → canonical value dict.

        Returns:
            ExtractionResult from NLP layer, or (None) if no match.
        """
        sorted_kw = sorted(keyword_map.items(), key=lambda x: len(x[0]), reverse=True)
        for token in tokens:
            for keyword, value in sorted_kw:
                if not value:
                    continue
                if keyword in token or token in keyword:
                    return ExtractionResult(value, _CONF_NLP, "nlp", token)
        return ExtractionResult(None)


# =============================================================================
# ── 5. Layer 3 — Fallback Resolver
# =============================================================================

class FallbackResolver:
    """
    Cross-attribute inference and default assignment.

    Applies after Layers 1 and 2. Fills any remaining None fields using:
      a. Cross-attribute inference rules:
         - "beach" occasion → "summer" season
         - "swimwear" category → "summer" season
         - "ski jacket" + "winter" → "outdoor" occasion
         - "luxury" style → "formal" occasion if not set
         - "athleisure" style → "sport" occasion if not set
         - "formal" style → gender hint from category (dress → women)
         - "dresses" category → "women" gender (unless overridden)
         - "jeans" or "t_shirts" + no gender → "unisex"
      b. Category → season defaults (e.g. jacket → all_season).
      c. Universal defaults for completely un-extracted fields.

    All fallback results get confidence = _CONF_CROSS_INFER (0.5) for
    inferences, or _CONF_DEFAULT (0.3) for pure defaults.
    """

    # ── Cross-inference rules (if field_a = val_a → set field_b = val_b) ──────
    _INFERENCE_RULES: List[Tuple[str, str, str, str]] = [
        # field_a,     value_a,           field_b,    inferred_value_b
        ("occasion",   "beach",           "season",   "summer"),
        ("occasion",   "sport",           "style",    "athleisure"),
        ("occasion",   "lounge",          "style",    "minimalist"),
        ("occasion",   "formal",          "style",    "formal"),
        ("occasion",   "wedding_festive", "style",    "luxury"),
        ("occasion",   "outdoor",         "season",   "autumn"),
        ("style",      "athleisure",      "occasion", "sport"),
        ("style",      "luxury",          "occasion", "formal"),
        ("style",      "streetwear",      "occasion", "casual"),
        ("style",      "minimalist",      "occasion", "casual"),
        ("style",      "formal",          "occasion", "formal"),
        ("style",      "techwear",        "occasion", "outdoor"),
        ("style",      "vintage",         "occasion", "casual"),
        ("category",   "shorts",          "season",   "summer"),
        ("category",   "dresses",         "gender",   "women"),
        ("category",   "hoodies",         "season",   "winter"),
        ("category",   "jackets",         "occasion", "outdoor"),
        ("category",   "footwear",        "gender",   "unisex"),
    ]

    # ── Category → style inference ────────────────────────────────────────────
    _CATEGORY_STYLE_MAP: Dict[str, str] = {
        "hoodies"    : "streetwear",
        "jeans"      : "streetwear",
        "ethnic_wear": "formal",
    }

    # ── Category → season defaults ────────────────────────────────────────────
    _CATEGORY_SEASON_DEFAULT: Dict[str, str] = {
        "dresses"    : "summer",
        "shorts"     : "summer",
        "hoodies"    : "winter",
        "jackets"    : "all_season",
        "pants"      : "all_season",
        "jeans"      : "all_season",
        "shirts"     : "all_season",
        "t_shirts"   : "all_season",
        "footwear"   : "all_season",
        "accessories": "all_season",
        "ethnic_wear": "all_season",
    }

    # ── Gender defaults by category ───────────────────────────────────────────
    _CATEGORY_GENDER_DEFAULT: Dict[str, str] = {
        "dresses"    : "women",
        "ethnic_wear": "unisex",
        "footwear"   : "unisex",
        "accessories": "unisex",
        "t_shirts"   : "unisex",
        "jeans"      : "unisex",
        "shorts"     : "unisex",
        "hoodies"    : "unisex",
        "jackets"    : "unisex",
    }

    def resolve(
        self,
        results  : Dict[str, ExtractionResult],
        text_norm: str,
    ) -> Dict[str, ExtractionResult]:
        """
        Fill missing fields using cross-attribute inference and defaults.

        Modifies the results dict in place and returns it.

        Args:
            results   : Dict from rule/NLP layers (may contain None values).
            text_norm : Normalised input text (for pattern-based fallbacks).

        Returns:
            Updated results dict with all 8 fields populated.
        """
        r = results  # alias for brevity

        # ── Step 1: Apply cross-inference rules ───────────────────────────────
        for field_a, val_a, field_b, inferred_val in self._INFERENCE_RULES:
            if (r[field_a].value == val_a and r[field_b].value is None):
                r[field_b] = ExtractionResult(
                    inferred_val, _CONF_CROSS_INFER, "fallback",
                    f"inferred from {field_a}={val_a}"
                )

        # ── Step 2: Category-driven style inference ───────────────────────────
        if r["style"].value is None and r["category"].value:
            inferred_style = self._CATEGORY_STYLE_MAP.get(r["category"].value)
            if inferred_style:
                r["style"] = ExtractionResult(
                    inferred_style, _CONF_CROSS_INFER, "fallback",
                    f"inferred from category={r['category'].value}"
                )

        # ── Step 3: Category-driven season default ────────────────────────────
        if r["season"].value is None and r["category"].value:
            season_def = self._CATEGORY_SEASON_DEFAULT.get(r["category"].value)
            if season_def:
                r["season"] = ExtractionResult(
                    season_def, _CONF_CROSS_INFER, "fallback",
                    f"default for category={r['category'].value}"
                )

        # ── Step 4: Category-driven gender default ────────────────────────────
        if r["gender"].value is None and r["category"].value:
            gender_def = self._CATEGORY_GENDER_DEFAULT.get(r["category"].value)
            if gender_def:
                r["gender"] = ExtractionResult(
                    gender_def, _CONF_CROSS_INFER, "fallback",
                    f"default for category={r['category'].value}"
                )

        # ── Step 5: Universal defaults for remaining None fields ──────────────
        _FINAL_DEFAULTS = {
            "style"   : "casual",
            "category": "accessories",
            "season"  : "all_season",
            "gender"  : "unisex",
            "occasion": "casual",
            "fit"     : "regular_fit",
            "pattern" : "solid",
            "color"   : None,          # color has no sensible default → leave None
        }
        for field, default_val in _FINAL_DEFAULTS.items():
            if r[field].value is None:
                r[field] = ExtractionResult(
                    default_val, _CONF_DEFAULT, "default",
                    f"no evidence found — using default"
                )

        logger.debug(
            "Fallback: "
            + ", ".join(f"{k}={v.value}({v.confidence:.2f})"
                        for k, v in r.items() if v.method in ("fallback", "default"))
        )
        return r


# =============================================================================
# ── 6. MetadataGeneratorEngine — Orchestrator
# =============================================================================

class MetadataGeneratorEngine:
    """
    Orchestrates the three-layer metadata extraction pipeline.

    Extraction Order:
      1. RuleBasedExtractor   → fast, zero-dependency keyword matching
      2. NLPExtractor         → spaCy noun-chunk / POS analysis
      3. FallbackResolver     → cross-inference and default assignment

    Merge Strategy:
      The highest-confidence result per field wins.
      Rule-based results beat NLP results at equal confidence.

    Logging:
      Every call to generate() logs at DEBUG level (per-field extractions)
      and INFO level (summary + processing time).

    Args:
        nlp_model   : spaCy model name (default: "en_core_web_sm").
        enable_nlp  : Set False to skip NLP layer even if spaCy is installed.
        log_level   : Loguru log level for engine-level messages.
    """

    def __init__(
        self,
        nlp_model : str  = "en_core_web_sm",
        enable_nlp: bool = True,
    ) -> None:
        logger.info("Initialising MetadataGeneratorEngine…")

        self._rule_extractor = RuleBasedExtractor()
        self._nlp_extractor  = NLPExtractor(nlp_model) if enable_nlp else NLPExtractor.__new__(NLPExtractor)
        if not enable_nlp:
            self._nlp_extractor._nlp = None

        self._fallback = FallbackResolver()

        nlp_status = "enabled" if (enable_nlp and self._nlp_extractor.available) else "disabled"
        logger.info(f"MetadataGeneratorEngine ready | NLP={nlp_status}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, description: str) -> MetadataResult:
        """
        Generate structured metadata from a single text description.

        Args:
            description : Raw fashion item description string.

        Returns:
            MetadataResult with all 8 attributes populated.

        Example:
            result = engine.generate("Black oversized hoodie with neon graphics")
            print(result.to_json())
        """
        if not description or not description.strip():
            logger.warning("Empty description received — returning all defaults.")
            return self._make_empty_result(description)

        t_start = time.perf_counter()

        # ── Layer 1: Rule-based ───────────────────────────────────────────────
        rule_results = self._rule_extractor.extract_all(description)

        # ── Layer 2: NLP-based ────────────────────────────────────────────────
        nlp_results = self._nlp_extractor.extract_all(description)

        # ── Merge: highest confidence wins ────────────────────────────────────
        merged = self._merge(rule_results, nlp_results)

        # ── Layer 3: Fallback resolution ──────────────────────────────────────
        norm    = description.lower()
        final   = self._fallback.resolve(merged, norm)

        # ── Build result ──────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        result = MetadataResult(
            description        = description,
            style              = final["style"].value,
            category           = final["category"].value,
            season             = final["season"].value,
            gender             = final["gender"].value,
            occasion           = final["occasion"].value,
            fit                = final["fit"].value,
            pattern            = final["pattern"].value,
            color              = final["color"].value,
            details            = final,
            overall_confidence = self._mean_confidence(final),
            processing_time_ms = elapsed_ms,
        )

        logger.info(
            f"Generated metadata | {result.to_summary()}"
        )
        return result

    def generate_batch(
        self,
        descriptions: List[str],
    ) -> List[MetadataResult]:
        """
        Generate metadata for a list of descriptions.

        Args:
            descriptions : List of raw description strings.

        Returns:
            List of MetadataResult objects in the same order.
        """
        logger.info(f"Batch generation: {len(descriptions)} descriptions")
        results = []
        for i, desc in enumerate(descriptions):
            try:
                results.append(self.generate(desc))
            except Exception as exc:
                logger.error(f"Error at index {i}: {exc}")
                results.append(self._make_empty_result(desc))
        logger.success(
            f"Batch complete: {len(results)} results | "
            f"avg_conf={sum(r.overall_confidence for r in results)/max(len(results),1):.2%}"
        )
        return results

    def generate_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a FashionGen/DeepFashion processed record dict with
        auto-generated metadata from its 'description' field.

        Fields NOT already set in the record are filled from generation.
        Existing non-None, non-empty fields are preserved.

        Args:
            record : Dict with at least 'description' key.

        Returns:
            A new enriched dict (original not mutated).
        """
        enriched   = dict(record)
        desc       = record.get("description") or ""
        result     = self.generate(desc)
        result_d   = result.to_dict()

        # Only fill fields that are missing or empty in the original record
        _FILLABLE = ("style", "season", "gender", "occasion", "fit", "pattern", "color")
        for field_name in _FILLABLE:
            current = enriched.get(field_name)
            if not current:
                enriched[field_name] = result_d.get(field_name)

        # category: only fill if record has an empty category
        if not enriched.get("category"):
            enriched["category"] = result_d.get("category")

        # Attach generation details
        enriched["_auto_generated"]        = True
        enriched["_gen_confidence"]        = round(result.overall_confidence, 4)
        enriched["_gen_processing_time_ms"] = round(result.processing_time_ms, 2)
        return enriched

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _merge(
        rule: Dict[str, ExtractionResult],
        nlp : Dict[str, ExtractionResult],
    ) -> Dict[str, ExtractionResult]:
        """
        Merge rule-based and NLP results.

        Rule wins at equal confidence.
        NLP wins only if rule has no result (None) AND NLP has a result.

        Args:
            rule : Results from RuleBasedExtractor.
            nlp  : Results from NLPExtractor.

        Returns:
            Merged dict.
        """
        merged: Dict[str, ExtractionResult] = {}
        for key in rule:
            r = rule[key]
            n = nlp[key]
            if r.value is not None and n.value is not None:
                merged[key] = r if r.confidence >= n.confidence else n
            elif r.value is not None:
                merged[key] = r
            elif n.value is not None:
                merged[key] = n
            else:
                merged[key] = ExtractionResult(None)
        return merged

    @staticmethod
    def _mean_confidence(results: Dict[str, ExtractionResult]) -> float:
        """Compute mean confidence across all 8 fields."""
        confs = [v.confidence for v in results.values() if v.value is not None]
        return round(sum(confs) / len(confs), 4) if confs else 0.0

    @staticmethod
    def _make_empty_result(description: str) -> MetadataResult:
        """Return a MetadataResult with all default values for an empty input."""
        return MetadataResult(
            description        = description,
            style              = "casual",
            category           = "accessories",
            season             = "all_season",
            gender             = "unisex",
            occasion           = "casual",
            fit                = "regular_fit",
            pattern            = "solid",
            color              = None,
            overall_confidence = _CONF_DEFAULT,
        )
