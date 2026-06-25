"""
week2/prompts/style_presets.py
================================
Fashion style preset registry for prompt building.

Each preset defines positive and negative tags, recommended generation
parameters, and category applicability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# ── Style Preset Dataclass
# =============================================================================

@dataclass
class StylePreset:
    """
    A named fashion style with all prompt engineering data.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. "streetwear").
    display_name : str
        Human-readable name (e.g. "Streetwear / Urban").
    positive_tags : list of str
        Tags appended to the positive prompt.
    negative_tags : list of str
        Tags appended to the negative prompt.
    quality_tags : list of str
        Technical/quality boosters specific to this style.
    recommended_steps : int
        Suggested num_inference_steps for this style.
    recommended_guidance : float
        Suggested guidance scale.
    compatible_categories : list of str
        Fashion categories this style works with.
    description : str
        Short style description for documentation.
    """
    name:                   str
    display_name:           str
    positive_tags:          List[str]   = field(default_factory=list)
    negative_tags:          List[str]   = field(default_factory=list)
    quality_tags:           List[str]   = field(default_factory=list)
    recommended_steps:      int         = 30
    recommended_guidance:   float       = 7.5
    compatible_categories:  List[str]   = field(default_factory=list)
    description:            str         = ""

    def get_positive_string(self, sep: str = ", ") -> str:
        return sep.join(self.positive_tags + self.quality_tags)

    def get_negative_string(self, sep: str = ", ") -> str:
        return sep.join(self.negative_tags)


# =============================================================================
# ── Preset Registry
# =============================================================================

_PRESETS: Dict[str, StylePreset] = {

    "streetwear": StylePreset(
        name               = "streetwear",
        display_name       = "Streetwear / Urban",
        description        = "Urban street style with bold graphics, oversized fits, and sneaker culture.",
        positive_tags      = [
            "urban streetwear fashion",
            "trendy street style",
            "oversized silhouette",
            "graphic print",
            "sneaker culture",
            "skate style",
            "hypebeast fashion",
        ],
        negative_tags      = ["formal", "business attire", "office wear", "suit and tie"],
        quality_tags       = ["editorial photography", "urban background", "natural light"],
        recommended_steps  = 30,
        recommended_guidance = 7.0,
        compatible_categories = ["t_shirts", "hoodies", "jackets", "pants", "jeans", "shorts", "footwear"],
    ),

    "formal": StylePreset(
        name               = "formal",
        display_name       = "Formal / Business",
        description        = "Elegant, tailored clothing for professional and formal occasions.",
        positive_tags      = [
            "elegant formal wear",
            "tailored silhouette",
            "luxury fashion",
            "sophisticated style",
            "haute couture",
            "polished look",
            "professional attire",
        ],
        negative_tags      = ["casual", "streetwear", "sporty", "ripped", "distressed"],
        quality_tags       = ["studio photography", "clean white background", "professional lighting"],
        recommended_steps  = 35,
        recommended_guidance = 8.0,
        compatible_categories = ["shirts", "jackets", "pants", "dresses", "accessories"],
    ),

    "casual": StylePreset(
        name               = "casual",
        display_name       = "Casual / Everyday",
        description        = "Relaxed, comfortable everyday fashion for any occasion.",
        positive_tags      = [
            "casual everyday fashion",
            "relaxed fit",
            "comfortable style",
            "minimalist aesthetic",
            "laid-back look",
        ],
        negative_tags      = ["formal", "evening wear", "ball gown"],
        quality_tags       = ["lifestyle photography", "natural lighting", "outdoor setting"],
        recommended_steps  = 25,
        recommended_guidance = 7.0,
        compatible_categories = ["t_shirts", "shirts", "jeans", "shorts", "hoodies", "footwear"],
    ),

    "athleisure": StylePreset(
        name               = "athleisure",
        display_name       = "Athleisure / Sporty",
        description        = "Athletic-meets-fashion — performance wear with a stylish edge.",
        positive_tags      = [
            "athletic fashion",
            "sporty chic",
            "performance wear",
            "active lifestyle",
            "gym fashion",
            "sportswear",
            "fitness aesthetic",
        ],
        negative_tags      = ["formal", "office wear", "suit", "heels", "evening gown"],
        quality_tags       = ["action photography", "clean background", "dynamic pose"],
        recommended_steps  = 28,
        recommended_guidance = 7.5,
        compatible_categories = ["t_shirts", "shorts", "pants", "jackets", "footwear"],
    ),

    "bohemian": StylePreset(
        name               = "bohemian",
        display_name       = "Bohemian / Boho-Chic",
        description        = "Free-spirited, earthy, vintage-inspired fashion with flowing fabrics.",
        positive_tags      = [
            "bohemian fashion",
            "boho chic",
            "free-spirited style",
            "flowy fabric",
            "earthy tones",
            "vintage-inspired",
            "eclectic accessories",
            "natural textures",
        ],
        negative_tags      = ["structured", "corporate", "stiff fabric", "minimalist"],
        quality_tags       = ["golden hour photography", "natural outdoor setting", "warm tones"],
        recommended_steps  = 30,
        recommended_guidance = 7.0,
        compatible_categories = ["dresses", "shirts", "accessories", "ethnic_wear"],
    ),

    "luxury": StylePreset(
        name               = "luxury",
        display_name       = "Luxury / High-End",
        description        = "Premium designer fashion with opulent materials and expert craftsmanship.",
        positive_tags      = [
            "luxury fashion",
            "high-end designer",
            "premium materials",
            "opulent style",
            "couture quality",
            "fashion week",
            "avant-garde tailoring",
        ],
        negative_tags      = ["fast fashion", "casual", "streetwear", "cheap fabric"],
        quality_tags       = ["luxury editorial", "high fashion magazine", "dramatic lighting", "8k"],
        recommended_steps  = 40,
        recommended_guidance = 8.5,
        compatible_categories = ["jackets", "dresses", "shirts", "accessories"],
    ),

    "minimalist": StylePreset(
        name               = "minimalist",
        display_name       = "Minimalist / Clean",
        description        = "Clean lines, neutral palette, and understated elegance.",
        positive_tags      = [
            "minimalist fashion",
            "clean lines",
            "neutral palette",
            "understated elegance",
            "Scandinavian style",
            "monochromatic look",
        ],
        negative_tags      = ["busy pattern", "maximalist", "heavy embellishment", "loud colors", "floral"],
        quality_tags       = ["clean white studio", "minimal background", "soft diffused lighting"],
        recommended_steps  = 30,
        recommended_guidance = 7.5,
        compatible_categories = ["t_shirts", "shirts", "pants", "jackets", "dresses"],
    ),

    "avant_garde": StylePreset(
        name               = "avant_garde",
        display_name       = "Avant-Garde / Experimental",
        description        = "Conceptual, boundary-pushing fashion with sculptural silhouettes.",
        positive_tags      = [
            "avant-garde fashion",
            "experimental design",
            "artistic fashion",
            "conceptual clothing",
            "sculptural silhouette",
            "fashion forward",
            "editorial statement",
        ],
        negative_tags      = ["conventional", "basic", "simple", "ordinary"],
        quality_tags       = ["fashion editorial", "dramatic composition", "artistic lighting", "vogue"],
        recommended_steps  = 45,
        recommended_guidance = 9.0,
        compatible_categories = ["dresses", "jackets", "accessories"],
    ),

    "ethnic": StylePreset(
        name               = "ethnic",
        display_name       = "Traditional / Ethnic",
        description        = "Traditional ethnic wear with cultural embroidery and authentic fabrics.",
        positive_tags      = [
            "traditional ethnic fashion",
            "cultural clothing",
            "embroidered fabric",
            "handcrafted details",
            "authentic textiles",
            "heritage fashion",
        ],
        negative_tags      = ["western", "casual", "modern minimalist"],
        quality_tags       = ["warm studio lighting", "cultural backdrop", "detailed close-up"],
        recommended_steps  = 35,
        recommended_guidance = 7.5,
        compatible_categories = ["ethnic_wear", "accessories", "dresses"],
    ),
}


# =============================================================================
# ── Public API
# =============================================================================

def get_preset(name: str) -> StylePreset:
    """
    Retrieve a style preset by name.

    Parameters
    ----------
    name : str  e.g. "streetwear", "formal", "luxury"

    Raises
    ------
    KeyError  if the preset name is not registered.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _PRESETS:
        available = list(_PRESETS.keys())
        raise KeyError(
            f"Unknown style preset {name!r}. Available: {available}"
        )
    return _PRESETS[key]


def list_presets() -> List[str]:
    """Return all registered style preset names."""
    return list(_PRESETS.keys())


def get_all_presets() -> Dict[str, StylePreset]:
    """Return the full preset registry."""
    return dict(_PRESETS)


def register_preset(preset: StylePreset) -> None:
    """
    Register a custom style preset at runtime.

    Parameters
    ----------
    preset : StylePreset
    """
    _PRESETS[preset.name.lower()] = preset
