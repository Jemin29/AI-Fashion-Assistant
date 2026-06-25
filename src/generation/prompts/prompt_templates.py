"""
week2/prompts/prompt_templates.py
====================================
Production-Ready Prompt Engineering Framework
AI-Powered Fashion Design Assistant — Week 2

╔══════════════════════════════════════════════════════════════╗
║              Fashion Prompt Template Engine                  ║
║                                                              ║
║  Provides three-layer prompt engineering:                    ║
║    1. Template layer  — style-specific sentence templates    ║
║    2. Enhancement layer — context boosters & modifiers       ║
║    3. Negative layer  — style-aware negative prompt builder  ║
║                                                              ║
║  Styles: streetwear · luxury · casual · formal ·            ║
║          techwear · vintage · athleisure                     ║
╚══════════════════════════════════════════════════════════════╝

Quick Start
-----------
    from src.generation.prompts.prompt_templates import (
        generate_prompt,
        generate_negative_prompt,
        prompt_enhancer,
    )

    # Dict-driven API
    positive = generate_prompt({
        "category": "hoodie",
        "style":    "streetwear",
        "color":    "black",
    })
    # → "Professional fashion photography of a black oversized streetwear
    #    hoodie, urban street style, hypebeast fashion, editorial photography..."

    negative = generate_negative_prompt("streetwear")

    # Enhance a raw prompt string
    enhanced = prompt_enhancer(
        "a red hoodie",
        style="streetwear",
        boost_quality=True,
    )

Extensibility
-------------
    # Register a custom style template at runtime
    from src.generation.prompts.prompt_templates import register_template, FashionTemplate
    register_template(FashionTemplate(
        style="y2k",
        display_name="Y2K / 2000s Revival",
        ...
    ))
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("prompt_templates")  # type: ignore[assignment]


# ── Re-use shared negative prompt library ────────────────────────────────────
from src.generation.prompts.negative_prompts import (
    get_fashion_negative,
    get_style_negative,
    format_negative,
    QUALITY_NEGATIVES,
    ARTEFACT_NEGATIVES,
    NSFW_NEGATIVES,
    ANATOMY_NEGATIVES,
    LIGHTING_NEGATIVES,
    FASHION_NEGATIVES,
    BACKGROUND_NEGATIVES,
    STYLE_NEGATIVES,
)


# =============================================================================
# ── Constants
# =============================================================================

# Default quality boosters appended to every positive prompt
_GLOBAL_QUALITY_BOOSTERS: List[str] = [
    "professional fashion photography",
    "ultra detailed",
    "sharp focus",
    "8k resolution",
    "photorealistic",
    "high resolution",
]

# Photography / lighting modifiers per shooting context
_PHOTO_STYLES: Dict[str, List[str]] = {
    "studio":   ["clean studio background", "professional studio lighting", "white backdrop"],
    "editorial":["fashion editorial", "high fashion magazine", "dramatic lighting"],
    "outdoor":  ["outdoor natural lighting", "lifestyle photography", "golden hour"],
    "urban":    ["urban environment", "city street background", "natural light"],
    "runway":   ["runway show", "fashion week", "catwalk photography"],
    "lookbook": ["lookbook style", "fashion lookbook", "model shoot"],
}

# Gender modifiers
_GENDER_PHRASES: Dict[str, str] = {
    "women":  "a woman wearing",
    "men":    "a man wearing",
    "unisex": "a model wearing",
    "girl":   "a young woman wearing",
    "boy":    "a young man wearing",
}

# Fit / silhouette modifiers by category
_CATEGORY_FIT_HINTS: Dict[str, str] = {
    "hoodie":      "oversized",
    "t-shirt":     "fitted",
    "t_shirt":     "fitted",
    "jacket":      "structured",
    "blazer":      "tailored",
    "pants":       "well-cut",
    "jeans":       "slim-fit",
    "dress":       "flowing",
    "skirt":       "A-line",
    "coat":        "longline",
    "shorts":      "casual",
    "sweatshirt":  "relaxed",
    "sneakers":    "clean",
    "boots":       "ankle-height",
    "suit":        "well-tailored",
    "tracksuit":   "matching",
    "jumpsuit":    "one-piece",
    "vest":        "slim",
    "cardigan":    "cosy",
    "bomber":      "oversized",
    "parka":       "heavy-duty",
    "polo":        "slim-fit",
    "turtleneck":  "ribbed",
    "leggings":    "high-waist",
    "joggers":     "tapered",
    "windbreaker": "lightweight",
    "puffer":      "quilted",
}

# Fabric descriptors per style
_STYLE_FABRICS: Dict[str, List[str]] = {
    "streetwear":  ["cotton", "fleece", "denim", "nylon"],
    "luxury":      ["silk", "cashmere", "wool", "satin", "velvet"],
    "casual":      ["cotton", "linen", "jersey", "denim"],
    "formal":      ["wool", "tweed", "broadcloth", "crepe"],
    "techwear":    ["ripstop nylon", "Gore-Tex", "polyester", "technical fabric"],
    "vintage":     ["tweed", "corduroy", "denim", "linen", "wool blend"],
    "athleisure":  ["spandex", "moisture-wicking", "mesh", "lycra", "performance fabric"],
}

# CLIP token budget (approximate)
_MAX_TOKENS: int = 77


# =============================================================================
# ── FashionTemplate Dataclass
# =============================================================================

@dataclass
class FashionTemplate:
    """
    Complete prompt template definition for a single fashion style.

    Attributes
    ----------
    style : str
        Unique style key (e.g. ``"streetwear"``).
    display_name : str
        Human-readable label (e.g. ``"Streetwear / Urban"``).
    sentence_template : str
        A Python format string for the core sentence.
        Available variables: ``{fit}``, ``{color}``, ``{category}``,
        ``{style_adj}``, ``{gender_phrase}``, ``{fabric}``.
    style_adjectives : list of str
        Style-specific adjective pool used to fill ``{style_adj}``.
    positive_tags : list of str
        Style tags always appended to the positive prompt.
    quality_tags : list of str
        Photography / quality tags specific to this style.
    negative_extra : list of str
        Additional style-specific negative tags (stacked on top of base).
    photo_style : str
        Default photography context key (see ``_PHOTO_STYLES``).
    recommended_steps : int
    recommended_guidance : float
    description : str
        One-line style description.
    """
    style:              str
    display_name:       str
    sentence_template:  str
    style_adjectives:   List[str]  = field(default_factory=list)
    positive_tags:      List[str]  = field(default_factory=list)
    quality_tags:       List[str]  = field(default_factory=list)
    negative_extra:     List[str]  = field(default_factory=list)
    photo_style:        str        = "studio"
    recommended_steps:  int        = 30
    recommended_guidance: float    = 7.5
    description:        str        = ""

    # ── Convenience ───────────────────────────────────────────────────────

    def primary_adjective(self) -> str:
        """Return the first style adjective (or style name as fallback)."""
        return self.style_adjectives[0] if self.style_adjectives else self.style

    def all_positive_tags(self) -> List[str]:
        """Merge style tags + quality tags (deduped)."""
        seen: set = set()
        out: List[str] = []
        for t in (self.positive_tags + self.quality_tags):
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    def render_sentence(
        self,
        category:      str = "",
        color:         str = "",
        gender_phrase: str = "a model wearing",
        fabric:        str = "",
        fit:           str = "",
    ) -> str:
        """
        Render the sentence template with provided values.
        Missing values fall back to sensible defaults.
        """
        style_adj = self.primary_adjective()
        _fit      = fit or _CATEGORY_FIT_HINTS.get(category.lower().replace(" ", "_"), "")
        _fabric   = fabric or ""

        try:
            return self.sentence_template.format(
                category      = category or "garment",
                color         = color     or "",
                style_adj     = style_adj,
                gender_phrase = gender_phrase,
                fabric        = _fabric,
                fit           = _fit,
                style         = self.style,
            ).strip()
        except KeyError as exc:
            logger.warning("Template key error {!r} — using fallback", exc)
            return (
                f"Professional fashion photography of a "
                f"{color} {_fit} {self.style} {category}".strip()
            )


# =============================================================================
# ── Template Registry — 7 Core Styles
# =============================================================================

_TEMPLATES: Dict[str, FashionTemplate] = {

    # ── 1. Streetwear ─────────────────────────────────────────────────────
    "streetwear": FashionTemplate(
        style         = "streetwear",
        display_name  = "Streetwear / Urban",
        description   = "Bold urban street style with oversized fits, graphics, and sneaker culture.",
        sentence_template = (
            "Professional fashion photography of {gender_phrase} "
            "a {color} {fit} {style_adj} {category}, "
            "urban street style"
        ),
        style_adjectives = [
            "streetwear", "urban", "hypebeast", "skate-inspired", "drop-culture",
        ],
        positive_tags = [
            "urban streetwear fashion",
            "oversized silhouette",
            "graphic print",
            "sneaker culture",
            "hypebeast aesthetic",
            "bold color blocking",
            "graffiti-inspired",
            "skate culture vibes",
        ],
        quality_tags  = [
            "editorial street photography",
            "urban background",
            "natural light",
            "candid pose",
            "full-body shot",
        ],
        negative_extra = [
            "formal wear", "business attire", "suit and tie",
            "evening gown", "tailored blazer",
        ],
        photo_style       = "urban",
        recommended_steps = 30,
        recommended_guidance = 7.0,
    ),

    # ── 2. Luxury ─────────────────────────────────────────────────────────
    "luxury": FashionTemplate(
        style         = "luxury",
        display_name  = "Luxury / High-End",
        description   = "Premium designer fashion with opulent materials and couture craftsmanship.",
        sentence_template = (
            "Luxury fashion editorial of {gender_phrase} "
            "an exquisite {color} {fit} {style_adj} {category}, "
            "{fabric} fabric"
        ),
        style_adjectives = [
            "haute couture", "designer", "luxury", "opulent", "premium",
        ],
        positive_tags = [
            "luxury fashion",
            "high-end designer wear",
            "premium materials",
            "couture craftsmanship",
            "fashion week quality",
            "avant-garde tailoring",
            "opulent details",
            "exquisite embellishment",
        ],
        quality_tags  = [
            "luxury fashion editorial",
            "Vogue magazine cover",
            "dramatic studio lighting",
            "8k ultra detail",
            "impeccable styling",
        ],
        negative_extra = [
            "fast fashion", "cheap fabric", "plastic accessories",
            "mass market", "synthetic material",
        ],
        photo_style       = "editorial",
        recommended_steps = 40,
        recommended_guidance = 8.5,
    ),

    # ── 3. Casual ─────────────────────────────────────────────────────────
    "casual": FashionTemplate(
        style         = "casual",
        display_name  = "Casual / Everyday",
        description   = "Relaxed, comfortable everyday fashion for any occasion.",
        sentence_template = (
            "Lifestyle fashion photography of {gender_phrase} "
            "a {color} {fit} {style_adj} {category}"
        ),
        style_adjectives = [
            "casual", "relaxed-fit", "everyday", "laid-back", "minimalist",
        ],
        positive_tags = [
            "casual everyday fashion",
            "relaxed comfortable style",
            "minimalist aesthetic",
            "effortless look",
            "clean and simple",
            "wardrobe staple",
            "versatile piece",
        ],
        quality_tags  = [
            "lifestyle photography",
            "natural lighting",
            "outdoor setting",
            "authentic candid style",
        ],
        negative_extra = [
            "evening wear", "formal gown", "tuxedo",
            "heavy embellishment", "ball gown",
        ],
        photo_style       = "outdoor",
        recommended_steps = 25,
        recommended_guidance = 7.0,
    ),

    # ── 4. Formal ─────────────────────────────────────────────────────────
    "formal": FashionTemplate(
        style         = "formal",
        display_name  = "Formal / Business",
        description   = "Elegant, tailored clothing for professional and formal occasions.",
        sentence_template = (
            "Professional fashion photography of {gender_phrase} "
            "an elegant {color} {fit} {style_adj} {category}"
        ),
        style_adjectives = [
            "formal", "tailored", "sophisticated", "polished", "refined",
        ],
        positive_tags = [
            "elegant formal wear",
            "perfectly tailored silhouette",
            "sophisticated style",
            "professional attire",
            "polished look",
            "high-quality workmanship",
            "classic formal elegance",
        ],
        quality_tags  = [
            "clean white studio background",
            "professional studio lighting",
            "sharp tailoring details",
            "three-quarter shot",
        ],
        negative_extra = [
            "casual", "sporty", "ripped", "distressed",
            "streetwear", "sneakers", "athletic wear",
        ],
        photo_style       = "studio",
        recommended_steps = 35,
        recommended_guidance = 8.0,
    ),

    # ── 5. Techwear ───────────────────────────────────────────────────────
    "techwear": FashionTemplate(
        style         = "techwear",
        display_name  = "Techwear / Functional Fashion",
        description   = "High-performance technical garments with futuristic aesthetic and utility.",
        sentence_template = (
            "Futuristic fashion photography of {gender_phrase} "
            "a {color} {fit} {style_adj} {category}, "
            "{fabric}"
        ),
        style_adjectives = [
            "techwear", "tactical", "futuristic", "utilitarian", "cyberpunk-inspired",
        ],
        positive_tags = [
            "techwear aesthetic",
            "functional fashion",
            "tactical utility pockets",
            "modular design",
            "waterproof construction",
            "futuristic silhouette",
            "cyberpunk fashion",
            "dystopian aesthetic",
            "urban warrior style",
            "technical performance wear",
        ],
        quality_tags  = [
            "dark futuristic editorial",
            "neon-lit urban background",
            "dramatic shadows",
            "cinematic composition",
        ],
        negative_extra = [
            "traditional", "conventional", "casual basics",
            "pastel colors", "floral print", "vintage",
        ],
        photo_style       = "editorial",
        recommended_steps = 35,
        recommended_guidance = 8.0,
    ),

    # ── 6. Vintage ────────────────────────────────────────────────────────
    "vintage": FashionTemplate(
        style         = "vintage",
        display_name  = "Vintage / Retro",
        description   = "Timeless retro-inspired fashion drawing from past decades.",
        sentence_template = (
            "Retro fashion photography of {gender_phrase} "
            "a {color} {fit} {style_adj} {category}, "
            "vintage aesthetic"
        ),
        style_adjectives = [
            "vintage", "retro", "classic", "timeless", "old-school",
        ],
        positive_tags = [
            "vintage fashion",
            "retro aesthetic",
            "classic silhouette",
            "nostalgic style",
            "thrift store chic",
            "70s inspired",
            "80s revival",
            "timeless elegance",
            "heritage fashion",
            "old Hollywood glamour",
        ],
        quality_tags  = [
            "warm film photography aesthetic",
            "golden hour lighting",
            "nostalgic colour grading",
            "retro editorial style",
        ],
        negative_extra = [
            "modern", "contemporary", "futuristic",
            "neon colors", "techwear", "fast fashion",
        ],
        photo_style       = "outdoor",
        recommended_steps = 32,
        recommended_guidance = 7.5,
    ),

    # ── 7. Athleisure ─────────────────────────────────────────────────────
    "athleisure": FashionTemplate(
        style         = "athleisure",
        display_name  = "Athleisure / Sport-Luxe",
        description   = "Athletic meets fashion — performance wear with a stylish, luxurious edge.",
        sentence_template = (
            "Dynamic fashion photography of {gender_phrase} "
            "a {color} {fit} {style_adj} {category}, "
            "sporty chic"
        ),
        style_adjectives = [
            "athleisure", "sport-luxe", "activewear", "performance", "athletic",
        ],
        positive_tags = [
            "athleisure fashion",
            "sport-luxe aesthetic",
            "athletic silhouette",
            "performance-meets-style",
            "gym-to-street style",
            "active lifestyle",
            "fitness fashion",
            "elevated sportswear",
        ],
        quality_tags  = [
            "clean studio background",
            "dynamic athletic pose",
            "bright studio lighting",
            "full-body active shot",
        ],
        negative_extra = [
            "formal wear", "office attire", "evening gown",
            "heels", "suit and tie",
        ],
        photo_style       = "studio",
        recommended_steps = 28,
        recommended_guidance = 7.5,
    ),
}


# =============================================================================
# ── Template Registry API
# =============================================================================

def register_template(template: FashionTemplate) -> None:
    """
    Register a custom ``FashionTemplate`` in the global registry.

    Parameters
    ----------
    template : FashionTemplate
        The template to register. Overwrites any existing template with
        the same style key.

    Example
    -------
        register_template(FashionTemplate(
            style="y2k",
            display_name="Y2K / 2000s Revival",
            sentence_template="...",
        ))
    """
    key = template.style.lower().replace(" ", "_").replace("-", "_")
    _TEMPLATES[key] = template
    logger.debug("Registered template for style {!r}", key)


def get_template(style: str) -> FashionTemplate:
    """
    Retrieve a ``FashionTemplate`` by style name.

    Parameters
    ----------
    style : str  e.g. ``"streetwear"``, ``"luxury"``

    Raises
    ------
    KeyError  if the style is not registered.
    """
    key = style.lower().replace(" ", "_").replace("-", "_")
    if key not in _TEMPLATES:
        available = list(_TEMPLATES.keys())
        raise KeyError(
            f"Unknown style {style!r}. Available: {available}. "
            f"Register custom styles with register_template()."
        )
    return _TEMPLATES[key]


def list_templates() -> List[str]:
    """Return all registered style keys."""
    return list(_TEMPLATES.keys())


def list_template_details() -> List[Dict[str, Any]]:
    """Return summary dicts for all templates (useful for UIs)."""
    return [
        {
            "style":       t.style,
            "display_name":t.display_name,
            "description": t.description,
            "photo_style": t.photo_style,
            "recommended_steps":    t.recommended_steps,
            "recommended_guidance": t.recommended_guidance,
        }
        for t in _TEMPLATES.values()
    ]


# =============================================================================
# ── Core Public Functions
# =============================================================================

def generate_prompt(
    item: Dict[str, Any],
    *,
    photo_style:      Optional[str]       = None,
    extra_tags:       Optional[List[str]] = None,
    boost_quality:    bool                = True,
    include_fabric:   bool                = True,
    gender:           Optional[str]       = None,
    max_tokens:       int                 = _MAX_TOKENS,
) -> str:
    """
    Generate a complete positive prompt from a fashion item dict.

    Parameters
    ----------
    item : dict
        Fashion item descriptor. Recognised keys:

        =====================  ================================================
        ``style``              Style name e.g. ``"streetwear"``
        ``category``           Garment type e.g. ``"hoodie"``
        ``color`` / ``colors`` Single colour string or list
        ``gender``             ``"men"`` | ``"women"`` | ``"unisex"``
        ``description``        Free-text override (used as sentence prefix)
        ``fabric``             Fabric override e.g. ``"cashmere"``
        ``fit``                Fit override e.g. ``"oversized"``
        ``occasion``           Occasion tag appended verbatim
        ``season``             Season tag appended verbatim
        ``extra_tags``         List of extra positive tags
        =====================  ================================================

    photo_style : str, optional
        Override photography context: ``"studio"``, ``"editorial"``,
        ``"outdoor"``, ``"urban"``, ``"runway"``, ``"lookbook"``.
    extra_tags : list of str, optional
        Extra tags appended after the template tags.
    boost_quality : bool
        Append global quality boosters (default True).
    include_fabric : bool
        Inject a fabric hint from the style's fabric palette (default True).
    gender : str, optional
        Override the item-level gender field.
    max_tokens : int
        CLIP token budget. Prompt is trimmed if over-budget.

    Returns
    -------
    str — Complete positive prompt ready for the SDXL pipeline.

    Examples
    --------
    >>> generate_prompt({"category": "hoodie", "style": "streetwear", "color": "black"})
    'Professional fashion photography of a model wearing a black oversized
     streetwear hoodie, urban street style, ...'

    >>> generate_prompt({
    ...     "category": "evening gown",
    ...     "style":    "luxury",
    ...     "color":    "deep emerald",
    ...     "gender":   "women",
    ... })
    'Luxury fashion editorial of a woman wearing an exquisite deep emerald
     flowing haute couture evening gown, silk fabric, ...'
    """
    # ── Extract item fields ───────────────────────────────────────────────
    style_key  = str(item.get("style", "casual")).strip().lower()
    category   = str(item.get("category", "")).strip()
    raw_color  = item.get("color") or item.get("colors") or ""
    color      = (raw_color[0] if isinstance(raw_color, list) else raw_color).strip()
    description= str(item.get("description", "")).strip()
    fabric_in  = str(item.get("fabric", "")).strip()
    fit_in     = str(item.get("fit", "")).strip()
    occasion   = str(item.get("occasion", "")).strip()
    season     = str(item.get("season", "")).strip()
    gender_key = (gender or str(item.get("gender", ""))).strip().lower()
    item_extras= item.get("extra_tags", [])

    # ── Resolve template ──────────────────────────────────────────────────
    try:
        tmpl = get_template(style_key)
    except KeyError:
        logger.warning(
            "Style {!r} not found in templates — falling back to 'casual'", style_key
        )
        tmpl = _TEMPLATES["casual"]

    # ── Resolve gender phrase ─────────────────────────────────────────────
    gender_phrase = _GENDER_PHRASES.get(gender_key, "a model wearing")

    # ── Resolve fabric ────────────────────────────────────────────────────
    fabric = fabric_in
    if not fabric and include_fabric:
        style_fabrics = _STYLE_FABRICS.get(style_key, [])
        fabric = style_fabrics[0] if style_fabrics else ""

    # ── Resolve fit ───────────────────────────────────────────────────────
    fit = fit_in or _CATEGORY_FIT_HINTS.get(
        category.lower().replace(" ", "_").replace("-", "_"), ""
    )

    # ── Build sentence (core) ─────────────────────────────────────────────
    if description:
        # User provided a free-text description → use it as the sentence
        sentence = description[:300]
    else:
        sentence = tmpl.render_sentence(
            category      = category,
            color         = color,
            gender_phrase = gender_phrase,
            fabric        = fabric,
            fit           = fit,
        )

    # ── Collect all positive parts ────────────────────────────────────────
    parts: List[str] = [sentence]
    parts.extend(tmpl.positive_tags)

    # Photography style
    ps_key    = photo_style or tmpl.photo_style
    ps_tags   = _PHOTO_STYLES.get(ps_key, _PHOTO_STYLES["studio"])
    parts.extend(ps_tags)

    # Occasion / season
    if occasion:
        parts.append(f"{occasion} occasion")
    if season:
        parts.append(f"{season} season")

    # Extra tags from item dict
    if item_extras:
        parts.extend(item_extras if isinstance(item_extras, list) else [item_extras])

    # Extra tags from caller
    if extra_tags:
        parts.extend(extra_tags)

    # Quality boosters
    if boost_quality:
        parts.extend(tmpl.quality_tags)
        parts.extend(_GLOBAL_QUALITY_BOOSTERS)

    # ── Deduplicate ───────────────────────────────────────────────────────
    parts = _deduplicate(parts)

    # ── Join & trim to token budget ───────────────────────────────────────
    prompt = _join_and_trim(parts, max_tokens)

    logger.debug(
        "generate_prompt | style={} | category={} | tokens≈{}",
        style_key, category, _estimate_tokens(prompt),
    )
    return prompt


def generate_negative_prompt(
    style:               Optional[str]       = None,
    *,
    include_nsfw:        bool                = True,
    include_anatomy:     bool                = True,
    include_artefacts:   bool                = True,
    include_lighting:    bool                = True,
    include_background:  bool                = True,
    extra_negatives:     Optional[List[str]] = None,
    item:                Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a comprehensive negative prompt for a given fashion style.

    Stacks: base quality + anatomy + artefact + fashion + style-specific
    negatives into a single deduplicated string.

    Parameters
    ----------
    style : str, optional
        Style name. If provided, adds style-specific negatives
        (e.g. ``"streetwear"`` adds ``"formal wear"``, ``"suit and tie"``).
    include_nsfw : bool
        Append NSFW suppression tags (default True).
    include_anatomy : bool
        Append anatomy / deformation tags (default True).
    include_artefacts : bool
        Append watermark / text / logo suppression tags (default True).
    include_lighting : bool
        Append bad-lighting suppression tags (default True).
    include_background : bool
        Append cluttered background tags (default True).
    extra_negatives : list of str, optional
        Additional caller-supplied negative tags.
    item : dict, optional
        If provided, the ``style`` field is read from ``item["style"]``
        unless the ``style`` argument is explicitly set.

    Returns
    -------
    str — Complete negative prompt string.

    Examples
    --------
    >>> generate_negative_prompt("luxury")
    'blurry, out of focus, low quality, ..., fast fashion, cheap fabric, ...'

    >>> generate_negative_prompt(item={"style": "streetwear", "category": "hoodie"})
    'blurry, ..., formal wear, business attire, ...'
    """
    # Resolve style
    effective_style = style
    if effective_style is None and item:
        effective_style = str(item.get("style", "")).strip().lower() or None

    # ── Build negative layers ─────────────────────────────────────────────
    neg: List[str] = []
    neg.extend(QUALITY_NEGATIVES)

    if include_anatomy:
        neg.extend(ANATOMY_NEGATIVES)

    if include_artefacts:
        neg.extend(ARTEFACT_NEGATIVES)

    neg.extend(FASHION_NEGATIVES)

    if include_lighting:
        neg.extend(LIGHTING_NEGATIVES)

    if include_background:
        neg.extend(BACKGROUND_NEGATIVES)

    if include_nsfw:
        neg.extend(NSFW_NEGATIVES)

    # ── Style-specific negatives ──────────────────────────────────────────
    if effective_style:
        # From the template registry
        try:
            tmpl = get_template(effective_style)
            neg.extend(tmpl.negative_extra)
        except KeyError:
            pass

        # From the shared STYLE_NEGATIVES dict
        style_specific = STYLE_NEGATIVES.get(effective_style.lower(), [])
        neg.extend(style_specific)

    # ── Extra caller negatives ────────────────────────────────────────────
    if extra_negatives:
        neg.extend(extra_negatives)

    result = format_negative(neg)
    logger.debug(
        "generate_negative_prompt | style={} | tags={}",
        effective_style, len(result.split(",")),
    )
    return result


def prompt_enhancer(
    prompt:           str,
    style:            Optional[str]       = None,
    *,
    photo_style:      Optional[str]       = None,
    boost_quality:    bool                = True,
    add_style_tags:   bool                = True,
    extra_tags:       Optional[List[str]] = None,
    max_tokens:       int                 = _MAX_TOKENS,
    preserve_order:   bool                = True,
) -> str:
    """
    Enhance a raw free-text prompt with style tags, quality boosters,
    and photography modifiers.

    This is the "turbocharge" function — take any existing prompt string
    and make it production-ready for SDXL without restructuring it.

    Parameters
    ----------
    prompt : str
        Raw prompt string (e.g. ``"a red hoodie"``).
    style : str, optional
        Fashion style to layer on top (e.g. ``"streetwear"``).
    photo_style : str, optional
        Photography context override.
    boost_quality : bool
        Append global quality boosters (default True).
    add_style_tags : bool
        Inject style-specific positive tags (default True).
    extra_tags : list of str, optional
        Additional tags to append.
    max_tokens : int
        CLIP token budget.
    preserve_order : bool
        Keep the raw prompt at position [0] (default True).

    Returns
    -------
    str — Enhanced prompt string.

    Examples
    --------
    >>> prompt_enhancer("a red hoodie", style="streetwear")
    'a red hoodie, urban streetwear fashion, oversized silhouette, ...'

    >>> prompt_enhancer("elegant navy blazer", style="formal", boost_quality=True)
    'elegant navy blazer, elegant formal wear, tailored silhouette, ..., 8k resolution'
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    parts: List[str] = [prompt.strip()] if preserve_order else []

    # ── Style layer ───────────────────────────────────────────────────────
    if style:
        try:
            tmpl = get_template(style)

            if add_style_tags:
                parts.extend(tmpl.positive_tags)

            # Photography style
            ps_key  = photo_style or tmpl.photo_style
            ps_tags = _PHOTO_STYLES.get(ps_key, _PHOTO_STYLES["studio"])
            parts.extend(ps_tags)

            if boost_quality:
                parts.extend(tmpl.quality_tags)

        except KeyError:
            logger.warning("Unknown style {!r} in prompt_enhancer — style tags skipped", style)

    # ── Fallback photography style (no style given) ───────────────────────
    elif photo_style:
        ps_tags = _PHOTO_STYLES.get(photo_style, _PHOTO_STYLES["studio"])
        parts.extend(ps_tags)

    # ── Global quality boosters ───────────────────────────────────────────
    if boost_quality:
        parts.extend(_GLOBAL_QUALITY_BOOSTERS)

    # ── Extra tags ────────────────────────────────────────────────────────
    if extra_tags:
        parts.extend(extra_tags)

    if not preserve_order:
        parts.insert(0, prompt.strip())

    parts  = _deduplicate(parts)
    result = _join_and_trim(parts, max_tokens)

    logger.debug(
        "prompt_enhancer | style={} | in_tokens≈{} | out_tokens≈{}",
        style,
        _estimate_tokens(prompt),
        _estimate_tokens(result),
    )
    return result


# =============================================================================
# ── Batch / Convenience Helpers
# =============================================================================

def generate_prompt_pair(
    item: Dict[str, Any],
    **kwargs,
) -> Tuple[str, str]:
    """
    Generate (positive, negative) prompt pair from a single item dict.

    Parameters
    ----------
    item : dict  (same as ``generate_prompt``)
    **kwargs     Forwarded to ``generate_prompt``.

    Returns
    -------
    (positive_prompt, negative_prompt)

    Example
    -------
    >>> pos, neg = generate_prompt_pair({"style": "luxury", "category": "dress", "color": "gold"})
    """
    positive = generate_prompt(item, **kwargs)
    negative = generate_negative_prompt(item=item)
    return positive, negative


def generate_batch_prompts(
    items:          Sequence[Dict[str, Any]],
    shared_kwargs:  Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """
    Generate prompt pairs for a list of fashion items.

    Parameters
    ----------
    items : list of dict
    shared_kwargs : dict, optional  — forwarded to every ``generate_prompt`` call.

    Returns
    -------
    List of (positive, negative) tuples (same length as items).

    Example
    -------
    >>> pairs = generate_batch_prompts([
    ...     {"style": "streetwear", "category": "hoodie", "color": "black"},
    ...     {"style": "luxury",     "category": "dress",  "color": "gold"},
    ... ])
    """
    kwargs  = shared_kwargs or {}
    results = []
    for idx, item in enumerate(items):
        try:
            pair = generate_prompt_pair(item, **kwargs)
        except Exception as exc:
            logger.error("generate_batch_prompts item[{}] failed: {}", idx, exc)
            pair = ("", "")
        results.append(pair)
    return results


def enhance_batch(
    prompts: Sequence[str],
    style:   Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Apply ``prompt_enhancer`` to a list of raw prompts.

    Parameters
    ----------
    prompts : list of str
    style : str, optional — applied to all prompts.
    **kwargs  — forwarded to ``prompt_enhancer``.

    Returns
    -------
    list of str — enhanced prompts (same order as input).
    """
    return [prompt_enhancer(p, style=style, **kwargs) for p in prompts]


# =============================================================================
# ── Inspection Utilities
# =============================================================================

def explain_prompt(
    item: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a prompt AND return a breakdown of every contributing layer.

    Useful for debugging and understanding prompt composition.

    Returns
    -------
    dict with keys:
        - ``full_prompt``  — final generated prompt string
        - ``negative``     — negative prompt string
        - ``layers``       — dict of each contributing segment
        - ``token_estimate``— estimated CLIP token count
        - ``style``        — resolved style name
        - ``template``     — FashionTemplate details
    """
    style_key = str(item.get("style", "casual")).lower()
    try:
        tmpl = get_template(style_key)
    except KeyError:
        tmpl = _TEMPLATES["casual"]

    category  = str(item.get("category", ""))
    color     = item.get("color", "")
    if isinstance(color, list):
        color = color[0] if color else ""
    gender_key= str(item.get("gender", "")).lower()
    ps_key    = kwargs.get("photo_style") or tmpl.photo_style

    layers = {
        "sentence":      tmpl.render_sentence(
            category      = category,
            color         = str(color),
            gender_phrase = _GENDER_PHRASES.get(gender_key, "a model wearing"),
        ),
        "style_tags":    tmpl.positive_tags,
        "photo_tags":    _PHOTO_STYLES.get(ps_key, []),
        "quality_tags":  tmpl.quality_tags,
        "global_quality":_GLOBAL_QUALITY_BOOSTERS,
    }

    full_prompt = generate_prompt(item, **kwargs)
    negative    = generate_negative_prompt(style=style_key)

    return {
        "full_prompt":      full_prompt,
        "negative":         negative,
        "layers":           layers,
        "token_estimate":   _estimate_tokens(full_prompt),
        "style":            tmpl.style,
        "template": {
            "display_name":       tmpl.display_name,
            "photo_style":        tmpl.photo_style,
            "recommended_steps":  tmpl.recommended_steps,
            "recommended_guidance": tmpl.recommended_guidance,
        },
    }


# =============================================================================
# ── Private Helpers
# =============================================================================

def _deduplicate(tags: List[str]) -> List[str]:
    """Remove duplicate tags (case-insensitive) preserving first-seen order."""
    seen: set = set()
    result: List[str] = []
    for tag in tags:
        tag_clean = tag.strip()
        if not tag_clean:
            continue
        key = re.sub(r"\s+", " ", tag_clean.lower())
        if key not in seen:
            seen.add(key)
            result.append(tag_clean)
    return result


def _estimate_tokens(text: str) -> int:
    """Rough CLIP token estimate: words × 1.3."""
    return int(len(text.split()) * 1.3)


def _join_and_trim(parts: List[str], max_tokens: int) -> str:
    """Join parts with ', ' and trim from the tail if over token budget."""
    joined = ", ".join(parts)
    if _estimate_tokens(joined) <= max_tokens:
        return joined

    kept = list(parts)
    while len(kept) > 1 and _estimate_tokens(", ".join(kept)) > max_tokens:
        kept.pop()

    trimmed = ", ".join(kept)
    logger.debug(
        "Prompt trimmed: {} → {} parts to fit {} token budget",
        len(parts), len(kept), max_tokens,
    )
    return trimmed
