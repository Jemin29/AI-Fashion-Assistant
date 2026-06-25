"""
week2/prompts/negative_prompts.py
===================================
Curated negative prompt library for SDXL fashion image generation.

Provides pre-built, categorised negative prompt lists that can be
combined to suppress common SDXL failure modes.
"""

from __future__ import annotations

from typing import Dict, List


# =============================================================================
# ── Base Negative Prompts (always recommended)
# =============================================================================

QUALITY_NEGATIVES: List[str] = [
    "blurry",
    "out of focus",
    "low quality",
    "pixelated",
    "jpeg artifacts",
    "compression artifacts",
    "grainy",
    "noise",
    "low resolution",
    "worst quality",
    "bad quality",
]

ANATOMY_NEGATIVES: List[str] = [
    "deformed",
    "distorted",
    "disfigured",
    "bad anatomy",
    "bad proportions",
    "extra limbs",
    "missing limbs",
    "extra fingers",
    "missing fingers",
    "fused fingers",
    "mutated hands",
    "malformed hands",
    "cloned face",
    "long neck",
    "cropped head",
]

ARTEFACT_NEGATIVES: List[str] = [
    "watermark",
    "text",
    "signature",
    "logo",
    "username",
    "artist name",
    "copyright",
    "frame",
    "border",
    "duplicate",
    "tiling",
]

COMPOSITION_NEGATIVES: List[str] = [
    "cropped",
    "cut off",
    "out of frame",
    "poorly framed",
    "bad framing",
    "awkward angle",
]

# =============================================================================
# ── Fashion-Specific Negatives
# =============================================================================

FASHION_NEGATIVES: List[str] = [
    "bad clothing",
    "wrinkled fabric",
    "dirty clothes",
    "stained",
    "torn fabric",
    "mismatched colors",
    "poor tailoring",
    "sloppy fit",
    "ill-fitting",
    "costume",
    "halloween costume",
]

BACKGROUND_NEGATIVES: List[str] = [
    "cluttered background",
    "busy background",
    "distracting background",
    "dark background",
    "ugly background",
]

LIGHTING_NEGATIVES: List[str] = [
    "harsh lighting",
    "overexposed",
    "underexposed",
    "bad lighting",
    "dark",
    "shadows on face",
    "red eye",
]

NSFW_NEGATIVES: List[str] = [
    "nsfw",
    "nude",
    "explicit",
    "suggestive",
    "inappropriate",
]

# =============================================================================
# ── Style-Specific Negatives
# =============================================================================

STYLE_NEGATIVES: Dict[str, List[str]] = {
    "formal": ["casual", "sporty", "streetwear", "ripped jeans", "sneakers"],
    "streetwear": ["formal", "business attire", "suit and tie"],
    "casual": ["formal wear", "evening gown", "tuxedo"],
    "athletic": ["formal", "office wear", "heels"],
    "luxury": ["fast fashion", "cheap fabric", "plastic accessories"],
    "minimalist": ["busy pattern", "heavy embellishment", "maximalist", "loud colors"],
    "bohemian": ["structured", "corporate", "stiff fabric"],
}

# =============================================================================
# ── Pre-built Negative Prompt Bundles
# =============================================================================

def get_base_negative() -> List[str]:
    """Return the standard base negative prompt list."""
    return (
        QUALITY_NEGATIVES +
        ANATOMY_NEGATIVES +
        ARTEFACT_NEGATIVES +
        COMPOSITION_NEGATIVES
    )


def get_fashion_negative() -> List[str]:
    """Return fashion-specific negative prompt list."""
    return (
        get_base_negative() +
        FASHION_NEGATIVES +
        LIGHTING_NEGATIVES
    )


def get_full_negative(include_nsfw: bool = True) -> List[str]:
    """Return the most comprehensive negative prompt list."""
    neg = get_fashion_negative() + BACKGROUND_NEGATIVES
    if include_nsfw:
        neg += NSFW_NEGATIVES
    return neg


def get_style_negative(style: str) -> List[str]:
    """
    Get negative prompts for a specific fashion style.

    Parameters
    ----------
    style : str  e.g. "formal", "streetwear", "casual"

    Returns
    -------
    list of str — base negatives + style-specific negatives.
    """
    base = get_fashion_negative()
    style_specific = STYLE_NEGATIVES.get(style.lower(), [])
    return base + style_specific


def format_negative(tags: List[str], separator: str = ", ") -> str:
    """Join a list of negative tags into a single string for the pipeline."""
    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in tags:
        t_clean = t.strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            unique.append(t_clean)
    return separator.join(unique)
