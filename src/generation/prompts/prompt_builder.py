"""
week2/prompts/prompt_builder.py
=================================
Structured prompt assembly engine for SDXL fashion generation.

Architecture
------------
- PromptComponents: typed container for all prompt parts
- PromptBuilder: assembles components into final positive/negative strings
- Supports style presets, season/gender modifiers, quality boosters
- Token counting and truncation to stay within CLIP's 77-token limit

Usage
-----
    from src.generation.prompts.prompt_builder import PromptBuilder
    from src.utils.config_manager import get_config

    builder = PromptBuilder(config=get_config())
    result = builder.build(
        subject     = "a woman wearing a red silk dress",
        style       = "luxury",
        gender      = "women",
        season      = "summer",
        occasion    = "evening",
    )
    print(result.positive)   # Full positive prompt string
    print(result.negative)   # Full negative prompt string
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from src.generation.prompts.style_presets import get_preset, StylePreset
from src.generation.prompts.negative_prompts import (
    get_fashion_negative,
    get_style_negative,
    format_negative,
)


# =============================================================================
# ── Built Prompt Result
# =============================================================================

@dataclass
class BuiltPrompt:
    """
    Result of PromptBuilder.build().

    Attributes
    ----------
    positive : str  Final positive prompt for the diffusion pipeline.
    negative : str  Final negative prompt.
    token_count_estimate : int  Rough token count for positive prompt.
    components : dict  All individual prompt parts for debugging.
    style_name : str  Name of the applied style preset.
    was_truncated : bool  True if positive prompt was truncated.
    """
    positive:               str
    negative:               str
    token_count_estimate:   int         = 0
    components:             Dict        = field(default_factory=dict)
    style_name:             str         = ""
    was_truncated:          bool        = False

    def __repr__(self) -> str:
        trunc = " [TRUNCATED]" if self.was_truncated else ""
        return (
            f"BuiltPrompt(style={self.style_name!r} | "
            f"tokens≈{self.token_count_estimate}{trunc})"
        )


# =============================================================================
# ── Prompt Builder
# =============================================================================

class PromptBuilder:
    """
    Assembles structured fashion prompts for SDXL.

    Parameters
    ----------
    config : Week2Config
        Loaded Week 2 config (prompts section used).

    Example
    -------
        builder = PromptBuilder(config=get_config())
        result  = builder.build(
            subject="oversized black hoodie with embroidery",
            style="streetwear",
            gender="unisex",
        )
    """

    # Token-to-word approximation factor (CLIP averages ~1.3 tokens/word)
    _TOKENS_PER_WORD: float = 1.3

    def __init__(self, config=None) -> None:
        self._cfg = config
        # Pull from config if available, else use sensible defaults
        if config is not None:
            self._quality_boosters: List[str] = config.prompts.quality_boosters
            self._fashion_technical: List[str] = config.prompts.fashion_technical
            self._global_negative:  List[str] = config.prompts.global_negative
            self._max_tokens:        int       = config.prompts.structure.max_tokens
            self._separator:         str       = config.prompts.structure.separator
            self._gender_mods:       Dict      = config.prompts.gender_modifiers
            self._season_mods:       Dict      = config.prompts.season_modifiers
            self._category_templates:Dict      = config.prompts.category_templates
        else:
            self._quality_boosters = [
                "photorealistic", "8k resolution", "ultra detailed",
                "sharp focus", "professional photography",
            ]
            self._fashion_technical = [
                "fashion editorial", "fashion photography", "lookbook style",
            ]
            self._global_negative   = get_fashion_negative()
            self._max_tokens        = 77
            self._separator         = ", "
            self._gender_mods       = {}
            self._season_mods       = {}
            self._category_templates= {}

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        subject:            str,
        style:              Optional[str] = None,
        gender:             Optional[str] = None,
        season:             Optional[str] = None,
        occasion:           Optional[str] = None,
        category:           Optional[str] = None,
        extra_positive:     Optional[List[str]] = None,
        extra_negative:     Optional[List[str]] = None,
        include_quality:    bool = True,
        include_fashion_tech: bool = True,
    ) -> BuiltPrompt:
        """
        Build a complete positive and negative prompt from components.

        Parameters
        ----------
        subject : str
            Core description of the garment/model (e.g. "a woman in a red gown").
        style : str, optional
            Style preset name (e.g. "luxury", "streetwear").
        gender : str, optional
            men | women | unisex
        season : str, optional
            spring | summer | autumn | winter | all_season
        occasion : str, optional
            casual | formal | evening | sport | etc.
        category : str, optional
            Fashion category (used for template lookup).
        extra_positive : list of str, optional
            Additional positive tags.
        extra_negative : list of str, optional
            Additional negative tags.
        include_quality : bool
            Whether to append global quality boosters.
        include_fashion_tech : bool
            Whether to append fashion-technical tags.

        Returns
        -------
        BuiltPrompt
        """
        components: Dict[str, List[str]] = {}

        # ── 1. Subject (core) ─────────────────────────────────────────────
        positive_parts: List[str] = [subject.strip()]
        components["subject"] = [subject.strip()]

        # ── 2. Style preset ───────────────────────────────────────────────
        preset: Optional[StylePreset] = None
        style_name = ""
        if style:
            try:
                preset      = get_preset(style)
                style_name  = preset.name
                positive_parts.extend(preset.positive_tags)
                components["style_positive"] = preset.positive_tags
            except KeyError:
                logger.warning("Unknown style preset {!r} — skipped", style)

        # ── 3. Gender modifier ────────────────────────────────────────────
        if gender:
            gender_tags = self._gender_mods.get(gender.lower(), [])
            positive_parts.extend(gender_tags)
            components["gender"] = gender_tags

        # ── 4. Season modifier ────────────────────────────────────────────
        if season:
            season_tags = self._season_mods.get(season.lower(), [])
            positive_parts.extend(season_tags)
            components["season"] = season_tags

        # ── 5. Occasion tag ───────────────────────────────────────────────
        if occasion:
            occ_tag = f"{occasion} occasion"
            positive_parts.append(occ_tag)
            components["occasion"] = [occ_tag]

        # ── 6. Extra user tags ────────────────────────────────────────────
        if extra_positive:
            positive_parts.extend(extra_positive)
            components["extra_positive"] = extra_positive

        # ── 7. Fashion technical boosters ─────────────────────────────────
        if include_fashion_tech:
            positive_parts.extend(self._fashion_technical)
            components["fashion_technical"] = self._fashion_technical

        # ── 8. Quality boosters ───────────────────────────────────────────
        if include_quality:
            positive_parts.extend(self._quality_boosters)
            components["quality"] = self._quality_boosters

        # ── Style quality tags ────────────────────────────────────────────
        if preset and preset.quality_tags:
            positive_parts.extend(preset.quality_tags)
            components["style_quality"] = preset.quality_tags

        # ── Assemble positive ─────────────────────────────────────────────
        positive_raw    = self._deduplicate(positive_parts)
        positive_str, was_truncated = self._truncate(positive_raw)

        # ── Assemble negative ─────────────────────────────────────────────
        negative_parts = list(self._global_negative)
        if preset and preset.negative_tags:
            negative_parts.extend(preset.negative_tags)
        if extra_negative:
            negative_parts.extend(extra_negative)

        negative_str = format_negative(negative_parts, self._separator)

        token_est = self._estimate_tokens(positive_str)

        if was_truncated:
            logger.warning(
                "Positive prompt truncated to ≈{} tokens | original length: {} chars",
                self._max_tokens,
                len(self._separator.join(positive_raw)),
            )
        else:
            logger.debug(
                "Prompt built | style={} | tokens≈{}", style_name, token_est
            )

        return BuiltPrompt(
            positive             = positive_str,
            negative             = negative_str,
            token_count_estimate = token_est,
            components           = components,
            style_name           = style_name,
            was_truncated        = was_truncated,
        )

    def build_from_metadata(self, record: Dict[str, Any]) -> BuiltPrompt:
        """
        Build a prompt from a Week 1 fashion dataset record.

        Parameters
        ----------
        record : dict
            A unified fashion record with keys like ``category``,
            ``gender``, ``style``, ``season``, ``description``, ``color``.

        Returns
        -------
        BuiltPrompt
        """
        category    = record.get("category", "")
        description = record.get("description", "")
        gender_raw  = record.get("gender", "")
        style_raw   = record.get("style", "")
        season      = record.get("season", "all_season")
        colors      = record.get("color", record.get("colors", []))

        # Build subject from template or description
        if category and category in self._category_templates:
            color_str = colors[0] if colors else ""
            subject   = self._category_templates[category].format(
                gender      = gender_raw or "model",
                style       = style_raw or "stylish",
                color       = color_str,
                description = description[:120] if description else "",
            )
        elif description:
            subject = description[:200]
        else:
            subject = f"a {gender_raw} wearing {category} fashion"

        return self.build(
            subject = subject,
            style   = style_raw or None,
            gender  = gender_raw or None,
            season  = season,
            category= category,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _deduplicate(self, tags: List[str]) -> List[str]:
        """Remove duplicates while preserving insertion order."""
        seen: set = set()
        result: List[str] = []
        for t in tags:
            t_clean = t.strip()
            if t_clean and t_clean.lower() not in seen:
                seen.add(t_clean.lower())
                result.append(t_clean)
        return result

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (words × 1.3)."""
        words = len(text.split())
        return int(words * self._TOKENS_PER_WORD)

    def _truncate(self, tags: List[str]) -> tuple[str, bool]:
        """
        Join tags and truncate if the estimated token count exceeds max_tokens.

        Returns
        -------
        (truncated_string, was_truncated)
        """
        joined = self._separator.join(tags)
        if self._estimate_tokens(joined) <= self._max_tokens:
            return joined, False

        # Drop tags from the end until we fit
        kept = list(tags)
        while kept and self._estimate_tokens(self._separator.join(kept)) > self._max_tokens:
            kept.pop()

        return self._separator.join(kept), True
