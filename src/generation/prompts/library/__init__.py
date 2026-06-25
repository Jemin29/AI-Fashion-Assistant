"""
week2/prompts/library/__init__.py
===================================
Fashion Prompt Library — Python interface to the .txt prompt collections.

Each .txt file contains 50 high-quality prompts organised into sections:
  - Photorealistic
  - Runway / Editorial
  - E-commerce / Product
  - Diverse Categories
  - Modern Trends
  - Colour Stories

Styles available
----------------
  streetwear · luxury · casual · formal · techwear · vintage · athleisure

Quick Start
-----------
    from src.generation.prompts.library import PromptLibrary

    lib = PromptLibrary()

    # Get all 50 prompts for a style
    prompts = lib.get_prompts("streetwear")

    # Get a random prompt
    prompt = lib.random_prompt("luxury")

    # Search across all styles
    results = lib.search("black hoodie")

    # Get prompts by section
    ecom = lib.get_section("streetwear", "e-commerce")

    # Stats
    print(lib.stats())
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("prompt_library")  # type: ignore[assignment]


# ── Library directory (same folder as this file) ──────────────────────────────
_LIB_DIR = Path(__file__).resolve().parent

# ── Available style files ─────────────────────────────────────────────────────
AVAILABLE_STYLES: List[str] = [
    "streetwear",
    "luxury",
    "casual",
    "formal",
    "techwear",
    "vintage",
    "athleisure",
]


# =============================================================================
# ── PromptLibrary
# =============================================================================

class PromptLibrary:
    """
    Interface to the curated fashion prompt `.txt` collections.

    Each style file contains 50 prompts split into labelled sections.

    Parameters
    ----------
    library_dir : Path, optional
        Override the default library directory.

    Example
    -------
        lib = PromptLibrary()
        prompts = lib.get_prompts("streetwear")
        prompt  = lib.random_prompt("luxury")
        results = lib.search("black hoodie")
    """

    # Section headers recognised inside .txt files
    _SECTION_PATTERNS: Dict[str, str] = {
        "photorealistic": "PHOTOREALISTIC",
        "runway":         "RUNWAY",
        "editorial":      "RUNWAY",   # alias
        "ecommerce":      "E-COMMERCE",
        "e-commerce":     "E-COMMERCE",
        "product":        "E-COMMERCE",
        "categories":     "DIVERSE",
        "diverse":        "DIVERSE",
        "trends":         "MODERN",
        "modern":         "MODERN",
        "colour":         "COLOUR",
        "color":          "COLOUR",
        "accessories":    "ACCESSORIES",
        "seasonal":       "SEASONAL",
        "night":          "NIGHT",
        "layering":       "LAYERING",
    }

    def __init__(self, library_dir: Optional[Path] = None) -> None:
        self._dir    = Path(library_dir) if library_dir else _LIB_DIR
        self._cache: Dict[str, Dict[str, List[str]]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def get_prompts(
        self,
        style:   str,
        section: Optional[str] = None,
    ) -> List[str]:
        """
        Return prompts for a style, optionally filtered to a section.

        Parameters
        ----------
        style : str  e.g. ``"streetwear"``
        section : str, optional
            Section keyword e.g. ``"runway"``, ``"e-commerce"``,
            ``"photorealistic"``, ``"colour"``.

        Returns
        -------
        list of str — prompt strings (may be empty if section not found).

        Raises
        ------
        ValueError  if style is not in AVAILABLE_STYLES.
        """
        data = self._load(style)
        if section is None:
            # All prompts flattened
            all_prompts: List[str] = []
            for prompts in data.values():
                all_prompts.extend(prompts)
            return all_prompts

        # Section lookup
        key = self._resolve_section_key(section)
        for sec_name, prompts in data.items():
            if key in sec_name.upper():
                return list(prompts)
        logger.warning("Section {!r} not found in style {!r}", section, style)
        return []

    def random_prompt(
        self,
        style:   str,
        section: Optional[str] = None,
    ) -> str:
        """
        Return a single random prompt from the style (optionally from a section).

        Parameters
        ----------
        style   : str
        section : str, optional

        Returns
        -------
        str — a single prompt string.

        Raises
        ------
        ValueError  if no prompts are available for the given parameters.
        """
        prompts = self.get_prompts(style, section=section)
        if not prompts:
            raise ValueError(
                f"No prompts found for style={style!r}, section={section!r}"
            )
        return random.choice(prompts)

    def random_batch(
        self,
        style:      str,
        n:          int            = 5,
        section:    Optional[str]  = None,
        unique:     bool           = True,
    ) -> List[str]:
        """
        Return a batch of random prompts from the style.

        Parameters
        ----------
        style   : str
        n       : int   Number of prompts to return.
        section : str, optional
        unique  : bool  If True, returns unique prompts only.

        Returns
        -------
        list of str
        """
        pool = self.get_prompts(style, section=section)
        if not pool:
            return []
        if unique:
            n = min(n, len(pool))
            return random.sample(pool, n)
        return [random.choice(pool) for _ in range(n)]

    def search(
        self,
        query:      str,
        styles:     Optional[List[str]] = None,
        case:       bool                = False,
    ) -> Dict[str, List[str]]:
        """
        Full-text search across prompt files.

        Parameters
        ----------
        query  : str  Search term.
        styles : list, optional  Limit to specific styles (default: all).
        case   : bool  Case-sensitive search (default: False).

        Returns
        -------
        dict mapping style name → list of matching prompts.
        """
        target_styles = styles or AVAILABLE_STYLES
        results: Dict[str, List[str]] = {}
        _q = query if case else query.lower()

        for style in target_styles:
            try:
                all_prompts = self.get_prompts(style)
            except (FileNotFoundError, ValueError):
                continue

            matches = [
                p for p in all_prompts
                if _q in (p if case else p.lower())
            ]
            if matches:
                results[style] = matches

        return results

    def get_all(self) -> Dict[str, List[str]]:
        """
        Return all 350 prompts across all 7 style files as a dict.

        Returns
        -------
        dict mapping style → list of prompts.
        """
        result: Dict[str, List[str]] = {}
        for style in AVAILABLE_STYLES:
            try:
                result[style] = self.get_prompts(style)
            except FileNotFoundError:
                logger.warning("Prompt file not found for style {!r}", style)
        return result

    def get_flat(self) -> List[str]:
        """Return all prompts as a flat list (all 7 styles combined)."""
        flat: List[str] = []
        for prompts in self.get_all().values():
            flat.extend(prompts)
        return flat

    def stats(self) -> Dict[str, int]:
        """
        Return prompt counts per style plus a total.

        Returns
        -------
        dict  e.g. ``{"streetwear": 50, ..., "total": 350}``
        """
        counts: Dict[str, int] = {}
        for style in AVAILABLE_STYLES:
            try:
                counts[style] = len(self.get_prompts(style))
            except FileNotFoundError:
                counts[style] = 0
        counts["total"] = sum(counts.values())
        return counts

    def list_sections(self, style: str) -> List[str]:
        """Return the section headers found in a style's prompt file."""
        data = self._load(style)
        return list(data.keys())

    # ── Private helpers ────────────────────────────────────────────────────

    def _load(self, style: str) -> Dict[str, List[str]]:
        """Load and cache a style's prompts, parsed by section."""
        key = style.lower().replace(" ", "_").replace("-", "_")

        if key not in AVAILABLE_STYLES:
            raise ValueError(
                f"Unknown style {style!r}. Available: {AVAILABLE_STYLES}"
            )

        if key in self._cache:
            return self._cache[key]

        path = self._dir / f"{key}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt library file not found: {path}\n"
                f"Expected one of: {[str(self._dir / s + '.txt') for s in AVAILABLE_STYLES]}"
            )

        sections = self._parse_file(path)
        self._cache[key] = sections
        logger.debug(
            "Loaded prompt library | style={} | sections={} | prompts={}",
            key,
            len(sections),
            sum(len(v) for v in sections.values()),
        )
        return sections

    @staticmethod
    def _parse_file(path: Path) -> Dict[str, List[str]]:
        """
        Parse a prompt .txt file into sections.

        Section headers look like: ``# ── PHOTOREALISTIC ──────``
        Prompts are non-comment, non-blank lines.
        Inline comments (``# 01``, ``# 02`` etc.) are stripped.
        """
        content = path.read_text(encoding="utf-8")
        lines   = content.splitlines()

        sections:       Dict[str, List[str]] = {}
        current_section = "GENERAL"
        sections[current_section] = []

        # Pattern: section header e.g. "# ── PHOTOREALISTIC ─────"
        section_re = re.compile(r"^#\s*[─\-─]+\s*([A-Z][A-Z /\-&]+?)\s*[─\-─]*$")

        for line in lines:
            stripped = line.strip()

            # Detect section header
            m = section_re.match(stripped)
            if m:
                current_section = m.group(1).strip()
                if current_section not in sections:
                    sections[current_section] = []
                continue

            # Skip file-level comment headers and blank lines
            if not stripped or stripped.startswith("# =") or stripped.startswith("# ─"):
                continue

            # Skip numbered inline markers like "# 01", "# 02"
            if re.match(r"^#\s*\d+\s*$", stripped):
                continue

            # Skip other full-line comments (file header lines)
            if stripped.startswith("#"):
                continue

            # Actual prompt line
            sections[current_section].append(stripped)

        # Drop any empty sections
        return {k: v for k, v in sections.items() if v}

    @staticmethod
    def _resolve_section_key(section: str) -> str:
        """Normalise a section query to the uppercase header substring."""
        norm = section.lower().replace(" ", "").replace("-", "")
        mapping = {
            "photorealistic": "PHOTOREALISTIC",
            "runway":         "RUNWAY",
            "editorial":      "RUNWAY",
            "ecommerce":      "E-COMMERCE",
            "product":        "E-COMMERCE",
            "categories":     "DIVERSE",
            "diverse":        "DIVERSE",
            "trends":         "MODERN",
            "modern":         "MODERN",
            "colour":         "COLOUR",
            "color":          "COLOUR",
            "accessories":    "ACCESSORIES",
            "seasonal":       "SEASONAL",
            "night":          "NIGHT",
            "layering":       "LAYERING",
        }
        return mapping.get(norm, section.upper())


# =============================================================================
# ── Module-level singleton convenience API
# =============================================================================

_DEFAULT_LIBRARY = PromptLibrary()


def get_prompts(style: str, section: Optional[str] = None) -> List[str]:
    """Module-level shortcut for ``PromptLibrary().get_prompts()``."""
    return _DEFAULT_LIBRARY.get_prompts(style, section=section)


def random_prompt(style: str, section: Optional[str] = None) -> str:
    """Module-level shortcut for ``PromptLibrary().random_prompt()``."""
    return _DEFAULT_LIBRARY.random_prompt(style, section=section)


def random_batch(
    style: str,
    n:     int           = 5,
    section: Optional[str] = None,
) -> List[str]:
    """Module-level shortcut for ``PromptLibrary().random_batch()``."""
    return _DEFAULT_LIBRARY.random_batch(style, n=n, section=section)


def search(
    query:  str,
    styles: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Module-level shortcut for ``PromptLibrary().search()``."""
    return _DEFAULT_LIBRARY.search(query, styles=styles)


def stats() -> Dict[str, int]:
    """Module-level shortcut for ``PromptLibrary().stats()``."""
    return _DEFAULT_LIBRARY.stats()
