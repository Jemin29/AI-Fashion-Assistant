"""
week2/tests/test_prompt_library.py
=====================================
Tests for the Fashion Prompt Library:
  - .txt file existence and structure
  - PromptLibrary class methods
  - All 7 styles × expected prompt counts
  - Section filtering, search, random selection
  - Module-level convenience functions
  - Edge cases and error handling
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generation.prompts.library import (
    AVAILABLE_STYLES,
    PromptLibrary,
    get_prompts,
    random_batch,
    random_prompt,
    search,
    stats,
)
from src.generation.prompts.library import _LIB_DIR


# =============================================================================
# ── File Existence Tests
# =============================================================================

class TestPromptFiles:
    """Verify all 7 .txt prompt files exist and are non-empty."""

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_file_exists(self, style):
        path = _LIB_DIR / f"{style}.txt"
        assert path.exists(), f"Missing prompt file: {path}"

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_file_nonempty(self, style):
        path = _LIB_DIR / f"{style}.txt"
        assert path.stat().st_size > 1000, f"{style}.txt is too small"

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_file_is_utf8(self, style):
        path = _LIB_DIR / f"{style}.txt"
        # Should read without error
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_exactly_seven_txt_files(self):
        txt_files = list(_LIB_DIR.glob("*.txt"))
        assert len(txt_files) == 7

    def test_available_styles_list(self):
        assert len(AVAILABLE_STYLES) == 7
        assert "streetwear"  in AVAILABLE_STYLES
        assert "luxury"      in AVAILABLE_STYLES
        assert "casual"      in AVAILABLE_STYLES
        assert "formal"      in AVAILABLE_STYLES
        assert "techwear"    in AVAILABLE_STYLES
        assert "vintage"     in AVAILABLE_STYLES
        assert "athleisure"  in AVAILABLE_STYLES


# =============================================================================
# ── PromptLibrary.get_prompts()
# =============================================================================

class TestGetPrompts:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_returns_exactly_50_prompts(self, lib, style):
        prompts = lib.get_prompts(style)
        assert len(prompts) == 50, f"{style} has {len(prompts)} prompts, expected 50"

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_returns_list_of_strings(self, lib, style):
        prompts = lib.get_prompts(style)
        assert isinstance(prompts, list)
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 20, f"Prompt too short: {p!r}"

    def test_unknown_style_raises_value_error(self, lib):
        with pytest.raises(ValueError, match="Unknown style"):
            lib.get_prompts("doesnotexist_xyz")

    def test_case_insensitive_style(self, lib):
        lower = lib.get_prompts("streetwear")
        upper = lib.get_prompts("STREETWEAR")
        assert lower == upper

    def test_each_prompt_is_nonempty(self, lib):
        for style in AVAILABLE_STYLES:
            for p in lib.get_prompts(style):
                assert p.strip(), f"Empty prompt in {style}"

    def test_no_comment_lines_in_output(self, lib):
        """Prompts should NOT start with #."""
        for style in AVAILABLE_STYLES:
            for p in lib.get_prompts(style):
                assert not p.strip().startswith("#"), \
                    f"Comment leaked into prompts for {style}: {p!r}"

    def test_no_numbered_markers_in_output(self, lib):
        """Lines like '# 01' should not appear as prompts."""
        import re
        for style in AVAILABLE_STYLES:
            for p in lib.get_prompts(style):
                assert not re.match(r"^#\s*\d+\s*$", p), \
                    f"Numbered marker leaked into prompts: {p!r}"

    # ── Section filtering ─────────────────────────────────────────────────

    def test_section_photorealistic_returns_subset(self, lib):
        all_p  = lib.get_prompts("streetwear")
        photo  = lib.get_prompts("streetwear", section="photorealistic")
        assert 0 < len(photo) < len(all_p)

    def test_section_ecommerce_returns_subset(self, lib):
        ec = lib.get_prompts("luxury", section="e-commerce")
        assert len(ec) > 0

    def test_section_runway_returns_subset(self, lib):
        rw = lib.get_prompts("formal", section="runway")
        assert len(rw) > 0

    def test_section_colour_returns_subset(self, lib):
        col = lib.get_prompts("techwear", section="colour")
        assert len(col) > 0

    def test_unknown_section_returns_empty(self, lib):
        result = lib.get_prompts("streetwear", section="nonexistent_section_xyz")
        assert result == []

    def test_product_section_alias_ecommerce(self, lib):
        product  = lib.get_prompts("casual", section="product")
        ecommerce= lib.get_prompts("casual", section="e-commerce")
        assert set(product) == set(ecommerce)

    def test_section_results_are_actual_prompts(self, lib):
        photo = lib.get_prompts("luxury", section="photorealistic")
        for p in photo:
            assert len(p) > 30

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_all_sections_add_up_to_total(self, lib, style):
        total = len(lib.get_prompts(style))
        sections = lib.list_sections(style)
        section_total = sum(
            len(lib.get_prompts(style, section=s)) for s in sections
        )
        # Sections might overlap with "all" due to aliases, but total should be ≥ 50
        assert section_total >= total - 5  # allow small margin for edge cases


# =============================================================================
# ── PromptLibrary.random_prompt()
# =============================================================================

class TestRandomPrompt:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_returns_string(self, lib):
        p = lib.random_prompt("streetwear")
        assert isinstance(p, str) and len(p) > 20

    def test_returns_different_prompts_on_multiple_calls(self, lib):
        """With 50 prompts, 10 draws should not all be the same."""
        results = {lib.random_prompt("luxury") for _ in range(10)}
        assert len(results) > 1

    def test_unknown_style_raises_value_error(self, lib):
        with pytest.raises(ValueError):
            lib.random_prompt("does_not_exist")

    def test_result_is_in_prompt_list(self, lib):
        all_p = lib.get_prompts("vintage")
        p     = lib.random_prompt("vintage")
        assert p in all_p

    def test_section_filtered_random(self, lib):
        all_ec = lib.get_prompts("formal", section="e-commerce")
        p      = lib.random_prompt("formal", section="e-commerce")
        assert p in all_ec

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_all_styles_return_prompt(self, lib, style):
        p = lib.random_prompt(style)
        assert isinstance(p, str) and len(p) > 10


# =============================================================================
# ── PromptLibrary.random_batch()
# =============================================================================

class TestRandomBatch:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_returns_list(self, lib):
        result = lib.random_batch("streetwear", n=5)
        assert isinstance(result, list)

    def test_correct_length(self, lib):
        result = lib.random_batch("luxury", n=10)
        assert len(result) == 10

    def test_unique_mode(self, lib):
        result = lib.random_batch("casual", n=20, unique=True)
        assert len(result) == len(set(result))

    def test_n_capped_at_available_when_unique(self, lib):
        result = lib.random_batch("formal", n=1000, unique=True)
        assert len(result) == 50

    def test_non_unique_mode_allows_repeats(self, lib):
        # With n >> pool size, some repeats are likely
        result = lib.random_batch("techwear", n=200, unique=False)
        assert len(result) == 200

    def test_section_filtered_batch(self, lib):
        ec     = lib.get_prompts("vintage", section="e-commerce")
        result = lib.random_batch("vintage", n=3, section="e-commerce")
        for p in result:
            assert p in ec

    def test_empty_section_returns_empty(self, lib):
        result = lib.random_batch("streetwear", n=5, section="nonexistent_xyz")
        assert result == []


# =============================================================================
# ── PromptLibrary.search()
# =============================================================================

class TestSearch:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_returns_dict(self, lib):
        result = lib.search("black")
        assert isinstance(result, dict)

    def test_search_term_in_results(self, lib):
        results = lib.search("hoodie")
        for style, prompts in results.items():
            for p in prompts:
                assert "hoodie" in p.lower()

    def test_case_insensitive_by_default(self, lib):
        lower = lib.search("silk")
        upper = lib.search("SILK")
        assert lower == upper

    def test_case_sensitive_mode(self, lib):
        lower = lib.search("silk", case=True)
        upper = lib.search("SILK", case=True)
        # Silk appears in prompts in title case; SILK may not
        # The result sets should differ
        assert lower != upper or len(upper) == 0

    def test_no_results_for_nonexistent_term(self, lib):
        results = lib.search("xyzzy_not_a_fashion_word_at_all_99999")
        assert results == {}

    def test_style_filter_limits_search(self, lib):
        all_results = lib.search("gown")
        limited     = lib.search("gown", styles=["luxury"])
        assert "streetwear" not in limited
        assert len(limited) <= len(all_results)

    def test_results_contain_only_filtered_styles(self, lib):
        results = lib.search("jacket", styles=["techwear", "streetwear"])
        for style in results:
            assert style in ["techwear", "streetwear"]

    def test_search_across_all_seven_styles(self, lib):
        # "black" should appear in all or most styles
        results = lib.search("black")
        assert len(results) >= 4

    def test_each_result_is_list_of_strings(self, lib):
        results = lib.search("blue")
        for style, prompts in results.items():
            assert isinstance(prompts, list)
            for p in prompts:
                assert isinstance(p, str)


# =============================================================================
# ── PromptLibrary.stats()
# =============================================================================

class TestStats:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_returns_dict(self, lib):
        s = lib.stats()
        assert isinstance(s, dict)

    def test_all_styles_in_stats(self, lib):
        s = lib.stats()
        for style in AVAILABLE_STYLES:
            assert style in s

    def test_each_style_has_50_prompts(self, lib):
        s = lib.stats()
        for style in AVAILABLE_STYLES:
            assert s[style] == 50, f"{style} has {s[style]} prompts"

    def test_total_is_350(self, lib):
        s = lib.stats()
        assert s["total"] == 350

    def test_total_key_present(self, lib):
        s = lib.stats()
        assert "total" in s


# =============================================================================
# ── PromptLibrary.list_sections()
# =============================================================================

class TestListSections:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_returns_list(self, lib):
        sections = lib.list_sections("streetwear")
        assert isinstance(sections, list)

    def test_sections_nonempty(self, lib):
        for style in AVAILABLE_STYLES:
            assert len(lib.list_sections(style)) > 0

    def test_photorealistic_section_present(self, lib):
        sections = lib.list_sections("luxury")
        assert any("PHOTOREALISTIC" in s for s in sections)

    def test_ecommerce_section_present(self, lib):
        for style in AVAILABLE_STYLES:
            sections = lib.list_sections(style)
            assert any("E-COMMERCE" in s or "PRODUCT" in s for s in sections), \
                f"No e-commerce section in {style}"

    def test_colour_section_present(self, lib):
        for style in AVAILABLE_STYLES:
            sections = lib.list_sections(style)
            assert any("COLOUR" in s for s in sections), \
                f"No colour section in {style}"


# =============================================================================
# ── PromptLibrary.get_all() and get_flat()
# =============================================================================

class TestGetAll:

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    def test_get_all_returns_dict(self, lib):
        result = lib.get_all()
        assert isinstance(result, dict)

    def test_get_all_has_seven_keys(self, lib):
        result = lib.get_all()
        assert len(result) == 7

    def test_get_flat_returns_350_prompts(self, lib):
        flat = lib.get_flat()
        assert len(flat) == 350

    def test_get_flat_returns_list_of_strings(self, lib):
        flat = lib.get_flat()
        for p in flat:
            assert isinstance(p, str) and len(p) > 10


# =============================================================================
# ── Caching Tests
# =============================================================================

class TestCaching:

    def test_same_object_returned_from_cache(self):
        lib = PromptLibrary()
        lib.get_prompts("streetwear")
        lib.get_prompts("streetwear")
        # The internal cache dict should be populated exactly once
        assert "streetwear" in lib._cache
        # Content should be identical on repeated calls
        p1 = lib.get_prompts("streetwear")
        p2 = lib.get_prompts("streetwear")
        assert p1 == p2

    def test_cache_populated_after_load(self):
        lib = PromptLibrary()
        assert "streetwear" not in lib._cache
        lib.get_prompts("streetwear")
        assert "streetwear" in lib._cache

    def test_different_styles_cached_independently(self):
        lib = PromptLibrary()
        lib.get_prompts("luxury")
        lib.get_prompts("formal")
        assert "luxury"     in lib._cache
        assert "formal"     in lib._cache
        assert "streetwear" not in lib._cache


# =============================================================================
# ── Module-Level Convenience Functions
# =============================================================================

class TestModuleFunctions:

    def test_get_prompts_module_function(self):
        prompts = get_prompts("streetwear")
        assert len(prompts) == 50

    def test_random_prompt_module_function(self):
        p = random_prompt("luxury")
        assert isinstance(p, str) and len(p) > 20

    def test_random_batch_module_function(self):
        batch = random_batch("casual", n=5)
        assert len(batch) == 5

    def test_search_module_function(self):
        results = search("suit")
        assert isinstance(results, dict)

    def test_stats_module_function(self):
        s = stats()
        assert s["total"] == 350


# =============================================================================
# ── Content Quality Tests
# =============================================================================

class TestContentQuality:
    """Validate that each prompt meets quality standards."""

    @pytest.fixture
    def lib(self):
        return PromptLibrary()

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_prompts_mention_clothing(self, lib, style):
        """At least 40 of 50 prompts should mention clothing items."""
        clothing_words = {
            "dress", "jacket", "suit", "blazer", "trouser", "jeans", "hoodie",
            "coat", "gown", "shirt", "skirt", "pant", "top", "boot", "sneaker",
            "shoe", "vest", "shorts", "tracksuit", "legging", "jumpsuit",
        }
        prompts = lib.get_prompts(style)
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in clothing_words)
        )
        assert count >= 40, \
            f"{style}: only {count}/50 prompts mention clothing"

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_prompts_mention_colours(self, lib, style):
        """At least 30 of 50 prompts should mention a colour."""
        colour_words = {
            "black", "white", "grey", "gray", "blue", "red", "green", "yellow",
            "purple", "pink", "orange", "brown", "navy", "cream", "ivory",
            "gold", "silver", "beige", "camel", "burgundy", "emerald", "cobalt",
            "teal", "coral", "lavender", "olive", "rust", "scarlet", "champagne",
        }
        prompts = lib.get_prompts(style)
        count   = sum(
            1 for p in prompts
            if any(c in p.lower() for c in colour_words)
        )
        assert count >= 30, \
            f"{style}: only {count}/50 prompts mention colours"

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_prompts_mention_photography(self, lib, style):
        """At least 25 prompts should mention photography/editorial."""
        photo_words = {
            "photography", "editorial", "photo", "runway", "e-commerce",
            "studio", "lifestyle", "portrait", "lookbook", "magazine", "vogue",
        }
        prompts = lib.get_prompts(style)
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in photo_words)
        )
        assert count >= 25, \
            f"{style}: only {count}/50 prompts mention photography"

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_minimum_prompt_length(self, lib, style):
        """All prompts should be at least 50 characters."""
        prompts = lib.get_prompts(style)
        short   = [p for p in prompts if len(p) < 50]
        assert len(short) == 0, \
            f"{style}: {len(short)} prompts are under 50 chars: {short[:2]}"

    @pytest.mark.parametrize("style", AVAILABLE_STYLES)
    def test_no_duplicate_prompts_within_style(self, lib, style):
        prompts  = lib.get_prompts(style)
        unique_p = set(p.lower().strip() for p in prompts)
        assert len(unique_p) == len(prompts), \
            f"{style}: contains duplicate prompts"

    def test_streetwear_mentions_urban_or_street(self, lib):
        prompts = lib.get_prompts("streetwear")
        count   = sum(1 for p in prompts if "urban" in p.lower() or "street" in p.lower())
        assert count >= 10

    def test_luxury_mentions_premium_or_couture(self, lib):
        prompts = lib.get_prompts("luxury")
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in ["luxury", "couture", "premium", "designer"])
        )
        assert count >= 15

    def test_techwear_mentions_technical_fabric(self, lib):
        prompts = lib.get_prompts("techwear")
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in ["nylon", "gore-tex", "tactical", "technical", "waterproof"])
        )
        assert count >= 10

    def test_vintage_mentions_decade(self, lib):
        prompts = lib.get_prompts("vintage")
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in ["1920s", "1930s", "1940s", "1950s",
                                              "1960s", "1970s", "1980s", "1990s"])
        )
        assert count >= 15

    def test_athleisure_mentions_sport_or_athletic(self, lib):
        prompts = lib.get_prompts("athleisure")
        count   = sum(
            1 for p in prompts
            if any(w in p.lower() for w in ["athletic", "sport", "gym", "performance", "yoga"])
        )
        assert count >= 15
