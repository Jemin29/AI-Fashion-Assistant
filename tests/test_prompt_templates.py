"""
week2/tests/test_prompt_templates.py
=======================================
Comprehensive tests for week2/prompts/prompt_templates.py

Coverage:
- FashionTemplate dataclass (render_sentence, primary_adjective, all_positive_tags)
- Template registry (get_template, register_template, list_templates)
- All 7 style templates registered and valid
- generate_prompt() — core scenarios, all styles, field combinations
- generate_negative_prompt() — all flags, style stacking, item dict input
- prompt_enhancer() — enhancement, style injection, quality boosting
- generate_prompt_pair() — returns (pos, neg) tuple
- generate_batch_prompts() — list processing, partial failure isolation
- enhance_batch() — list enhancement
- explain_prompt() — layer breakdown structure
- Private helpers: _deduplicate, _estimate_tokens, _join_and_trim
- Edge cases: empty fields, unknown styles, missing keys, colour lists
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generation.prompts.prompt_templates import (
    FashionTemplate,
    _TEMPLATES,
    _GLOBAL_QUALITY_BOOSTERS,
    _PHOTO_STYLES,
    _GENDER_PHRASES,
    _CATEGORY_FIT_HINTS,
    _STYLE_FABRICS,
    _deduplicate,
    _estimate_tokens,
    _join_and_trim,
    enhance_batch,
    explain_prompt,
    generate_batch_prompts,
    generate_negative_prompt,
    generate_prompt,
    generate_prompt_pair,
    get_template,
    list_template_details,
    list_templates,
    prompt_enhancer,
    register_template,
)


# =============================================================================
# ── FashionTemplate Dataclass Tests
# =============================================================================

class TestFashionTemplate:

    @pytest.fixture
    def basic_tmpl(self) -> FashionTemplate:
        return FashionTemplate(
            style             = "test_style",
            display_name      = "Test Style",
            sentence_template = "Photo of {gender_phrase} a {color} {fit} {style_adj} {category}",
            style_adjectives  = ["cool", "trendy"],
            positive_tags     = ["tag_a", "tag_b"],
            quality_tags      = ["hd quality"],
            negative_extra    = ["bad_tag"],
            photo_style       = "studio",
        )

    def test_primary_adjective_returns_first(self, basic_tmpl):
        assert basic_tmpl.primary_adjective() == "cool"

    def test_primary_adjective_fallback_to_style(self):
        t = FashionTemplate(
            style="minimal", display_name="Min", sentence_template="X", style_adjectives=[]
        )
        assert t.primary_adjective() == "minimal"

    def test_all_positive_tags_merged(self, basic_tmpl):
        combined = basic_tmpl.all_positive_tags()
        assert "tag_a"      in combined
        assert "tag_b"      in combined
        assert "hd quality" in combined

    def test_all_positive_tags_no_duplicates(self):
        t = FashionTemplate(
            style="dup", display_name="Dup", sentence_template="X",
            positive_tags=["one", "two"],
            quality_tags =["two", "three"],
        )
        tags = t.all_positive_tags()
        assert tags.count("two") == 1

    def test_render_sentence_basic(self, basic_tmpl):
        result = basic_tmpl.render_sentence(
            category="hoodie", color="black", gender_phrase="a man wearing"
        )
        assert "black" in result
        assert "hoodie" in result
        assert "a man wearing" in result

    def test_render_sentence_empty_color(self, basic_tmpl):
        result = basic_tmpl.render_sentence(category="jacket", color="")
        assert "jacket" in result  # Should still work

    def test_render_sentence_uses_fit_hint(self):
        t = FashionTemplate(
            style="x", display_name="X",
            sentence_template="Photo of a {fit} {category}",
            style_adjectives=["bold"],
        )
        result = t.render_sentence(category="hoodie", color="red")
        assert "oversized" in result  # from _CATEGORY_FIT_HINTS

    def test_render_sentence_bad_key_fallback(self):
        t = FashionTemplate(
            style="x", display_name="X",
            sentence_template="{undefined_key} {category}",
            style_adjectives=["bold"],
        )
        result = t.render_sentence(category="dress")
        assert "dress" in result  # fallback triggered, should contain category


# =============================================================================
# ── Template Registry Tests
# =============================================================================

class TestTemplateRegistry:

    EXPECTED_STYLES = [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ]

    def test_all_seven_styles_registered(self):
        registered = list_templates()
        for style in self.EXPECTED_STYLES:
            assert style in registered, f"Missing style: {style}"

    def test_get_template_returns_fashion_template(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert isinstance(tmpl, FashionTemplate)

    def test_get_template_case_insensitive(self):
        assert get_template("STREETWEAR").style == "streetwear"
        assert get_template("Luxury").style     == "luxury"

    def test_get_template_hyphen_normalised(self):
        # register a hyphenated name
        t = FashionTemplate(
            style="hi-tech", display_name="Hi-Tech", sentence_template="X",
            style_adjectives=["hi-tech"],
        )
        register_template(t)
        result = get_template("hi-tech")
        assert result.style == "hi-tech"

    def test_get_template_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown style"):
            get_template("does_not_exist_xyz")

    def test_register_template_adds_to_registry(self):
        before = len(list_templates())
        custom = FashionTemplate(
            style             = f"custom_{id(object())}",
            display_name      = "Custom",
            sentence_template = "Photo of {category}",
            style_adjectives  = ["custom"],
        )
        register_template(custom)
        assert custom.style in list_templates()
        assert len(list_templates()) == before + 1

    def test_register_template_overwrites_existing(self):
        t1 = FashionTemplate(
            style="overwrite_test", display_name="V1",
            sentence_template="V1 photo", style_adjectives=["old"],
        )
        t2 = FashionTemplate(
            style="overwrite_test", display_name="V2",
            sentence_template="V2 photo", style_adjectives=["new"],
        )
        register_template(t1)
        register_template(t2)
        assert get_template("overwrite_test").display_name == "V2"

    def test_list_template_details_returns_all(self):
        details = list_template_details()
        assert isinstance(details, list)
        names = [d["style"] for d in details]
        for style in self.EXPECTED_STYLES:
            assert style in names

    def test_list_template_details_structure(self):
        details = list_template_details()
        for d in details:
            assert "style"                in d
            assert "display_name"         in d
            assert "description"          in d
            assert "photo_style"          in d
            assert "recommended_steps"    in d
            assert "recommended_guidance" in d

    def test_each_template_has_nonempty_positive_tags(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert len(tmpl.positive_tags) > 0, f"{style} has no positive_tags"

    def test_each_template_has_nonempty_quality_tags(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert len(tmpl.quality_tags) > 0, f"{style} has no quality_tags"

    def test_each_template_has_negative_extra(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert len(tmpl.negative_extra) > 0, f"{style} has no negative_extra"

    def test_each_template_recommended_steps_in_range(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert 10 <= tmpl.recommended_steps <= 60

    def test_each_template_guidance_in_range(self):
        for style in self.EXPECTED_STYLES:
            tmpl = get_template(style)
            assert 5.0 <= tmpl.recommended_guidance <= 12.0


# =============================================================================
# ── generate_prompt() Tests
# =============================================================================

class TestGeneratePrompt:

    # ── Basic structure ───────────────────────────────────────────────────

    def test_returns_nonempty_string(self):
        result = generate_prompt({"category": "hoodie", "style": "streetwear", "color": "black"})
        assert isinstance(result, str)
        assert len(result) > 20

    def test_contains_category(self):
        result = generate_prompt({"category": "blazer", "style": "formal"})
        assert "blazer" in result.lower()

    def test_contains_color(self):
        result = generate_prompt({"category": "dress", "style": "luxury", "color": "gold"})
        assert "gold" in result.lower()

    def test_contains_style_tag(self):
        result = generate_prompt({"style": "streetwear", "category": "hoodie"})
        # Should contain at least one streetwear-related tag
        assert any(
            word in result.lower()
            for word in ["streetwear", "urban", "hypebeast", "street style"]
        )

    def test_output_is_comma_separated(self):
        result = generate_prompt({"style": "casual", "category": "t-shirt", "color": "white"})
        assert "," in result

    # ── All 7 styles ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_all_styles_produce_output(self, style):
        result = generate_prompt({"style": style, "category": "jacket", "color": "black"})
        assert isinstance(result, str)
        assert len(result) > 30

    # ── Token budget ──────────────────────────────────────────────────────

    def test_respects_max_tokens(self):
        result = generate_prompt(
            {"style": "luxury", "category": "evening gown", "color": "deep emerald"},
            max_tokens=77,
        )
        assert _estimate_tokens(result) <= 77

    def test_custom_max_tokens_tighter(self):
        result = generate_prompt(
            {"style": "streetwear", "category": "hoodie"},
            max_tokens=30,
        )
        assert _estimate_tokens(result) <= 30

    # ── Gender phrases ────────────────────────────────────────────────────

    def test_gender_women_phrase(self):
        result = generate_prompt({
            "style": "luxury", "category": "dress",
            "color": "red", "gender": "women",
        })
        assert "woman" in result.lower()

    def test_gender_men_phrase(self):
        result = generate_prompt({
            "style": "formal", "category": "suit",
            "color": "navy", "gender": "men",
        })
        assert "man" in result.lower()

    def test_gender_kwarg_overrides_item(self):
        result = generate_prompt(
            {"style": "casual", "category": "hoodie", "gender": "men"},
            gender="women",
        )
        assert "woman" in result.lower()

    # ── Color handling ────────────────────────────────────────────────────

    def test_color_list_uses_first_element(self):
        result = generate_prompt({
            "style": "casual", "category": "t-shirt",
            "colors": ["crimson", "white"],
        })
        assert "crimson" in result.lower()

    def test_color_key_alias(self):
        result = generate_prompt({"style": "casual", "category": "shirt", "color": "teal"})
        assert "teal" in result.lower()

    # ── Description override ──────────────────────────────────────────────

    def test_description_used_as_sentence(self):
        desc = "A breathtaking silk kimono with dragon embroidery"
        result = generate_prompt({
            "style": "luxury", "category": "kimono",
            "description": desc,
        })
        assert "kimono" in result.lower()
        assert "silk" in result.lower()

    # ── Optional fields ───────────────────────────────────────────────────

    def test_occasion_appended(self):
        result = generate_prompt({
            "style": "formal", "category": "suit", "occasion": "wedding",
        })
        assert "wedding" in result.lower()

    def test_season_appended(self):
        result = generate_prompt({
            "style": "casual", "category": "jacket", "season": "winter",
        })
        assert "winter" in result.lower()

    def test_extra_tags_from_item(self):
        result = generate_prompt({
            "style": "streetwear", "category": "hoodie",
            "extra_tags": ["limited edition drop"],
        })
        assert "limited edition drop" in result.lower()

    def test_extra_tags_kwarg(self):
        result = generate_prompt(
            {"style": "casual", "category": "shirt"},
            extra_tags=["summer vibes"],
        )
        assert "summer vibes" in result.lower()

    # ── Quality boosters ──────────────────────────────────────────────────

    def test_quality_boost_included_by_default(self):
        result = generate_prompt({"style": "luxury", "category": "coat"})
        quality_found = any(q in result for q in _GLOBAL_QUALITY_BOOSTERS[:3])
        assert quality_found

    def test_quality_boost_can_be_disabled(self):
        result = generate_prompt(
            {"style": "luxury", "category": "coat"},
            boost_quality=False,
        )
        # Should have fewer tags
        assert _estimate_tokens(result) < _estimate_tokens(
            generate_prompt({"style": "luxury", "category": "coat"})
        )

    # ── Fabric handling ───────────────────────────────────────────────────

    def test_fabric_override_from_item(self):
        result = generate_prompt({
            "style": "luxury", "category": "dress",
            "fabric": "velvet",
        })
        assert "velvet" in result.lower()

    def test_fabric_auto_injected_by_style(self):
        result = generate_prompt({"style": "luxury", "category": "coat"})
        assert any(f in result.lower() for f in _STYLE_FABRICS["luxury"])

    def test_fabric_injection_can_be_disabled(self):
        result_with    = generate_prompt({"style": "luxury", "category": "coat"}, include_fabric=True)
        result_without = generate_prompt({"style": "luxury", "category": "coat"}, include_fabric=False)
        assert len(result_with) >= len(result_without)

    # ── Photo style ───────────────────────────────────────────────────────

    def test_photo_style_override(self):
        result = generate_prompt(
            {"style": "streetwear", "category": "hoodie"},
            photo_style="runway",
        )
        runway_found = any(t in result for t in _PHOTO_STYLES["runway"])
        assert runway_found

    # ── Unknown style fallback ────────────────────────────────────────────

    def test_unknown_style_falls_back_to_casual(self):
        result = generate_prompt({"style": "totally_unknown_style", "category": "shirt"})
        assert isinstance(result, str)
        assert len(result) > 10

    # ── Empty / minimal input ─────────────────────────────────────────────

    def test_empty_dict_does_not_crash(self):
        result = generate_prompt({})
        assert isinstance(result, str)

    def test_category_only_does_not_crash(self):
        result = generate_prompt({"category": "hoodie"})
        assert "hoodie" in result.lower()

    def test_no_duplicates_in_output(self):
        result = generate_prompt({"style": "streetwear", "category": "hoodie", "color": "black"})
        tags = [t.strip().lower() for t in result.split(",")]
        # Check no exact duplicate tags
        assert len(tags) == len(set(tags))


# =============================================================================
# ── generate_negative_prompt() Tests
# =============================================================================

class TestGenerateNegativePrompt:

    def test_returns_nonempty_string(self):
        result = generate_negative_prompt("streetwear")
        assert isinstance(result, str)
        assert len(result) > 20

    def test_always_contains_quality_negatives(self):
        for style in ["streetwear", "luxury", "casual", "formal"]:
            result = generate_negative_prompt(style)
            assert "blurry" in result

    def test_contains_anatomy_negatives_by_default(self):
        result = generate_negative_prompt()
        assert "deformed" in result

    def test_anatomy_negatives_can_be_disabled(self):
        result = generate_negative_prompt(include_anatomy=False)
        assert "deformed" not in result

    def test_contains_artefact_negatives_by_default(self):
        result = generate_negative_prompt()
        assert "watermark" in result

    def test_artefact_negatives_can_be_disabled(self):
        result = generate_negative_prompt(include_artefacts=False)
        assert "watermark" not in result

    def test_nsfw_included_by_default(self):
        result = generate_negative_prompt()
        assert "nsfw" in result

    def test_nsfw_can_be_disabled(self):
        result = generate_negative_prompt(include_nsfw=False)
        assert "nsfw" not in result

    def test_style_specific_negatives_added(self):
        result = generate_negative_prompt("streetwear")
        # streetwear negative_extra includes "formal wear"
        assert "formal wear" in result

    def test_luxury_specific_negatives(self):
        result = generate_negative_prompt("luxury")
        assert "fast fashion" in result

    def test_formal_specific_negatives(self):
        result = generate_negative_prompt("formal")
        assert "casual" in result

    def test_item_dict_input(self):
        result = generate_negative_prompt(item={"style": "streetwear", "category": "hoodie"})
        assert "blurry" in result
        assert "formal wear" in result

    def test_style_arg_overrides_item_style(self):
        result = generate_negative_prompt(
            "luxury",
            item={"style": "streetwear"},
        )
        # luxury negatives should be present
        assert "fast fashion" in result

    def test_extra_negatives_appended(self):
        result = generate_negative_prompt(extra_negatives=["my_custom_negative"])
        assert "my_custom_negative" in result

    def test_no_style_returns_base_negatives(self):
        result = generate_negative_prompt()
        assert "blurry" in result
        assert "deformed" in result

    def test_no_duplicate_tags(self):
        result = generate_negative_prompt("streetwear", include_nsfw=True)
        tags = [t.strip().lower() for t in result.split(",")]
        assert len(tags) == len(set(tags))

    def test_all_seven_styles_produce_output(self):
        for style in ["streetwear", "luxury", "casual", "formal", "techwear", "vintage", "athleisure"]:
            result = generate_negative_prompt(style)
            assert isinstance(result, str) and len(result) > 10


# =============================================================================
# ── prompt_enhancer() Tests
# =============================================================================

class TestPromptEnhancer:

    def test_returns_string(self):
        result = prompt_enhancer("a red hoodie")
        assert isinstance(result, str)

    def test_input_prompt_preserved_at_start(self):
        result = prompt_enhancer("a red hoodie", style="streetwear")
        assert result.startswith("a red hoodie")

    def test_style_tags_injected(self):
        result = prompt_enhancer("a red dress", style="luxury")
        assert any(t in result for t in get_template("luxury").positive_tags[:3])

    def test_quality_boosters_appended(self):
        result = prompt_enhancer("a shirt", boost_quality=True)
        assert any(q in result for q in _GLOBAL_QUALITY_BOOSTERS[:3])

    def test_quality_boost_disabled(self):
        result_with    = prompt_enhancer("a shirt", boost_quality=True)
        result_without = prompt_enhancer("a shirt", boost_quality=False)
        assert len(result_without) < len(result_with)

    def test_style_tags_can_be_disabled(self):
        result_with    = prompt_enhancer("a hoodie", style="streetwear", add_style_tags=True)
        result_without = prompt_enhancer("a hoodie", style="streetwear", add_style_tags=False)
        assert len(result_without) < len(result_with)

    def test_extra_tags_appended(self):
        result = prompt_enhancer("a coat", extra_tags=["runway model"])
        assert "runway model" in result

    def test_token_budget_respected(self):
        result = prompt_enhancer("a coat", style="luxury", max_tokens=30)
        assert _estimate_tokens(result) <= 30

    def test_photo_style_injected(self):
        result = prompt_enhancer("a suit", photo_style="runway")
        assert any(t in result for t in _PHOTO_STYLES["runway"])

    def test_all_styles_enhance_without_error(self):
        for style in ["streetwear", "luxury", "casual", "formal", "techwear", "vintage", "athleisure"]:
            result = prompt_enhancer("a garment", style=style)
            assert isinstance(result, str)

    def test_empty_prompt_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            prompt_enhancer("")

    def test_whitespace_prompt_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            prompt_enhancer("   ")

    def test_unknown_style_does_not_crash(self):
        result = prompt_enhancer("a hoodie", style="totally_unknown_style_xyz")
        assert result.startswith("a hoodie")

    def test_no_duplicates_in_output(self):
        result = prompt_enhancer("a hoodie", style="streetwear", boost_quality=True)
        tags = [t.strip().lower() for t in result.split(",")]
        assert len(tags) == len(set(tags))

    def test_preserve_order_false_still_includes_original(self):
        result = prompt_enhancer("my special prompt", style="casual", preserve_order=False)
        assert "my special prompt" in result


# =============================================================================
# ── generate_prompt_pair() Tests
# =============================================================================

class TestGeneratePromptPair:

    def test_returns_tuple(self):
        pair = generate_prompt_pair({"style": "casual", "category": "shirt"})
        assert isinstance(pair, tuple)
        assert len(pair) == 2

    def test_positive_is_nonempty(self):
        pos, neg = generate_prompt_pair({"style": "luxury", "category": "dress"})
        assert len(pos) > 10

    def test_negative_is_nonempty(self):
        pos, neg = generate_prompt_pair({"style": "formal", "category": "suit"})
        assert len(neg) > 10

    def test_negative_contains_blurry(self):
        _, neg = generate_prompt_pair({"style": "streetwear", "category": "hoodie"})
        assert "blurry" in neg

    def test_positive_contains_category(self):
        pos, _ = generate_prompt_pair({"style": "vintage", "category": "leather jacket"})
        assert "leather jacket" in pos.lower()

    @pytest.mark.parametrize("style", [
        "streetwear", "luxury", "casual", "formal",
        "techwear", "vintage", "athleisure",
    ])
    def test_all_styles_return_valid_pair(self, style):
        pos, neg = generate_prompt_pair({"style": style, "category": "jacket"})
        assert isinstance(pos, str) and len(pos) > 10
        assert isinstance(neg, str) and len(neg) > 10


# =============================================================================
# ── generate_batch_prompts() Tests
# =============================================================================

class TestGenerateBatchPrompts:

    def test_returns_list(self):
        result = generate_batch_prompts([{"style": "casual", "category": "shirt"}])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        items = [
            {"style": "streetwear", "category": "hoodie"},
            {"style": "luxury",     "category": "dress"},
            {"style": "formal",     "category": "suit"},
        ]
        result = generate_batch_prompts(items)
        assert len(result) == 3

    def test_each_element_is_tuple_of_strings(self):
        items  = [{"style": "casual", "category": "shirt", "color": "white"}]
        result = generate_batch_prompts(items)
        pos, neg = result[0]
        assert isinstance(pos, str)
        assert isinstance(neg, str)

    def test_empty_list_returns_empty(self):
        assert generate_batch_prompts([]) == []

    def test_shared_kwargs_applied_to_all(self):
        items = [
            {"style": "streetwear", "category": "hoodie"},
            {"style": "luxury",     "category": "dress"},
        ]
        results = generate_batch_prompts(items, shared_kwargs={"boost_quality": False})
        for pos, _ in results:
            # Global quality boosters should be absent
            assert "8k resolution" not in pos

    def test_batch_all_seven_styles(self):
        items = [
            {"style": s, "category": "jacket"}
            for s in ["streetwear", "luxury", "casual", "formal", "techwear", "vintage", "athleisure"]
        ]
        results = generate_batch_prompts(items)
        assert len(results) == 7
        for pos, neg in results:
            assert len(pos) > 10
            assert len(neg) > 10


# =============================================================================
# ── enhance_batch() Tests
# =============================================================================

class TestEnhanceBatch:

    def test_returns_list_same_length(self):
        prompts = ["a red dress", "a blue hoodie", "a white shirt"]
        result  = enhance_batch(prompts, style="casual")
        assert len(result) == len(prompts)

    def test_each_element_starts_with_original(self):
        prompts = ["a red dress", "a blue hoodie"]
        result  = enhance_batch(prompts, style="casual")
        for orig, enhanced in zip(prompts, result):
            assert enhanced.startswith(orig)

    def test_empty_list_returns_empty(self):
        assert enhance_batch([]) == []

    def test_style_applied_to_all(self):
        prompts = ["garment A", "garment B"]
        result  = enhance_batch(prompts, style="streetwear")
        for enhanced in result:
            assert any(
                t in enhanced for t in get_template("streetwear").positive_tags[:2]
            )


# =============================================================================
# ── explain_prompt() Tests
# =============================================================================

class TestExplainPrompt:

    def test_returns_dict(self):
        result = explain_prompt({"style": "streetwear", "category": "hoodie"})
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = explain_prompt({"style": "luxury", "category": "gown"})
        for key in ("full_prompt", "negative", "layers", "token_estimate", "style", "template"):
            assert key in result, f"Missing key: {key}"

    def test_layers_has_required_sublayers(self):
        result = explain_prompt({"style": "formal", "category": "suit"})
        layers = result["layers"]
        assert "sentence"      in layers
        assert "style_tags"    in layers
        assert "photo_tags"    in layers
        assert "quality_tags"  in layers

    def test_full_prompt_nonempty(self):
        result = explain_prompt({"style": "casual", "category": "t-shirt"})
        assert len(result["full_prompt"]) > 10

    def test_token_estimate_positive(self):
        result = explain_prompt({"style": "vintage", "category": "coat"})
        assert result["token_estimate"] > 0

    def test_style_field_matches_requested(self):
        result = explain_prompt({"style": "techwear", "category": "jacket"})
        assert result["style"] == "techwear"

    def test_template_has_recommended_steps(self):
        result = explain_prompt({"style": "luxury", "category": "dress"})
        assert "recommended_steps" in result["template"]

    def test_negative_contains_blurry(self):
        result = explain_prompt({"style": "streetwear", "category": "hoodie"})
        assert "blurry" in result["negative"]


# =============================================================================
# ── Private Helpers Tests
# =============================================================================

class TestPrivateHelpers:

    # ── _deduplicate ──────────────────────────────────────────────────────

    def test_deduplicate_removes_exact_duplicates(self):
        tags = ["one", "two", "one", "three"]
        result = _deduplicate(tags)
        assert result == ["one", "two", "three"]

    def test_deduplicate_case_insensitive(self):
        tags = ["Urban", "urban", "URBAN"]
        result = _deduplicate(tags)
        assert len(result) == 1

    def test_deduplicate_strips_whitespace(self):
        tags = ["  tag  ", "tag"]
        result = _deduplicate(tags)
        assert len(result) == 1

    def test_deduplicate_removes_empty_strings(self):
        tags = ["", "  ", "real_tag"]
        result = _deduplicate(tags)
        assert result == ["real_tag"]

    def test_deduplicate_preserves_order(self):
        tags = ["c", "a", "b", "a", "c"]
        result = _deduplicate(tags)
        assert result == ["c", "a", "b"]

    # ── _estimate_tokens ──────────────────────────────────────────────────

    def test_estimate_tokens_single_word(self):
        assert _estimate_tokens("hello") == 1

    def test_estimate_tokens_two_words(self):
        # int(2 * 1.3) = 2
        assert _estimate_tokens("hello world") == 2

    def test_estimate_tokens_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_scales_with_length(self):
        short = _estimate_tokens("a b c")
        long  = _estimate_tokens("a b c d e f g h i j")
        assert long > short

    # ── _join_and_trim ────────────────────────────────────────────────────

    def test_join_and_trim_short_list(self):
        parts  = ["one", "two", "three"]
        result = _join_and_trim(parts, 77)
        assert result == "one, two, three"

    def test_join_and_trim_respects_budget(self):
        # Create a list that definitely exceeds 10 tokens
        parts  = ["word"] * 50
        result = _join_and_trim(parts, 10)
        assert _estimate_tokens(result) <= 10

    def test_join_and_trim_preserves_first_element(self):
        parts  = ["first important part"] + ["filler"] * 100
        result = _join_and_trim(parts, 10)
        assert "first important part" in result

    def test_join_and_trim_single_part(self):
        result = _join_and_trim(["single"], 77)
        assert result == "single"


# =============================================================================
# ── Constants / Configuration Tests
# =============================================================================

class TestConstants:

    def test_global_quality_boosters_nonempty(self):
        assert len(_GLOBAL_QUALITY_BOOSTERS) >= 4

    def test_photo_styles_has_required_keys(self):
        for key in ("studio", "editorial", "outdoor", "urban"):
            assert key in _PHOTO_STYLES

    def test_each_photo_style_has_tags(self):
        for key, tags in _PHOTO_STYLES.items():
            assert len(tags) > 0, f"Photo style {key!r} has no tags"

    def test_gender_phrases_cover_all_genders(self):
        for gender in ("women", "men", "unisex"):
            assert gender in _GENDER_PHRASES

    def test_style_fabrics_covers_all_seven_styles(self):
        for style in ["streetwear", "luxury", "casual", "formal", "techwear", "vintage", "athleisure"]:
            assert style in _STYLE_FABRICS
            assert len(_STYLE_FABRICS[style]) > 0


# =============================================================================
# ── Integration / Scenario Tests
# =============================================================================

class TestIntegrationScenarios:

    def test_exact_example_from_requirements(self):
        """
        Exact example from requirements:
            Input:  {"category":"hoodie","style":"streetwear","color":"black"}
            Output: "Professional fashion photography of a black oversized streetwear hoodie..."
        """
        result = generate_prompt({
            "category": "hoodie",
            "style":    "streetwear",
            "color":    "black",
        })
        assert "hoodie" in result.lower()
        assert "black"  in result.lower()
        # Should mention streetwear or urban context
        assert any(w in result.lower() for w in ["streetwear", "urban", "street"])

    def test_luxury_dress_scenario(self):
        pos, neg = generate_prompt_pair({
            "category": "evening gown",
            "style":    "luxury",
            "color":    "deep emerald",
            "gender":   "women",
        })
        assert "emerald"   in pos.lower()
        assert "gown"      in pos.lower() or "evening" in pos.lower()
        assert "woman"     in pos.lower()
        assert "blurry"    in neg
        assert "fast fashion" in neg

    def test_techwear_jacket_scenario(self):
        pos, neg = generate_prompt_pair({
            "category": "jacket",
            "style":    "techwear",
            "color":    "matte black",
        })
        assert "jacket"   in pos.lower()
        assert "matte black" in pos.lower()
        assert any(t in pos.lower() for t in ["techwear", "technical", "tactical"])

    def test_vintage_outfit_scenario(self):
        pos, neg = generate_prompt_pair({
            "category": "trench coat",
            "style":    "vintage",
            "color":    "camel",
            "gender":   "women",
        })
        assert "camel"  in pos.lower()
        assert "woman"  in pos.lower()
        assert any(t in pos.lower() for t in ["vintage", "retro", "classic"])

    def test_athleisure_scenario(self):
        pos, neg = generate_prompt_pair({
            "category": "leggings",
            "style":    "athleisure",
            "color":    "coral",
            "gender":   "women",
        })
        assert "coral"  in pos.lower()
        assert any(t in pos.lower() for t in ["athleisure", "athletic", "sport"])

    def test_enhancer_then_generator_pipeline(self):
        """Simulate: enhance a raw description, then use as generate_prompt input."""
        raw = "A cozy winter coat with fur trim"
        enhanced_subject = prompt_enhancer(raw, style="luxury")
        result = generate_prompt({
            "description": enhanced_subject,
            "style":       "luxury",
            "category":    "coat",
        })
        assert "coat" in result.lower()
        assert len(result) > 50

    def test_full_batch_pipeline(self):
        """Simulate a real production batch run with 5 items."""
        items = [
            {"style": "streetwear", "category": "hoodie",    "color": "black",  "gender": "unisex"},
            {"style": "luxury",     "category": "dress",     "color": "gold",   "gender": "women"},
            {"style": "casual",     "category": "t-shirt",   "color": "white",  "gender": "men"},
            {"style": "formal",     "category": "blazer",    "color": "navy",   "gender": "men"},
            {"style": "athleisure", "category": "joggers",   "color": "grey",   "gender": "women"},
        ]
        results = generate_batch_prompts(items)
        assert len(results) == 5
        for pos, neg in results:
            assert len(pos) > 20
            assert len(neg) > 20
            assert "blurry" in neg

    def test_custom_template_end_to_end(self):
        """Register a brand-new Y2K style and use it end-to-end."""
        y2k_tmpl = FashionTemplate(
            style             = "y2k",
            display_name      = "Y2K / 2000s Revival",
            description       = "2000s pop culture revival fashion.",
            sentence_template = (
                "Y2K fashion photograph of {gender_phrase} "
                "a {color} {fit} {style_adj} {category}"
            ),
            style_adjectives  = ["Y2K", "2000s-inspired", "retro-futuristic"],
            positive_tags     = [
                "Y2K aesthetic", "2000s pop culture", "butterfly clips",
                "low-rise jeans", "platform shoes",
            ],
            quality_tags      = ["fashion editorial", "vibrant colors"],
            negative_extra    = ["minimalist", "monochrome"],
            photo_style       = "outdoor",
            recommended_steps = 28,
            recommended_guidance = 7.0,
        )
        register_template(y2k_tmpl)

        pos, neg = generate_prompt_pair({
            "style":    "y2k",
            "category": "mini skirt",
            "color":    "hot pink",
            "gender":   "women",
        })
        assert "y2k" in pos.lower() or "2000s" in pos.lower()
        assert "hot pink"    in pos.lower()
        assert "mini skirt"  in pos.lower()
        assert "woman"       in pos.lower()
        assert "minimalist"  in neg
