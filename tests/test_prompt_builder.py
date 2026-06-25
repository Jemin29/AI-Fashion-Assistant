"""
week2/tests/test_prompt_builder.py
=====================================
Tests for PromptBuilder, PromptValidator, style presets,
and negative prompt utilities. No model loading required.
"""

from __future__ import annotations

import pytest

from src.generation.prompts.prompt_builder import PromptBuilder, BuiltPrompt
from src.generation.prompts.prompt_validator import PromptValidator, ValidationResult
from src.generation.prompts.style_presets import (
    get_preset, list_presets, register_preset, StylePreset
)
from src.generation.prompts.negative_prompts import (
    get_base_negative, get_fashion_negative, get_full_negative,
    get_style_negative, format_negative, QUALITY_NEGATIVES,
)


# =============================================================================
# ── PromptBuilder Tests
# =============================================================================

class TestPromptBuilder:

    @pytest.fixture
    def builder(self, mock_config):
        return PromptBuilder(config=mock_config)

    def test_build_returns_built_prompt(self, builder):
        result = builder.build(subject="a red dress")
        assert isinstance(result, BuiltPrompt)

    def test_positive_contains_subject(self, builder):
        result = builder.build(subject="a red silk dress")
        assert "a red silk dress" in result.positive

    def test_positive_contains_quality_tags(self, builder):
        result = builder.build(subject="a blue hoodie")
        assert "photorealistic" in result.positive or "8k" in result.positive

    def test_style_tags_applied(self, builder):
        result = builder.build(subject="a jacket", style="streetwear")
        assert result.style_name == "streetwear"
        # Should have urban/streetwear tags in positive
        assert len(result.positive) > len("a jacket")

    def test_unknown_style_does_not_crash(self, builder):
        result = builder.build(subject="a jacket", style="nonexistent_style_xyz")
        assert isinstance(result, BuiltPrompt)
        assert result.style_name == ""

    def test_gender_modifier_applied(self, builder):
        result = builder.build(subject="a dress", gender="women")
        assert "female model" in result.positive

    def test_season_modifier_applied(self, builder):
        result = builder.build(subject="a light dress", season="summer")
        assert "summer fashion" in result.positive

    def test_negative_not_empty(self, builder):
        result = builder.build(subject="a dress")
        assert len(result.negative) > 0
        assert "blurry" in result.negative

    def test_extra_positive_included(self, builder):
        result = builder.build(subject="a coat", extra_positive=["wool fabric"])
        assert "wool fabric" in result.positive

    def test_extra_negative_included(self, builder):
        result = builder.build(subject="a jacket", extra_negative=["red color"])
        assert "red color" in result.negative

    def test_deduplication_works(self, builder):
        # Passing the same tag twice should only appear once
        result = builder.build(
            subject="a dress",
            extra_positive=["photorealistic"],   # already in quality_boosters
        )
        count = result.positive.lower().count("photorealistic")
        assert count == 1

    def test_token_count_is_int(self, builder):
        result = builder.build(subject="a shirt")
        assert isinstance(result.token_count_estimate, int)
        assert result.token_count_estimate > 0

    def test_no_quality_boosters_when_disabled(self, builder):
        result = builder.build(subject="a dress", include_quality=False)
        assert "photorealistic" not in result.positive

    def test_build_from_metadata_uses_description(self, builder):
        record = {
            "category":    "dresses",
            "description": "A flowing floral summer dress",
            "gender":      "women",
            "style":       "casual",
            "season":      "summer",
        }
        result = builder.build_from_metadata(record)
        assert isinstance(result, BuiltPrompt)
        assert len(result.positive) > 0


# =============================================================================
# ── PromptValidator Tests
# =============================================================================

class TestPromptValidator:

    @pytest.fixture
    def validator(self):
        return PromptValidator()

    def test_valid_prompt_passes(self, validator):
        r = validator.validate("A woman in a red silk evening gown")
        assert r.is_valid is True
        assert r.errors == []

    def test_none_prompt_fails(self, validator):
        r = validator.validate(None)
        assert r.is_valid is False
        assert len(r.errors) > 0

    def test_empty_prompt_fails(self, validator):
        r = validator.validate("")
        assert r.is_valid is False

    def test_short_prompt_fails(self, validator):
        r = validator.validate("hi")
        assert r.is_valid is False

    def test_nsfw_blocked(self, validator):
        r = validator.validate("a nude woman")
        assert r.is_valid is False
        assert any("blocked" in e.lower() for e in r.errors)

    def test_injection_stripped(self, validator):
        r = validator.validate("a dress <lora:fashion:1.0> in summer")
        assert r.is_valid is True
        assert "<lora:" not in r.sanitized_prompt

    def test_whitespace_normalised(self, validator):
        r = validator.validate("a   red    dress")
        assert "  " not in r.sanitized_prompt

    def test_long_prompt_warns(self, validator):
        long_prompt = " ".join(["fashion"] * 100)
        r = validator.validate(long_prompt)
        # May be valid but should warn about tokens
        assert len(r.warnings) > 0

    def test_token_estimate_is_int(self, validator):
        r = validator.validate("A stylish red dress")
        assert isinstance(r.token_estimate, int)

    def test_validate_pair_empty_negative_valid(self, validator):
        pos, neg = validator.validate_pair("A red dress", "")
        assert neg.is_valid is True

    def test_validate_pair_both_valid(self, validator):
        pos, neg = validator.validate_pair("A red dress", "blurry, deformed")
        assert pos.is_valid is True
        assert neg.is_valid is True

    def test_bool_conversion(self, validator):
        r = validator.validate("A fashionable coat")
        assert bool(r) is True


# =============================================================================
# ── Style Presets Tests
# =============================================================================

class TestStylePresets:

    def test_all_presets_accessible(self):
        names = list_presets()
        assert len(names) >= 5
        for name in names:
            preset = get_preset(name)
            assert isinstance(preset, StylePreset)

    def test_streetwear_has_positive_tags(self):
        p = get_preset("streetwear")
        assert len(p.positive_tags) > 0

    def test_streetwear_has_negative_tags(self):
        p = get_preset("streetwear")
        assert len(p.negative_tags) > 0

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            get_preset("does_not_exist_xyz")

    def test_custom_preset_registration(self):
        custom = StylePreset(
            name         = "test_custom",
            display_name = "Test Custom",
            positive_tags= ["test tag"],
        )
        register_preset(custom)
        retrieved = get_preset("test_custom")
        assert retrieved.name == "test_custom"

    def test_preset_positive_string(self):
        p = get_preset("luxury")
        s = p.get_positive_string()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_preset_recommended_steps_positive(self):
        for name in list_presets():
            p = get_preset(name)
            assert p.recommended_steps > 0

    def test_preset_recommended_guidance_positive(self):
        for name in list_presets():
            p = get_preset(name)
            assert p.recommended_guidance > 0


# =============================================================================
# ── Negative Prompt Tests
# =============================================================================

class TestNegativePrompts:

    def test_base_negative_not_empty(self):
        neg = get_base_negative()
        assert len(neg) > 0
        assert all(isinstance(t, str) for t in neg)

    def test_fashion_negative_superset_of_base(self):
        base    = set(get_base_negative())
        fashion = set(get_fashion_negative())
        assert base.issubset(fashion)

    def test_full_negative_includes_nsfw(self):
        full = get_full_negative(include_nsfw=True)
        assert any("nsfw" in t.lower() for t in full)

    def test_full_negative_no_nsfw(self):
        full = get_full_negative(include_nsfw=False)
        assert not any("nsfw" in t.lower() for t in full)

    def test_style_negative_for_formal(self):
        neg = get_style_negative("formal")
        assert "casual" in neg or "streetwear" in neg

    def test_format_negative_deduplicates(self):
        tags   = ["blurry", "deformed", "blurry"]
        result = format_negative(tags)
        assert result.count("blurry") == 1

    def test_format_negative_separator(self):
        result = format_negative(["a", "b", "c"], separator=" | ")
        assert " | " in result

    def test_quality_negatives_list_not_empty(self):
        assert len(QUALITY_NEGATIVES) > 0
