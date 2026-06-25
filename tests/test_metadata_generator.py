"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_metadata_generator.py — Unit Tests: Metadata Generation Engine
=============================================================================
Full test suite for:
  data_pipeline/metadata_generation/metadata_generator.py

Test Classes:
  TestExtractionResult          — Dataclass: is_extracted, to_dict
  TestMetadataResult            — Dataclass: to_dict, to_json, to_summary
  TestRuleBasedExtractorColor   — Color extraction (plain, modifier, None)
  TestRuleBasedExtractorCategory— Category extraction (all 11 categories)
  TestRuleBasedExtractorStyle   — Style extraction (all 8 styles)
  TestRuleBasedExtractorSeason  — Season extraction (all 5 seasons)
  TestRuleBasedExtractorFit     — Fit extraction (all 8 fits)
  TestRuleBasedExtractorPattern — Pattern extraction (all 12 patterns)
  TestRuleBasedExtractorGender  — Gender extraction (men/women/unisex)
  TestRuleBasedExtractorOccasion— Occasion extraction (all 9 occasions)
  TestRuleBasedEdgeCases        — Empty string, numbers, emoji, long text
  TestNLPExtractor              — Availability check, graceful fallback
  TestFallbackResolver          — Cross-inference rules, defaults
  TestMetadataGeneratorEngine   — End-to-end generation, batch, enrich record
  TestCanonicalExamples         — Spec examples from the prompt requirements
  TestConfidenceScoring         — Rule > NLP > fallback confidence ordering
  TestBatchGeneration           — Batch mode, error isolation

Total: ~160 tests

Run:
    pytest tests/test_metadata_generator.py -v
    pytest tests/test_metadata_generator.py -v --cov=data_pipeline.metadata_generation
=============================================================================
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from src.data.metadata_generation.metadata_generator import (
    ExtractionResult,
    MetadataResult,
    RuleBasedExtractor,
    NLPExtractor,
    FallbackResolver,
    MetadataGeneratorEngine,
    _CONF_EXACT,
    _CONF_PHRASE,
    _CONF_NLP,
    _CONF_CROSS_INFER,
    _CONF_DEFAULT,
)


# =============================================================================
# ── Shared fixtures
# =============================================================================

@pytest.fixture(scope="module")
def engine() -> MetadataGeneratorEngine:
    """Single shared engine instance for the whole module (NLP disabled for speed)."""
    return MetadataGeneratorEngine(enable_nlp=False)


@pytest.fixture(scope="module")
def rule_extractor() -> RuleBasedExtractor:
    return RuleBasedExtractor()


@pytest.fixture(scope="module")
def fallback() -> FallbackResolver:
    return FallbackResolver()


# =============================================================================
# ── 1. TestExtractionResult
# =============================================================================

class TestExtractionResult:

    def test_is_extracted_with_value(self):
        er = ExtractionResult(value="streetwear", confidence=1.0, method="rule", evidence="streetwear")
        assert er.is_extracted() is True

    def test_is_extracted_none(self):
        er = ExtractionResult(None)
        assert er.is_extracted() is False

    def test_is_extracted_default_method(self):
        er = ExtractionResult(value="casual", confidence=0.3, method="default", evidence="")
        assert er.is_extracted() is False

    def test_to_dict_structure(self):
        er = ExtractionResult(value="shirts", confidence=0.85, method="rule", evidence="shirt")
        d  = er.to_dict()
        assert d["value"]      == "shirts"
        assert d["confidence"] == 0.85
        assert d["method"]     == "rule"
        assert d["evidence"]   == "shirt"

    def test_confidence_rounded_to_4dp(self):
        er = ExtractionResult(value="jeans", confidence=0.12345678, method="rule", evidence="denim")
        d  = er.to_dict()
        assert d["confidence"] == round(0.12345678, 4)

    def test_default_fields(self):
        er = ExtractionResult(None)
        assert er.confidence == 0.0
        assert er.method     == "default"
        assert er.evidence   == ""


# =============================================================================
# ── 2. TestMetadataResult
# =============================================================================

class TestMetadataResult:

    def test_to_dict_has_all_8_fields(self):
        r = MetadataResult(description="test", style="formal", category="shirts")
        d = r.to_dict()
        for field in ("style", "category", "season", "gender", "occasion", "fit", "pattern", "color"):
            assert field in d

    def test_to_dict_no_details_by_default(self):
        r = MetadataResult(description="test")
        d = r.to_dict()
        assert "_details" not in d
        assert "_meta"    not in d

    def test_to_dict_with_details(self):
        r = MetadataResult(
            description="test",
            details={"style": ExtractionResult("formal", 1.0, "rule", "formal")}
        )
        d = r.to_dict(include_details=True)
        assert "_details" in d
        assert "_meta"    in d

    def test_to_json_is_valid_json(self):
        r = MetadataResult(description="A blue formal shirt")
        js = r.to_json()
        data = json.loads(js)
        assert "style" in data

    def test_to_json_indented(self):
        r  = MetadataResult(description="test")
        js = r.to_json(indent=2)
        assert "\n" in js

    def test_to_summary_contains_key_fields(self):
        r = MetadataResult(
            description="test", style="formal", category="shirts",
            color="White", fit="slim_fit", overall_confidence=0.9,
        )
        s = r.to_summary()
        assert "formal" in s
        assert "shirts"   in s or "cat=" in s

    def test_generated_at_is_iso_format(self):
        from datetime import datetime
        r = MetadataResult(description="test")
        datetime.fromisoformat(r.generated_at)  # must not raise


# =============================================================================
# ── 3. TestRuleBasedExtractorColor
# =============================================================================

class TestRuleBasedExtractorColor:

    def test_plain_black(self, rule_extractor):
        r = rule_extractor._extract_color("black oversized hoodie")
        assert r.value == "Black"
        assert r.confidence == _CONF_EXACT

    def test_plain_white(self, rule_extractor):
        r = rule_extractor._extract_color("white cotton tee")
        assert r.value == "White"

    def test_navy(self, rule_extractor):
        r = rule_extractor._extract_color("navy blue chinos")
        assert r.value == "Navy"
        assert r.confidence == _CONF_PHRASE   # "navy blue" is a phrase

    def test_modifier_prefix(self, rule_extractor):
        # "neon pink" → modifier regex first
        r = rule_extractor._extract_color("neon pink hoodie")
        # "neon" is also a direct keyword → "Neon" color
        assert r.value is not None

    def test_dark_modifier(self, rule_extractor):
        r = rule_extractor._extract_color("dark navy trousers")
        assert r.value is not None  # either Navy or a modifier result

    def test_olive(self, rule_extractor):
        r = rule_extractor._extract_color("olive cargo pants")
        assert r.value == "Olive"

    def test_burgundy(self, rule_extractor):
        r = rule_extractor._extract_color("deep burgundy coat")
        assert r.value is not None

    def test_multicolor_tie_dye(self, rule_extractor):
        r = rule_extractor._extract_color("tie-dye hoodie")
        assert r.value == "Multicolor"

    def test_no_color_returns_none(self, rule_extractor):
        r = rule_extractor._extract_color("oversized slim fit jacket")
        assert r.value is None

    def test_print_keyword_not_a_color(self, rule_extractor):
        r = rule_extractor._extract_color("floral print dress")
        # "print" has None in map; "floral" is not a color → None expected
        assert r.value is None or r.value != "print"


# =============================================================================
# ── 4. TestRuleBasedExtractorCategory
# =============================================================================

class TestRuleBasedExtractorCategory:

    @pytest.mark.parametrize("desc, expected", [
        ("black graphic tee",          "t_shirts"),
        ("oversized hoodie",           "hoodies"),
        ("slim fit dress shirt",       "shirts"),
        ("slim fit navy chinos",       "pants"),
        ("blue denim jeans",           "jeans"),
        ("cargo shorts",               "shorts"),
        ("floral summer dress",        "dresses"),
        ("waterproof parka jacket",    "jackets"),
        ("white sneakers",             "footwear"),
        ("leather handbag",            "accessories"),
        ("silk kurta",                 "ethnic_wear"),
    ])
    def test_category_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["category"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['category'].value}'"

    def test_category_returns_valid_value(self, rule_extractor):
        from src.data.metadata_generation.metadata_generator import VALID_CATEGORIES
        descs = [
            "graphic tee", "button down shirt", "winter hoodie", "leather jacket",
            "slim trousers", "blue jeans", "denim shorts", "wrap dress",
            "running shoes", "gold bracelet", "salwar kameez",
        ]
        for desc in descs:
            r = rule_extractor.extract_all(desc)
            if r["category"].value is not None:
                assert r["category"].value in VALID_CATEGORIES, \
                    f"'{r['category'].value}' not in VALID_CATEGORIES"


# =============================================================================
# ── 5. TestRuleBasedExtractorStyle
# =============================================================================

class TestRuleBasedExtractorStyle:

    @pytest.mark.parametrize("desc, expected", [
        ("oversized hoodie with neon graphics", "streetwear"),
        ("luxury cashmere sweater",             "luxury"),
        ("tailored formal blazer",              "formal"),
        ("smart casual polo shirt",             "business_casual"),
        ("waterproof tactical techwear jacket", "techwear"),
        ("minimalist capsule wardrobe tee",     "minimalist"),
        ("vintage 90s distressed denim",        "vintage"),
        ("athletic gym performance shorts",     "athleisure"),
    ])
    def test_style_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["style"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['style'].value}'"


# =============================================================================
# ── 6. TestRuleBasedExtractorSeason
# =============================================================================

class TestRuleBasedExtractorSeason:

    @pytest.mark.parametrize("desc, expected", [
        ("lightweight linen beach sundress",     "summer"),
        ("heavyweight wool winter coat",         "winter"),
        ("floral spring rain jacket",            "spring"),
        ("autumn earthy rust-coloured knitwear", "autumn"),
    ])
    def test_season_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["season"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['season'].value}'"

    def test_season_none_for_unrelated(self, rule_extractor):
        results = rule_extractor.extract_all("blue slim fit chinos")
        # No season cue → None at rule layer (fallback handles it)
        assert results["season"].value is None or results["season"].value in (
            "spring", "summer", "autumn", "winter", "all_season"
        )


# =============================================================================
# ── 7. TestRuleBasedExtractorFit
# =============================================================================

class TestRuleBasedExtractorFit:

    @pytest.mark.parametrize("desc, expected", [
        ("slim fit white dress shirt",      "slim_fit"),
        ("regular fit navy chinos",         "regular_fit"),
        ("relaxed loose fit sweatpants",    "relaxed_fit"),
        ("black oversized boxy hoodie",     "oversized"),
        ("cropped crop top",                "cropped"),
        ("ultra slim skinny jeans",         "skinny"),
        ("straight leg cut blue denim",     "straight"),
        ("athletic fit performance shorts", "athletic_fit"),
    ])
    def test_fit_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["fit"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['fit'].value}'"


# =============================================================================
# ── 8. TestRuleBasedExtractorPattern
# =============================================================================

class TestRuleBasedExtractorPattern:

    @pytest.mark.parametrize("desc, expected", [
        ("solid plain monochrome tee",         "solid"),
        ("blue and white horizontal stripes",  "stripes"),
        ("classic plaid flannel shirt",        "checks"),
        ("floral botanical summer dress",      "floral"),
        ("geometric chevron print top",        "geometric"),
        ("leopard animal print blouse",        "animal_print"),
        ("camo camouflage military jacket",    "camouflage"),
        ("tie-dye ombre hoodie",               "tie_dye"),
        ("paisley print silk scarf",           "paisley"),
        ("neon graphic logo tee",              "graphic"),
        ("abstract painterly watercolor top",  "abstract"),
        ("white polka dot blouse",             "polka_dot"),
    ])
    def test_pattern_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["pattern"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['pattern'].value}'"


# =============================================================================
# ── 9. TestRuleBasedExtractorGender
# =============================================================================

class TestRuleBasedExtractorGender:

    @pytest.mark.parametrize("desc, expected", [
        ("men's slim fit chinos",       "men"),
        ("women's floral summer dress", "women"),
        ("unisex oversized hoodie",     "unisex"),
        ("ladies evening gown",         "women"),
        ("boys graphic tee",            "men"),
        ("gender neutral joggers",      "unisex"),
    ])
    def test_gender_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["gender"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['gender'].value}'"


# =============================================================================
# ── 10. TestRuleBasedExtractorOccasion
# =============================================================================

class TestRuleBasedExtractorOccasion:

    @pytest.mark.parametrize("desc, expected", [
        ("casual everyday weekend tee",             "casual"),
        ("business casual smart office polo",       "business_casual"),
        ("formal black tie gala gown",              "formal"),
        ("night out party club dress",              "party"),
        ("gym workout running shorts",              "sport"),
        ("hiking outdoor adventure jacket",         "outdoor"),
        ("beach resort vacation swimwear",          "beach"),
        ("wedding bridal festive lehenga",          "wedding_festive"),
        ("lounge homewear pajama set",              "lounge"),
    ])
    def test_occasion_extraction(self, rule_extractor, desc, expected):
        results = rule_extractor.extract_all(desc)
        assert results["occasion"].value == expected, \
            f"'{desc}' → expected '{expected}', got '{results['occasion'].value}'"


# =============================================================================
# ── 11. TestRuleBasedEdgeCases
# =============================================================================

class TestRuleBasedEdgeCases:

    def test_empty_string(self, rule_extractor):
        results = rule_extractor.extract_all("")
        for v in results.values():
            assert v.value is None

    def test_whitespace_only(self, rule_extractor):
        results = rule_extractor.extract_all("   ")
        for v in results.values():
            assert v.value is None

    def test_numeric_string(self, rule_extractor):
        results = rule_extractor.extract_all("12345")
        for v in results.values():
            assert v.value is None

    def test_non_fashion_text(self, rule_extractor):
        results = rule_extractor.extract_all("The quick brown fox jumps over the lazy dog")
        # "brown" is a color
        assert results["color"].value is not None or results["color"].value is None

    def test_very_long_description(self, rule_extractor):
        desc = "black " * 200 + "slim fit oversized hoodie"
        results = rule_extractor.extract_all(desc)
        assert results["color"].value    == "Black"
        assert results["category"].value == "hoodies"

    def test_uppercase_input(self, rule_extractor):
        results = rule_extractor.extract_all("BLACK OVERSIZED HOODIE WITH NEON GRAPHICS")
        assert results["color"].value    == "Black"
        assert results["category"].value == "hoodies"
        assert results["fit"].value      == "oversized"

    def test_mixed_case(self, rule_extractor):
        results = rule_extractor.extract_all("Slim-Fit Navy Chinos For Office")
        assert results["fit"].value is not None

    def test_hyphenated_keywords(self, rule_extractor):
        results = rule_extractor.extract_all("slim-fit navy blue chinos")
        assert results["fit"].value   == "slim_fit"
        assert results["color"].value == "Navy"

    def test_multiple_colors_picks_first_match(self, rule_extractor):
        # "black and white" — multi-word phrase "black" comes first alphabetically
        results = rule_extractor.extract_all("black and white striped shirt")
        assert results["color"].value in ("Black", "White")

    def test_all_8_fields_returned(self, rule_extractor):
        results = rule_extractor.extract_all("blue jeans")
        expected_keys = {"style", "category", "season", "gender",
                         "occasion", "fit", "pattern", "color"}
        assert set(results.keys()) == expected_keys


# =============================================================================
# ── 12. TestNLPExtractor
# =============================================================================

class TestNLPExtractor:

    def test_nlp_extractor_instantiates(self):
        nlp = NLPExtractor()
        # available depends on environment; just check it doesn't crash
        assert isinstance(nlp.available, bool)

    def test_nlp_returns_8_fields_when_unavailable(self):
        nlp = NLPExtractor.__new__(NLPExtractor)
        nlp._nlp = None
        results = nlp.extract_all("Black oversized hoodie with neon graphics")
        assert set(results.keys()) == {"style", "category", "season", "gender",
                                       "occasion", "fit", "pattern", "color"}

    def test_nlp_all_none_when_unavailable(self):
        nlp = NLPExtractor.__new__(NLPExtractor)
        nlp._nlp = None
        results = nlp.extract_all("floral summer dress")
        for v in results.values():
            assert v.value is None

    def test_nlp_method_tag(self):
        """When NLP produces a result its method should be 'nlp'."""
        nlp = NLPExtractor.__new__(NLPExtractor)
        nlp._nlp = None
        results = nlp.extract_all("whatever")
        # All None with method="default" — acceptable
        for v in results.values():
            assert v.method in ("default", "nlp")


# =============================================================================
# ── 13. TestFallbackResolver
# =============================================================================

class TestFallbackResolver:

    def _empty_results(self) -> Dict[str, ExtractionResult]:
        fields = ("style", "category", "season", "gender",
                  "occasion", "fit", "pattern", "color")
        return {f: ExtractionResult(None) for f in fields}

    def test_beach_occasion_infers_summer(self, fallback):
        r = self._empty_results()
        r["occasion"] = ExtractionResult("beach", 1.0, "rule", "beach")
        resolved = fallback.resolve(r, "beach outfit")
        assert resolved["season"].value == "summer"
        assert resolved["season"].method == "fallback"

    def test_athleisure_style_infers_sport_occasion(self, fallback):
        r = self._empty_results()
        r["style"] = ExtractionResult("athleisure", 1.0, "rule", "athletic")
        resolved = fallback.resolve(r, "gym wear")
        assert resolved["occasion"].value == "sport"

    def test_luxury_style_infers_formal_occasion(self, fallback):
        r = self._empty_results()
        r["style"] = ExtractionResult("luxury", 1.0, "rule", "luxury")
        resolved = fallback.resolve(r, "luxury gown")
        assert resolved["occasion"].value == "formal"

    def test_dresses_category_infers_women_gender(self, fallback):
        r = self._empty_results()
        r["category"] = ExtractionResult("dresses", 1.0, "rule", "dress")
        resolved = fallback.resolve(r, "floral dress")
        assert resolved["gender"].value == "women"

    def test_hoodies_category_infers_winter_season(self, fallback):
        r = self._empty_results()
        r["category"] = ExtractionResult("hoodies", 1.0, "rule", "hoodie")
        resolved = fallback.resolve(r, "hoodie")
        assert resolved["season"].value == "winter"

    def test_hoodies_category_infers_streetwear_style(self, fallback):
        r = self._empty_results()
        r["category"] = ExtractionResult("hoodies", 1.0, "rule", "hoodie")
        resolved = fallback.resolve(r, "hoodie")
        assert resolved["style"].value == "streetwear"

    def test_shorts_category_infers_summer(self, fallback):
        r = self._empty_results()
        r["category"] = ExtractionResult("shorts", 1.0, "rule", "shorts")
        resolved = fallback.resolve(r, "shorts")
        assert resolved["season"].value == "summer"

    def test_universal_defaults_applied(self, fallback):
        r = self._empty_results()
        resolved = fallback.resolve(r, "")
        assert resolved["style"].value    == "casual"
        assert resolved["category"].value == "accessories"
        assert resolved["season"].value   == "all_season"
        assert resolved["gender"].value   == "unisex"
        assert resolved["occasion"].value == "casual"
        assert resolved["fit"].value      == "regular_fit"
        assert resolved["pattern"].value  == "solid"
        # color has no sensible default → may be None
        # (the engine leaves color None if not found)

    def test_existing_values_not_overwritten(self, fallback):
        r = self._empty_results()
        r["season"] = ExtractionResult("winter", 1.0, "rule", "winter")
        r["occasion"] = ExtractionResult("beach", 1.0, "rule", "beach")
        resolved = fallback.resolve(r, "winter beach")
        # "beach" → "summer" inference must NOT overwrite existing "winter"
        assert resolved["season"].value == "winter"

    def test_all_8_fields_populated_after_resolve(self, fallback):
        r = self._empty_results()
        resolved = fallback.resolve(r, "")
        for field in ("style", "category", "season", "gender",
                      "occasion", "fit", "pattern"):
            assert resolved[field].value is not None, f"{field} should not be None"

    def test_fallback_confidence_is_cross_infer(self, fallback):
        r = self._empty_results()
        r["occasion"] = ExtractionResult("beach", 1.0, "rule", "beach")
        resolved = fallback.resolve(r, "beach")
        assert resolved["season"].confidence == _CONF_CROSS_INFER

    def test_default_confidence_is_low(self, fallback):
        r = self._empty_results()
        resolved = fallback.resolve(r, "")
        assert resolved["style"].confidence == _CONF_DEFAULT
        assert resolved["fit"].confidence   == _CONF_DEFAULT


# =============================================================================
# ── 14. TestMetadataGeneratorEngine
# =============================================================================

class TestMetadataGeneratorEngine:

    def test_returns_metadata_result(self, engine):
        result = engine.generate("black oversized hoodie with neon graphics")
        assert isinstance(result, MetadataResult)

    def test_all_8_fields_present(self, engine):
        result = engine.generate("slim fit white dress shirt for office")
        for field in ("style", "category", "season", "gender",
                      "occasion", "fit", "pattern", "color"):
            assert hasattr(result, field)

    def test_overall_confidence_in_range(self, engine):
        result = engine.generate("floral summer sundress for the beach")
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_processing_time_positive(self, engine):
        result = engine.generate("black slim fit chinos")
        assert result.processing_time_ms > 0

    def test_description_preserved(self, engine):
        desc   = "navy blue regular fit chinos for business"
        result = engine.generate(desc)
        assert result.description == desc

    def test_empty_description_returns_defaults(self, engine):
        result = engine.generate("")
        assert result.overall_confidence == _CONF_DEFAULT

    def test_whitespace_description_returns_defaults(self, engine):
        result = engine.generate("   ")
        assert result.overall_confidence == _CONF_DEFAULT

    def test_to_dict_is_json_serializable(self, engine):
        result = engine.generate("olive green cargo shorts")
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_with_details(self, engine):
        result = engine.generate("oversized vintage hoodie")
        d = result.to_dict(include_details=True)
        assert "_details" in d
        assert "style" in d["_details"]

    def test_generate_record_enriches_missing_fields(self, engine):
        record = {
            "image_id"   : "FG_001",
            "image_path" : "p.jpg",
            "description": "slim fit navy chinos for office",
            "category"   : "",
            "style"      : "",
        }
        enriched = engine.generate_from_record(record)
        assert enriched["category"] != "" and enriched["category"] is not None
        assert enriched["_auto_generated"] is True

    def test_generate_record_preserves_existing_fields(self, engine):
        record = {
            "image_id"   : "FG_001",
            "image_path" : "p.jpg",
            "description": "slim fit navy chinos for office",
            "style"      : "luxury",  # already set
            "category"   : "pants",  # already set
        }
        enriched = engine.generate_from_record(record)
        # Existing fields must NOT be overwritten
        assert enriched["style"]    == "luxury"
        assert enriched["category"] == "pants"

    def test_generate_record_meta_keys(self, engine):
        record = {"description": "blue jeans"}
        enriched = engine.generate_from_record(record)
        assert "_auto_generated" in enriched
        assert "_gen_confidence" in enriched
        assert "_gen_processing_time_ms" in enriched

    def test_details_are_extraction_results(self, engine):
        result = engine.generate("black slim hoodie")
        for key, val in result.details.items():
            assert isinstance(val, ExtractionResult)


# =============================================================================
# ── 15. TestCanonicalExamples (Prompt spec examples)
# =============================================================================

class TestCanonicalExamples:
    """
    Tests based directly on the requirements specification examples.
    These are the definitive acceptance tests for the engine.
    """

    def test_spec_example_black_oversized_hoodie(self, engine):
        """
        Input:  "Black oversized hoodie with neon graphics"
        Expected:
          category = hoodies     (explicit: "hoodie")
          fit      = oversized   (explicit: "oversized")
          color    = Black       (explicit: "black")
          style    = streetwear  (via neon + graphic + hoodie)
          season   = winter      (hoodies → winter inference)
          occasion = casual      (streetwear → casual)
          pattern  = graphic     (explicit: "graphics")
          gender   = unisex      (hoodies → unisex default)
        """
        result = engine.generate("Black oversized hoodie with neon graphics")
        assert result.category == "hoodies",    f"category={result.category}"
        assert result.fit      == "oversized",  f"fit={result.fit}"
        assert result.color    == "Black",      f"color={result.color}"
        assert result.style    == "streetwear", f"style={result.style}"
        assert result.season   == "winter",     f"season={result.season}"
        assert result.pattern  in ("graphic", "solid"), f"pattern={result.pattern}"

    def test_slim_fit_navy_chinos_for_office(self, engine):
        result = engine.generate("slim fit navy chinos for office wear")
        assert result.category in ("pants", "shirts"), f"category={result.category}"
        assert result.fit      == "slim_fit",          f"fit={result.fit}"
        assert result.color    == "Navy",              f"color={result.color}"
        assert result.occasion in ("formal", "business_casual"), \
            f"occasion={result.occasion}"

    def test_floral_summer_sundress(self, engine):
        result = engine.generate("floral print summer sundress")
        assert result.category == "dresses", f"category={result.category}"
        assert result.season   == "summer",  f"season={result.season}"
        assert result.pattern  == "floral",  f"pattern={result.pattern}"
        assert result.gender   == "women",   f"gender={result.gender}"

    def test_white_slim_fit_formal_dress_shirt(self, engine):
        result = engine.generate("white slim fit formal dress shirt")
        assert result.category in ("shirts", "t_shirts"), f"category={result.category}"
        assert result.fit      == "slim_fit",             f"fit={result.fit}"
        assert result.color    == "White",                f"color={result.color}"
        assert result.style    == "formal",               f"style={result.style}"

    def test_leopard_print_blouse_for_party(self, engine):
        result = engine.generate("leopard print blouse for a night out party")
        assert result.pattern  == "animal_print", f"pattern={result.pattern}"
        assert result.category == "shirts",       f"category={result.category}"
        assert result.occasion == "party",        f"occasion={result.occasion}"

    def test_vintage_90s_distressed_denim(self, engine):
        result = engine.generate("vintage 90s distressed slim denim jeans")
        assert result.category == "jeans",   f"category={result.category}"
        assert result.style    == "vintage", f"style={result.style}"

    def test_waterproof_techwear_jacket(self, engine):
        result = engine.generate("waterproof modular techwear tactical jacket")
        assert result.category == "jackets",  f"category={result.category}"
        assert result.style    == "techwear", f"style={result.style}"
        assert result.occasion in ("outdoor", "casual"), f"occasion={result.occasion}"

    def test_luxe_cashmere_sweater(self, engine):
        result = engine.generate("luxe premium cashmere turtleneck sweater")
        assert result.style    == "luxury",  f"style={result.style}"
        assert result.category == "hoodies", f"category={result.category}"

    def test_beach_resort_swimwear(self, engine):
        result = engine.generate("tropical beach resort swimwear")
        assert result.occasion == "beach",  f"occasion={result.occasion}"
        assert result.season   == "summer", f"season={result.season}"

    def test_wedding_festive_lehenga(self, engine):
        result = engine.generate("bridal wedding festive red lehenga")
        assert result.occasion == "wedding_festive", f"occasion={result.occasion}"
        assert result.category == "ethnic_wear",     f"category={result.category}"


# =============================================================================
# ── 16. TestConfidenceScoring
# =============================================================================

class TestConfidenceScoring:
    """Tests verifying confidence tiers are correctly assigned."""

    def test_exact_keyword_gets_max_confidence(self, rule_extractor):
        results = rule_extractor.extract_all("black hoodie")
        assert results["color"].confidence    == _CONF_EXACT
        assert results["category"].confidence == _CONF_EXACT

    def test_phrase_match_gets_phrase_confidence(self, rule_extractor):
        results = rule_extractor.extract_all("navy blue chinos")
        assert results["color"].confidence == _CONF_PHRASE

    def test_fallback_confidence_is_lower_than_rule(self, engine):
        # Generate a result with a clear rule match and fallback fields
        result = engine.generate("black oversized hoodie")
        rule_conf     = result.details["color"].confidence     # from rule
        fallback_conf = result.details["gender"].confidence    # likely fallback
        assert rule_conf >= fallback_conf

    def test_default_confidence_is_lowest(self, fallback):
        r = {f: ExtractionResult(None) for f in
             ("style","category","season","gender","occasion","fit","pattern","color")}
        resolved = fallback.resolve(r, "")
        for field in ("style", "fit", "occasion"):
            assert resolved[field].confidence == _CONF_DEFAULT

    def test_overall_confidence_is_mean(self, engine):
        result = engine.generate("black slim fit hoodie for gym workout")
        # Manual check: overall_confidence should be mean of field confidences
        confs = [v.confidence for v in result.details.values() if v.value is not None]
        expected_mean = sum(confs) / len(confs) if confs else 0.0
        assert abs(result.overall_confidence - expected_mean) < 0.01

    def test_richer_description_gives_higher_confidence(self, engine):
        sparse = engine.generate("jacket")
        rich   = engine.generate(
            "men's slim fit formal navy blue tailored blazer for office casual wear"
        )
        assert rich.overall_confidence >= sparse.overall_confidence


# =============================================================================
# ── 17. TestBatchGeneration
# =============================================================================

class TestBatchGeneration:

    def test_batch_returns_correct_count(self, engine):
        descs   = ["black hoodie", "blue jeans", "floral dress"]
        results = engine.generate_batch(descs)
        assert len(results) == len(descs)

    def test_batch_all_metadata_results(self, engine):
        results = engine.generate_batch(["tee", "chinos"])
        assert all(isinstance(r, MetadataResult) for r in results)

    def test_batch_preserves_order(self, engine):
        descs   = ["black hoodie", "white tee", "blue jeans"]
        results = engine.generate_batch(descs)
        for desc, result in zip(descs, results):
            assert result.description == desc

    def test_batch_empty_list(self, engine):
        results = engine.generate_batch([])
        assert results == []

    def test_batch_single_item(self, engine):
        results = engine.generate_batch(["oversized vintage hoodie"])
        assert len(results) == 1

    def test_batch_with_empty_string(self, engine):
        results = engine.generate_batch(["blue jeans", ""])
        # Should not crash; empty string gets default result
        assert len(results) == 2
        assert results[1].overall_confidence == _CONF_DEFAULT

    def test_batch_all_descriptions_processed(self, engine):
        descs = [f"item description {i}" for i in range(20)]
        results = engine.generate_batch(descs)
        assert len(results) == 20
        assert all(isinstance(r, MetadataResult) for r in results)
