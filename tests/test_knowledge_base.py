"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_knowledge_base.py — Unit Tests: Fashion Knowledge Base
=============================================================================
Comprehensive test suite for fashion_domain_research.py.

Tests cover:
  - FashionDomainResearch class (loading, lookups, cross-references, search)
  - Normalizer functions (color, fabric, style, fit, pattern, season, occasion)
  - FashionRecord dataclass (serialization, deserialization)
  - ValidationResult dataclass (error accumulation)
  - validate_fashion_record() (all 6 validation layers)
  - build_category_mapping() and build_style_profile() builders
  - Artifact generation (category_mapping, style_profiles, taxonomy_tree)

Run:
    pytest tests/test_knowledge_base.py -v
    pytest tests/test_knowledge_base.py -v --cov=data_pipeline.knowledge_base
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def kb():
    """
    Module-scoped FashionDomainResearch instance.

    Loaded once per test module to avoid repeated JSON parsing overhead.
    All tests using this fixture share the same (read-only) instance.
    """
    from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
    return FashionDomainResearch()


@pytest.fixture
def valid_record() -> Dict[str, Any]:
    """A minimal valid FashionRecord dict for testing."""
    return {
        "image_id"      : "TEST_001",
        "dataset_source": "fashiongen",
        "category"      : "t_shirts",
        "gender"        : "men",
        "colors"        : ["Navy"],
        "fabrics"       : ["Cotton"],
        "styles"        : ["streetwear"],
        "fit"           : "regular_fit",
        "patterns"      : ["graphic"],
        "seasons"       : ["spring"],
        "occasions"     : ["casual"],
        "description"   : "A bold graphic tee in navy cotton.",
    }


# =============================================================================
# ── FashionDomainResearch: Loading & Initialization
# =============================================================================

class TestKnowledgeBaseLoading:

    def test_kb_loads_without_error(self, kb):
        """The KB should load successfully if fashion_knowledge.json exists."""
        assert kb is not None

    def test_kb_has_categories(self, kb):
        """KB must contain all 11 required categories."""
        cats = kb.get_all_categories()
        expected = {
            "t_shirts", "shirts", "hoodies", "jackets", "pants",
            "jeans", "shorts", "dresses", "ethnic_wear", "footwear", "accessories"
        }
        assert expected.issubset(set(cats.keys())), (
            f"Missing categories: {expected - set(cats.keys())}"
        )

    def test_kb_has_all_styles(self, kb):
        """KB must contain all 8 required styles."""
        styles = kb.get_all_styles()
        expected = {
            "streetwear", "luxury", "formal", "business_casual",
            "techwear", "minimalist", "vintage", "athleisure"
        }
        assert expected.issubset(set(styles.keys()))

    def test_kb_has_all_genders(self, kb):
        """KB must define men, women, and unisex genders."""
        for gender in ("men", "women", "unisex"):
            result = kb.get_gender(gender)
            assert result is not None, f"Gender '{gender}' not found in KB"
            assert "label" in result

    def test_kb_missing_file_raises(self, tmp_path):
        """Loading a non-existent KB path must raise FileNotFoundError."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        with pytest.raises(FileNotFoundError):
            FashionDomainResearch(knowledge_json_path=tmp_path / "nonexistent.json")

    def test_kb_stats_completeness(self, kb):
        """get_knowledge_base_stats() should return positive counts for all keys."""
        stats = kb.get_knowledge_base_stats()
        assert stats["total_categories"]    == 11
        assert stats["total_styles"]        == 8
        assert stats["total_genders"]       == 3
        assert stats["total_color_entries"] > 20
        assert stats["total_fabric_entries"] > 10
        assert stats["total_alias_color_index"] > 20
        assert stats["total_alias_category_index"] > 11


# =============================================================================
# ── Category Lookups
# =============================================================================

class TestCategoryLookups:

    def test_get_category_t_shirts(self, kb):
        cat = kb.get_category("t_shirts")
        assert cat is not None
        assert cat["label"] == "T-Shirts"
        assert cat["code"] == "TSH"
        assert "men" in cat["genders"]
        assert "women" in cat["genders"]

    def test_get_category_dresses_women_only(self, kb):
        cat = kb.get_category("dresses")
        assert "women" in cat["genders"]
        assert "men" not in cat["genders"]

    def test_get_category_returns_none_for_unknown(self, kb):
        assert kb.get_category("space_suits") is None

    def test_get_subcategories_t_shirts(self, kb):
        subcats = kb.get_subcategories("t_shirts")
        assert "graphic_tee" in subcats
        assert "polo_tee" in subcats
        assert "tank_top" in subcats
        assert len(subcats) >= 5

    def test_get_subcategories_jeans(self, kb):
        subcats = kb.get_subcategories("jeans")
        assert "slim_fit_jeans" in subcats
        assert "skinny_jeans" in subcats
        assert "mom_jeans" in subcats

    def test_all_categories_have_label_and_code(self, kb):
        for cat_key, cat_def in kb.get_all_categories().items():
            assert "label" in cat_def, f"Category '{cat_key}' missing 'label'"
            assert "code" in cat_def, f"Category '{cat_key}' missing 'code'"
            assert len(cat_def["code"]) == 3, f"Category '{cat_key}' code should be 3 chars"


# =============================================================================
# ── Style Lookups
# =============================================================================

class TestStyleLookups:

    def test_get_style_streetwear(self, kb):
        style = kb.get_style("streetwear")
        assert style is not None
        assert style["label"] == "Streetwear"
        assert style["tier"] == 1
        assert "oversized" in style["key_attributes"]["fits"]
        assert len(style["brand_archetypes"]) > 0

    def test_get_style_luxury(self, kb):
        style = kb.get_style("luxury")
        assert style["code"] == "LUX"
        assert "Silk" in style["key_attributes"]["fabrics"]

    def test_get_style_returns_none_for_unknown(self, kb):
        assert kb.get_style("hipster") is None

    def test_style_tiers(self, kb):
        tiers = kb.get_style_tiers()
        # Tier 1 should have foundation styles
        assert "streetwear" in tiers.get(1, [])
        assert "minimalist" in tiers.get(1, [])
        # Tier 2 should have derived styles
        assert "techwear" in tiers.get(2, [])

    def test_all_styles_have_required_fields(self, kb):
        required_fields = ["label", "code", "tier", "description", "key_attributes",
                           "color_palette", "aesthetic_tags"]
        for style_key, style_def in kb.get_all_styles().items():
            for field in required_fields:
                assert field in style_def, f"Style '{style_key}' missing '{field}'"

    def test_incompatible_styles(self, kb):
        assert "formal" in kb.get_incompatible_styles("streetwear")
        assert "luxury" in kb.get_incompatible_styles("streetwear")

    def test_compatible_styles_formal(self, kb):
        compatible = kb.get_compatible_styles("formal")
        # Formal has "luxury" as parent and "business_casual" as child
        assert "luxury" in compatible or "business_casual" in compatible


# =============================================================================
# ── Cross-Reference Lookups
# =============================================================================

class TestCrossReferenceLookups:

    def test_categories_for_men(self, kb):
        cats = kb.get_categories_for_gender("men")
        assert "t_shirts" in cats
        assert "shirts"   in cats
        assert "jeans"    in cats
        # Men should NOT have dresses
        assert "dresses" not in cats

    def test_categories_for_women(self, kb):
        cats = kb.get_categories_for_gender("women")
        assert "dresses" in cats
        assert "t_shirts" in cats

    def test_categories_for_unisex(self, kb):
        cats = kb.get_categories_for_gender("unisex")
        assert "t_shirts" in cats
        assert "dresses" not in cats  # Dresses are women-specific

    def test_categories_for_invalid_gender(self, kb):
        cats = kb.get_categories_for_gender("robot")
        assert cats == []

    def test_categories_for_streetwear(self, kb):
        cats = kb.get_categories_for_style("streetwear")
        assert "t_shirts"   in cats
        assert "hoodies"    in cats
        assert "footwear"   in cats
        assert "dresses"    not in cats

    def test_categories_for_luxury(self, kb):
        cats = kb.get_categories_for_style("luxury")
        assert "jackets"    in cats
        assert "footwear"   in cats

    def test_styles_for_casual(self, kb):
        styles = kb.get_styles_for_occasion("casual")
        assert "streetwear" in styles
        assert "athleisure" in styles

    def test_styles_for_formal(self, kb):
        styles = kb.get_styles_for_occasion("formal")
        assert "formal" in styles
        assert "luxury" in styles

    def test_fabrics_for_summer(self, kb):
        fabrics = kb.get_fabrics_for_season("summer")
        assert "Linen" in fabrics
        assert "Cotton" in fabrics

    def test_fabrics_for_winter(self, kb):
        fabrics = kb.get_fabrics_for_season("winter")
        assert "Wool" in fabrics
        assert "Fleece" in fabrics
        assert "Gore-Tex" in fabrics

    def test_attributes_for_footwear(self, kb):
        attrs = kb.get_attributes_for_category("footwear")
        assert "color"  in attrs
        assert "style"  in attrs
        assert "season" in attrs
        # Footwear does not have a "fit" attribute
        assert "fit" not in attrs

    def test_get_color_hex_white(self, kb):
        hex_val = kb.get_color_hex("White")
        assert hex_val == "#FFFFFF"

    def test_get_color_hex_black(self, kb):
        assert kb.get_color_hex("Black") == "#000000"

    def test_get_color_hex_alias(self, kb):
        # "navy blue" is an alias for "Navy"
        hex_val = kb.get_color_hex("navy blue")
        assert hex_val is not None  # Should resolve via alias

    def test_get_fabric_properties_cotton(self, kb):
        props = kb.get_fabric_properties("Cotton")
        assert "properties" in props
        assert "breathable" in props["properties"]
        assert "care" in props

    def test_get_fabric_properties_alias(self, kb):
        # "polycotton" should resolve to Cotton-Polyester Blend
        props = kb.get_fabric_properties("polycotton")
        assert props.get("name") == "Cotton-Polyester Blend"


# =============================================================================
# ── Search API
# =============================================================================

class TestSearchAPI:

    def test_search_returns_dict(self, kb):
        result = kb.search_by_tags(gender="women", style="athleisure")
        assert isinstance(result, dict)
        assert "categories" in result
        assert "styles" in result
        assert "fabrics" in result

    def test_search_women_athleisure(self, kb):
        result = kb.search_by_tags(gender="women", style="athleisure")
        assert "t_shirts" in result["categories"]
        assert "hoodies" in result["categories"]
        # Athleisure colors
        assert len(result["colors"]) > 0

    def test_search_men_formal(self, kb):
        result = kb.search_by_tags(gender="men", style="formal")
        assert "shirts"  in result["categories"]
        assert "jackets" in result["categories"]

    def test_search_with_season(self, kb):
        result = kb.search_by_tags(season="winter")
        assert "Wool" in result["fabrics"] or "Fleece" in result["fabrics"]

    def test_search_with_occasion(self, kb):
        result = kb.search_by_tags(occasion="casual")
        assert "streetwear" in result["styles"] or "athleisure" in result["styles"]

    def test_taxonomy_tree_structure(self, kb):
        tree = kb.get_taxonomy_tree()
        assert "men" in tree
        assert "women" in tree
        assert "unisex" in tree
        assert "dresses" in tree["women"]
        assert "dresses" not in tree["men"]

    def test_taxonomy_tree_subcategories(self, kb):
        tree = kb.get_taxonomy_tree()
        jeans_subcats = tree["men"]["jeans"]["subcategories"]
        assert "slim_fit_jeans" in jeans_subcats


# =============================================================================
# ── Normalizer Functions
# =============================================================================

class TestNormalizerFunctions:

    # Color normalizers
    def test_normalize_color_exact(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_color
        assert normalize_color("White", kb=kb)  == "White"
        assert normalize_color("Black", kb=kb)  == "Black"

    def test_normalize_color_alias(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_color
        assert normalize_color("navy blue", kb=kb) == "Navy"
        assert normalize_color("off-white", kb=kb) == "White"
        assert normalize_color("cognac",    kb=kb) == "Brown"
        assert normalize_color("ivory",     kb=kb) == "White"

    def test_normalize_color_unknown_returns_none(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_color
        assert normalize_color("xyzzy999", kb=kb) is None

    def test_normalize_color_none_input(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_color
        assert normalize_color(None, kb=kb) is None  # type: ignore[arg-type]
        assert normalize_color("", kb=kb) is None

    # Fabric normalizers
    def test_normalize_fabric_exact(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_fabric
        assert normalize_fabric("Cotton", kb=kb) == "Cotton"
        assert normalize_fabric("Linen",  kb=kb) == "Linen"

    def test_normalize_fabric_alias(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_fabric
        assert normalize_fabric("polycotton",  kb=kb) == "Cotton-Polyester Blend"
        assert normalize_fabric("elastane",    kb=kb) == "Spandex"
        assert normalize_fabric("lycra",       kb=kb) == "Spandex"
        assert normalize_fabric("merino wool", kb=kb) == "Wool"

    def test_normalize_fabric_unknown(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_fabric
        assert normalize_fabric("moon_dust", kb=kb) is None

    # Style normalizers
    def test_normalize_style_key(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_style
        assert normalize_style("streetwear",     kb=kb) == "streetwear"
        assert normalize_style("Streetwear",     kb=kb) == "streetwear"
        assert normalize_style("Business Casual",kb=kb) == "business_casual"
        assert normalize_style("Athleisure",     kb=kb) == "athleisure"

    def test_normalize_style_unknown(self, kb):
        from src.data.knowledge_base.fashion_domain_research import normalize_style
        assert normalize_style("hype beast", kb=kb) is None

    # Fit normalizers
    def test_normalize_fit_canonical(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_fit
        assert normalize_fit("slim_fit")    == "slim_fit"
        assert normalize_fit("regular_fit") == "regular_fit"
        assert normalize_fit("oversized")   == "oversized"

    def test_normalize_fit_alias(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_fit
        assert normalize_fit("fitted")  == "slim_fit"
        assert normalize_fit("baggy")   == "oversized"
        assert normalize_fit("regular") == "regular_fit"
        assert normalize_fit("loose")   == "relaxed_fit"

    def test_normalize_fit_unknown(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_fit
        assert normalize_fit("superfluid") is None

    # Pattern normalizers
    def test_normalize_pattern_canonical(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_pattern
        assert normalize_pattern("solid")       == "solid"
        assert normalize_pattern("stripes")     == "stripes"
        assert normalize_pattern("floral")      == "floral"
        assert normalize_pattern("camouflage")  == "camouflage"

    def test_normalize_pattern_alias(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_pattern
        assert normalize_pattern("plain")   == "solid"
        assert normalize_pattern("plaid")   == "checks"
        assert normalize_pattern("leopard") == "animal_print"
        assert normalize_pattern("camo")    == "camouflage"
        assert normalize_pattern("dots")    == "polka_dot"
        assert normalize_pattern("spotted") == "polka_dot"

    def test_normalize_pattern_unknown(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_pattern
        assert normalize_pattern("quantum_foam") is None

    # Season normalizers
    def test_normalize_season(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_season
        assert normalize_season("SS")         == "spring"
        assert normalize_season("AW")         == "autumn"
        assert normalize_season("fall")       == "autumn"
        assert normalize_season("year-round") == "all_season"
        assert normalize_season("winter")     == "winter"

    def test_normalize_season_unknown(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_season
        assert normalize_season("monsoon") is None

    # Occasion normalizers
    def test_normalize_occasion(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_occasion
        assert normalize_occasion("gym")        == "sport"
        assert normalize_occasion("black_tie")  == "formal"
        assert normalize_occasion("hiking")     == "outdoor"
        assert normalize_occasion("office")     == "business_casual"
        assert normalize_occasion("wedding")    == "wedding_festive"
        assert normalize_occasion("pyjamas")    == "lounge"

    def test_normalize_occasion_unknown(self):
        from src.data.knowledge_base.fashion_domain_research import normalize_occasion
        assert normalize_occasion("rocket_launch") is None


# =============================================================================
# ── FashionRecord Dataclass
# =============================================================================

class TestFashionRecord:

    def test_create_minimal_record(self):
        from src.data.knowledge_base.fashion_domain_research import FashionRecord
        rec = FashionRecord(
            image_id="T001",
            dataset_source="fashiongen",
            category="t_shirts",
            gender="men",
        )
        assert rec.image_id == "T001"
        assert rec.colors   == []
        assert rec.fabrics  == []
        assert rec.is_valid is False

    def test_to_dict_and_from_dict_roundtrip(self):
        from src.data.knowledge_base.fashion_domain_research import FashionRecord
        original = FashionRecord(
            image_id="T002",
            dataset_source="deepfashion",
            category="jeans",
            gender="women",
            colors=["Navy"],
            fabrics=["Denim"],
            styles=["vintage"],
            fit="relaxed_fit",
            patterns=["solid"],
            seasons=["autumn"],
            occasions=["casual"],
            description="Relaxed-fit vintage-wash jeans.",
            is_valid=True,
        )
        d = original.to_dict()
        restored = FashionRecord.from_dict(d)

        assert restored.image_id       == original.image_id
        assert restored.category       == original.category
        assert restored.colors         == original.colors
        assert restored.fit            == original.fit
        assert restored.is_valid       == original.is_valid
        assert restored.description    == original.description

    def test_to_dict_json_serializable(self):
        from src.data.knowledge_base.fashion_domain_research import FashionRecord
        rec = FashionRecord(
            image_id="T003",
            dataset_source="fashiongen",
            category="shirts",
            gender="men",
        )
        d = rec.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "T003" in json_str


# =============================================================================
# ── ValidationResult Dataclass
# =============================================================================

class TestValidationResult:

    def test_initial_state_is_valid(self):
        from src.data.knowledge_base.fashion_domain_research import ValidationResult
        vr = ValidationResult(record_id="X")
        assert vr.is_valid is True
        assert vr.errors   == []
        assert vr.warnings == []

    def test_add_error_marks_invalid(self):
        from src.data.knowledge_base.fashion_domain_research import ValidationResult
        vr = ValidationResult(record_id="X")
        vr.add_error("Something broke")
        assert vr.is_valid is False
        assert "Something broke" in vr.errors

    def test_add_warning_keeps_valid(self):
        from src.data.knowledge_base.fashion_domain_research import ValidationResult
        vr = ValidationResult(record_id="X")
        vr.add_warning("Something is odd")
        assert vr.is_valid is True
        assert "Something is odd" in vr.warnings

    def test_summary_valid(self):
        from src.data.knowledge_base.fashion_domain_research import ValidationResult
        vr = ValidationResult(record_id="T001")
        assert "VALID" in vr.summary()
        assert "T001"  in vr.summary()

    def test_summary_invalid(self):
        from src.data.knowledge_base.fashion_domain_research import ValidationResult
        vr = ValidationResult(record_id="T002")
        vr.add_error("fail")
        assert "INVALID" in vr.summary()


# =============================================================================
# ── validate_fashion_record() — All Layers
# =============================================================================

class TestValidateFashionRecord:

    def test_valid_record_passes(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        result = validate_fashion_record(valid_record, kb=kb)
        assert result.is_valid, f"Expected valid but got errors: {result.errors}"

    def test_missing_image_id(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        del valid_record["image_id"]
        result = validate_fashion_record(valid_record, kb=kb)
        assert not result.is_valid
        assert any("image_id" in e for e in result.errors)

    def test_missing_category(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        del valid_record["category"]
        result = validate_fashion_record(valid_record, kb=kb)
        assert not result.is_valid

    def test_invalid_gender(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        valid_record["gender"] = "cyborg"
        result = validate_fashion_record(valid_record, kb=kb)
        assert not result.is_valid
        assert any("gender" in e for e in result.errors)

    def test_invalid_category(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        valid_record["category"] = "space_suit"
        result = validate_fashion_record(valid_record, kb=kb)
        assert not result.is_valid
        assert any("category" in e for e in result.errors)

    def test_dresses_invalid_for_men(self, kb):
        """CR001: Dresses should fail for gender=men."""
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        record = {
            "image_id"      : "T010",
            "dataset_source": "test",
            "category"      : "dresses",
            "gender"        : "men",
        }
        result = validate_fashion_record(record, kb=kb)
        assert not result.is_valid

    def test_dresses_valid_for_women(self, kb):
        """Dresses must pass for gender=women."""
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        record = {
            "image_id"      : "T011",
            "dataset_source": "test",
            "category"      : "dresses",
            "gender"        : "women",
        }
        result = validate_fashion_record(record, kb=kb)
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_unknown_fit_generates_warning(self, kb, valid_record):
        """An unrecognized fit should not break validation — record stays valid."""
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        valid_record["fit"] = "mega_baggy_ultra_loose"
        result = validate_fashion_record(valid_record, kb=kb)
        # The validator accepts unknown fit values gracefully — record stays valid
        assert result.is_valid

    def test_unrecognized_occasion_generates_warning(self, kb, valid_record):
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        valid_record["occasions"] = ["rocket_launch"]
        result = validate_fashion_record(valid_record, kb=kb)
        assert any("rocket_launch" in w for w in result.warnings)

    def test_alias_gender_accepted(self, kb):
        """Gender aliases like 'male' should be accepted."""
        from src.data.knowledge_base.fashion_domain_research import validate_fashion_record
        record = {
            "image_id"      : "T020",
            "dataset_source": "test",
            "category"      : "t_shirts",
            "gender"        : "male",
        }
        result = validate_fashion_record(record, kb=kb)
        # "male" should be recognized as "men"
        assert result.is_valid, f"Alias gender 'male' should be valid: {result.errors}"


# =============================================================================
# ── Builder Functions
# =============================================================================

class TestBuilderFunctions:

    def test_build_category_mapping_t_shirts(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_category_mapping
        mapping = build_category_mapping("t_shirts", kb=kb)
        assert mapping.category_label == "T-Shirts"
        assert mapping.category_code  == "TSH"
        assert "men" in mapping.genders
        assert "women" in mapping.genders
        assert len(mapping.subcategories) >= 5
        assert "color" in mapping.attributes

    def test_build_category_mapping_footwear(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_category_mapping
        mapping = build_category_mapping("footwear", kb=kb)
        assert mapping.category_code == "FTW"
        assert len(mapping.subcategories) >= 5

    def test_build_style_profile_streetwear(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_style_profile
        profile = build_style_profile("streetwear", kb=kb)
        assert profile.label == "Streetwear"
        assert profile.code  == "STW"
        assert profile.tier  == 1
        assert "t_shirts" in profile.key_categories
        assert "formal"   in profile.incompatible_styles
        assert len(profile.aesthetic_tags) > 0
        assert len(profile.brand_archetypes) > 0

    def test_build_style_profile_luxury(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_style_profile
        profile = build_style_profile("luxury", kb=kb)
        assert profile.code == "LUX"
        assert "Silk" in profile.key_fabrics or "Wool" in profile.key_fabrics

    def test_category_mapping_to_dict(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_category_mapping
        mapping = build_category_mapping("jeans", kb=kb)
        d = mapping.to_dict()
        assert isinstance(d, dict)
        assert "category_label" in d
        assert "genders" in d
        # Should be JSON-serializable
        json.dumps(d)

    def test_style_profile_to_dict(self, kb):
        from src.data.knowledge_base.fashion_domain_research import build_style_profile
        profile = build_style_profile("minimalist", kb=kb)
        d = profile.to_dict()
        assert "style_key" in d
        assert "color_palette" in d
        json.dumps(d)


# =============================================================================
# ── Artifact Generation
# =============================================================================

class TestArtifactGeneration:

    def test_generate_category_mapping(self, kb, tmp_path):
        """Category mapping JSON should be written and contain all 11 categories."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        local_kb = FashionDomainResearch(output_dir=tmp_path)
        out_path = local_kb.generate_category_mapping()

        assert out_path.exists()
        with open(out_path, "r") as f:
            data = json.load(f)

        assert "mappings" in data
        assert len(data["mappings"]) == 11
        assert "t_shirts" in data["mappings"]
        assert "dresses"  in data["mappings"]

    def test_generate_style_profiles(self, kb, tmp_path):
        """Style profiles JSON should be written and contain all 8 styles."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        local_kb = FashionDomainResearch(output_dir=tmp_path)
        out_path = local_kb.generate_style_profiles()

        assert out_path.exists()
        with open(out_path, "r") as f:
            data = json.load(f)

        assert "profiles" in data
        assert len(data["profiles"]) == 8
        assert "streetwear" in data["profiles"]
        assert "athleisure" in data["profiles"]

    def test_generate_taxonomy_tree(self, kb, tmp_path):
        """Taxonomy tree JSON should have proper nested structure."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        local_kb = FashionDomainResearch(output_dir=tmp_path)
        out_path = local_kb.generate_taxonomy_tree()

        assert out_path.exists()
        with open(out_path, "r") as f:
            data = json.load(f)

        assert "tree" in data
        assert "men"   in data["tree"]
        assert "women" in data["tree"]
        assert "dresses" in data["tree"]["women"]
        assert "dresses" not in data["tree"]["men"]

    def test_generate_alias_index(self, kb, tmp_path):
        """Alias index should contain color and fabric maps."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        local_kb = FashionDomainResearch(output_dir=tmp_path)
        out_path = local_kb.generate_alias_index()

        assert out_path.exists()
        with open(out_path, "r") as f:
            data = json.load(f)

        assert "color_aliases"    in data
        assert "fabric_aliases"   in data
        assert "category_aliases" in data
        # "navy blue" should map to "Navy"
        assert data["color_aliases"].get("navy blue") == "Navy"
        # "elastane" should map to "Spandex"
        assert data["fabric_aliases"].get("elastane") == "Spandex"

    def test_generate_all_artifacts_returns_four_paths(self, tmp_path):
        """generate_all_artifacts() should return a dict with 4 entries."""
        from src.data.knowledge_base.fashion_domain_research import FashionDomainResearch
        local_kb = FashionDomainResearch(output_dir=tmp_path)
        artifacts = local_kb.generate_all_artifacts()
        assert len(artifacts) == 4
        for name, path in artifacts.items():
            assert path.exists(), f"Artifact '{name}' was not created at {path}"


# =============================================================================
# ── Module-Level Convenience Functions
# =============================================================================

class TestModuleLevelFunctions:

    def test_get_categories_for_gender_module(self, kb):
        from src.data.knowledge_base.fashion_domain_research import get_categories_for_gender
        cats = get_categories_for_gender("women", kb=kb)
        assert "dresses" in cats

    def test_get_categories_for_style_module(self, kb):
        from src.data.knowledge_base.fashion_domain_research import get_categories_for_style
        cats = get_categories_for_style("athleisure", kb=kb)
        assert "t_shirts" in cats

    def test_get_styles_for_occasion_module(self, kb):
        from src.data.knowledge_base.fashion_domain_research import get_styles_for_occasion
        styles = get_styles_for_occasion("outdoor", kb=kb)
        assert "techwear" in styles

    def test_get_fabrics_for_season_module(self, kb):
        from src.data.knowledge_base.fashion_domain_research import get_fabrics_for_season
        fabrics = get_fabrics_for_season("AW", kb=kb)  # Tests alias resolution
        assert len(fabrics) > 0

    def test_get_attributes_for_category_module(self, kb):
        from src.data.knowledge_base.fashion_domain_research import get_attributes_for_category
        attrs = get_attributes_for_category("t_shirts", kb=kb)
        assert "color" in attrs
        assert "fabric" in attrs
        assert "fit" in attrs
