"""Quick smoke test for the Fashion Knowledge Base module."""
import sys
sys.path.insert(0, ".")

from src.data.knowledge_base import (
    FashionDomainResearch,
    normalize_color, normalize_fabric, normalize_style,
    validate_fashion_record,
)

kb = FashionDomainResearch()

# ── Statistics ─────────────────────────────────────────────────────────────
stats = kb.get_knowledge_base_stats()
print("=== KB Statistics ===")
for k, v in stats.items():
    print("  " + str(k) + ": " + str(v))

# ── Generate artifacts ─────────────────────────────────────────────────────
print("\n=== Generating Artifacts ===")
artifacts = kb.generate_all_artifacts()
for name, path in artifacts.items():
    size_kb = path.stat().st_size / 1024
    print("  " + name + " -> " + path.name + "  (" + str(round(size_kb, 1)) + " KB)")

# ── Normalizers ────────────────────────────────────────────────────────────
print("\n=== Normalizer Tests ===")
tests = [
    ("color",  "navy blue",       normalize_color("navy blue", kb=kb)),
    ("color",  "off-white",       normalize_color("off-white", kb=kb)),
    ("color",  "cognac",          normalize_color("cognac", kb=kb)),
    ("fabric", "elastane",        normalize_fabric("elastane", kb=kb)),
    ("fabric", "polycotton",      normalize_fabric("polycotton", kb=kb)),
    ("style",  "Business Casual", normalize_style("Business Casual", kb=kb)),
    ("style",  "Athleisure",      normalize_style("Athleisure", kb=kb)),
]
for attr, raw, result in tests:
    ok = "OK" if result else "MISS"
    print("  [" + ok + "] normalize_" + attr + "(" + raw + ") -> " + str(result))

# ── Cross-reference queries ────────────────────────────────────────────────
print("\n=== Cross-Reference Queries ===")
women_cats = kb.get_categories_for_gender("women")
print("  Women categories (" + str(len(women_cats)) + "):", women_cats)

casual_styles = kb.get_styles_for_occasion("casual")
print("  Casual styles:", casual_styles)

summer_fabrics = kb.get_fabrics_for_season("summer")
print("  Summer fabrics:", summer_fabrics)

# ── Validation ─────────────────────────────────────────────────────────────
print("\n=== Validation Tests ===")
r1 = validate_fashion_record({
    "image_id": "T1", "category": "t_shirts",
    "gender": "men", "dataset_source": "fashiongen"
}, kb=kb)
r2 = validate_fashion_record({
    "image_id": "T2", "category": "dresses",
    "gender": "men", "dataset_source": "fashiongen"
}, kb=kb)
r3 = validate_fashion_record({
    "image_id": "T3", "category": "dresses",
    "gender": "women", "dataset_source": "deepfashion",
    "styles": ["luxury"], "occasions": ["formal"]
}, kb=kb)

print("  T-shirt/men   is_valid=" + str(r1.is_valid) + "  errors=" + str(r1.errors))
print("  Dress/men     is_valid=" + str(r2.is_valid) + "  errors=" + str(r2.errors))
print("  Dress/women   is_valid=" + str(r3.is_valid) + "  errors=" + str(r3.errors))

print("\nALL SMOKE TESTS PASSED")
