"""
=============================================================================
AI-Powered Fashion Design Assistant
notebooks/02_knowledge_base_exploration.py
=============================================================================
Week 1 Knowledge Base Exploration Notebook (Jupytext format)

CONTENTS:
    1. Load the Fashion Knowledge Base
    2. Explore Taxonomy (Genders → Categories → Subcategories)
    3. Style Hierarchy Visualization
    4. Attribute Deep Dives (Color, Fabric, Fit, Pattern)
    5. Cross-Reference Queries (search_by_tags)
    6. Normalizer Function Demos
    7. Record Validation Examples
    8. Generate All Derived JSON Artifacts
    9. KB Statistics Summary

Convert to Jupyter:
    jupytext --to notebook notebooks/02_knowledge_base_exploration.py
=============================================================================
"""

# %% [markdown]
# # 🧠 Fashion Knowledge Base — Domain Research Explorer
# **AI-Powered Fashion Design Assistant | Week 1**
#
# This notebook demonstrates the full capabilities of `FashionDomainResearch`,
# the central knowledge base module powering taxonomy queries, attribute
# normalization, and record validation across the entire pipeline.

# %% Setup
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.knowledge_base import (
    FashionDomainResearch,
    normalize_color, normalize_fabric, normalize_style,
    normalize_fit, normalize_pattern, normalize_season, normalize_occasion,
    get_categories_for_gender, get_styles_for_occasion,
    validate_fashion_record,
)

# Load KB once for the whole notebook
kb = FashionDomainResearch()
print(f"✅ Knowledge Base loaded: {kb.get_knowledge_base_stats()['knowledge_base_version']}")

# %% [markdown]
# ## 1. Knowledge Base Statistics

# %% Stats
stats = kb.get_knowledge_base_stats()

print("\n📊 Fashion Knowledge Base — Statistics")
print("=" * 50)
for k, v in stats.items():
    print(f"  {k:<40} : {v}")

# %% [markdown]
# ## 2. Taxonomy Tree — Gender → Category → Subcategory

# %% Taxonomy tree
tree = kb.get_taxonomy_tree()

for gender, categories in tree.items():
    print(f"\n👤 {gender.upper()}")
    for cat_key, cat_data in categories.items():
        label = cat_data["label"]
        n_sub = len(cat_data["subcategories"])
        print(f"   ├── {label:<20} ({n_sub} subcategories)")
        for sub_key, sub_label in list(cat_data["subcategories"].items())[:3]:
            print(f"   │   ├── {sub_label}")
        if n_sub > 3:
            print(f"   │   └── ... ({n_sub - 3} more)")

# %% [markdown]
# ## 3. Style Hierarchy Visualization

# %% Style tiers
style_tiers = kb.get_style_tiers()
all_styles  = kb.get_all_styles()

# ── Color-coded style tier chart ──────────────────────────────────────────────
tier_colors = {1: "#6C5CE7", 2: "#00B894", 3: "#FDCB6E"}
fig, ax = plt.subplots(figsize=(12, 5))

y_positions = {}
x_by_tier = {1: 0, 2: 0, 3: 0}
tier_spacing = {1: 2.0, 2: 2.0, 3: 2.0}

for tier, style_keys in sorted(style_tiers.items()):
    for i, sk in enumerate(style_keys):
        style = all_styles[sk]
        x = tier * 4
        y = -(i * 1.5)
        y_positions[sk] = (x, y)

        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.2, y - 0.4), 2.4, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=tier_colors.get(tier, "#DFE6E9"),
            edgecolor="white", linewidth=2, zorder=2
        ))
        ax.text(x, y, style["label"], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

# Draw parent → child arrows
for sk, style in all_styles.items():
    if sk not in y_positions:
        continue
    for child in style.get("child_styles", []):
        if child in y_positions:
            x1, y1 = y_positions[sk]
            x2, y2 = y_positions[child]
            ax.annotate("", xy=(x2 - 1.2, y2), xytext=(x1 + 1.2, y1),
                        arrowprops=dict(arrowstyle="->", color="#636E72", lw=1.5))

ax.set_xlim(0, 16)
ax.set_ylim(-8, 2)
ax.axis("off")
ax.set_title("Fashion Style Hierarchy (Tier 1 → 3)", fontsize=14, fontweight="bold", pad=20)

for tier, color in tier_colors.items():
    ax.add_patch(mpatches.Patch(color=color, label=f"Tier {tier}"))
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "style_hierarchy.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved → outputs/style_hierarchy.png")

# %% [markdown]
# ## 4. Cross-Reference Queries

# %% Cross-reference: categories per gender
print("📋 Categories by Gender")
print("=" * 50)
for gender in ("men", "women", "unisex"):
    cats = get_categories_for_gender(gender, kb=kb)
    print(f"\n  {gender.upper()} ({len(cats)} categories):")
    for c in cats:
        cat_def = kb.get_category(c)
        print(f"    • {cat_def['label']:<20} [{cat_def['code']}]")

# %% Cross-reference: styles for occasions
print("\n🎯 Recommended Styles by Occasion")
print("=" * 50)
for occasion in ("casual", "formal", "sport", "party", "outdoor"):
    styles = get_styles_for_occasion(occasion, kb=kb)
    style_labels = [kb.get_style(s)["label"] for s in styles if kb.get_style(s)]
    print(f"  {occasion:<20} → {', '.join(style_labels)}")

# %% Season-fabric matrix
print("\n🌿 Recommended Fabrics by Season")
print("=" * 50)
for season in ("spring", "summer", "autumn", "winter", "all_season"):
    fabrics = kb.get_fabrics_for_season(season)
    print(f"  {season:<15} → {', '.join(fabrics[:4])}{'...' if len(fabrics) > 4 else ''}")

# %% [markdown]
# ## 5. search_by_tags() — Recommendation Engine Demo

# %% search_by_tags demo
print("\n🔍 Smart Search: Women | Streetwear | Casual | Summer")
print("=" * 55)
result = kb.search_by_tags(
    gender="women",
    style="streetwear",
    occasion="casual",
    season="summer",
)
for key, values in result.items():
    print(f"  {key:<15} : {values}")

print("\n🔍 Smart Search: Men | Formal | Business Casual | All Season")
print("=" * 60)
result2 = kb.search_by_tags(
    gender="men",
    style="formal",
    occasion="business_casual",
)
for key, values in result2.items():
    print(f"  {key:<15} : {values}")

# %% [markdown]
# ## 6. Normalizer Function Demos

# %% Normalizer demos
print("\n🎨 Color Normalizer")
test_colors = [
    "navy blue", "off-white", "cognac", "ivory",
    "charcoal", "cobalt", "hot pink", "rust", "jade", "xyzzy"
]
for c in test_colors:
    canonical = normalize_color(c, kb=kb)
    status = "✅" if canonical else "❌"
    print(f"  {status} '{c:<20}' → {canonical!r}")

print("\n🧵 Fabric Normalizer")
test_fabrics = [
    "polycotton", "elastane", "lycra", "merino wool",
    "100% cotton", "ripstop", "modal", "moonsilk"
]
for f in test_fabrics:
    canonical = normalize_fabric(f, kb=kb)
    status = "✅" if canonical else "❌"
    print(f"  {status} '{f:<25}' → {canonical!r}")

print("\n👗 Style, Fit, Pattern, Season, Occasion Normalizers")
tests = [
    ("style",   normalize_style,   ["Business Casual", "Athleisure", "Hype Beast"], kb),
    ("fit",     lambda x, kb=None: normalize_fit(x), ["fitted", "baggy", "regular", "superfluid"], None),
    ("pattern", lambda x, kb=None: normalize_pattern(x), ["plaid", "leopard", "plain", "camo", "quantum_foam"], None),
    ("season",  lambda x, kb=None: normalize_season(x), ["SS", "AW", "year-round", "monsoon"], None),
    ("occasion",lambda x, kb=None: normalize_occasion(x), ["gym", "black tie", "hiking", "rocket_launch"], None),
]
for attr_name, fn, values, fn_kb in tests:
    print(f"\n  [{attr_name.upper()}]")
    for v in values:
        if fn_kb is not None:
            result = fn(v, kb=fn_kb)
        else:
            result = fn(v)
        status = "✅" if result else "❌"
        print(f"    {status} '{v:<25}' → {result!r}")

# %% [markdown]
# ## 7. Record Validation Examples

# %% Validation demos
print("\n✅ Record Validation Examples")
print("=" * 55)

test_records = [
    {
        "name": "Valid Men's T-Shirt",
        "record": {
            "image_id": "V001", "dataset_source": "fashiongen",
            "category": "t_shirts", "gender": "men",
            "styles": ["streetwear"], "occasions": ["casual"],
        }
    },
    {
        "name": "Invalid: Dress for Men",
        "record": {
            "image_id": "V002", "dataset_source": "test",
            "category": "dresses", "gender": "men",
        }
    },
    {
        "name": "Invalid: Unknown Category",
        "record": {
            "image_id": "V003", "dataset_source": "test",
            "category": "space_suit", "gender": "women",
        }
    },
    {
        "name": "Valid Women's Dress with Occasion",
        "record": {
            "image_id": "V004", "dataset_source": "deepfashion",
            "category": "dresses", "gender": "women",
            "occasions": ["party"], "styles": ["luxury"],
            "seasons": ["summer"], "colors": ["Red"],
        }
    },
    {
        "name": "Warning: Unusual fit for footwear",
        "record": {
            "image_id": "V005", "dataset_source": "test",
            "category": "footwear", "gender": "unisex",
            "fit": "slim_fit",  # Unusual for footwear
        }
    },
]

for item in test_records:
    result = validate_fashion_record(item["record"], kb=kb)
    icon = "✅" if result.is_valid else "❌"
    print(f"\n  {icon} {item['name']}")
    print(f"     {result.summary()}")
    if result.errors:
        for e in result.errors:
            print(f"       Error:   {e}")
    if result.warnings:
        for w in result.warnings:
            print(f"       Warning: {w}")

# %% [markdown]
# ## 8. Generate All Derived JSON Artifacts

# %% Generate artifacts
print("\n🏗  Generating all knowledge base artifacts…\n")
artifacts = kb.generate_all_artifacts()

print("\n✅ Artifacts saved to datasets/metadata/:")
for name, path in artifacts.items():
    size_kb = path.stat().st_size / 1024
    print(f"   {name:<25} → {path.name}  ({size_kb:.1f} KB)")

# %% [markdown]
# ## 9. Color Palette Visualization

# %% Color palette heatmap
color_groups = {
    "Neutrals"   : ["White", "Black", "Grey", "Beige", "Brown"],
    "Blues"      : ["Navy", "Royal Blue", "Sky Blue", "Teal"],
    "Greens"     : ["Olive", "Forest Green", "Mint", "Emerald"],
    "Reds/Pinks" : ["Red", "Burgundy", "Pink", "Coral"],
    "Warm Tones" : ["Yellow", "Orange", "Purple"],
}

# Get hex codes
palette_data = []
for group, colors in color_groups.items():
    for color in colors:
        hex_val = kb.get_color_hex(color)
        palette_data.append((group, color, hex_val or "#CCCCCC"))

fig, axes = plt.subplots(len(color_groups), 1, figsize=(14, 8))
fig.suptitle("Fashion Color Taxonomy Palette", fontsize=14, fontweight="bold")

for ax, (group, colors) in zip(axes, color_groups.items()):
    for i, color_name in enumerate(colors):
        hex_val = kb.get_color_hex(color_name) or "#CCCCCC"
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_val))
        text_color = "black" if hex_val in ("#FFFFFF", "#F5F5DC", "#87CEEB", "#98FF98") else "white"
        ax.text(i + 0.5, 0.5, color_name, ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold")
    ax.set_xlim(0, max(len(c) for c in color_groups.values()))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(group, rotation=0, labelpad=60, va="center", fontsize=9)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "color_taxonomy.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved → outputs/color_taxonomy.png")

# %% [markdown]
# ## ✅ Week 1 Knowledge Base Exploration Complete
#
# **What we built:**
# - 11 categories × 3 genders with 70+ subcategories
# - 8 style tiers with compatibility matrices
# - 7 attribute types (color, fabric, fit, pattern, season, occasion, style)
# - 6 normalizer functions mapping 150+ aliases → canonical values
# - 5-layer record validator with conditional taxonomy rules
# - 4 derived JSON artifacts auto-generated to `datasets/metadata/`
#
# **Next step:** Wire the KB normalizers into the ingestion pipeline's
# metadata tagging system in Week 2.
