"""
Demo: Run the 7-stage preprocessing pipeline and generate clean_dataset.json.
Simulates a realistic mixed-dataset batch (FashionGen + DeepFashion records).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline, PipelineConfig
)

cfg = PipelineConfig(
    target_size           = (256, 256),
    dedup_strategy        = "path_hash",
    lowercase_description = False,
    drop_unknown_categories=False,
    keep_normalization_log = True,
    balance_target_per_class=None,
)

pipeline = PreprocessingPipeline(config=cfg)

# ── Simulate a realistic mixed-quality batch ──────────────────────────────────
records = [
    # ── Valid FashionGen records (varied categories) ──
    {
        "image_id": "FG_0001", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0001.jpg",
        "category": "shirts", "gender": "men",
        "style": "formal", "fit": "slim_fit", "season": "all_season",
        "occasion": ["formal"], "color": ["White"],
        "description": "  A <b>classic</b> slim fit white cotton shirt.  ",
        "attributes": ["collar", "button-down"],
    },
    {
        "image_id": "FG_0002", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0002.jpg",
        "category": "hoodies", "gender": "unisex",
        "style": "streetwear", "fit": "oversized", "season": "winter",
        "occasion": ["casual"], "color": ["Black"],
        "description": "An oversized black hoodie with neon graphics and bold logo.",
        "attributes": ["graphic print"],
    },
    {
        "image_id": "FG_0003", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0003.jpg",
        "category": "dresses", "gender": "women",
        "style": "luxury", "fit": "regular_fit", "season": "summer",
        "occasion": ["party", "formal"], "color": ["Red", "Gold"],
        "description": "An elegant red and gold evening dress for cocktail parties.",
        "attributes": ["sleeveless", "midi"],
    },
    {
        "image_id": "FG_0004", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0004.jpg",
        "category": "jeans", "gender": "men",
        "style": "casual", "fit": "slim", "season": "all_season",
        "occasion": ["casual"], "color": ["navy blue"],
        "description": "  Slim fit navy blue stretch jeans with a tapered leg.  ",
        "attributes": ["five-pocket", "stretch"],
    },
    {
        "image_id": "FG_0005", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0005.jpg",
        "category": "jackets", "gender": "women",
        "style": "minimalist", "fit": "regular fit", "season": "autumn",
        "occasion": ["business casual"], "color": ["Beige"],
        "description": "A minimalist beige trench coat for business casual settings.",
        "attributes": ["double-breasted", "belt"],
    },
    # ── DeepFashion records ──
    {
        "image_id": "DF_0001", "source_dataset": "deepfashion",
        "image_path": "datasets/deepfashion/img/Blouse/img_00000001.jpg",
        "category": "blouse", "gender": None,
        "style": None, "fit": None, "season": "all_season",
        "occasion": [], "color": [],
        "description": None,
        "attributes": ["v-neck", "stripe"],
        "landmarks": [
            {"name": "left_collar",  "x": 0.3, "y": 0.1, "visible": True},
            {"name": "right_collar", "x": 0.7, "y": 0.1, "visible": True},
        ],
    },
    {
        "image_id": "DF_0002", "source_dataset": "deepfashion",
        "image_path": "datasets/deepfashion/img/Jeans/img_00000002.jpg",
        "category": "denim jeans", "gender": None,
        "style": "retro", "fit": "straight leg", "season": None,
        "occasion": ["everyday"], "color": [],
        "description": None,
        "attributes": ["high-waist", "distressed"],
    },
    # ── Records needing normalization ──
    {
        "image_id": "FG_0006", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0006.jpg",
        "category": "tshirt", "gender": "male",
        "style": "athleisure", "fit": "baggy", "season": "summer",
        "occasion": ["sport", "gym"], "color": ["navy"],
        "description": "A&amp;F athletic tee — <i>perfect</i> for training!!!\n\nMade from moisture-wicking fabric.",
        "attributes": ["crew neck", "moisture-wicking"],
    },
    {
        "image_id": "FG_0007", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0007.jpg",
        "category": "sneakers", "gender": "unisex",
        "style": "streetwear", "fit": None, "season": "all_season",
        "occasion": ["casual"], "color": ["White", "white"],  # duplicate color
        "description": "Classic white leather sneakers with rubber sole.",
        "attributes": ["lace-up", "leather"],
    },
    # ── Duplicate image ──
    {
        "image_id": "FG_0001_DUP", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0001.jpg",  # same path as FG_0001
        "category": "shirts", "gender": "men",
        "description": "Duplicate of FG_0001",
        "attributes": [],
    },
    # ── Unknown / uncategorized ──
    {
        "image_id": "FG_0008", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0008.jpg",
        "category": "swimwear", "gender": "women",
        "style": "beach", "fit": None, "season": "summer",
        "occasion": ["beach"], "color": ["Coral"],
        "description": "A vibrant coral swimsuit with UV protection.",
        "attributes": ["one-piece"],
    },
    # ── Ethnic wear ──
    {
        "image_id": "FG_0009", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0009.jpg",
        "category": "kurta", "gender": "men",
        "style": "formal", "fit": "regular_fit", "season": "all_season",
        "occasion": ["wedding"], "color": ["Ivory"],
        "description": "A traditional ivory silk kurta for festive occasions.",
        "attributes": ["silk", "hand-embroidered"],
    },
    # ── Accessories ──
    {
        "image_id": "FG_0010", "source_dataset": "fashiongen",
        "image_path": "datasets/fashiongen/images/FG_0010.jpg",
        "category": "handbag", "gender": "women",
        "style": "luxury", "fit": None, "season": "all_season",
        "occasion": ["formal", "party"], "color": ["Brown"],
        "description": "A structured brown leather tote with gold hardware.",
        "attributes": ["leather", "structured"],
    },
]

print(f"\nRunning preprocessing pipeline on {len(records)} records...\n")
result = pipeline.run(records)

# Save output
path = pipeline.save(result, "datasets/processed/clean_dataset.json")

import json
data = json.loads(path.read_text(encoding="utf-8"))
print("\n=== clean_dataset.json SUMMARY ===")
import pprint
pprint.pprint(data["summary"])

print("\n=== BALANCE STATS: Category ===")
cat_stats = data["balance_stats"]["category"]
for cat, count in sorted(cat_stats["counts"].items()):
    share = cat_stats["shares_pct"].get(cat, 0)
    print(f"  {cat:<20} {count:>3} records  ({share:.1f}%)")

print("\n=== BALANCE RECOMMENDATION ===")
for cat, info in data["balance_stats"]["recommended_balance"]["categories"].items():
    print(f"  {cat:<20} count={info['current_count']}  target={info['target_count']}  action={info['action']}")

print(f"\nOutput: {path}  ({path.stat().st_size/1024:.1f} KB)")
print(f"Records in file: {len(data['records'])}")
