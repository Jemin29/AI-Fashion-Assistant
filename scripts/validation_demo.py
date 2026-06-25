"""Integration demo: run FashionDataValidator on a mixed batch and save the report."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data.validation import FashionDataValidator, ValidationConfig
import json

cfg = ValidationConfig(verify_image_exists=False, verify_image_readable=False)
validator = FashionDataValidator(config=cfg)

records = [
    {
        "image_id": "FG_0001", "image_path": "datasets/fashiongen/images/FG_0001.jpg",
        "category": "shirts", "source_dataset": "fashiongen", "gender": "men",
        "color": ["White"], "season": "all_season", "occasion": ["formal"],
        "description": "A classic slim fit white cotton dress shirt for formal occasions.",
        "attributes": [], "is_valid": True, "processed_at": "2026-06-03T12:00:00+00:00",
    },
    {
        "image_id": "FG_0002", "image_path": "datasets/fashiongen/images/FG_0002.jpg",
        "category": "hoodies", "source_dataset": "fashiongen", "gender": "unisex",
        "color": ["Black"], "season": "winter", "occasion": ["casual"],
        "description": "An oversized black hoodie with neon graphics and bold logo.",
        "attributes": ["graphic"], "is_valid": True, "processed_at": "2026-06-03T12:00:00+00:00",
    },
    {
        "image_id": "FG_0003", "image_path": "datasets/test.gif",
        "category": "UNKNOWN_CATEGORY", "source_dataset": "fashiongen",
        "description": "Short", "is_valid": True, "processed_at": "2026-06-03T12:00:00+00:00",
    },
    {
        "image_id": "FG_0004", "image_path": "datasets/test.jpg",
        "category": "jeans", "source_dataset": "fashiongen",
        "gender": "alien",
        "description": "A pair of blue straight-leg jeans for everyday casual wear.",
        "season": "monsoon", "is_valid": True, "processed_at": "2026-06-03T12:00:00+00:00",
    },
    {
        "image_id": "DF_0001", "image_path": "datasets/deepfashion/img/item.jpg",
        "category": "dresses", "source_dataset": "deepfashion", "gender": None,
        "color": [], "season": "summer",
        "landmarks": [{"name": "left_collar", "x": 0.3, "y": 0.1, "visible": True}],
        "bounding_box": {"x1": 10, "y1": 20, "x2": 200, "y2": 240},
        "description": None, "attributes": ["floral"],
        "is_valid": True, "processed_at": "2026-06-03T12:00:00+00:00",
    },
    {
        "image_id": 99999, "image_path": "datasets/corrupted.jpg",
        "category": ["shirts"], "source_dataset": "fashiongen",
        "color": "White",
        "description": "A shirt.", "is_valid": "yes",
        "processed_at": "NOT-A-DATE",
    },
]

print("Running batch validation...")
batch = validator.validate_batch(records)
print()
print(batch.summary())

report_path = validator.save_report(
    batch,
    "datasets/metadata/validation_report.json",
    include_valid=False,
)

report = json.loads(report_path.read_text(encoding="utf-8"))
print()
print("=== validation_report.json (summary section) ===")
print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
print()
print(f"Records in report : {len(report['records'])}")
sz = report_path.stat().st_size / 1024
print(f"Report size       : {sz:.1f} KB")
print(f"Report path       : {report_path}")
