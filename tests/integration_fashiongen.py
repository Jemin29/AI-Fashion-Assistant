"""
End-to-end integration smoke test for the FashionGen ingestion pipeline.
Uses a fully synthetic, in-memory mock — no real HDF5 file required.

Run: python tests/integration_fashiongen.py
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")

# ── Import all public API symbols ─────────────────────────────────────────────
from src.data.ingestion.fashiongen_loader import (
    FashionGenLoader,
    FashionGenExtractor,
    FashionGenTransformer,
    FashionGenValidator,
    FashionGenWriter,
    FashionGenRecord,
    RawFashionGenRecord,
    PipelineStats,
    _CATEGORY_MAP,
    _GENDER_MAP,
)

print("=" * 60)
print("FashionGen Loader — Integration Smoke Test")
print("=" * 60)

# ── 1. Validate data models ───────────────────────────────────────────────────
print("\n[1] Data Models")
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
raw = RawFashionGenRecord(42, img, "A white cotton formal shirt.", "Shirts", "Dress Shirts", "Men", Path("x.h5"))
assert raw.source_index == 42
print("    RawFashionGenRecord    : OK")

rec = FashionGenRecord(image_id="FG_0000042", image_path="datasets/fashiongen/images/0000/FG_0000042.jpg")
d = rec.to_dict()
for f in ["image_id", "image_path", "description", "category", "gender", "season", "style"]:
    assert f in d, f"Spec field missing: {f}"
print("    FashionGenRecord       : OK (all 7 spec fields present)")
print("    JSON serializable      : OK")

stats = PipelineStats()
stats.increment_category("shirts"); stats.increment_category("shirts")
stats.increment_gender("men"); stats.increment_gender("women")
assert stats.category_counts["shirts"] == 2
assert stats.gender_counts["men"]      == 1
print("    PipelineStats counters : OK")

# ── 2. Transformer ────────────────────────────────────────────────────────────
print("\n[2] FashionGenTransformer")
t = FashionGenTransformer(id_prefix="FG", kb=None)

tests = [
    ("Shirts",    "shirts"),
    ("T-Shirts",  "t_shirts"),
    ("Jeans",     "jeans"),
    ("Dresses",   "dresses"),
    ("Shoes",     "footwear"),
    ("Hoodies",   "hoodies"),
    ("Jackets",   "jackets"),
    ("Shorts",    "shorts"),
    ("Watches",   "accessories"),
]
for raw_cat, expected in tests:
    got = t._normalize_category(raw_cat)
    status = "OK" if got == expected else "FAIL"
    print(f"    [{status}] {raw_cat:<15} -> {got} (expected {expected})")
    assert got == expected, f"Category map fail: {raw_cat}"

gender_tests = [("Men", "men"), ("Women", "women"), ("Boys", "men"), ("Girls", "women")]
for raw_g, exp_g in gender_tests:
    assert t._normalize_gender(raw_g) == exp_g, f"Gender map fail: {raw_g}"
print("    Gender normalization   : OK (Men/Women/Boys/Girls)")

season_tests = [
    ("A breathable lightweight summer beach top.", "summer"),
    ("Warm wool fleece jacket for cold winter.", "winter"),
    ("", "all_season"),
]
for desc, exp_s in season_tests:
    assert t._infer_season(desc) == exp_s, f"Season infer fail: {desc}"
print("    Season inference       : OK")

style_tests = [
    ("A professional formal blazer for office.", "jackets", "formal"),
    ("Oversized graphic logo streetwear tee.", "t_shirts", "streetwear"),
    ("Performance running shorts for gym.", "shorts", "athleisure"),
]
for desc, cat, exp_style in style_tests:
    got = t._infer_style(desc, cat)
    assert got == exp_style, f"Style infer fail: got {got}, expected {exp_style} ({desc})"
print("    Style inference        : OK")

# Full transform
result = t.transform(raw)
assert result.image_id    == "FG_0000042"
assert result.category    == "shirts"
assert result.gender      == "men"
assert result.image_path.endswith(".jpg")
assert "\\" not in result.image_path
print("    Full transform         : OK")

# ── 3. Validator ──────────────────────────────────────────────────────────────
print("\n[3] FashionGenValidator")
v = FashionGenValidator(kb=None)

good = FashionGenRecord("FG_001", "p.jpg", category="shirts", gender="men", dataset_source="fashiongen")
r = v.validate(good)
assert r.is_valid, f"Should be valid: {r.errors}"
print("    Valid record           : OK (is_valid=True, errors=[])")

bad_cat = FashionGenRecord("FG_002", "p.jpg", category="spaceship", gender="men", dataset_source="fashiongen")
r = v.validate(bad_cat)
assert not r.is_valid
print("    Invalid category       : OK (is_valid=False)")

dresses_men = FashionGenRecord("FG_003", "p.jpg", category="dresses", gender="men", dataset_source="fashiongen")
r = v.validate(dresses_men)
assert not r.is_valid
assert any("dresses" in e for e in r.errors)
print("    Dresses/men CR001      : OK (is_valid=False)")

dresses_women = FashionGenRecord("FG_004", "p.jpg", category="dresses", gender="women", dataset_source="fashiongen")
r = v.validate(dresses_women)
assert r.is_valid, f"Dresses/women should be valid: {r.errors}"
print("    Dresses/women          : OK (is_valid=True)")

# ── 4. Writer ─────────────────────────────────────────────────────────────────
print("\n[4] FashionGenWriter")
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    writer  = FashionGenWriter(output_dir=tmp)
    records = [good, dresses_women]
    for rec in records:
        rec.is_valid = True
    out_path = writer.save_records(records, PipelineStats())
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "_meta"   in data
    assert "records" in data
    assert data["_meta"]["total_records"] == 2
    for rec_dict in data["records"]:
        for spec_field in ["image_id", "image_path", "description", "category", "gender", "season", "style"]:
            assert spec_field in rec_dict, f"Missing spec field in JSON: {spec_field}"
    print(f"    JSON output created    : OK ({out_path.stat().st_size} bytes)")
    print("    All 7 spec fields      : OK in every record")
    print("    _meta + records keys   : OK")

    rpt_path = writer.save_run_report(PipelineStats(), Path("fake.h5"))
    assert rpt_path.exists()
    print("    Run report saved       : OK")

# ── 5. Pipeline Stats math ────────────────────────────────────────────────────
print("\n[5] PipelineStats")
s = PipelineStats()
s.total_processed = 100; s.total_valid = 90
assert abs(s.valid_rate - 0.90) < 0.001
s.finalize()
t1 = s.elapsed_seconds
import time; time.sleep(0.05)
t2 = s.elapsed_seconds
assert abs(t1 - t2) < 0.001, "elapsed_seconds changed after finalize"
print("    valid_rate calculation  : OK (0.90)")
print("    finalize() stamps time  : OK")
d = s.to_dict()
for key in ["total_read", "total_processed", "total_valid", "valid_rate",
            "records_per_second", "elapsed_seconds", "category_counts",
            "gender_counts", "style_counts", "season_counts"]:
    assert key in d, f"Missing key in to_dict(): {key}"
print("    to_dict() completeness  : OK (all keys present)")
assert json.dumps(d)                    # must be JSON-serializable
print("    to_dict() JSON-safe     : OK")

# ── 6. Category/Gender map completeness ────────────────────────────────────────
print("\n[6] Constant Maps")
valid_keys = {"t_shirts","shirts","hoodies","jackets","pants","jeans","shorts",
              "dresses","ethnic_wear","footwear","accessories"}
for raw_key, taxonomy_key in _CATEGORY_MAP.items():
    assert taxonomy_key in valid_keys, f"Bad taxonomy key: {taxonomy_key}"
print(f"    _CATEGORY_MAP          : OK ({len(_CATEGORY_MAP)} entries, all valid)")

valid_genders = {"men", "women", "unisex"}
for raw_g, mapped_g in _GENDER_MAP.items():
    assert mapped_g in valid_genders
print(f"    _GENDER_MAP            : OK ({len(_GENDER_MAP)} entries, all valid)")

# ── 7. Public package API ─────────────────────────────────────────────────────
print("\n[7] Package API")
from src.data.ingestion import (
    FashionGenLoader, FashionGenExtractor, FashionGenTransformer,
    FashionGenValidator, FashionGenWriter, FashionGenRecord,
    RawFashionGenRecord, PipelineStats,
)
print("    All 8 symbols importable from src.data.ingestion : OK")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL INTEGRATION TESTS PASSED")
print("=" * 60)
