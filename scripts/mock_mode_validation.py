"""
scripts/mock_mode_validation.py
================================
Comprehensive Mock Mode Validation for the AI Fashion Creative Studio.
Tests all 7 service modules + Gradio page builders in mock=True mode
(CPU-only, no GPU, no real model weights).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOCK_MODE", "true")

# Silence loguru below WARNING in test output
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR", colorize=False,
           format="{level} | {message}")

# ── primitives ─────────────────────────────────────────────────────────────────

@dataclass
class AR:          # AssertionResult
    name: str
    passed: bool
    detail: str = ""
    latency_ms: float = 0.0

@dataclass
class MR:          # ModuleResult
    module: str
    assertions: List[AR] = field(default_factory=list)

    @property
    def passed(self): return sum(1 for a in self.assertions if a.passed)
    @property
    def total(self): return len(self.assertions)
    @property
    def ok(self): return self.passed == self.total


def chk(r: MR, name: str, cond: bool, detail: str = "", ms: float = 0.0) -> None:
    icon = "  [PASS]" if cond else "  [FAIL]"
    lat  = f"  [{ms:.0f}ms]" if ms else ""
    print(f"{icon}  {name}{lat}")
    if not cond and detail:
        trimmed = detail.strip().split("\n")[0][:200]
        print(f"         -> {trimmed}")
    r.assertions.append(AR(name, cond, detail, ms))


def timed(fn: Callable):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def sec(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  [MODULE] {title}")
    print(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════

def validate_generation_service() -> MR:
    sec("Text-to-Fashion -- GenerationService (mock_mode=True)")
    r = MR("Text-to-Fashion / GenerationService")
    try:
        from week6.services.generation_service import GenerationService
        svc = GenerationService(mock_mode=True)

        health, ms = timed(lambda: svc.health_check())
        chk(r, "health_check() returns ServiceResult", hasattr(health, "success"), ms=ms)
        chk(r, "health_check().success is True", health.success)

        # Basic generation
        res, ms = timed(lambda: svc.generate(
            prompt="elegant minimalist silk dress, soft studio lighting",
            negative_prompt="blurry, low quality",
            num_inference_steps=4,
            guidance_scale=7.5,
            width=512, height=512,
            seed=42,
        ))
        chk(r, "generate() returns ServiceResult", hasattr(res, "success"), ms=ms)
        chk(r, "generate().success is True", res.success, res.message)
        chk(r, "result.data is not None", res.data is not None)
        chk(r, "result.metadata is populated", bool(res.metadata))

        # Empty prompt validation
        res_empty = svc.generate(prompt="")
        chk(r, "Empty prompt -> success=False (validation guard)", not res_empty.success)

        # Batch: 3 different seeds
        for seed_offset in range(3):
            rb = svc.generate(prompt="streetwear hoodie", seed=100 + seed_offset)
            chk(r, f"Batch seed {100+seed_offset} -> success", rb.success)

        # Style presets
        for preset in ["Minimalist", "Streetwear", "Haute Couture"]:
            rp = svc.generate(prompt=f"{preset} look editorial", style_label=preset)
            chk(r, f"Style preset '{preset}' -> success", rp.success)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_controlnet_service() -> MR:
    sec("Sketch2Design -- ControlNetService (mock_mode=True)")
    r = MR("Sketch2Design / ControlNetService")
    try:
        from week6.services.controlnet_service import ControlNetService
        from PIL import Image as PILImage

        svc = ControlNetService(mock_mode=True)
        health, ms = timed(lambda: svc.health_check())
        chk(r, "health_check() -> success", health.success, ms=ms)

        dummy_img = PILImage.new("RGB", (512, 512), color=(120, 80, 60))

        # Preprocess all 4 modes
        for mode in ["canny", "sketch", "pose", "depth"]:
            res, ms = timed(lambda m=mode: svc.preprocess_image(dummy_img, mode=m))
            chk(r, f"preprocess_image(mode={mode}) -> success", res.success,
                res.message if not res.success else "", ms)
            chk(r, f"preprocess_image(mode={mode}) -> data not None", res.data is not None)

        # Conditioned generation
        for mode in ["canny", "sketch"]:
            res, ms = timed(lambda m=mode: svc.generate_conditioned(
                prompt="fashion model in couture evening gown",
                control_image=dummy_img,
                mode=m,
                conditioning_scale=0.7,
                num_inference_steps=4,
                guidance_scale=7.0,
            ))
            chk(r, f"generate_conditioned(mode={mode}) -> success", res.success,
                res.message if not res.success else "", ms)
            if res.success and isinstance(res.data, dict):
                chk(r, f"generate_conditioned(mode={mode}) -> 'image' key in data",
                    "image" in res.data)

        # None image guard
        res_none = svc.preprocess_image(None, mode="canny")
        chk(r, "None image -> success=False", not res_none.success)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_lora_service() -> MR:
    sec("Brand Studio -- LoRAService (mock_mode=True)")
    r = MR("Brand Studio / LoRAService")
    try:
        from week6.services.lora_service import LoRAService
        svc = LoRAService(mock_mode=True)

        health, ms = timed(lambda: svc.health_check())
        chk(r, "health_check() -> success", health.success, ms=ms)

        brands = svc.get_brands()
        chk(r, "get_brands() returns list", isinstance(brands, list))
        chk(r, "At least 1 brand available", len(brands) >= 1)

        # Brand info cards
        for b in brands:
            info = svc.get_brand_info(b)
            chk(r, f"get_brand_info({b}) -> dict", isinstance(info, dict))

        # Single brand generation
        for b in brands[:3]:
            res, ms = timed(lambda brand=b: svc.generate_with_brand(
                prompt="tailored jacket with technical details",
                brand=brand,
                lora_scale=0.85,
            ))
            chk(r, f"generate_with_brand({b}) -> success", res.success,
                res.message if not res.success else "", ms)

        # Style mix (multi-brand LoRA blend)
        if len(brands) >= 2:
            weights = {brands[0]: 0.6, brands[1]: 0.4}
            res_mix, ms = timed(lambda: svc.mix_styles(
                prompt="luxury streetwear capsule ensemble",
                brand_weights=weights,
            ))
            chk(r, "mix_styles() -> success", res_mix.success,
                res_mix.message if not res_mix.success else "", ms)
            if res_mix.success and isinstance(res_mix.data, dict):
                chk(r, "mix_styles() -> 'image' in data", "image" in res_mix.data)

        # Empty prompt guard
        res_empty = svc.generate_with_brand(prompt="", brand=brands[0])
        chk(r, "Empty prompt -> success=False", not res_empty.success)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_rag_service() -> MR:
    sec("Fashion Assistant -- RAGService (mock_mode=True)")
    r = MR("Fashion Assistant / RAGService")
    try:
        from week6.services.rag_service import RAGService
        svc = RAGService(mock_mode=True)

        health, ms = timed(lambda: svc.health_check())
        chk(r, "health_check() -> success", health.success, ms=ms)

        questions = [
            "What is quiet luxury in fashion?",
            "How do I build a capsule wardrobe?",
            "Tell me about Nike aesthetics.",
            "What fabrics work best for summer?",
        ]
        for q in questions:
            res, ms = timed(lambda question=q: svc.chat(question, history=[]))
            chk(r, f"chat('{q[:35]}...') -> success", res.success,
                res.message if not res.success else "", ms)
            if res.success:
                chk(r, "  -> 'response' key in data", "response" in (res.data or {}))
                chk(r, "  -> 'citations' key in data", "citations" in (res.data or {}))
                chk(r, "  -> 'history' key in data",  "history"   in (res.data or {}))

        # Semantic search
        res_s, ms = timed(lambda: svc.semantic_search("silk fabric properties", n_results=3))
        chk(r, "semantic_search() -> success", res_s.success, ms=ms)
        if res_s.success:
            chk(r, "semantic_search() -> list result", isinstance(res_s.data, list))

        # Q&A mode
        res_qa, ms = timed(lambda: svc.answer_question("What is Gucci known for?"))
        chk(r, "answer_question() -> success", res_qa.success, ms=ms)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_recommendation_service() -> MR:
    sec("Recommendation Hub -- RecommendationService & TrendService (mock)")
    r = MR("Recommendation Hub / RecommendationService")
    try:
        from week6.services.recommendation_service import RecommendationService
        from week6.services.trend_service import TrendService

        rec_svc = RecommendationService(mock_mode=True)
        health, ms = timed(lambda: rec_svc.health_check())
        chk(r, "RecommendationService.health_check() -> success", health.success, ms=ms)

        # Style recommendations
        for occasion in ["casual", "formal", "business casual"]:
            res, ms = timed(lambda o=occasion: rec_svc.recommend_styles(
                gender="unisex", style="minimalist", occasion=o, fit="regular", n=4
            ))
            chk(r, f"recommend_styles(occasion={occasion}) -> success", res.success, ms=ms)
            if res.success:
                chk(r, f"  -> data is list", isinstance(res.data, list))
                chk(r, f"  -> len(data) > 0", len(res.data) > 0)

        # Brand recommendations
        for style in ["streetwear", "luxury", "minimalist"]:
            res, ms = timed(lambda s=style: rec_svc.recommend_brands(
                styles=[s], aesthetic="clean understated quality", n=4
            ))
            chk(r, f"recommend_brands(style={style}) -> success", res.success, ms=ms)
            if res.success:
                chk(r, f"  -> data is list", isinstance(res.data, list))

        # User profile
        for uid in ["demo_user", "user_001"]:
            res_p, ms = timed(lambda u=uid: rec_svc.get_user_profile(u))
            chk(r, f"get_user_profile({uid}) -> success", res_p.success, ms=ms)
            if res_p.success:
                chk(r, f"  -> data is dict", isinstance(res_p.data, dict))

        # TrendService
        trend_svc = TrendService(mock_mode=True)
        health_t, ms = timed(lambda: trend_svc.health_check())
        chk(r, "TrendService.health_check() -> success", health_t.success, ms=ms)

        for season in ["spring_summer", "autumn_winter"]:
            res, ms = timed(lambda s=season: trend_svc.forecast_season(s))
            chk(r, f"forecast_season({season}) -> success", res.success, ms=ms)
            if res.success:
                chk(r, f"  -> data is list", isinstance(res.data, list))

        res_all, ms = timed(lambda: trend_svc.get_all_trends())
        chk(r, "get_all_trends() -> success", res_all.success, ms=ms)

        res_explain, ms = timed(lambda: trend_svc.explain_trend("Quiet Luxury"))
        chk(r, "explain_trend('Quiet Luxury') -> success", res_explain.success, ms=ms)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_gallery_service() -> MR:
    sec("Gallery -- HistoryManager (disk-backed, CPU-only)")
    r = MR("Gallery / HistoryManager")
    try:
        from week6.services.history_manager import HistoryManager
        from PIL import Image as PILImage

        mgr = HistoryManager()

        # List entries (may be empty)
        (entries, total), ms = timed(lambda: mgr.list_entries(page=1, page_size=10))
        chk(r, "list_entries() returns (list, int)", isinstance(entries, list) and isinstance(total, int), ms=ms)

        # Record a synthetic entry
        dummy_img = PILImage.new("RGB", (256, 256), color=(200, 150, 100))
        tmp_img_path = ROOT / "outputs" / "__test_mock_img.png"
        tmp_img_path.parent.mkdir(exist_ok=True)
        dummy_img.save(tmp_img_path)

        entry, ms = timed(lambda: mgr.record(
            prompt="silk evening gown validation test",
            image_path=str(tmp_img_path),
            model="mock-sdxl",
            service="generation",
            seed=42,
            steps=25,
            guidance_scale=7.5,
        ))
        chk(r, "record() returns HistoryEntry", entry is not None, ms=ms)
        entry_id = entry.id if entry else None

        if entry_id:
            # Retrieve
            fetched = mgr.get_entry(entry_id)
            chk(r, "get_entry() returns HistoryEntry", fetched is not None)
            if fetched:
                chk(r, "  -> prompt matches", "silk evening gown" in fetched.prompt)

            # Update (rating + notes)
            updated = mgr.update_entry(entry_id, rating=5, notes="Mock validation note")
            chk(r, "update_entry() -> not None", updated is not None)

            # Search
            results, n_hits = mgr.search("silk")
            chk(r, "search('silk') -> list", isinstance(results, list))

            # Export formats
            for fmt in ["json", "csv", "markdown"]:
                try:
                    if fmt == "json":
                        path = mgr.export_json()
                    elif fmt == "csv":
                        path = mgr.export_csv()
                    else:
                        path = mgr.export_markdown()
                    chk(r, f"export_{fmt}() -> path returned", bool(path))
                except Exception as ex:
                    chk(r, f"export_{fmt}()", False, str(ex))

            # Delete
            deleted = mgr.delete_entry(entry_id, delete_file=False)
            chk(r, "delete_entry() -> True", deleted)
            gone = mgr.get_entry(entry_id)
            chk(r, "After delete: get_entry() -> None", gone is None)

        # Stats
        stats = mgr.get_stats()
        chk(r, "get_stats() -> dict", isinstance(stats, dict))

        # Cleanup
        if tmp_img_path.exists():
            tmp_img_path.unlink()

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_evaluation_service() -> MR:
    sec("Evaluation Dashboard -- EvaluationService (mock_mode=True)")
    r = MR("Evaluation / EvaluationService")
    try:
        from week6.services.eval_service import EvaluationService
        svc = EvaluationService(mock_mode=True)

        health, ms = timed(lambda: svc.health_check())
        chk(r, "health_check() -> success", health.success, ms=ms)

        # Run evaluation (first call)
        res, ms = timed(lambda: svc.run_evaluation())
        chk(r, "run_evaluation() -> success", res.success,
            res.message if not res.success else "", ms)

        if res.success:
            report = res.data or {}
            chk(r, "  -> 'summary' key in report",    "summary"    in report)
            chk(r, "  -> 'test_cases' key in report", "test_cases" in report)
            chk(r, "  -> 'timestamp' key in report",  "timestamp"  in report)
            chk(r, "  -> test_cases is list",
                isinstance(report.get("test_cases", []), list))
            summary = report.get("summary", {})
            chk(r, "  -> summary is non-empty dict", isinstance(summary, dict) and len(summary) > 0)

        # Second run (idempotent)
        res2, ms2 = timed(lambda: svc.run_evaluation())
        chk(r, "run_evaluation() second run -> success", res2.success, ms=ms2)

    except Exception:
        chk(r, "Module import / init", False, traceback.format_exc())
    return r


def validate_gradio_page_build() -> MR:
    sec("Gradio UI -- All Pages Build Without Error (mock mode)")
    r = MR("Gradio Pages / UI Build")
    try:
        import gradio as gr
        from week6.services.generation_service import GenerationService
        from week6.services.controlnet_service import ControlNetService
        from week6.services.lora_service import LoRAService
        from week6.services.rag_service import RAGService
        from week6.services.recommendation_service import RecommendationService
        from week6.services.trend_service import TrendService
        from week6.services.eval_service import EvaluationService

        gen_svc   = GenerationService(mock_mode=True)
        cn_svc    = ControlNetService(mock_mode=True)
        lora_svc  = LoRAService(mock_mode=True)
        rag_svc   = RAGService(mock_mode=True)
        rec_svc   = RecommendationService(mock_mode=True)
        trend_svc = TrendService(mock_mode=True)
        eval_svc  = EvaluationService(mock_mode=True)

        pages = {
            "Text-to-Fashion":   ("week6.pages.text_to_fashion",  "build_text_to_fashion_page",  [gen_svc]),
            "Style Studio":      ("week6.pages.style_studio",     "build_style_studio_page",     [gen_svc]),
            "Sketch2Design":     ("week6.pages.sketch_to_design", "build_sketch_to_design_page", [cn_svc]),
            "Brand Switcher":    ("week6.pages.style_switcher",   "build_style_switcher_page",   [lora_svc]),
            "Brand Mixer":       ("week6.pages.style_mixer",      "build_style_mixer_page",      [lora_svc]),
            "ControlNet Page":   ("week6.pages.controlnet_page",  "build_controlnet_page",       [cn_svc]),
            "Fashion Assistant": ("week6.pages.fashion_assistant","build_fashion_assistant_page",[rag_svc]),
            "Fashion QA":        ("week6.pages.fashion_qa",       "build_fashion_qa_page",       [rag_svc]),
            "Recommendations":   ("week6.pages.recommendations",  "build_recommendations_page",  [rec_svc, trend_svc]),
            "Recommend Hub":     ("week6.pages.recommend_hub",    "build_recommend_hub_page",    [rec_svc]),
            "Trend Explorer":    ("week6.pages.trend_explorer",   "build_trend_explorer_page",   [trend_svc]),
            "Gallery":           ("week6.pages.gallery",          "build_gallery_page",          []),
            "Eval Dashboard":    ("week6.pages.eval_dashboard",   "build_eval_dashboard_page",   [eval_svc]),
        }

        for page_name, (module_path, fn_name, args) in pages.items():
            try:
                mod = importlib.import_module(module_path)
                fn  = getattr(mod, fn_name)
                t0  = time.perf_counter()
                with gr.Blocks():
                    fn(*args)
                ms = (time.perf_counter() - t0) * 1000
                chk(r, f"{page_name} page builds OK", True, ms=ms)
            except Exception as ex:
                chk(r, f"{page_name} page builds OK", False, str(ex)[:250])

    except Exception:
        chk(r, "Gradio import / service init", False, traceback.format_exc())
    return r


# ══════════════════════════════════════════════════════════════════════════════
# REPORT RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_markdown(modules: List[MR], total_ms: float, ts: str) -> str:
    total_pass = sum(m.passed for m in modules)
    total_all  = sum(m.total  for m in modules)
    all_ok = total_pass == total_all
    badge  = "ALL PASS" if all_ok else "FAILURES DETECTED"

    md = f"""# Mock Mode Validation Report — AI Fashion Creative Studio

**Generated**: {ts}
**Environment**: CPU-only | Mock Mode Enabled | No GPU Required
**Overall Status**: {badge}
**Total Assertions**: **{total_pass} / {total_all} passed**
**Total Runtime**: {total_ms/1000:.2f}s

---

## Module Summary

| Module | Pass | Total | Status | Notes |
|:---|:---:|:---:|:---:|:---|
"""
    for m in modules:
        status = "PASS" if m.ok else "FAIL"
        failed = [a.name for a in m.assertions if not a.passed]
        notes  = (", ".join(a.name[:50] for a in m.assertions if not a.passed))[:80] if failed else "All assertions passed"
        md += f"| {m.module} | {m.passed} | {m.total} | {status} | {notes} |\n"

    md += "\n---\n\n## Detailed Assertion Results\n"

    for m in modules:
        md += f"\n### {m.module}\n\n"
        md += "| # | Assertion | Result | Latency |\n"
        md += "|:---:|:---|:---:|---:|\n"
        for i, a in enumerate(m.assertions, 1):
            icon   = "PASS" if a.passed else "FAIL"
            lat    = f"{a.latency_ms:.0f}ms" if a.latency_ms else "—"
            detail = f" ← _{a.detail.strip().split(chr(10))[0][:100]}_" if not a.passed and a.detail else ""
            md += f"| {i} | {a.name}{detail} | {icon} | {lat} |\n"

    md += f"""
---

## Mock Mode Architecture

All services implement a **dual-mode** design pattern:

| Mode | Behaviour |
|:---|:---|
| `mock_mode=True` | Returns deterministic CPU-generated PIL images, seeded metadata, JSON fixtures — **zero GPU required** |
| `mock_mode=False` | Routes to real SDXL / ControlNet / LoRA inference (requires CUDA GPU + model weights) |

### Per-Service Mock Strategy

| Service | Mock Strategy |
|:---|:---|
| `GenerationService` | Generates a programmatic gradient PIL image with metadata; no diffusion model loaded |
| `ControlNetService` | Applies CPU OpenCV Canny edge detection for preprocessing; returns mock generated image |
| `LoRAService` | Applies brand colour-tint to base image; `mix_styles` blends colour channels proportionally |
| `RAGService` | Uses in-memory ChromaDB with deterministic mock embeddings; real retrieval pipeline runs on CPU |
| `RecommendationService` | Returns seeded style/brand fixtures from `fashion_knowledge_base.json` |
| `TrendService` | Returns forecasts from pre-loaded `trend_dataset.json` |
| `EvaluationService` | Computes CLIP/FID scores with fallback cosine similarity against mock image pairs |
| `HistoryManager` | Full disk I/O via SQLite-backed storage; tested end-to-end without GPU |

---

## Conclusion

{"**All modules pass every assertion in mock mode.** The application is production-ready for CPU/demo deployment and can be presented without a GPU." if all_ok else "**Some modules have assertion failures.** See the detailed table above for specifics."}

> Generated by `scripts/mock_mode_validation.py`
"""
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n{'='*62}")
    print(f"  AI FASHION STUDIO -- MOCK MODE VALIDATION")
    print(f"  {ts}")
    print(f"{'='*62}")

    t_start = time.perf_counter()

    validators = [
        validate_generation_service,
        validate_controlnet_service,
        validate_lora_service,
        validate_rag_service,
        validate_recommendation_service,
        validate_gallery_service,
        validate_evaluation_service,
        validate_gradio_page_build,
    ]

    modules: List[MR] = []
    for validator in validators:
        try:
            modules.append(validator())
        except Exception:
            name = validator.__name__.replace("validate_", "").replace("_", " ").title()
            mr = MR(name)
            mr.assertions.append(AR(name, False, traceback.format_exc()))
            modules.append(mr)

    total_ms = (time.perf_counter() - t_start) * 1000
    total_pass = sum(m.passed for m in modules)
    total_all  = sum(m.total  for m in modules)

    print(f"\n{'='*62}")
    print(f"  VALIDATION COMPLETE")
    print(f"  {total_pass}/{total_all} assertions passed  |  {total_ms/1000:.2f}s")
    for m in modules:
        icon = "OK" if m.ok else "!!"
        print(f"  [{icon}]  {m.module}: {m.passed}/{m.total}")
    print(f"{'='*62}\n")

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    md = render_markdown(modules, total_ms, ts)

    md_path = out_dir / "mock_validation_report.md"
    md_path.write_text(md, encoding="utf-8")

    root_path = ROOT / "mock_validation_report.md"
    root_path.write_text(md, encoding="utf-8")

    json_data = {
        "timestamp": ts,
        "total_pass": total_pass,
        "total_assertions": total_all,
        "runtime_ms": round(total_ms, 2),
        "overall_status": "PASS" if total_pass == total_all else "FAIL",
        "modules": [
            {
                "module": m.module,
                "passed": m.passed,
                "total": m.total,
                "status": "PASS" if m.ok else "FAIL",
                "assertions": [
                    {"name": a.name, "passed": a.passed,
                     "detail": a.detail[:200], "latency_ms": round(a.latency_ms, 2)}
                    for a in m.assertions
                ],
            }
            for m in modules
        ],
    }
    json_path = out_dir / "mock_validation_results.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"Report  -> {md_path}")
    print(f"JSON    -> {json_path}")
    print(f"Root MD -> {root_path}\n")

    sys.exit(0 if total_pass == total_all else 1)


if __name__ == "__main__":
    main()
