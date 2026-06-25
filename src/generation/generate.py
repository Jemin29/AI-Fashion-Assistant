"""
week2/generate.py
==================
One-click CLI entry point for the FashionSDXLGenerator.

Usage
-----
    # Basic usage (single image)
    python week2/generate.py --prompt "A woman in an elegant red silk gown"

    # With style preset and custom size
    python week2/generate.py \
        --prompt "Tailored navy suit with gold buttons" \
        --negative "blurry, deformed, watermark" \
        --style formal \
        --size portrait_1024 \
        --steps 35 \
        --guidance 8.0 \
        --seed 42

    # Batch mode (from file or inline)
    python week2/generate.py \
        --batch-file prompts.txt \
        --style streetwear \
        --n 2

    # High-quality run
    python week2/generate.py \
        --prompt "A luxurious fur coat" \
        --steps 50 \
        --scheduler dpm++ \
        --refiner

    # List available options
    python week2/generate.py --list-sizes
    python week2/generate.py --list-schedulers
    python week2/generate.py --info
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ── Make week2 importable from project root ────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
except ImportError:
    import logging
    logger = logging.getLogger("generate")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from src.generation.generator.sdxl_generator import (
    FashionSDXLGenerator,
    SIZE_PRESETS,
    SCHEDULER_MAP,
)
from src.generation.prompts.style_presets import list_presets, get_preset
from src.generation.prompts.negative_prompts import get_fashion_negative, format_negative


# =============================================================================
# ── CLI Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python week2/generate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        ╔══════════════════════════════════════════════════════╗
        ║      AI Fashion Design Assistant — Week 2            ║
        ║       FashionSDXLGenerator CLI                       ║
        ╚══════════════════════════════════════════════════════╝
        Generate fashion images using Stable Diffusion XL.
        """),
        epilog=textwrap.dedent("""
        Examples:
          python week2/generate.py --prompt "A red silk dress" --style luxury
          python week2/generate.py --batch-file prompts.txt --style casual --n 1
          python week2/generate.py --list-sizes
        """),
    )

    # ── Prompt inputs ─────────────────────────────────────────────────────
    input_grp = parser.add_argument_group("Prompt")
    input_grp.add_argument(
        "--prompt", "-p",
        type=str, default=None,
        help="Positive prompt text.",
    )
    input_grp.add_argument(
        "--negative", "-n-prompt",
        type=str, default=None,
        help="Negative prompt (default: built-in fashion negatives).",
    )
    input_grp.add_argument(
        "--batch-file", "-b",
        type=Path, default=None,
        help="Path to a .txt file with one prompt per line (batch mode).",
    )
    input_grp.add_argument(
        "--style", "-s",
        type=str, default=None,
        choices=list_presets() + [None],
        help="Style preset name to apply (adds quality/style tags).",
        metavar="STYLE",
    )

    # ── Generation parameters ─────────────────────────────────────────────
    gen_grp = parser.add_argument_group("Generation")
    gen_grp.add_argument(
        "--size", "-sz",
        type=str, default="square_1024",
        choices=list(SIZE_PRESETS.keys()),
        help="Image size preset (default: square_1024 = 1024×1024).",
    )
    gen_grp.add_argument(
        "--width", "-W",
        type=int, default=None,
        help="Custom width in pixels (overrides --size).",
    )
    gen_grp.add_argument(
        "--height", "-H",
        type=int, default=None,
        help="Custom height in pixels (overrides --size).",
    )
    gen_grp.add_argument(
        "--steps", "-st",
        type=int, default=30,
        help="Denoising steps (default: 30). Use 15-20 for drafts, 50 for HQ.",
    )
    gen_grp.add_argument(
        "--guidance", "-g",
        type=float, default=7.5,
        help="Guidance scale (default: 7.5). Higher = more prompt-faithful.",
    )
    gen_grp.add_argument(
        "--seed",
        type=int, default=-1,
        help="Seed for reproducibility (-1 = random, default).",
    )
    gen_grp.add_argument(
        "--n",
        type=int, default=1,
        dest="num_images",
        help="Number of images per prompt (default: 1).",
    )
    gen_grp.add_argument(
        "--scheduler", "-sc",
        type=str, default="euler",
        choices=list(SCHEDULER_MAP.keys()),
        help="Noise scheduler (default: euler).",
    )
    gen_grp.add_argument(
        "--refiner",
        action="store_true",
        default=False,
        help="Enable SDXL refiner pass (requires ~16 GB VRAM total).",
    )
    gen_grp.add_argument(
        "--refiner-strength",
        type=float, default=0.3,
        help="Refiner denoising strength (0.1–0.5, default: 0.3).",
    )

    # ── Model / hardware ──────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model & Hardware")
    model_grp.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model repo ID.",
    )
    model_grp.add_argument(
        "--device", "-d",
        type=str, default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device (default: auto-detect).",
    )
    model_grp.add_argument(
        "--dtype",
        type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype (default: float16).",
    )
    model_grp.add_argument(
        "--low-vram",
        action="store_true",
        help="Force model CPU offload (good for 8 GB VRAM).",
    )
    model_grp.add_argument(
        "--sequential-offload",
        action="store_true",
        help="Enable sequential CPU offload (minimum VRAM, slower).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output-dir", "-o",
        type=Path, default=Path("week2/outputs/generated"),
        help="Output directory (default: week2/outputs/generated).",
    )
    out_grp.add_argument(
        "--format", "-f",
        type=str, default="png",
        choices=["png", "jpg", "webp"],
        help="Image format (default: png).",
    )
    out_grp.add_argument(
        "--quality",
        type=int, default=95,
        help="JPEG/WebP quality 1-100 (default: 95, ignored for PNG).",
    )
    out_grp.add_argument(
        "--prefix",
        type=str, default="fashion",
        help="Filename prefix (default: fashion).",
    )
    out_grp.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip JSON metadata sidecar files.",
    )

    # ── Info flags ────────────────────────────────────────────────────────
    info_grp = parser.add_argument_group("Information")
    info_grp.add_argument(
        "--list-sizes",
        action="store_true",
        help="List all available size presets and exit.",
    )
    info_grp.add_argument(
        "--list-schedulers",
        action="store_true",
        help="List all available schedulers and exit.",
    )
    info_grp.add_argument(
        "--list-styles",
        action="store_true",
        help="List all available style presets and exit.",
    )
    info_grp.add_argument(
        "--info",
        action="store_true",
        help="Show generator info (device, VRAM, etc.) and exit.",
    )

    return parser


# =============================================================================
# ── Info helpers
# =============================================================================

def print_sizes() -> None:
    print("\n  Available Size Presets")
    print("  ─────────────────────────────────────────")
    for name, (w, h) in SIZE_PRESETS.items():
        print(f"  {name:<20} {w}×{h}")
    print()


def print_schedulers() -> None:
    print("\n  Available Schedulers")
    print("  ─────────────────────────────────────────")
    for name, cls in SCHEDULER_MAP.items():
        print(f"  {name:<14}  {cls}")
    print()


def print_styles() -> None:
    print("\n  Available Style Presets")
    print("  ─────────────────────────────────────────")
    for name in list_presets():
        try:
            p = get_preset(name)
            print(f"  {name:<18}  {p.description[:55]}")
        except Exception:
            print(f"  {name}")
    print()


def print_generator_info(gen: FashionSDXLGenerator) -> None:
    info = gen.get_info()
    print("\n  FashionSDXLGenerator Info")
    print("  ─────────────────────────────────────────")
    for k, v in info.items():
        if isinstance(v, list):
            print(f"  {k:<25} [{', '.join(str(x) for x in v[:4])}{'...' if len(v) > 4 else ''}]")
        else:
            print(f"  {k:<25} {v}")
    print()


# =============================================================================
# ── Prompt building helpers
# =============================================================================

def build_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    """
    Return a list of (positive, negative) prompt pairs.
    Sources: --prompt, --batch-file.
    Applies style preset tags if --style is given.
    """
    raw_prompts: list[str] = []

    if args.batch_file:
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            logger.error("Batch file not found: {}", batch_path)
            sys.exit(1)
        raw_prompts = [
            line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        logger.info("Loaded {} prompts from {}", len(raw_prompts), batch_path)

    if args.prompt:
        raw_prompts.append(args.prompt)

    if not raw_prompts:
        logger.error("No prompt provided. Use --prompt or --batch-file.")
        sys.exit(1)

    # ── Apply style preset to each prompt ────────────────────────────────
    pairs: list[tuple[str, str]] = []
    for raw in raw_prompts:
        positive = raw

        if args.style:
            try:
                preset   = get_preset(args.style)
                tags     = ", ".join(preset.positive_tags + preset.quality_tags)
                positive = f"{raw}, {tags}" if tags else raw
            except KeyError:
                logger.warning("Unknown style {!r} — skipping style tags", args.style)

        negative = args.negative if args.negative else format_negative(get_fashion_negative())
        pairs.append((positive, negative))

    return pairs


# =============================================================================
# ── Main
# =============================================================================

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Info-only flags ───────────────────────────────────────────────────
    if args.list_sizes:
        print_sizes()
        return 0

    if args.list_schedulers:
        print_schedulers()
        return 0

    if args.list_styles:
        print_styles()
        return 0

    if args.info:
        gen = FashionSDXLGenerator(
            model_id   = args.model,
            device     = args.device,
            torch_dtype= args.dtype,
            output_dir = args.output_dir,
        )
        print_generator_info(gen)
        return 0

    # ── Resolve width / height ────────────────────────────────────────────
    if args.width and args.height:
        width, height = args.width, args.height
        size_preset   = None
    else:
        size_preset   = args.size
        width, height = SIZE_PRESETS.get(size_preset, (1024, 1024))

    # ── Build prompt pairs ────────────────────────────────────────────────
    prompt_pairs = build_prompts(args)
    n_prompts    = len(prompt_pairs)
    logger.info(
        "Starting generation | prompts={} | images_each={} | {}×{}",
        n_prompts, args.num_images, width, height,
    )

    # ── Initialise generator ──────────────────────────────────────────────
    gen = FashionSDXLGenerator(
        model_id       = args.model,
        device         = args.device,
        torch_dtype    = args.dtype,
        output_dir     = args.output_dir,
        enable_refiner = args.refiner,
        scheduler      = args.scheduler,
    )

    logger.info("Loading model…")
    try:
        gen.load_model(
            low_vram_mode      = args.low_vram,
            sequential_offload = args.sequential_offload,
        )
    except RuntimeError as e:
        logger.error("Model load failed: {}", e)
        return 1

    # ── Generate ──────────────────────────────────────────────────────────
    all_saved: list[Path] = []
    all_results           = []

    if n_prompts == 1:
        prompt, negative = prompt_pairs[0]
        result = gen.generate_image(
            prompt               = prompt,
            negative_prompt      = negative,
            width                = width,
            height               = height,
            num_inference_steps  = args.steps,
            guidance_scale       = args.guidance,
            seed                 = args.seed,
            num_images           = args.num_images,
            scheduler            = args.scheduler,
            use_refiner          = args.refiner,
            refiner_strength     = args.refiner_strength,
        )
        all_results.append(result)
    else:
        positives = [p for p, _ in prompt_pairs]
        negatives = [n for _, n in prompt_pairs]
        results   = gen.generate_batch(
            prompts              = positives,
            negative_prompts     = negatives,
            width                = width,
            height               = height,
            num_inference_steps  = args.steps,
            guidance_scale       = args.guidance,
            seeds                = [args.seed] * n_prompts,
            num_images_per_prompt= args.num_images,
            scheduler            = args.scheduler,
            use_refiner          = args.refiner,
            refiner_strength     = args.refiner_strength,
        )
        all_results.extend(results)

    # ── Save outputs ──────────────────────────────────────────────────────
    passed = 0
    for result in all_results:
        if not result.success:
            logger.error("Generation failed: {}", result.error)
            continue
        if not result.images:
            logger.warning("Result has no images — skipping save")
            continue
        try:
            paths = gen.save_output(
                result,
                output_dir      = args.output_dir,
                fmt             = args.format,
                quality         = args.quality,
                save_metadata   = not args.no_metadata,
                filename_prefix = args.prefix,
            )
            all_saved.extend(paths)
            passed += 1
        except Exception as e:
            logger.error("Save failed: {}", e)

    # ── Summary ───────────────────────────────────────────────────────────
    total_images = sum(len(r.images) for r in all_results if r.success)
    print()
    print("=" * 55)
    print(f"  Generation Complete")
    print("=" * 55)
    print(f"  Prompts processed  : {len(all_results)}")
    print(f"  Images generated   : {total_images}")
    print(f"  Images saved       : {len(all_saved)}")
    if all_saved:
        print(f"  Output directory   : {all_saved[0].parent}")
    print("=" * 55)

    if all_saved:
        print("\n  Saved files:")
        for p in all_saved[:10]:
            print(f"    {p}")
        if len(all_saved) > 10:
            print(f"    … and {len(all_saved) - 10} more")

    gen.unload_model()
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
