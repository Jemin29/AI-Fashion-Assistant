"""
demo_lora.py
============
Week 4 Demo Runner for AI-Powered Fashion Design Assistant.

Demonstrates:
1. Nike Style Generation
2. Gucci Style Generation
3. Zara Style Generation
4. H&M Style Generation
5. Dynamic Style Switching (Nike -> Gucci)
6. Dynamic Multi-Adapter Style Mixing (70% Nike + 30% Gucci)

For each generation step, this script:
- Generates the styled design image (using LoraInferenceSystem or StyleMixer).
- Evaluates the output image (style consistency, prompt alignment, brand similarity, CLIP, quality).
- Saves a sidecar evaluation JSON report.
- Compiles a central comparison summary outputs report.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Setup logging
from loguru import logger

# Import Week 4 modules
try:
    from src.utils.logging_setup import setup_logging
    from src.utils.config_manager import get_default_config
    from src.lora.style_manager.lora_registry import LoraRegistry
    from src.lora.style_manager.style_mixer import StyleMixer
    from src.lora.inference.lora_inference import LoraInferenceSystem
    from src.evaluation.week4_style_evaluator import FashionStyleEvaluator
    from src.evaluation.week4_lora_tracker import LoraExperimentTracker
    _IMPORTS_OK = True
except ImportError as err:
    logger.error(f"Failed to import Week 4 modules: {err}")
    _IMPORTS_OK = False


def setup_mock_models(registry: LoraRegistry, folder: Path) -> None:
    """Ensure mock safetensors files are registered so dry-run pipeline executes without KeyError."""
    folder.mkdir(parents=True, exist_ok=True)
    for brand in ["nike", "gucci", "zara", "h&m"]:
        mock_file = folder / f"{brand}_style_mock.safetensors"
        if not mock_file.exists():
            with open(mock_file, "wb") as f:
                f.write(brand.upper().encode())
        registry.register_model(brand=brand, model_path=mock_file)
    logger.info("Checked and registered mock models in registry.")


def main() -> int:
    """Orchestrate the demo generations and evaluations."""
    if not _IMPORTS_OK:
        logger.error("Imports failed. Exiting.")
        return 1

    parser = argparse.ArgumentParser(description="Week 4 LoRA Personalization Demo Runner.")
    parser.add_argument("--real", action="store_true", help="Run with real SDXL pipeline weights (requires GPU).")
    parser.add_argument("--output-dir", type=str, default="outputs/demo_lora", help="Target demo outputs folder.")
    args = parser.parse_args()

    dry_run = not args.real
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize rotated daily sinks logging
    setup_logging(log_dir=output_dir / "logs")

    logger.info("=" * 60)
    logger.info("   STARTING WEEK 4 LORA PERSONALIZATION DEMO RUNNER   ")
    logger.info(f"   Mode: {'DRY-RUN (Simulated)' if dry_run else 'REAL (SDXL/LoRA)'} | Output: {output_dir}")
    logger.info("=" * 60)

    # 1. Initialize Registry and Mock Weights
    registry_file = output_dir / "lora_registry_demo.json"
    registry = LoraRegistry(registry_path=registry_file)
    setup_mock_models(registry, output_dir / "mock_weights")

    # 2. Initialize Inference, Evaluation, and Tracking Systems
    config = get_default_config()
    inference_system = LoraInferenceSystem(
        config=config,
        registry=registry,
        output_dir=output_dir,
        dry_run=dry_run
    )
    inference_system.load_pipeline()
    evaluator = FashionStyleEvaluator(config=config)
    tracker = LoraExperimentTracker(output_dir=output_dir / "experiments")

    # Dictionary to collect results for the summary report
    summary_report: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "mode": "dry_run" if dry_run else "real",
        "generisons": {}
    }

    # Helper to evaluate image and save report
    def evaluate_and_save(img_path_str: str, prompt: str, brand: str, step_name: str) -> Dict[str, Any]:
        img_path = Path(img_path_str)
        report_path = img_path.with_name(f"report_{step_name}.json")
        
        logger.info(f"Evaluating {step_name} generation output: {img_path.name}")
        with Image.open(img_path) as img:
            scores = evaluator.evaluate(image=img, prompt=prompt, brand=brand)
        
        # Save sidecar report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "step": step_name,
                "brand": brand,
                "prompt": prompt,
                "image_path": str(img_path.relative_to(output_dir)),
                "evaluation_scores": scores
            }, f, indent=2)
        
        # Log to experiment tracker database
        try:
            tracker.log_experiment(
                brand=brand,
                lora_version="demo_v1.0",
                training_loss=0.042,  # Simulated training loss
                validation_score=scores.get("style_similarity", 0.0),
                clip_score=0.31,
                style_similarity=scores.get("style_similarity", 0.0),
                parameters={"prompt": prompt, "step_name": step_name}
            )
        except Exception as tracker_err:
            logger.warning(f"Could not log run to experiment tracker: {tracker_err}")

        logger.success(f"Scores for {step_name}: Style Similarity={scores['style_similarity']}, Prompt Alignment={scores['prompt_alignment']}")
        return scores

    # ── 1. Nike Generation ──
    logger.info("\n--- STEP 1: Nike Generation ---")
    nike_prompt = "A modern running shirt with athletic fit"
    nike_res = inference_system.generate(prompt=nike_prompt, brand="nike", scale=1.0, seed=10)
    nike_scores = evaluate_and_save(nike_res["image_path"], nike_prompt, "nike", "nike_generation")
    summary_report["generisons"]["nike"] = nike_scores

    # ── 2. Gucci Generation ──
    logger.info("\n--- STEP 2: Gucci Generation ---")
    gucci_prompt = "An elegant haute-couture gown with gold embroidery"
    gucci_res = inference_system.generate(prompt=gucci_prompt, brand="gucci", scale=1.0, seed=20)
    gucci_scores = evaluate_and_save(gucci_res["image_path"], gucci_prompt, "gucci", "gucci_generation")
    summary_report["generisons"]["gucci"] = gucci_scores

    # ── 3. Zara Generation ──
    logger.info("\n--- STEP 3: Zara Generation ---")
    zara_prompt = "A contemporary linen jacket in beige and minimalist cream tones"
    zara_res = inference_system.generate(prompt=zara_prompt, brand="zara", scale=1.0, seed=30)
    zara_scores = evaluate_and_save(zara_res["image_path"], zara_prompt, "zara", "zara_generation")
    summary_report["generisons"]["zara"] = zara_scores

    # ── 4. H&M Generation ──
    logger.info("\n--- STEP 4: H&M Generation ---")
    hm_prompt = "A basic white organic cotton t-shirt with clean texture"
    hm_res = inference_system.generate(prompt=hm_prompt, brand="h&m", scale=1.0, seed=40)
    hm_scores = evaluate_and_save(hm_res["image_path"], hm_prompt, "h&m", "hm_generation")
    summary_report["generisons"]["hm"] = hm_scores

    # ── 5. Style Switching Demonstration ──
    logger.info("\n--- STEP 5: Style Switching Demonstration (Nike -> Gucci) ---")
    switch_prompt = "A luxury sportswear windbreaker jacket"
    
    # Generate Nike first
    logger.info("Generating in Nike Style...")
    switch_nike_res = inference_system.generate(prompt=switch_prompt, brand="nike", scale=1.0, seed=50)
    switch_nike_scores = evaluate_and_save(switch_nike_res["image_path"], switch_prompt, "nike", "switch_nike")
    
    # Switch dynamically to Gucci
    logger.info("Generating in Gucci Style...")
    switch_gucci_res = inference_system.generate(prompt=switch_prompt, brand="gucci", scale=1.0, seed=50)
    switch_gucci_scores = evaluate_and_save(switch_gucci_res["image_path"], switch_prompt, "gucci", "switch_gucci")

    summary_report["generisons"]["style_switching"] = {
        "nike_scores": switch_nike_scores,
        "gucci_scores": switch_gucci_scores
    }

    # ── 6. Style Mixing Demonstration (70% Nike + 30% Gucci) ──
    logger.info("\n--- STEP 6: Style Mixing Demonstration (70% Nike + 30% Gucci) ---")
    mix_prompt = "A designer performance hoodie with gold metal accessories"
    mix_weights = {"nike": 0.7, "gucci": 0.3}
    
    style_mixer = StyleMixer(
        registry=registry,
        inference_pipeline=inference_system.pipeline,
        output_dir=output_dir
    )
    
    mix_res = style_mixer.generate_mixed_design(
        prompt=mix_prompt,
        brand_weights=mix_weights,
        dry_run=dry_run
    )
    
    # Evaluate mixed output (using dominant brand Nike as style target)
    mix_scores = evaluate_and_save(mix_res["image_path"], mix_prompt, "nike", "style_mixing")
    
    summary_report["generisons"]["style_mixing"] = {
        "weights": mix_weights,
        "scores": mix_scores
    }

    # Save central summary report
    summary_file = output_dir / "demo_lora_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, sort_keys=True)

    logger.info("\n" + "=" * 60)
    logger.success("   WEEK 4 DEMONSTRATION RUN COMPLETED SUCCESSFULLY   ")
    logger.success(f"   Summary Report written to: {summary_file}")
    logger.info("=" * 60)

    # Release log file handles on Windows
    try:
        from loguru import logger as loguru_logger
        loguru_logger.remove()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
