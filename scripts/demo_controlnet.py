"""
demo_controlnet.py
==================
Week 3 Demo Runner for AI-Powered Fashion Design Assistant.

Coordinates and demonstrates:
1. Sketch2Design (Sketch + Prompt -> Design)
2. Pose2Fashion (Pose + Prompt -> Design)
3. Depth2Fashion (Depth + Prompt -> Design)

For each pipeline, this script:
  - Loads an input image (generates default mock inputs if missing).
  - Preprocesses the input (edge extraction, skeleton joints extraction, depth map estimation).
  - Generates the standard unconditioned SDXL output and ControlNet conditioned output.
  - Evaluates both outputs using CLIP text-image similarity, grayscale SSIM, and structural layout scores.
  - Saves the generated design images alongside metadata and a sidecar report.
  - Aggregates results into a final unified `comparison_report.json`.

Usage
-----
    # Default: Run in fast mock/simulation mode (no GPU/heavy models required)
    python demo_controlnet.py

    # Run in real mode (requires CUDA GPU, downloaded SDXL and ControlNet checkpoints)
    python demo_controlnet.py --real
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Setup logging
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo_controlnet")

# Import Week 3 classes
try:
    from src.utils.config_manager import get_config
    from src.utils.logging_setup import setup_logging
    
    from src.controlnet.controlnet.sketch2design import Sketch2Design
    from src.controlnet.controlnet.pose2fashion import Pose2Fashion
    from src.controlnet.controlnet.depth2fashion import Depth2Fashion
    
    from src.evaluation.week3_comparison_engine import ComparisonEngine
    from src.evaluation.week3_structure_evaluator import StructureEvaluator
    
    _IMPORTS_OK = True
except ImportError as err:
    logger.error(f"Failed to import Week 3 modules: {err}")
    _IMPORTS_OK = False


# =============================================================================
# ── Input Image Generators (Fallback Mocks)
# =============================================================================

def create_mock_sketch(path: Path) -> Image.Image:
    """Draw a mock fashion sketch of a hoodie using black line art on a white canvas."""
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Hood
    draw.arc([180, 50, 332, 180], start=180, end=360, fill=(0, 0, 0), width=4)
    draw.line([180, 115, 220, 180], fill=(0, 0, 0), width=4)
    draw.line([332, 115, 292, 180], fill=(0, 0, 0), width=4)
    
    # Collar/neck crossover & drawstrings
    draw.line([220, 180, 292, 180], fill=(0, 0, 0), width=4)
    draw.line([240, 180, 245, 220], fill=(0, 0, 0), width=4)
    draw.line([270, 180, 265, 220], fill=(0, 0, 0), width=4)
    
    # Torso body
    draw.rectangle([180, 180, 332, 420], outline=(0, 0, 0), width=4)
    
    # Sleeves
    draw.line([180, 180, 120, 350], fill=(0, 0, 0), width=4)
    draw.line([120, 350, 150, 350], fill=(0, 0, 0), width=4)
    draw.line([150, 350, 180, 220], fill=(0, 0, 0), width=4)
    
    draw.line([332, 180, 392, 350], fill=(0, 0, 0), width=4)
    draw.line([392, 350, 362, 350], fill=(0, 0, 0), width=4)
    draw.line([362, 350, 332, 220], fill=(0, 0, 0), width=4)
    
    # Kangaroo pocket
    draw.polygon([(210, 320), (300, 320), (320, 380), (190, 380)], outline=(0, 0, 0), width=3)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    logger.info(f"Generated default mock sketch outline at: {path}")
    return img


def create_mock_pose_photo(path: Path) -> Image.Image:
    """Draw a simple mannequin model silhouette representing a human pose photo."""
    img = Image.new("RGB", (512, 512), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Head
    draw.ellipse([230, 60, 282, 120], fill=(225, 185, 150), outline=(50, 50, 50), width=2)
    # Neck
    draw.rectangle([250, 120, 262, 140], fill=(225, 185, 150), outline=(50, 50, 50), width=2)
    # Torso (colored shirt/body block)
    draw.polygon([(210, 140), (302, 140), (312, 320), (200, 320)], fill=(70, 130, 180), outline=(50, 50, 50), width=2)
    
    # Left Arm
    draw.line([210, 140, 160, 260], fill=(225, 185, 150), width=18)
    draw.ellipse([150, 250, 170, 270], fill=(225, 185, 150))
    
    # Right Arm
    draw.line([302, 140, 352, 260], fill=(225, 185, 150), width=18)
    draw.ellipse([342, 250, 362, 270], fill=(225, 185, 150))
    
    # Legs (trousers block)
    draw.rectangle([210, 320, 245, 480], fill=(50, 60, 120), outline=(50, 50, 50), width=2)
    draw.rectangle([267, 320, 302, 480], fill=(50, 60, 120), outline=(50, 50, 50), width=2)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    logger.info(f"Generated default mock pose mannequin photo at: {path}")
    return img


def create_mock_depth_photo(path: Path) -> Image.Image:
    """Draw a dress form mannequin with nested concentric gradients to simulate depth folds."""
    img = Image.new("RGB", (512, 512), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    cx, cy = 256, 270
    for offset in range(50, 0, -2):
        ratio = offset / 50.0
        # Draw nested torso polygons to simulate depth mapping (lighter center = closer)
        pts = [
            (int(cx + (220 - cx) * ratio), int(cy + (120 - cy) * ratio)),
            (int(cx + (292 - cx) * ratio), int(cy + (120 - cy) * ratio)),
            (int(cx + (332 - cx) * ratio), int(cy + (420 - cy) * ratio)),
            (int(cx + (180 - cx) * ratio), int(cy + (420 - cy) * ratio)),
        ]
        c_val = int(35 + (255 - 35) * (1.0 - ratio))
        draw.polygon(pts, fill=(c_val, c_val, c_val))
        
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    logger.info(f"Generated default mock depth mannequin photo at: {path}")
    return img


def create_mock_standard_image(prompt: str, size: Tuple[int, int], task_type: str) -> Image.Image:
    """Creates a basic textured image to represent standard unconditioned generation in mock mode."""
    w, h = size
    bg_colors = {
        "sketch": (100, 100, 120),
        "pose": (120, 100, 100),
        "depth": (100, 120, 100)
    }
    bg_color = bg_colors.get(task_type, (80, 80, 80))
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([15, 15, w - 15, h - 15], outline=(180, 180, 180), width=2)
    
    text_lines = [
        "STANDARD SDXL OUTPUT",
        f"Prompt: {prompt[:35]}...",
        "[UNCONDITIONED GENERATION]"
    ]
    y = h // 3
    for line in text_lines:
        draw.text((w // 2, y), line, fill=(255, 255, 255), anchor="mm")
        y += 40
        
    return img


# =============================================================================
# ── Core Demonstration Runner
# =============================================================================

def run_demo(mock_mode: bool, output_dir: Path, seed: int) -> int:
    """Run the complete Week 3 ControlNet orchestrator pipeline & evaluation demo."""
    logger.info(f"Starting ControlNet Demo Runner | Mode={'MOCK' if mock_mode else 'REAL'} | Seed={seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize config
    config = get_config() if _IMPORTS_OK else None
    
    # Initialize orchestrators & evaluation engines
    logger.info("Initializing Orchestrator and Evaluation instances...")
    sketch_orch = Sketch2Design(config=config, mock=mock_mode)
    pose_orch = Pose2Fashion(config=config, mock=mock_mode)
    depth_orch = Depth2Fashion(config=config, mock=mock_mode)
    
    comparison_engine = ComparisonEngine(config=config, mock=mock_mode)
    structure_evaluator = StructureEvaluator()
    
    # Setup inputs
    datasets_dir = Path("week3/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    sketch_in_path = datasets_dir / "sketch_input.png"
    pose_in_path = datasets_dir / "pose_input.png"
    depth_in_path = datasets_dir / "depth_input.png"
    
    # Load or generate default inputs
    if sketch_in_path.exists():
        logger.info(f"Loading existing sketch input: {sketch_in_path}")
        sketch_img = Image.open(sketch_in_path).convert("RGB")
    else:
        sketch_img = create_mock_sketch(sketch_in_path)
        
    if pose_in_path.exists():
        logger.info(f"Loading existing pose input: {pose_in_path}")
        pose_img = Image.open(pose_in_path).convert("RGB")
    else:
        pose_img = create_mock_pose_photo(pose_in_path)
        
    if depth_in_path.exists():
        logger.info(f"Loading existing depth input: {depth_in_path}")
        depth_img = Image.open(depth_in_path).convert("RGB")
    else:
        depth_img = create_mock_depth_photo(depth_in_path)

    # Collection structures for final batch comparison report
    standard_images = []
    controlnet_images = []
    condition_images = []  # The preprocessed edge/skeleton/depth maps
    prompts_list = []
    structural_metrics_list = []
    
    # =============================================================================
    # ── 1. Sketch2Design Pipeline
    # =============================================================================
    logger.info("\n=== RUNNING 1. SKETCH-TO-DESIGN PIPELINE ===")
    sketch_prompt = "Oversized charcoal gray streetwear hoodie with front kangaroo pocket, ribbed trim details"
    
    # Preprocess
    logger.info("Step 1.1: Preprocessing Sketch...")
    preprocessed_sketch = sketch_orch.processor.preprocess_sketch(sketch_img, method="canny")
    
    # Generate outputs
    logger.info("Step 1.2: Running Generation...")
    # Generate standard unconditioned image
    if mock_mode:
        std_sketch_out = create_mock_standard_image(sketch_prompt, sketch_img.size, "sketch")
    else:
        # Real standard generation uses same pipeline but with scale 0.0
        res_std = sketch_orch.generate_design(
            sketch=sketch_img,
            prompt=sketch_prompt,
            style="streetwear",
            method="canny",
            conditioning_scale=0.0,
            seed=seed
        )
        std_sketch_out = res_std.images[0] if res_std.success else create_mock_standard_image(sketch_prompt, sketch_img.size, "sketch")

    # Generate ControlNet conditioned image
    res_cnet_sketch = sketch_orch.generate_design(
        sketch=sketch_img,
        prompt=sketch_prompt,
        style="streetwear",
        method="canny",
        conditioning_scale=0.85,
        seed=seed
    )
    cnet_sketch_out = res_cnet_sketch.images[0]
    
    # Save outputs
    logger.info("Step 1.3: Saving Results...")
    sketch_orch.save_results(res_cnet_sketch, output_dir=output_dir)
    std_sketch_path = output_dir / "standard_sketch.png"
    std_sketch_out.save(std_sketch_path, format="PNG")
    logger.info(f"Saved standard unconditioned output to: {std_sketch_path}")
    
    # Evaluate
    logger.info("Step 1.4: Evaluating Structure and Prompt Alignment...")
    pair_eval = comparison_engine.evaluate_pair(
        standard_img=std_sketch_out,
        controlnet_img=cnet_sketch_out,
        condition_img=preprocessed_sketch,
        prompt=sketch_prompt
    )
    struct_eval = structure_evaluator.compare_images(
        standard_img=std_sketch_out,
        controlnet_img=cnet_sketch_out,
        reference_img=preprocessed_sketch
    )
    
    # Save Sidecar Report
    sketch_report_path = output_dir / "sketch2design_report.json"
    with open(sketch_report_path, "w", encoding="utf-8") as f:
        json.dump({"comparison": pair_eval, "structural": struct_eval}, f, indent=2)
    logger.success(f"Sketch2Design sidecar report saved: {sketch_report_path}")
    
    # Log summary console stats
    logger.info(f"  - Standard CLIP Score: {pair_eval['metrics']['standard']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['standard']['ssim']:.4f}")
    logger.info(f"  - ControlNet CLIP Score: {pair_eval['metrics']['controlnet']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['controlnet']['ssim']:.4f}")
    logger.info(f"  - Edge Preservation Improvement: +{struct_eval['improvements']['edge_preservation'] * 100:.1f}%")
    logger.info(f"  - ControlNet Dominance: {struct_eval['controlnet_advantage']}")
    
    # Collect
    standard_images.append(std_sketch_out)
    controlnet_images.append(cnet_sketch_out)
    condition_images.append(preprocessed_sketch)
    prompts_list.append(sketch_prompt)
    structural_metrics_list.append(struct_eval)

    # =============================================================================
    # ── 2. Pose2Fashion Pipeline
    # =============================================================================
    logger.info("\n=== RUNNING 2. POSE-TO-FASHION PIPELINE ===")
    pose_prompt = "Elegant double-breasted navy blue tweed jacket with structured shoulders and gold crest button detailing"
    
    # Preprocess
    logger.info("Step 2.1: Preprocessing Pose...")
    preprocessed_pose = pose_orch.preprocess_pose(pose_img)
    
    # Generate outputs
    logger.info("Step 2.2: Running Generation...")
    # Generate standard unconditioned image
    if mock_mode:
        std_pose_out = create_mock_standard_image(pose_prompt, pose_img.size, "pose")
    else:
        res_std = pose_orch.generate_fashion(
            pose_image=pose_img,
            prompt=pose_prompt,
            style="luxury",
            conditioning_scale=0.0,
            seed=seed
        )
        std_pose_out = res_std.images[0] if res_std.success else create_mock_standard_image(pose_prompt, pose_img.size, "pose")

    # Generate ControlNet conditioned image
    res_cnet_pose = pose_orch.generate_fashion(
        pose_image=pose_img,
        prompt=pose_prompt,
        style="luxury",
        conditioning_scale=0.90,
        seed=seed
    )
    cnet_pose_out = res_cnet_pose.images[0]
    
    # Save outputs
    logger.info("Step 2.3: Saving Results...")
    pose_orch.save_results(res_cnet_pose, output_dir=output_dir)
    std_pose_path = output_dir / "standard_pose.png"
    std_pose_out.save(std_pose_path, format="PNG")
    logger.info(f"Saved standard unconditioned output to: {std_pose_path}")
    
    # Evaluate
    logger.info("Step 2.4: Evaluating Structure and Prompt Alignment...")
    pair_eval = comparison_engine.evaluate_pair(
        standard_img=std_pose_out,
        controlnet_img=cnet_pose_out,
        condition_img=preprocessed_pose,
        prompt=pose_prompt
    )
    struct_eval = structure_evaluator.compare_images(
        standard_img=std_pose_out,
        controlnet_img=cnet_pose_out,
        reference_img=preprocessed_pose
    )
    
    # Save Sidecar Report
    pose_report_path = output_dir / "pose2fashion_report.json"
    with open(pose_report_path, "w", encoding="utf-8") as f:
        json.dump({"comparison": pair_eval, "structural": struct_eval}, f, indent=2)
    logger.success(f"Pose2Fashion sidecar report saved: {pose_report_path}")
    
    # Log summary console stats
    logger.info(f"  - Standard CLIP Score: {pair_eval['metrics']['standard']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['standard']['ssim']:.4f}")
    logger.info(f"  - ControlNet CLIP Score: {pair_eval['metrics']['controlnet']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['controlnet']['ssim']:.4f}")
    logger.info(f"  - Layout Consistency Improvement: +{struct_eval['improvements']['layout_consistency'] * 100:.1f}%")
    logger.info(f"  - ControlNet Dominance: {struct_eval['controlnet_advantage']}")
    
    # Collect
    standard_images.append(std_pose_out)
    controlnet_images.append(cnet_pose_out)
    condition_images.append(preprocessed_pose)
    prompts_list.append(pose_prompt)
    structural_metrics_list.append(struct_eval)

    # =============================================================================
    # ── 3. Depth2Fashion Pipeline
    # =============================================================================
    logger.info("\n=== RUNNING 3. DEPTH-TO-FASHION PIPELINE ===")
    depth_prompt = "A flowy sage green A-line midi dress with gathered waist and square neckline, made of organic linen"
    
    # Preprocess
    logger.info("Step 3.1: Preprocessing Depth...")
    preprocessed_depth = depth_orch.preprocess_depth(depth_img)
    
    # Generate outputs
    logger.info("Step 3.2: Running Generation...")
    # Generate standard unconditioned image
    if mock_mode:
        std_depth_out = create_mock_standard_image(depth_prompt, depth_img.size, "depth")
    else:
        res_std = depth_orch.generate_fashion(
            depth_image=depth_img,
            prompt=depth_prompt,
            style="minimalist",
            conditioning_scale=0.0,
            seed=seed
        )
        std_depth_out = res_std.images[0] if res_std.success else create_mock_standard_image(depth_prompt, depth_img.size, "depth")

    # Generate ControlNet conditioned image
    res_cnet_depth = depth_orch.generate_fashion(
        depth_image=depth_img,
        prompt=depth_prompt,
        style="minimalist",
        conditioning_scale=0.85,
        seed=seed
    )
    cnet_depth_out = res_cnet_depth.images[0]
    
    # Save outputs
    logger.info("Step 3.3: Saving Results...")
    depth_orch.save_results(res_cnet_depth, output_dir=output_dir)
    std_depth_path = output_dir / "standard_depth.png"
    std_depth_out.save(std_depth_path, format="PNG")
    logger.info(f"Saved standard unconditioned output to: {std_depth_path}")
    
    # Evaluate
    logger.info("Step 3.4: Evaluating Structure and Prompt Alignment...")
    pair_eval = comparison_engine.evaluate_pair(
        standard_img=std_depth_out,
        controlnet_img=cnet_depth_out,
        condition_img=preprocessed_depth,
        prompt=depth_prompt
    )
    struct_eval = structure_evaluator.compare_images(
        standard_img=std_depth_out,
        controlnet_img=cnet_depth_out,
        reference_img=preprocessed_depth
    )
    
    # Save Sidecar Report
    depth_report_path = output_dir / "depth2fashion_report.json"
    with open(depth_report_path, "w", encoding="utf-8") as f:
        json.dump({"comparison": pair_eval, "structural": struct_eval}, f, indent=2)
    logger.success(f"Depth2Fashion sidecar report saved: {depth_report_path}")
    
    # Log summary console stats
    logger.info(f"  - Standard CLIP Score: {pair_eval['metrics']['standard']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['standard']['ssim']:.4f}")
    logger.info(f"  - ControlNet CLIP Score: {pair_eval['metrics']['controlnet']['clip_score']:.4f} | SSIM: {pair_eval['metrics']['controlnet']['ssim']:.4f}")
    logger.info(f"  - Shape Preservation Improvement: +{struct_eval['improvements']['shape_preservation'] * 100:.1f}%")
    logger.info(f"  - ControlNet Dominance: {struct_eval['controlnet_advantage']}")
    
    # Collect
    standard_images.append(std_depth_out)
    controlnet_images.append(cnet_depth_out)
    condition_images.append(preprocessed_depth)
    prompts_list.append(depth_prompt)
    structural_metrics_list.append(struct_eval)

    # =============================================================================
    # ── 4. Unified Comparison Report & Aggregation
    # =============================================================================
    logger.info("\n=== COMPILING UNIFIED COMPARISON REPORT ===")
    
    # Output file paths
    project_report_path = Path("comparison_report.json")
    demo_report_path = output_dir / "comparison_report.json"
    
    # Generate the base ComparisonEngine report
    master_report = comparison_engine.evaluate_batch(
        standard_imgs=standard_images,
        controlnet_imgs=controlnet_images,
        condition_imgs=condition_images,
        prompts=prompts_list,
        output_json=demo_report_path
    )
    
    # Enrich report with structural evaluator metrics
    for idx, pair in enumerate(master_report["pairs"]):
        se_metrics = structural_metrics_list[idx]
        pair["structural_metrics"] = se_metrics["metrics"]
        pair["structural_improvements"] = se_metrics["improvements"]
        pair["controlnet_advantage"] = se_metrics["controlnet_advantage"]
        
    # Also calculate and add average structural metrics to the summary
    avg_std_struct = {
        "ssim_mean": float(np.mean([m["metrics"]["standard"]["ssim"] for m in structural_metrics_list])),
        "edge_preservation_mean": float(np.mean([m["metrics"]["standard"]["edge_preservation"] for m in structural_metrics_list])),
        "layout_consistency_mean": float(np.mean([m["metrics"]["standard"]["layout_consistency"] for m in structural_metrics_list])),
        "shape_preservation_mean": float(np.mean([m["metrics"]["standard"]["shape_preservation"] for m in structural_metrics_list])),
    }
    avg_cnet_struct = {
        "ssim_mean": float(np.mean([m["metrics"]["controlnet"]["ssim"] for m in structural_metrics_list])),
        "edge_preservation_mean": float(np.mean([m["metrics"]["controlnet"]["edge_preservation"] for m in structural_metrics_list])),
        "layout_consistency_mean": float(np.mean([m["metrics"]["controlnet"]["layout_consistency"] for m in structural_metrics_list])),
        "shape_preservation_mean": float(np.mean([m["metrics"]["controlnet"]["shape_preservation"] for m in structural_metrics_list])),
    }
    
    master_report["summary"]["aggregate_metrics"]["standard"]["structural_averages"] = avg_std_struct
    master_report["summary"]["aggregate_metrics"]["controlnet"]["structural_averages"] = avg_cnet_struct
    master_report["summary"]["seed"] = seed
    master_report["summary"]["run_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    # Write enriched report to the demo folder and the project root folder
    for path in [demo_report_path, project_report_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(master_report, f, indent=2)
        logger.success(f"Enriched unified report saved to: {path.resolve()}")
        
    logger.info("\n=== DEMO RUN COMPLETED ===")
    logger.info(f"Results can be found in the directory: {output_dir.resolve()}")
    logger.info(f"Unified report created at: {project_report_path.resolve()}")
    return 0


# =============================================================================
# ── CLI Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Week 3 ControlNet Pipeline & Evaluation Demo.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run real Stable Diffusion XL and ControlNet execution (requires GPU and CUDA dependencies)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="week3/outputs/demo",
        help="Path where generated mock outputs and reports will be saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    
    # Parse output directory
    out_dir = Path(args.output_dir)
    
    # Setup config-level logging
    try:
        cfg = get_config()
        setup_logging(log_dir=cfg.log_dir)
    except Exception:
        pass
        
    return run_demo(mock_mode=not args.real, output_dir=out_dir, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
