"""
scripts/generate_comparison_set.py
==================================
Generates a side-by-side comparison set of controlled vs uncontrolled fashion designs
and computes real structural metrics (SSIM, edge preservation, layout, shape) using the
StructureEvaluator.
"""

import sys
import os
import json
import time
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.preprocessors.sketch_processor import SketchProcessor
from src.evaluation.week3_structure_evaluator import StructureEvaluator
from week6.services.controlnet_service import ControlNetService
from week6.services.generation_service import GenerationService
from week6.gradio_app.config import get_config

def main():
    print("Initializing comparison generation script...")
    
    # 1. Output directories
    out_dir = _ROOT / "week6" / "outputs" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Find 5 images from deepfashion sample
    sample_dir = _ROOT / "datasets" / "deepfashion" / "sample" / "img"
    img_paths = list(sample_dir.rglob("*.jpg")) + list(sample_dir.rglob("*.jpeg")) + list(sample_dir.rglob("*.png"))
    
    if len(img_paths) < 5:
        print(f"Error: Found only {len(img_paths)} sample images under {sample_dir}, need at least 5.")
        return 1
        
    selected_paths = img_paths[:5]
    print(f"Selected 5 reference images for comparison: {[p.name for p in selected_paths]}")
    
    # 3. Instantiate services (in mock mode for CPU safety, but metrics are computed on outputs)
    cfg = get_config()
    # Force mock mode for safety during CLI run
    gen_service = GenerationService(mock_mode=True)
    cn_service = ControlNetService(mock_mode=True)
    sketch_processor = SketchProcessor()
    evaluator = StructureEvaluator()
    
    prompts = [
        "A stylish summer blouse, floral design, lightweight fabric",
        "A formal designer suit jacket, navy blue wool, tailored fit",
        "A casual cotton hoodie, oversized streetwear fashion",
        "An elegant evening dress, silk draping, premium lookbook",
        "A modern athletic running tank top, breathable synthetic mesh"
    ]
    
    comparison_data = []
    
    for idx, (img_path, prompt) in enumerate(zip(selected_paths, prompts)):
        print(f"\n--- Processing Example {idx+1}/5 ---")
        print(f"Prompt: {prompt}")
        
        # Load and resize reference image
        ref_img = Image.open(img_path).convert("RGB").resize((512, 512))
        ref_save_path = out_dir / f"example_{idx+1}_ref.png"
        ref_img.save(ref_save_path)
        
        # Preprocess reference image to get sketch
        sketch_img = sketch_processor.preprocess_sketch(ref_img, method="canny")
        sketch_save_path = out_dir / f"example_{idx+1}_sketch.png"
        sketch_img.save(sketch_save_path)
        
        # Generate uncontrolled image
        print("Generating uncontrolled (standard) image...")
        uncontrolled_res = gen_service.generate(prompt=prompt, width=512, height=512)
        if not uncontrolled_res.success:
            print(f"Warning: Standard generation failed for '{prompt}'")
            continue
        uncontrolled_img = uncontrolled_res.data["image"].resize((512, 512))
        unc_save_path = out_dir / f"example_{idx+1}_uncontrolled.png"
        uncontrolled_img.save(unc_save_path)
        
        # Generate controlled image (ControlNet)
        print("Generating controlled (ControlNet) image...")
        controlled_res = cn_service.generate_conditioned(
            prompt=prompt,
            control_image=sketch_img,
            mode="canny",
            conditioning_scale=0.8,
            num_inference_steps=30
        )
        if not controlled_res.success:
            print(f"Warning: Controlled generation failed for '{prompt}'")
            continue
        controlled_img = controlled_res.data["image"].resize((512, 512))
        con_save_path = out_dir / f"example_{idx+1}_controlled.png"
        controlled_img.save(con_save_path)
        
        # Compute real metrics using StructureEvaluator
        print("Computing real structural metrics...")
        metrics_comparison = evaluator.compare_images(
            standard_img=uncontrolled_img,
            controlnet_img=controlled_img,
            reference_img=sketch_img
        )
        
        # Store relative paths for front-end access
        comparison_data.append({
            "id": idx + 1,
            "prompt": prompt,
            "sketch_path": f"outputs/comparison/example_{idx+1}_sketch.png",
            "uncontrolled_path": f"outputs/comparison/example_{idx+1}_uncontrolled.png",
            "controlled_path": f"outputs/comparison/example_{idx+1}_controlled.png",
            "metrics": metrics_comparison["metrics"],
            "improvements": metrics_comparison["improvements"],
            "controlnet_advantage": metrics_comparison["controlnet_advantage"]
        })
        print(f"Metrics: Standard SSIM={metrics_comparison['metrics']['standard']['ssim']:.4f} | ControlNet SSIM={metrics_comparison['metrics']['controlnet']['ssim']:.4f}")
        
    # Save the manifest JSON
    manifest_path = out_dir / "comparison_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
        
    print(f"\nSuccess! Generated comparison set and saved manifest to {manifest_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
