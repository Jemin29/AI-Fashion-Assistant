import os
import sys
import json
import time
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger
import torch
from diffusers import StableDiffusionXLPipeline
from src.evaluation.week2_clip_evaluator import CLIPEvaluator

def main():
    logger.info("Starting Real CLIP similarity evaluation of trained Nike LoRA...")
    
    # 1. Paths setup
    output_dir = Path("outputs/evaluation/nike_real_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_path = Path("weights/lora/nike/nike_lora_adapter.safetensors")
    ref_dir = Path("outputs/datasets/nike")
    
    if not adapter_path.exists():
        logger.error(f"Nike LoRA adapter not found at {adapter_path}! Run lora training first.")
        return 1
        
    if not ref_dir.exists():
        logger.error(f"Reference Nike imagery directory not found at {ref_dir}!")
        return 1

    # Get reference image paths
    ref_paths = (list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.png")))[:5]
    if not ref_paths:
        logger.error("No reference images found in reference directory!")
        return 1
    logger.info(f"Using {len(ref_paths)} reference Nike images for evaluation.")

    # 2. Load base SDXL pipeline (using tiny model for speed and CPU safety)
    base_model_id = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
    logger.info(f"Loading tiny SDXL base pipeline: {base_model_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using target device: {device}")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        use_safetensors=False
    ).to(device)
    logger.info("Base pipeline loaded successfully.")

    # 3. Load CLIPEvaluator
    logger.info("Loading CLIP model for similarity calculation...")
    evaluator = CLIPEvaluator(device=device)
    if not evaluator.load_clip():
        logger.error("Failed to load CLIP evaluator!")
        return 1
        
    # Pre-compute reference image embeddings
    logger.info("Pre-computing embeddings for reference brand imagery...")
    ref_embeddings = []
    for ref_path in ref_paths:
        try:
            with Image.open(ref_path) as ref_img:
                ref_img = ref_img.convert("RGB")
                inputs = evaluator._processor(text=[""], images=[ref_img], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = evaluator._model(**inputs)
                    emb = outputs.image_embeds
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    ref_embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Failed to process reference image {ref_path}: {e}")
            
    if not ref_embeddings:
        logger.error("Failed to extract any reference embeddings!")
        return 1
    logger.info(f"Extracted embeddings for {len(ref_embeddings)} reference images.")

    # 4. Prompts and Seeds for generations
    prompt = "A high-fidelity fashion photo of a nike sneakers, sportswear, techwear style, black fabric, A custom brand nike design."
    seeds = [10, 20, 30, 40]
    
    # ── GENERATION & EVALUATION WITHOUT ADAPTER (BASE) ──
    logger.info("--- Generating images WITHOUT adapter (Base Model) ---")
    base_images = []
    base_prompt_scores = []
    base_image_similarities = []
    
    for seed in seeds:
        logger.info(f"Generating Base (Seed {seed})...")
        generator = torch.Generator(device=device).manual_seed(seed)
        out = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=64,
            height=64
        )
        img = out.images[0]
        img_save_path = output_dir / f"base_seed_{seed}.png"
        img.save(img_save_path)
        base_images.append(img)
        
        # Evaluate prompt-to-image similarity
        clip_res = evaluator.evaluate(img, prompt)
        base_prompt_scores.append(clip_res.clip_score)
        
        # Evaluate image-to-image similarity against references
        inputs = evaluator._processor(text=[""], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = evaluator._model(**inputs)
            gen_emb = outputs.image_embeds
            gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
            
        sims = []
        for ref_emb in ref_embeddings:
            sim = (gen_emb @ ref_emb.T).squeeze().item()
            sims.append(sim)
        mean_sim = sum(sims) / len(sims)
        base_image_similarities.append(mean_sim)
        logger.info(f"Base Seed {seed} | Prompt Sim: {clip_res.clip_score:.4f} | Image-to-Image Sim: {mean_sim:.4f}")

    # ── GENERATION & EVALUATION WITH NIKE ADAPTER (LORA) ──
    logger.info("--- Loading Nike LoRA Adapter onto pipeline ---")
    pipeline.load_lora_weights(str(adapter_path))
    logger.info("Nike LoRA weights loaded and activated.")
    
    lora_images = []
    lora_prompt_scores = []
    lora_image_similarities = []
    
    for seed in seeds:
        logger.info(f"Generating with LoRA Adapter (Seed {seed})...")
        generator = torch.Generator(device=device).manual_seed(seed)
        out = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=64,
            height=64
        )
        img = out.images[0]
        img_save_path = output_dir / f"lora_seed_{seed}.png"
        img.save(img_save_path)
        lora_images.append(img)
        
        # Evaluate prompt-to-image similarity
        clip_res = evaluator.evaluate(img, prompt)
        lora_prompt_scores.append(clip_res.clip_score)
        
        # Evaluate image-to-image similarity against references
        inputs = evaluator._processor(text=[""], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = evaluator._model(**inputs)
            gen_emb = outputs.image_embeds
            gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
            
        sims = []
        for ref_emb in ref_embeddings:
            sim = (gen_emb @ ref_emb.T).squeeze().item()
            sims.append(sim)
        mean_sim = sum(sims) / len(sims)
        lora_image_similarities.append(mean_sim)
        logger.info(f"LoRA Seed {seed} | Prompt Sim: {clip_res.clip_score:.4f} | Image-to-Image Sim: {mean_sim:.4f}")

    # 5. Compile and save evaluation stats
    mean_base_prompt = sum(base_prompt_scores) / len(base_prompt_scores)
    mean_base_image = sum(base_image_similarities) / len(base_image_similarities)
    mean_lora_prompt = sum(lora_prompt_scores) / len(lora_prompt_scores)
    mean_lora_image = sum(lora_image_similarities) / len(lora_image_similarities)
    
    summary = {
        "prompt": prompt,
        "base_model": {
            "prompt_similarities": base_prompt_scores,
            "mean_prompt_similarity": mean_base_prompt,
            "image_similarities": base_image_similarities,
            "mean_image_similarity": mean_base_image
        },
        "lora_adapter": {
            "prompt_similarities": lora_prompt_scores,
            "mean_prompt_similarity": mean_lora_prompt,
            "image_similarities": lora_image_similarities,
            "mean_image_similarity": mean_lora_image
        },
        "improvements": {
            "prompt_similarity_delta": mean_lora_prompt - mean_base_prompt,
            "image_similarity_delta": mean_lora_image - mean_base_image
        }
    }
    
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.success("=" * 60)
    logger.success("   REAL LORA CLIP EVALUATION COMPLETED SUCCESSFULLY   ")
    logger.success(f"   Summary JSON report saved at: {summary_path}")
    logger.success(f"   Base model average Prompt Sim: {mean_base_prompt:.4f}")
    logger.success(f"   Base model average Image Sim (vs References): {mean_base_image:.4f}")
    logger.success(f"   LoRA adapter average Prompt Sim: {mean_lora_prompt:.4f}")
    logger.success(f"   LoRA adapter average Image Sim (vs References): {mean_lora_image:.4f}")
    logger.success(f"   Image-to-Image similarity improvement: {summary['improvements']['image_similarity_delta']:.4f}")
    logger.success("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
