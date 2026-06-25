"""
week4/inference/lora_inference.py
=================================
Production-ready LoRA Inference System.
Coordinates base SDXL pipeline model loading, dynamic adapter activation/blending,
single and batch design generations, and metadata sidecar serialization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from loguru import logger

from src.utils.config_manager import get_default_config
from src.lora.style_manager.lora_registry import LoraRegistry


class LoraInferenceSystem:
    """
    Manages SDXL diffusion loading, LoRA parameter-efficient style blending,
    runtime switching, batch design generations, and metadata storage.
    """

    def __init__(
        self,
        config: Any = None,
        registry: Optional[LoraRegistry] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        dry_run: bool = True
    ) -> None:
        """
        Initialize the LoraInferenceSystem.

        Parameters
        ----------
        config : Week4Config, optional
        registry : LoraRegistry, optional
        output_dir : Path or str, optional
            Folder to save outputs (default: outputs/inference).
        device : str
            Target computing device ("cpu" or "cuda").
        dry_run : bool
            If True, runs simulated image outputs without loading heavy models.
        """
        self.config = config or get_default_config()
        self.registry = registry or LoraRegistry(config=self.config)
        self.device = device.lower().strip()
        self.dry_run = dry_run
        
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = Path(self.config.output_root).resolve() / "inference"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Any = None
        self.switcher: Any = None
        logger.info(f"Initialized LoraInferenceSystem | device={self.device} | dry_run={self.dry_run} | output={self.output_dir}")

    def load_pipeline(self) -> None:
        """Load the base SDXL pipeline model and setup StyleSwitcher."""
        if self.pipeline is not None:
            return

        if self.dry_run:
            logger.info("DRY-RUN mode active. Instantiating mock diffusion pipeline...")
            
            class MockPipeline:
                """Mock HuggingFace diffusers pipeline wrapper for CPU test runs."""
                def __init__(self) -> None:
                    self.loaded_adapters: Dict[str, str] = {}
                    self.active_adapters: List[str] = []
                    self.active_weights: List[float] = []

                def load_lora_weights(self, path: str, adapter_name: str) -> None:
                    self.loaded_adapters[adapter_name] = path
                    logger.debug(f"Mock pipeline: loaded weights for '{adapter_name}' from {path}")

                def unload_lora_weights(self) -> None:
                    self.loaded_adapters.clear()
                    self.active_adapters.clear()
                    self.active_weights.clear()
                    logger.debug("Mock pipeline: unloaded all LoRA adapter weights.")

                def set_adapters(self, adapters: List[str], adapter_weights: List[float]) -> None:
                    self.active_adapters = adapters
                    self.active_weights = adapter_weights
                    logger.debug(f"Mock pipeline: set active adapters to {adapters} with scales {adapter_weights}")

                def __call__(self, prompt: str, **kwargs: Any) -> Any:
                    class MockPipelineOutput:
                        def __init__(self) -> None:
                            # Solid slate green mock image
                            self.images = [Image.new("RGB", (512, 512), color=(46, 139, 87))]
                    return MockPipelineOutput()

            self.pipeline = MockPipeline()
        else:
            logger.info(f"Loading base SDXL model pipeline onto {self.device}...")
            import torch
            from diffusers import StableDiffusionXLPipeline
            
            model_id = self.config.inference.base_model_id
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.pipeline.to(self.device)

        # Setup style switcher
        from src.lora.style_manager.style_switcher import StyleSwitcher
        self.switcher = StyleSwitcher(
            registry=self.registry,
            inference_pipeline=self.pipeline,
            output_dir=self.output_dir
        )
        logger.success("Base SDXL model pipeline loaded successfully.")

    def generate(
        self,
        prompt: str,
        brand: str,
        scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Switch style, preprocess prompts, generate image, and serialize sidecar metadata.

        Parameters
        ----------
        prompt : str
            Base text prompt.
        brand : str
            Target brand style (nike, gucci, zara, h&m).
        scale : float
            LoRA adapter weight scale (0.0 to 2.0).
        seed : int, optional
            Deterministic seed.

        Returns
        -------
        dict
            Inference status detailing outputs and prompt conditions.
        """
        brand_key = brand.lower().strip()
        if self.pipeline is None:
            self.load_pipeline()

        # 1. Trigger style switcher weights loading & prompt updates
        self.switcher.switch_style(brand_key, scale=scale)
        enriched_prompt = self.switcher.preprocess_prompt(prompt, brand_key)

        timestamp = int(time.time())
        img_name = f"design_{brand_key}_{timestamp}.png"
        meta_name = f"metadata_{brand_key}_{timestamp}.json"
        
        image_path = self.output_dir / img_name
        metadata_path = self.output_dir / meta_name

        if self.dry_run:
            logger.info("Simulating style inference design generation...")
            # Use switcher dry_run drawer
            res = self.switcher.generate_styled_design(prompt, brand_key, scale=scale, dry_run=True)
            shutil_src = Path(res["image_path"])
            if shutil_src.exists():
                shutil_src.replace(image_path)
        else:
            logger.info("Running real SDXL pipeline style generation...")
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
            out = self.pipeline(
                prompt=enriched_prompt,
                generator=generator,
                num_inference_steps=self.config.inference.num_inference_steps,
                guidance_scale=self.config.inference.guidance_scale
            )
            img = out.images[0]
            img.save(image_path)

        # 2. Metadata Storage (save sidecar details)
        meta_content = {
            "prompt": prompt,
            "enriched_prompt": enriched_prompt,
            "brand": brand_key,
            "scale": scale,
            "seed": seed,
            "device": self.device,
            "timestamp": timestamp,
            "resolution": "512x512" if self.dry_run else "1024x1024",
            "image_path": str(image_path.relative_to(self.output_dir.parent.parent))
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2, sort_keys=True)

        logger.success(f"Generated design output sidecar details persisted at: {metadata_path}")
        return {
            "success": True,
            "prompt": enriched_prompt,
            "brand": brand_key,
            "scale": scale,
            "image_path": str(image_path.as_posix()),
            "metadata_path": str(metadata_path.as_posix()),
            "dry_run": self.dry_run
        }

    def generate_batch(
        self,
        prompts: List[str],
        brands: List[str],
        scales: List[float],
        seeds: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute batch style generations, padding mismatched parameter arrays.

        Parameters
        ----------
        prompts : list of str
        brands : list of str
        scales : list of float
        seeds : list of int, optional

        Returns
        -------
        list of dict
            Collection of generation output results.
        """
        if not prompts or not brands or not scales:
            raise ValueError("In batch inference, prompts, brands, and scales cannot be empty.")

        batch_size = max(len(prompts), len(brands), len(scales))
        logger.info(f"Starting batch inference processing | batch_size={batch_size}")

        # Pad arrays
        padded_prompts = prompts + [prompts[-1]] * (batch_size - len(prompts))
        padded_brands = brands + [brands[-1]] * (batch_size - len(brands))
        padded_scales = scales + [scales[-1]] * (batch_size - len(scales))
        
        if seeds:
            padded_seeds = seeds + [seeds[-1]] * (batch_size - len(seeds))
        else:
            padded_seeds = [None] * batch_size # type: ignore[list-item]

        results = []
        for i in range(batch_size):
            logger.info(f"Processing batch item [{i+1}/{batch_size}]...")
            res = self.generate(
                prompt=padded_prompts[i],
                brand=padded_brands[i],
                scale=padded_scales[i],
                seed=padded_seeds[i]
            )
            results.append(res)

        logger.success(f"Batch generation completed | processed_count={len(results)}")
        return results
