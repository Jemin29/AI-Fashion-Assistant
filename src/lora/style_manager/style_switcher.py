"""
week4/style_manager/style_switcher.py
=====================================
Dynamic Style Switching Engine.
Coordinates real-time style adapter activations, text prompt conditioning,
and styled design output generations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from src.lora.style_manager.lora_registry import LoraRegistry


class StyleSwitcher:
    """
    Manages runtime LoRA style switching.
    Injects brand style tokens into prompts, manages memory load state registry,
    and coordinates active adapters inside the diffusion pipeline.
    """

    BRAND_PRESETS = {
        "nike": "sportswear, techwear style, athletic fit, performance fabrics",
        "gucci": "luxury, haute-couture style, avant-garde design, gold metal accents",
        "zara": "casual, contemporary style, minimalist look, trend-driven silhouette",
        "h&m": "basic, casual style, minimalist essential design, organic textures"
    }

    BRAND_COLORS = {
        "nike": (10, 10, 10),      # Dark charcoal
        "gucci": (139, 69, 19),    # Luxurious brown
        "zara": (245, 245, 220),   # Beige
        "h&m": (128, 128, 128)     # Grey
    }

    def __init__(
        self,
        registry: LoraRegistry,
        inference_pipeline: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the StyleSwitcher.

        Parameters
        ----------
        registry : LoraRegistry
            LoRA adapter model registry.
        inference_pipeline : Any, optional
            Active diffusion model generation pipeline.
        output_dir : Path or str, optional
            Folder to save generated images (default: outputs/style_switcher).
        """
        self.registry = registry
        self.inference_pipeline = inference_pipeline
        
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        elif registry.config and getattr(registry.config, "output_root", None):
            self.output_dir = Path(registry.config.output_root).resolve() / "style_switcher"
        else:
            self.output_dir = Path("outputs/style_switcher").resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized StyleSwitcher | output_dir={self.output_dir}")

    def switch_style(self, brand: str, scale: float = 1.0) -> None:
        """
        Deactivate active adapters, load if missing, and activate target brand adapter.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        scale : float
            Active blend scale (0.0 to 2.0).
        """
        brand_key = brand.lower().strip()
        
        # 1. Deactivate other active models in registry
        for active_brand in list(self.registry.list_models(filter_active=True).keys()):
            if active_brand != brand_key:
                self.registry.deactivate_model(active_brand)

        # 2. Check and load the target model
        if brand_key not in self.registry.models:
            raise KeyError(f"Brand style '{brand_key}' is not registered in the LoraRegistry.")

        record = self.registry.models[brand_key]
        if not record["loaded"]:
            self.registry.load_model(brand_key)

        # 3. Activate model in registry
        self.registry.activate_model(brand_key, scale=scale)

        # 4. Trigger weights loading/blending on the active inference pipeline if present
        if self.inference_pipeline is not None:
            logger.info(f"Switching weights at runtime for model adapter: '{brand_key}'...")
            
            # If the pipeline has unload method, clear previous LoRAs
            if hasattr(self.inference_pipeline, "unload_lora_weights"):
                try:
                    self.inference_pipeline.unload_lora_weights()
                except Exception as err:
                    logger.debug(f"Pipeline unload weights warning: {err}")

            # Load LoRA weights
            if hasattr(self.inference_pipeline, "load_lora_weights"):
                try:
                    self.inference_pipeline.load_lora_weights(
                        record["model_path"],
                        adapter_name=brand_key
                    )
                except Exception as err:
                    logger.warning(f"Failed to load LoRA weights to inference pipeline: {err}")

            # Apply active scale blending
            if hasattr(self.inference_pipeline, "set_adapters"):
                try:
                    self.inference_pipeline.set_adapters([brand_key], adapter_weights=[scale])
                except Exception as err:
                    logger.warning(f"Failed to set active adapter scale blending: {err}")

        logger.success(f"Dynamic adapter switching successful | style={brand_key} | scale={scale}")

    def preprocess_prompt(self, prompt: str, brand: str) -> str:
        """
        Append brand-specific styling trigger tokens to the input prompt.

        Parameters
        ----------
        prompt : str
        brand : str

        Returns
        -------
        str
            Enriched prompt.
        """
        brand_key = brand.lower().strip()
        tokens = self.BRAND_PRESETS.get(brand_key, "fashion style")
        
        # Prevent double appending
        if tokens.split(",")[0] in prompt.lower():
            return prompt
            
        return f"{prompt}, {tokens}"

    def generate_styled_design(
        self,
        prompt: str,
        brand: str,
        scale: float = 1.0,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Switch style, expand prompt, and generate styled design outputs.

        Parameters
        ----------
        prompt : str
            Base prompt (e.g. "Black oversized hoodie").
        brand : str
            Target brand (nike, gucci, zara, h&m).
        scale : float
            LoRA adapter active scale.
        dry_run : bool
            If True, generates a simulated output design on CPU.

        Returns
        -------
        dict
            Status logs and output paths.
        """
        brand_key = brand.lower().strip()
        
        # 1. Switch Style
        self.switch_style(brand_key, scale)

        # 2. Preprocess Prompt
        enriched_prompt = self.preprocess_prompt(prompt, brand_key)
        logger.info(f"Conditioning text prompt: '{enriched_prompt}'")

        # Create output file target
        timestamp = int(time.time())
        img_name = f"styled_{brand_key}_{timestamp}.png"
        dest_path = self.output_dir / img_name

        if dry_run:
            logger.info("Simulating style generation (dry_run)...")
            # Create a mock styling image frame
            color = self.BRAND_COLORS.get(brand_key, (255, 255, 255))
            img = Image.new("RGB", (512, 512), color=color)
            
            # Simple canvas overlay drawing
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 502, 502], outline=(255, 255, 255), width=2)
            
            # Write labels (fallback simple text overlays)
            text_color = (255, 255, 255) if brand_key != "zara" else (0, 0, 0)
            draw.text((20, 30), f"BRAND STYLE: {brand_key.upper()}", fill=text_color)
            draw.text((20, 60), f"PROMPT: {prompt}", fill=text_color)
            draw.text((20, 90), f"BLEND SCALE: {scale}", fill=text_color)
            draw.text((20, 120), f"TOKENS: {self.BRAND_PRESETS.get(brand_key)}", fill=text_color)

            img.save(dest_path)
            logger.success(f"Simulated output design saved to: {dest_path}")
            
            return {
                "success": True,
                "prompt": enriched_prompt,
                "brand": brand_key,
                "scale": scale,
                "image_path": str(dest_path.as_posix()),
                "dry_run": True
            }
        else:
            logger.info("Invoking active pipeline style generation...")
            if not self.inference_pipeline:
                raise ValueError("Active inference pipeline must be loaded to run in real mode.")
            
            # Call pipeline
            try:
                # Expecting pipeline to return an image list when called
                pipeline_output = self.inference_pipeline(prompt=enriched_prompt)
                img = pipeline_output.images[0]
                img.save(dest_path)
                logger.success(f"Styled design output saved to: {dest_path}")
                
                return {
                    "success": True,
                    "prompt": enriched_prompt,
                    "brand": brand_key,
                    "scale": scale,
                    "image_path": str(dest_path.as_posix()),
                    "dry_run": False
                }
            except Exception as err:
                logger.error(f"Inference pipeline execution failed: {err}")
                return {
                    "success": False,
                    "error": str(err),
                    "prompt": enriched_prompt,
                    "brand": brand_key,
                    "scale": scale,
                    "dry_run": False
                }
