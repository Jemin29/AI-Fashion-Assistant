"""
week4/style_manager/style_mixer.py
==================================
LoRA Style Mixer System.
Coordinates multi-LoRA adapter style blending, validation, weight normalization,
prompt token interpolation, and mixed design output generations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw
from loguru import logger

from src.lora.style_manager.lora_registry import LoraRegistry


class StyleMixer:
    """
    Manages multi-LoRA style blending ratios.
    Normalizes custom blending weights, interpolates trigger prompts,
    and configures multiple concurrent active adapters in the diffusion pipeline.
    """

    BRAND_PRESETS = {
        "nike": "sportswear techwear style",
        "gucci": "luxury haute-couture style",
        "zara": "casual contemporary style",
        "h&m": "basic casual style"
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
        Initialize the StyleMixer.

        Parameters
        ----------
        registry : LoraRegistry
            Active registry manager for style adapters.
        inference_pipeline : Any, optional
            Active diffusion model generation pipeline.
        output_dir : Path or str, optional
            Folder to save mixed images (default: outputs/style_mixer).
        """
        self.registry = registry
        self.inference_pipeline = inference_pipeline

        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        elif registry.config and getattr(registry.config, "output_root", None):
            self.output_dir = Path(registry.config.output_root).resolve() / "style_mixer"
        else:
            self.output_dir = Path("outputs/style_mixer").resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized StyleMixer | output_dir={self.output_dir}")

    def mix_styles(self, brand_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate, normalize weights, and activate multiple style adapters in registry.

        Parameters
        ----------
        brand_weights : dict of str to float
            Ratios for mixing styles (e.g. {'nike': 0.7, 'gucci': 0.3}).

        Returns
        -------
        dict of str to float
            Normalized style weights.
        """
        if not brand_weights:
            raise ValueError("No brand style weights specified for mixing.")

        # 1. Clean brand keys & validate support
        cleaned_weights = {}
        for brand, weight in brand_weights.items():
            brand_key = brand.lower().strip()
            if brand_key not in self.registry.SUPPORTED_BRANDS:
                raise ValueError(f"Brand '{brand}' not supported. Supported: {self.registry.SUPPORTED_BRANDS}")
            if weight < 0.0:
                raise ValueError(f"Blend weight for '{brand}' cannot be negative ({weight}).")
            cleaned_weights[brand_key] = weight

        # 2. Weight Normalization
        total_weight = sum(cleaned_weights.values())
        if total_weight <= 0.0:
            raise ValueError("Total blend weights sum must be greater than 0.")
        
        normalized_weights = {b: w / total_weight for b, w in cleaned_weights.items()}

        # 3. Deactivate other active models in registry
        for active_brand in list(self.registry.list_models(filter_active=True).keys()):
            if active_brand not in normalized_weights:
                self.registry.deactivate_model(active_brand)

        # 4. Load & Activate target models in registry
        for brand_key, weight in normalized_weights.items():
            if brand_key not in self.registry.models:
                raise KeyError(f"Brand style '{brand_key}' is not registered in the LoraRegistry.")
            
            record = self.registry.models[brand_key]
            if not record["loaded"]:
                self.registry.load_model(brand_key)
            
            self.registry.activate_model(brand_key, scale=weight)

        # 5. Load and scale adapters in pipeline if present
        if self.inference_pipeline is not None:
            logger.info(f"Configuring multi-LoRA active adapters: {normalized_weights}")
            
            if hasattr(self.inference_pipeline, "unload_lora_weights"):
                try:
                    self.inference_pipeline.unload_lora_weights()
                except Exception as err:
                    logger.debug(f"Pipeline unload weights warning: {err}")

            # Load LoRA weights for all active styles
            active_adapters = []
            active_scales = []
            
            for brand_key, weight in normalized_weights.items():
                record = self.registry.models[brand_key]
                if hasattr(self.inference_pipeline, "load_lora_weights"):
                    try:
                        self.inference_pipeline.load_lora_weights(
                            record["model_path"],
                            adapter_name=brand_key
                        )
                        active_adapters.append(brand_key)
                        active_scales.append(weight)
                    except Exception as err:
                        logger.warning(f"Failed to load LoRA weights for '{brand_key}' to pipeline: {err}")

            # Apply multi-adapter active blend scales
            if active_adapters and hasattr(self.inference_pipeline, "set_adapters"):
                try:
                    self.inference_pipeline.set_adapters(active_adapters, adapter_weights=active_scales)
                except Exception as err:
                    logger.warning(f"Failed to set active multi-adapters blending: {err}")

        logger.success(f"Configured style mixing adapters: {normalized_weights}")
        return normalized_weights

    def generate_blended_prompt(self, prompt: str, brand_weights: Dict[str, float]) -> str:
        """
        Interpolate text prompt trigger tokens proportionally according to blend weights.

        Parameters
        ----------
        prompt : str
        brand_weights : dict of str to float
            Normalized style weights.

        Returns
        -------
        str
            Enriched blended prompt.
        """
        blend_parts = []
        # Sort brands by weight descending for prompt flow readability
        sorted_weights = sorted(brand_weights.items(), key=lambda x: x[1], reverse=True)
        
        for brand, weight in sorted_weights:
            if weight > 0.0:
                pct = int(weight * 100)
                preset_tag = self.BRAND_PRESETS.get(brand, "fashion style")
                blend_parts.append(f"{pct}% {preset_tag}")
                
        if not blend_parts:
            return prompt
            
        return f"{prompt}, blended style of " + " and ".join(blend_parts)

    def generate_mixed_design(
        self,
        prompt: str,
        brand_weights: Dict[str, float],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestrate style mixing, prompt blending, and output design generation.

        Parameters
        ----------
        prompt : str
            Base prompt (e.g. "Oversized hoodie").
        brand_weights : dict of str to float
            Ratios for mixing styles (e.g. {'nike': 0.7, 'gucci': 0.3}).
        dry_run : bool
            If True, generates a simulated output design on CPU.

        Returns
        -------
        dict
            Status details and outputs.
        """
        # 1. Mix Styles
        normalized_weights = self.mix_styles(brand_weights)

        # 2. Blended prompt
        blended_prompt = self.generate_blended_prompt(prompt, normalized_weights)
        logger.info(f"Conditioning blended prompt: '{blended_prompt}'")

        # Create output file target
        timestamp = int(time.time())
        img_name = f"mixed_{timestamp}.png"
        dest_path = self.output_dir / img_name

        if dry_run:
            logger.info("Simulating style mixing generation (dry_run)...")
            
            # Interpolate a blended color background
            r, g, b = 0.0, 0.0, 0.0
            for brand, weight in normalized_weights.items():
                bc = self.BRAND_COLORS.get(brand, (255, 255, 255))
                r += bc[0] * weight
                g += bc[1] * weight
                b += bc[2] * weight
            mixed_color = (int(r), int(g), int(b))

            img = Image.new("RGB", (512, 512), color=mixed_color)
            
            # Draw labels
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 502, 502], outline=(255, 255, 255), width=2)
            
            text_color = (255, 255, 255)
            # If color is too bright, use black text (Zara is very bright beige)
            # Simple luminance check
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            if luminance > 200:
                text_color = (0, 0, 0)

            draw.text((20, 30), "STYLE MIXER SYSTEM ACTIVE", fill=text_color)
            draw.text((20, 60), f"PROMPT: {prompt}", fill=text_color)
            
            y_offset = 90
            for brand, weight in sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True):
                draw.text((20, y_offset), f"- {brand.upper()}: {weight*100:.1f}%", fill=text_color)
                y_offset += 25
                
            draw.text((20, y_offset + 10), f"BLENDED: {blended_prompt[:60]}...", fill=text_color)

            img.save(dest_path)
            logger.success(f"Simulated mixed design saved to: {dest_path}")
            
            return {
                "success": True,
                "prompt": blended_prompt,
                "weights": normalized_weights,
                "image_path": str(dest_path.as_posix()),
                "dry_run": True
            }
        else:
            logger.info("Invoking active pipeline multi-style generation...")
            if not self.inference_pipeline:
                raise ValueError("Active inference pipeline must be loaded to run in real mode.")
            
            try:
                pipeline_output = self.inference_pipeline(prompt=blended_prompt)
                img = pipeline_output.images[0]
                img.save(dest_path)
                logger.success(f"Mixed design output saved to: {dest_path}")
                
                return {
                    "success": True,
                    "prompt": blended_prompt,
                    "weights": normalized_weights,
                    "image_path": str(dest_path.as_posix()),
                    "dry_run": False
                }
            except Exception as err:
                logger.error(f"Inference pipeline execution failed: {err}")
                return {
                    "success": False,
                    "error": str(err),
                    "prompt": blended_prompt,
                    "weights": normalized_weights,
                    "dry_run": False
                }
