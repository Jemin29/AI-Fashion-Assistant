"""
week3/controlnet/sketch2design.py
=================================
Sketch2Design Orchestrator Module.
AI-Powered Fashion Design Assistant — Week 3.

Coordinates preprocessing of raw hand-drawn outlines or CAD fashion sketches,
and runs SDXL-ControlNet diffusion models to generate photorealistic designs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image

from src.controlnet.preprocessors.sketch_processor import SketchProcessor
from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine, GenerationOutput

# ── Dynamic import for Week 2 PromptBuilder ───────────────────────────────
try:
    from src.generation.prompts.prompt_builder import PromptBuilder
    _HAS_PROMPT_BUILDER = True
except ImportError:
    PromptBuilder = None
    _HAS_PROMPT_BUILDER = False


# =============================================================================
# ── Sketch2Design Class
# =============================================================================

class Sketch2Design:
    """
    Main orchestrator converting fashion sketches into photorealistic designs
    using ControlNet structure preservation and text prompt styling.
    """

    def __init__(self, config=None, mock: bool = False) -> None:
        """
        Initialize the Sketch2Design orchestrator.

        Parameters
        ----------
        config : Week3Config, optional
        mock : bool
            Force simulated execution (avoids loading heavy neural networks).
        """
        self.config = config
        self.mock = mock
        self.processor = SketchProcessor(config)
        self.engine = FashionControlNetEngine(config, mock=mock)
        
        # Instantiate prompt builder if Week 2 is importable
        self.prompt_builder = None
        if _HAS_PROMPT_BUILDER and config is not None:
            try:
                # We need to adapt the config to mock or load Week 2 config
                # PromptBuilder requires Week2Config structure
                self.prompt_builder = PromptBuilder(config)
                logger.debug("PromptBuilder imported and linked successfully.")
            except Exception as err:
                logger.debug(f"Failed to initialize PromptBuilder with current config: {err}")

    # ── Public APIs: Orchestration Methods ────────────────────────────────────

    def generate_design(
        self,
        sketch: Image.Image,
        prompt: str,
        style: Optional[str] = None,
        method: str = "canny",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        save_preprocessed: bool = True,
        **kwargs
    ) -> GenerationOutput:
        """
        Convert a single sketch into a stylized fashion design.

        Parameters
        ----------
        sketch : PIL.Image.Image
            Input black-and-white fashion sketch.
        prompt : str
            Creative text prompt detailing fabrics, fit, or details (e.g. "red silk evening gown").
        style : str, optional
            Style preset name (e.g. "luxury", "streetwear", "casual").
        method : str
            Edge extraction strategy: "canny" | "hed" | "lineart".
        conditioning_scale : float, optional
            ControlNet weight indicating strictness of sketch alignment (0.0 to 1.0).
        seed : int
            Fixed seed for reproducibility (-1 = random).
        num_inference_steps : int
            Denoising loop steps.
        guidance_scale : float
            CFG guidance scale.
        negative_prompt : str
            Additional negative prompt descriptors.
        save_preprocessed : bool
            Whether to save the preprocessed edge map alongside output files.
        """
        t0 = time.perf_counter()
        logger.info("Initializing Sketch2Design pipeline run...")

        # ── Step 1: Preprocess the input sketch ──
        preprocessed_sketch = self.processor.preprocess_sketch(sketch, method=method)

        # ── Step 2: Build / Expand Prompt ──
        final_positive = prompt
        final_negative = negative_prompt
        style_applied = style or "none"

        # Apply Week 2 PromptBuilder if available
        if self.prompt_builder and style:
            try:
                built = self.prompt_builder.build(
                    subject=prompt,
                    style=style,
                    extra_negative=[negative_prompt] if negative_prompt else None
                )
                final_positive = built.positive
                final_negative = built.negative
                style_applied = built.style_name
                logger.debug(f"Prompt expanded with style '{style_applied}'")
            except Exception as err:
                logger.warning(f"PromptBuilder failed to run: {err}. Using default prompt.")

        # If PromptBuilder is not available but style is requested, add simple tags
        if not self.prompt_builder and style:
            # Fallback simple expansion tags
            style_tags = {
                "luxury": "luxury fashion, high-end designer, haute couture, premium materials",
                "streetwear": "streetwear style, urban look, graphic design, oversized fit",
                "casual": "casual everyday fashion, minimalist style, relaxed fit, neutral colors",
                "formal": "formal attire, structured tailoring, elegant silhouette, premium fabric",
            }
            if style.lower() in style_tags:
                final_positive = f"{prompt}, {style_tags[style.lower()]}, highly detailed, photorealistic lookbook"
                logger.debug(f"Applied fallback style tags for '{style}'")

        # ── Step 3: Run Generation Engine ──
        # Sketch processor outputs canny/scribble edges. We route these to generate_from_sketch
        output = self.engine.generate_from_sketch(
            prompt=final_positive,
            sketch_image=preprocessed_sketch,
            negative_prompt=final_negative,
            conditioning_scale=conditioning_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        # ── Step 4: Metadata enrichment ──
        if output.success:
            # Add preprocessor details to output metadata
            output.metadata["preprocessor"] = {
                "method": method,
                "input_size": sketch.size,
                "style_preset": style_applied,
                "pipeline": "Sketch2Design"
            }
            output.metadata["timings"]["total_orchestration_s"] = round(time.perf_counter() - t0, 2)
            
            # Save the preprocessed edge map to output for auditability
            if save_preprocessed:
                # We store a reference to the preprocessed PIL image in metadata
                output.metadata["preprocessed_image"] = preprocessed_sketch

        return output

    def generate_batch(
        self,
        sketches: List[Image.Image],
        prompts: List[str],
        styles: Optional[List[str]] = None,
        method: str = "canny",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        **kwargs
    ) -> List[GenerationOutput]:
        """
        Process a batch list of sketches and prompts sequentially.
        """
        logger.info(f"Starting Batch Sketch2Design processing | size={len(sketches)} sketches")
        
        # Normalize lists
        if styles is None:
            styles = [None] * len(sketches)
        elif len(styles) < len(sketches):
            # Pad styles
            styles = styles + [styles[-1]] * (len(sketches) - len(styles))

        if len(prompts) < len(sketches):
            # Pad prompts
            prompts = prompts + [prompts[-1]] * (len(sketches) - len(prompts))

        outputs = []
        base_seed = seed if seed >= 0 else random.randint(0, 10000)

        for idx, (sk, pr, st) in enumerate(zip(sketches, prompts, styles)):
            logger.info(f"Batch Item [{idx+1}/{len(sketches)}] Processing prompt: '{pr[:30]}...'")
            
            # Resolve unique seed per batch item if using a base seed
            item_seed = base_seed + idx if seed >= 0 else -1
            
            out = self.generate_design(
                sketch=sk,
                prompt=pr,
                style=st,
                method=method,
                conditioning_scale=conditioning_scale,
                seed=item_seed,
                **kwargs
            )
            outputs.append(out)

        passed = sum(1 for o in outputs if o.success)
        logger.success(f"Batch Sketch2Design completed | {passed}/{len(sketches)} passed successfully")
        return outputs

    def save_results(
        self,
        output: GenerationOutput,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Saves output design images, JSON metadata sidecar, and preprocessed sketches.
        """
        if not output.success or not output.images:
            logger.warning("save_results skipped: empty output or failed run.")
            return []

        # Pop the PIL image reference out of metadata so it is not JSON serialized
        preprocessed_img = output.metadata.pop("preprocessed_image", None)

        # 1. Save standard generated image and metadata JSON via engine
        saved_paths = self.engine.save_output(output, output_dir=output_dir)

        # 2. Save preprocessed edge map if present
        if preprocessed_img and saved_paths:
            try:
                # Find output directory path
                target_dir = saved_paths[0].parent
                # Edge map gets identical ID suffix but has '_preprocessed' in name
                base_name = saved_paths[0].stem
                edge_filename = f"{base_name}_preprocessed.png"
                edge_path = target_dir / edge_filename
                
                preprocessed_img.save(edge_path, format="PNG")
                logger.info(f"Saved preprocessed sketch outline mapping to: {edge_path}")
                saved_paths.append(edge_path)
                
                # Restore the image reference in metadata
                output.metadata["preprocessed_image"] = preprocessed_img
            except Exception as exc:
                logger.warning(f"Could not write preprocessed sketch outline to disk: {exc}")

        return saved_paths
