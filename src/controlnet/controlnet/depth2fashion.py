"""
week3/controlnet/depth2fashion.py
=================================
Depth-Guided Fashion Generation Orchestrator.
AI-Powered Fashion Design Assistant — Week 3.

Processes depth maps (representing fabric folds and 3D shapes) and runs
SDXL-ControlNet models to generate garments aligned with the specified depth contours.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image, ImageDraw, ImageOps

from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine, GenerationOutput

# ── Dynamic imports ──────────────────────────────────────────────────────────
try:
    from src.generation.prompts.prompt_builder import PromptBuilder
    _HAS_PROMPT_BUILDER = True
except ImportError:
    PromptBuilder = None
    _HAS_PROMPT_BUILDER = False

controlnet_aux = None


# =============================================================================
# ── Depth2Fashion Class
# =============================================================================

class Depth2Fashion:
    """
    Orchestrates depth-conditioned image generation, preserving garment structures,
    creases, and fabric drapes.
    """

    def __init__(self, config=None, mock: bool = False) -> None:
        """
        Initialize the Depth2Fashion orchestrator.

        Parameters
        ----------
        config : Week3Config, optional
        mock : bool
            Force simulated execution (replaces deep learning with PIL wrappers).
        """
        self.config = config
        self.mock = mock
        self.engine = FashionControlNetEngine(config, mock=mock)
        self._depth_detector: Any = None

        # Link PromptBuilder if importable
        self.prompt_builder = None
        if _HAS_PROMPT_BUILDER and config is not None:
            try:
                self.prompt_builder = PromptBuilder(config)
                logger.debug("PromptBuilder linked successfully in Depth2Fashion.")
            except Exception as err:
                logger.debug(f"Failed to link PromptBuilder: {err}")

    # ── Public APIs: Core Methods ─────────────────────────────────────────────

    def preprocess_depth(self, image: Image.Image) -> Image.Image:
        """
        Extract a depth gradient map from a raw photo.
        Returns a grayscale image where light parts are near and dark parts are far.
        """
        w, h = image.size
        logger.info(f"Extracting depth map | input_size={image.size}...")

        if self.mock:
            # Generate simulated depth map (radial gradient: white center, black edges)
            time.sleep(0.3)
            return self._create_mock_depth(w, h)

        # ── Real Depth Extraction ─────────────────────────────────────────────
        global controlnet_aux
        try:
            from controlnet_aux import MidasDetector
            if self._depth_detector is None:
                logger.debug("Loading MiDaS preprocessor model...")
                self._depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            
            # Extract depth
            depth_map = self._depth_detector(image)
            logger.success("Depth map extracted successfully.")
            return depth_map.convert("RGB")
        except Exception as err:
            logger.warning(f"MidasDetector failed: {err}. Using simulated fallback depth map.")
            return self._create_mock_depth(w, h)

    def generate_fashion(
        self,
        depth_image: Image.Image,
        prompt: str,
        style: Optional[str] = None,
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        save_preprocessed: bool = True,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate fashion design aligned with the specified depth contours.

        Parameters
        ----------
        depth_image : PIL.Image.Image
            Human reference photo or pre-extracted depth map image.
        prompt : str
            Creative text details describing the garment (e.g. "oversized wool trench coat").
        style : str, optional
            Fashion style preset (e.g. "minimalist", "luxury").
        conditioning_scale : float, optional
            ControlNet weight strictness (0.0 to 1.0).
        seed : int
            Reproducibility seed.
        num_inference_steps : int
            Denoising steps.
        guidance_scale : float
        negative_prompt : str
        save_preprocessed : bool
            Save the extracted depth map PNG alongside results.
        """
        t0 = time.perf_counter()
        logger.info("Initializing Depth2Fashion pipeline run...")

        # ── Step 1: Preprocess depth reference ──
        # Check if the image is already a depth map (grayscale representation)
        if self._is_depth_image(depth_image):
            logger.info("Input appears to be a pre-extracted depth map. Skipping preprocessing.")
            depth_map = depth_image.convert("RGB")
        else:
            depth_map = self.preprocess_depth(depth_image)

        # ── Step 2: Build / Expand Prompt ──
        final_prompt = prompt
        final_negative = negative_prompt
        style_applied = style or "none"

        if self.prompt_builder and style:
            try:
                built = self.prompt_builder.build(
                    subject=prompt,
                    style=style,
                    extra_negative=[negative_prompt] if negative_prompt else None
                )
                final_prompt = built.positive
                final_negative = built.negative
                style_applied = built.style_name
                logger.debug(f"Prompt expanded with style '{style_applied}'")
            except Exception as err:
                logger.warning(f"PromptBuilder failed to run: {err}. Using default prompt.")

        if not self.prompt_builder and style:
            # Fallback simple expansion tags
            style_tags = {
                "luxury": "luxury fashion, high-end designer, premium silk, couturier details",
                "minimalist": "minimalist aesthetic, clean Scandi lines, high-quality wool, monochrome",
                "casual": "casual everyday fit, relaxed fit, comfortable fabric, clean stitching",
            }
            if style.lower() in style_tags:
                final_prompt = f"{prompt}, {style_tags[style.lower()]}, lookbook photography"
                logger.debug(f"Applied fallback style tags for '{style}'")

        # ── Step 3: Run ControlNet Depth Generation ──
        output = self.engine.generate_from_depth(
            prompt=final_prompt,
            depth_image=depth_map,
            negative_prompt=final_negative,
            conditioning_scale=conditioning_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        # ── Step 4: Metadata enrichment ──
        if output.success:
            output.metadata["preprocessor"] = {
                "method": "depth",
                "input_size": depth_image.size,
                "style_preset": style_applied,
                "pipeline": "Depth2Fashion"
            }
            output.metadata["timings"]["total_orchestration_s"] = round(time.perf_counter() - t0, 2)
            
            # Save the preprocessed depth map reference
            if save_preprocessed:
                output.metadata["preprocessed_depth"] = depth_map

        return output

    def generate_batch(
        self,
        depth_images: List[Image.Image],
        prompts: List[str],
        styles: Optional[List[str]] = None,
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        **kwargs
    ) -> List[GenerationOutput]:
        """
        Process a batch list of depth images and prompts sequentially.
        """
        logger.info(f"Starting Batch Depth2Fashion processing | size={len(depth_images)} depth images")

        # Pad lists if shorter
        if styles is None:
            styles = [None] * len(depth_images)
        elif len(styles) < len(depth_images):
            styles = styles + [styles[-1]] * (len(depth_images) - len(styles))

        if len(prompts) < len(depth_images):
            prompts = prompts + [prompts[-1]] * (len(depth_images) - len(prompts))

        outputs = []
        base_seed = seed if seed >= 0 else random.randint(0, 10000)

        for idx, (depth, pr, st) in enumerate(zip(depth_images, prompts, styles)):
            logger.info(f"Batch Item [{idx+1}/{len(depth_images)}] Processing prompt: '{pr[:30]}...'")
            item_seed = base_seed + idx if seed >= 0 else -1

            out = self.generate_fashion(
                depth_image=depth,
                prompt=pr,
                style=st,
                conditioning_scale=conditioning_scale,
                seed=item_seed,
                **kwargs
            )
            outputs.append(out)

        passed = sum(1 for o in outputs if o.success)
        logger.success(f"Batch Depth2Fashion completed | {passed}/{len(depth_images)} passed successfully")
        return outputs

    def save_results(
        self,
        output: GenerationOutput,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Saves output design images, JSON metadata configs, and preprocessed depth maps.
        """
        if not output.success or not output.images:
            logger.warning("save_results skipped: empty output or failed run.")
            return []

        # Pop the PIL image reference out of metadata so it is not JSON serialized
        depth_map = output.metadata.pop("preprocessed_depth", None)

        # 1. Save standard generated image and metadata JSON via engine
        saved_paths = self.engine.save_output(output, output_dir=output_dir)

        # 2. Save preprocessed depth map if present
        if depth_map and saved_paths:
            try:
                target_dir = saved_paths[0].parent
                base_name = saved_paths[0].stem
                depth_filename = f"{base_name}_depth_map.png"
                depth_path = target_dir / depth_filename
                
                depth_map.save(depth_path, format="PNG")
                logger.info(f"Saved preprocessed depth mapping to: {depth_path}")
                saved_paths.append(depth_path)
                
                # Restore reference
                output.metadata["preprocessed_depth"] = depth_map
            except Exception as exc:
                logger.warning(f"Could not write preprocessed depth map to disk: {exc}")

        return saved_paths

    # ── Private Utility & Fallback Methods ────────────────────────────────────

    def _is_depth_image(self, img: Image.Image) -> bool:
        """Heuristically checks if the input is already a grayscale depth map."""
        # Depth maps are typically single-channel grayscale (L) or have near-identical R, G, B channels
        try:
            if img.mode == "L" or img.mode == "1":
                return True
            
            # Check standard deviation of color channels (RGB difference)
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                rgb_diff = np.abs(arr[:, :, 0] - arr[:, :, 1]) + np.abs(arr[:, :, 1] - arr[:, :, 2])
                mean_diff = np.mean(rgb_diff)
                # True grayscale images converted to RGB will have identical channels (mean difference = 0.0)
                return float(mean_diff) < 2.0
            return False
        except Exception:
            return False

    def _create_mock_depth(self, w: int, h: int) -> Image.Image:
        """Create a mock radial gradient depth map (white in center, black at borders)."""
        img = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(img)

        # Draw concentric filled circles from dark to light to simulate depth values
        center_x = w // 2
        center_y = h // 2
        max_r = max(w, h) // 2

        steps = 16
        for r_idx in range(steps):
            r = int(max_r * (1.0 - (r_idx / steps)))
            luma = int(255 * (r_idx / steps))
            draw.ellipse(
                [center_x - r, center_y - r, center_x + r, center_y + r],
                fill=luma
            )
            
        return img.convert("RGB")
