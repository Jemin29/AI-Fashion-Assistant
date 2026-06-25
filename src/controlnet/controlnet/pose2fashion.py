"""
week3/controlnet/pose2fashion.py
================================
Pose-Guided Fashion Generation Orchestrator.
AI-Powered Fashion Design Assistant — Week 3.

Processes human pose reference photos (extracting pose skeletons) and runs
SDXL-ControlNet models to generate garments aligned with the specified pose.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image, ImageDraw

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
# ── Pose2Fashion Class
# =============================================================================

class Pose2Fashion:
    """
    Orchestrates pose-conditioned image generation, converting reference models
    or poses into custom fashion design catalogs.
    """

    def __init__(self, config=None, mock: bool = False) -> None:
        """
        Initialize the Pose2Fashion orchestrator.

        Parameters
        ----------
        config : Week3Config, optional
        mock : bool
            Force simulated execution (replaces deep learning with PIL wrappers).
        """
        self.config = config
        self.mock = mock
        self.engine = FashionControlNetEngine(config, mock=mock)
        self._pose_detector: Any = None

        # Link PromptBuilder if importable
        self.prompt_builder = None
        if _HAS_PROMPT_BUILDER and config is not None:
            try:
                self.prompt_builder = PromptBuilder(config)
                logger.debug("PromptBuilder linked successfully in Pose2Fashion.")
            except Exception as err:
                logger.debug(f"Failed to link PromptBuilder: {err}")

    # ── Public APIs: Core Methods ─────────────────────────────────────────────

    def preprocess_pose(self, image: Image.Image) -> Image.Image:
        """
        Extract Openpose skeleton joints from a human photo.
        Returns a black image containing multi-colored joint nodes and lines.
        """
        w, h = image.size
        logger.info(f"Extracting pose map skeleton | input_size={image.size}...")

        if self.mock:
            # Generate simulated skeleton map (black canvas with colored lines)
            time.sleep(0.3)
            return self._create_mock_skeleton(w, h)

        # ── Real Openpose Extraction ──────────────────────────────────────────
        global controlnet_aux
        try:
            from controlnet_aux import OpenposeDetector
            if self._pose_detector is None:
                logger.debug("Loading Openpose preprocessor model...")
                self._pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            
            # Extract pose joints
            pose_map = self._pose_detector(image, detect_hand_and_face=True)
            logger.success("Pose skeleton extracted successfully.")
            return pose_map.convert("RGB")
        except Exception as err:
            logger.warning(f"OpenposeDetector failed: {err}. Using simulated fallback skeleton.")
            return self._create_mock_skeleton(w, h)

    def generate_fashion(
        self,
        pose_image: Image.Image,
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
        Generate fashion design aligned with the specified pose skeleton.

        Parameters
        ----------
        pose_image : PIL.Image.Image
            Human reference photo or pre-extracted skeleton joint image.
        prompt : str
            Creative text details describing the clothing item (e.g. "navy blue blazer").
        style : str, optional
            Fashion style preset (e.g. "streetwear", "luxury").
        conditioning_scale : float, optional
            ControlNet weight strictness (0.0 to 1.0).
        seed : int
            Reproducibility seed.
        num_inference_steps : int
            Denoising steps.
        guidance_scale : float
        negative_prompt : str
        save_preprocessed : bool
            Save the extracted pose skeleton PNG alongside results.
        """
        t0 = time.perf_counter()
        logger.info("Initializing Pose2Fashion pipeline run...")

        # ── Step 1: Preprocess pose reference ──
        # Check if the image is already a skeleton map (black background, colored lines)
        if self._is_skeleton_image(pose_image):
            logger.info("Input appears to be a pre-extracted skeleton map. Skipping preprocessing.")
            pose_map = pose_image.convert("RGB")
        else:
            pose_map = self.preprocess_pose(pose_image)

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
                "luxury": "luxury fashion, high-end designer, couture look, premium materials",
                "streetwear": "streetwear style, urban look, graphic design details, oversized fit",
                "casual": "casual everyday outfit, relaxed fit, clean Scandi lines, neutral colors",
            }
            if style.lower() in style_tags:
                final_prompt = f"{prompt}, {style_tags[style.lower()]}, photorealistic lookbook photography"
                logger.debug(f"Applied fallback style tags for '{style}'")

        # ── Step 3: Run ControlNet Pose Generation ──
        output = self.engine.generate_from_pose(
            prompt=final_prompt,
            pose_image=pose_map,
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
                "method": "openpose",
                "input_size": pose_image.size,
                "style_preset": style_applied,
                "pipeline": "Pose2Fashion"
            }
            output.metadata["timings"]["total_orchestration_s"] = round(time.perf_counter() - t0, 2)
            
            # Save the preprocessed pose skeleton map reference
            if save_preprocessed:
                output.metadata["preprocessed_pose"] = pose_map

        return output

    def generate_batch(
        self,
        pose_images: List[Image.Image],
        prompts: List[str],
        styles: Optional[List[str]] = None,
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        **kwargs
    ) -> List[GenerationOutput]:
        """
        Process a batch list of pose images and prompts sequentially.
        """
        logger.info(f"Starting Batch Pose2Fashion processing | size={len(pose_images)} poses")

        # Pad lists if shorter
        if styles is None:
            styles = [None] * len(pose_images)
        elif len(styles) < len(pose_images):
            styles = styles + [styles[-1]] * (len(pose_images) - len(styles))

        if len(prompts) < len(pose_images):
            prompts = prompts + [prompts[-1]] * (len(pose_images) - len(prompts))

        outputs = []
        base_seed = seed if seed >= 0 else random.randint(0, 10000)

        for idx, (pose, pr, st) in enumerate(zip(pose_images, prompts, styles)):
            logger.info(f"Batch Item [{idx+1}/{len(pose_images)}] Processing prompt: '{pr[:30]}...'")
            item_seed = base_seed + idx if seed >= 0 else -1

            out = self.generate_fashion(
                pose_image=pose,
                prompt=pr,
                style=st,
                conditioning_scale=conditioning_scale,
                seed=item_seed,
                **kwargs
            )
            outputs.append(out)

        passed = sum(1 for o in outputs if o.success)
        logger.success(f"Batch Pose2Fashion completed | {passed}/{len(pose_images)} passed successfully")
        return outputs

    def save_results(
        self,
        output: GenerationOutput,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Saves output design images, JSON metadata configs, and preprocessed pose maps.
        """
        if not output.success or not output.images:
            logger.warning("save_results skipped: empty output or failed run.")
            return []

        # Pop the PIL image reference out of metadata so it is not JSON serialized
        pose_map = output.metadata.pop("preprocessed_pose", None)

        # 1. Save standard generated image and metadata JSON via engine
        saved_paths = self.engine.save_output(output, output_dir=output_dir)

        # 2. Save preprocessed pose skeleton map if present
        if pose_map and saved_paths:
            try:
                target_dir = saved_paths[0].parent
                base_name = saved_paths[0].stem
                pose_filename = f"{base_name}_pose_map.png"
                pose_path = target_dir / pose_filename
                
                pose_map.save(pose_path, format="PNG")
                logger.info(f"Saved preprocessed skeleton pose mapping to: {pose_path}")
                saved_paths.append(pose_path)
                
                # Restore the image reference in metadata
                output.metadata["preprocessed_pose"] = pose_map
            except Exception as exc:
                logger.warning(f"Could not write preprocessed pose map to disk: {exc}")

        return saved_paths

    # ── Private Utility & Fallback Methods ────────────────────────────────────

    def _is_skeleton_image(self, img: Image.Image) -> bool:
        """Heuristically checks if the input is already a black-background skeleton map."""
        # Crop a tiny 10x10 corner block and check its average luma
        try:
            corner = img.convert("L").crop((0, 0, min(20, img.width), min(20, img.height)))
            import numpy as np
            mean_val = np.mean(np.array(corner))
            # Black backgrounds typically average very close to 0 luma
            return float(mean_val) < 10.0
        except Exception:
            return False

    def _create_mock_skeleton(self, w: int, h: int) -> Image.Image:
        """Create a mock multi-colored joint skeleton on black background."""
        img = Image.new("RGB", (w, h), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw a stick figure pose skeleton representing torso, limbs, and head
        mid_x = w // 2
        head_r = min(w, h) // 12
        
        # Joint nodes coordinates
        head = (mid_x, h // 4)
        neck = (mid_x, h // 4 + head_r)
        pelvis = (mid_x, h // 2 + head_r)
        
        l_shoulder = (mid_x - w // 6, h // 3)
        r_shoulder = (mid_x + w // 6, h // 3)
        l_elbow = (mid_x - w // 5, h // 2)
        r_elbow = (mid_x + w // 5, h // 2)
        
        l_hip = (mid_x - w // 10, h // 2 + head_r)
        r_hip = (mid_x + w // 10, h // 2 + head_r)
        l_knee = (mid_x - w // 8, h // 4 * 3)
        r_knee = (mid_x + w // 8, h // 4 * 3)

        # Colors corresponding to Openpose channels (torso=blue, arms=green, legs=red)
        draw.line([neck, l_shoulder], fill=(0, 255, 255), width=4)
        draw.line([neck, r_shoulder], fill=(0, 255, 255), width=4)
        draw.line([l_shoulder, l_elbow], fill=(0, 255, 0), width=3)
        draw.line([r_shoulder, r_elbow], fill=(0, 255, 0), width=3)
        
        draw.line([neck, pelvis], fill=(255, 255, 0), width=5)
        
        draw.line([pelvis, l_hip], fill=(255, 0, 0), width=4)
        draw.line([pelvis, r_hip], fill=(255, 0, 0), width=4)
        draw.line([l_hip, l_knee], fill=(255, 0, 100), width=3)
        draw.line([r_hip, r_knee], fill=(255, 0, 100), width=3)
        
        draw.ellipse([head[0] - head_r, head[1] - head_r, head[0] + head_r, head[1] + head_r], outline=(255, 0, 255), width=3)

        return img
