"""
week3/pipelines/multi_control_pipeline.py
=========================================
Advanced Multi-Control Conditioning Fashion Pipeline.
AI-Powered Fashion Design Assistant — Week 3.

Combines Sketch, Pose, and Depth ControlNets simultaneously to generate fashion designs
matching specific shapes, postures, and spatial drapings.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from PIL import Image, ImageDraw

from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine, GenerationOutput
from src.controlnet.preprocessors.sketch_processor import SketchProcessor
from src.controlnet.controlnet.pose2fashion import Pose2Fashion
from src.controlnet.controlnet.depth2fashion import Depth2Fashion

# ── Dynamic imports ──────────────────────────────────────────────────────────
try:
    from src.generation.prompts.prompt_builder import PromptBuilder
    _HAS_PROMPT_BUILDER = True
except ImportError:
    PromptBuilder = None
    _HAS_PROMPT_BUILDER = False

torch = None
ControlNetModel = None
StableDiffusionXLControlNetPipeline = None


class MultiControlFashionPipeline:
    """
    Coordinates multi-control SDXL model execution using Sketch, Pose, and Depth.
    Handles dynamic preprocessing, weighted conditioning scales, and sidecar saving.
    """

    def __init__(self, config: Any = None, mock: bool = False) -> None:
        """
        Initialize the MultiControlFashionPipeline.

        Parameters
        ----------
        config : Week3Config, optional
        mock : bool
            Force mock/simulated execution mode.
        """
        self.config = config
        self.mock = mock

        # Instantiate sub-preprocessors/orchestrators
        self.sketch_processor = SketchProcessor(config=config)
        self.pose_orchestrator = Pose2Fashion(config=config, mock=mock)
        self.depth_orchestrator = Depth2Fashion(config=config, mock=mock)

        # Base engine helper for standard saves and configurations
        self.engine = FashionControlNetEngine(config=config, mock=mock)

        self._pipe: Any = None
        self._loaded_controlnets: List[str] = []

        # Link PromptBuilder if importable
        self.prompt_builder = None
        if _HAS_PROMPT_BUILDER and config is not None:
            try:
                self.prompt_builder = PromptBuilder(config)
                logger.debug("PromptBuilder linked successfully in MultiControlFashionPipeline.")
            except Exception as err:
                logger.debug(f"Failed to link PromptBuilder: {err}")

    # ── Public APIs: Core Methods ─────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        sketch_image: Optional[Image.Image] = None,
        pose_image: Optional[Image.Image] = None,
        depth_image: Optional[Image.Image] = None,
        sketch_scale: float = 0.5,
        pose_scale: float = 0.8,
        depth_scale: float = 0.3,
        style: Optional[str] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        save_preprocessed: bool = True,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate fashion design aligned with multiple active spatial conditions.
        """
        t0 = time.perf_counter()
        logger.info("Initializing Advanced Multi-Control pipeline run...")

        # 1. Verify inputs
        if sketch_image is None and pose_image is None and depth_image is None:
            raise ValueError("At least one conditioning image (sketch, pose, or depth) must be provided.")

        # ── Step 1: Preprocess active inputs ──
        active_images: List[Image.Image] = []
        active_scales: List[float] = []
        active_types: List[str] = []
        preprocessed_maps: Dict[str, Image.Image] = {}

        # Sketch Conditioning
        if sketch_image is not None:
            logger.info("Sketch condition detected. Resolving edges...")
            sketch_map = self.sketch_processor.preprocess_sketch(sketch_image)
            active_images.append(sketch_map)
            active_scales.append(sketch_scale)
            active_types.append("sketch")
            preprocessed_maps["sketch"] = sketch_map

        # Pose Conditioning
        if pose_image is not None:
            logger.info("Pose condition detected. Resolving skeleton joints...")
            # Use Pose2Fashion's heuristic to check if it's already a pose stick map
            if self.pose_orchestrator._is_skeleton_image(pose_image):
                pose_map = pose_image.convert("RGB")
            else:
                pose_map = self.pose_orchestrator.preprocess_pose(pose_image)
            active_images.append(pose_map)
            active_scales.append(pose_scale)
            active_types.append("pose")
            preprocessed_maps["pose"] = pose_map

        # Depth Conditioning
        if depth_image is not None:
            logger.info("Depth condition detected. Resolving 3D contours...")
            # Use Depth2Fashion's heuristic to check if it's already a depth map
            if self.depth_orchestrator._is_depth_image(depth_image):
                depth_map = depth_image.convert("RGB")
            else:
                depth_map = self.depth_orchestrator.preprocess_depth(depth_image)
            active_images.append(depth_map)
            active_scales.append(depth_scale)
            active_types.append("depth")
            preprocessed_maps["depth"] = depth_map

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
            style_tags = {
                "luxury": "luxury fashion, premium couturier details, lookbook photography",
                "streetwear": "streetwear style, oversized silhouette, urban fashion, lookbook",
                "casual": "casual everyday style, comfortable fit, clean lookbook",
            }
            if style.lower() in style_tags:
                final_prompt = f"{prompt}, {style_tags[style.lower()]}"

        # ── Step 3: Run Generation ──
        gen_seed = seed if seed >= 0 else random.randint(0, 1000000)

        if self.mock:
            # Mock mode: generate dummy composite image and timings
            time.sleep(0.4)
            logger.info("Mock mode active. Generating simulated multi-control design...")
            w, h = (1024, 1024)
            if self.config:
                # Default SDXL dimensions
                w = getattr(self.config.preprocessors.canny, "resolution", 1024)
                h = w
            
            # Draw mock design with overlays
            img = Image.new("RGB", (w, h), color=(40, 45, 55))
            draw = ImageDraw.Draw(img)
            
            # Draw border
            draw.rectangle([10, 10, w - 10, h - 10], outline=(150, 150, 150), width=4)
            
            # Draw labels
            label_text = f"MOCK MULTI-CONTROL DESIGN\nSeed: {gen_seed}\nPrompt: '{final_prompt[:30]}...'\n\nActive Controls:"
            for t, sc in zip(active_types, active_scales):
                label_text += f"\n- {t.upper()} (scale={sc})"
            
            draw.text((w // 2, h // 2), label_text, fill=(255, 255, 255), anchor="mm", align="center")
            
            output = GenerationOutput(
                images=[img],
                metadata={
                    "pipeline": "MultiControlFashionPipeline",
                    "device": "cpu",
                    "is_mock": True,
                    "timings": {"generation_s": 0.4}
                },
                prompt=final_prompt,
                negative_prompt=final_negative,
                seed=gen_seed,
                control_type="multi_control",
                success=True,
                elapsed_time_s=0.4
            )
        else:
            output = self._run_real_generation(
                active_images=active_images,
                active_scales=active_scales,
                active_types=active_types,
                prompt=final_prompt,
                negative_prompt=final_negative,
                seed=gen_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )

        # ── Step 4: Metadata enrichment ──
        if output.success:
            output.metadata["prompt"] = output.prompt
            output.metadata["negative_prompt"] = output.negative_prompt
            output.metadata["generation"] = {
                "seed": output.seed,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_type": "multi_control",
                "mock": self.mock
            }
            output.metadata["preprocessor"] = {
                "method": "multi_control",
                "active_types": active_types,
                "conditioning_scales": active_scales,
                "style_preset": style_applied,
                "pipeline": "MultiControlFashionPipeline"
            }
            output.metadata["timings"]["total_orchestration_s"] = round(time.perf_counter() - t0, 2)
            
            if save_preprocessed:
                # Save references in metadata
                for k, v in preprocessed_maps.items():
                    output.metadata[f"preprocessed_{k}"] = v

        return output

    def save_results(
        self,
        output: GenerationOutput,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Saves output design images, JSON metadata configs, and all preprocessed condition sidecar maps.
        """
        if not output.success or not output.images:
            logger.warning("save_results skipped: empty output or failed run.")
            return []

        # Pop the PIL image references out of metadata so they are not JSON serialized
        sketch_map = output.metadata.pop("preprocessed_sketch", None)
        pose_map = output.metadata.pop("preprocessed_pose", None)
        depth_map = output.metadata.pop("preprocessed_depth", None)

        # 1. Save standard generated image and metadata JSON via engine
        saved_paths = self.engine.save_output(output, output_dir=output_dir)

        # 2. Save preprocessed condition maps if present
        if saved_paths:
            try:
                target_dir = saved_paths[0].parent
                base_name = saved_paths[0].stem
                
                maps_to_save = {
                    "sketch": sketch_map,
                    "pose": pose_map,
                    "depth": depth_map
                }
                
                for k, v in maps_to_save.items():
                    if v:
                        filename = f"{base_name}_{k}_map.png"
                        map_path = target_dir / filename
                        v.save(map_path, format="PNG")
                        logger.info(f"Saved preprocessed {k} mapping to: {map_path}")
                        saved_paths.append(map_path)
                        
                        # Restore reference
                        output.metadata[f"preprocessed_{k}"] = v
            except Exception as exc:
                logger.warning(f"Could not write preprocessed condition sidecar maps to disk: {exc}")

        return saved_paths

    # ── Private Weight/Pipeline Execution Helpers ─────────────────────────────

    def _run_real_generation(
        self,
        active_images: List[Image.Image],
        active_scales: List[float],
        active_types: List[str],
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        **kwargs
    ) -> GenerationOutput:
        """Executes real multi-ControlNet SDXL inference."""
        t0 = time.perf_counter()

        # Import heavy dependencies
        global torch, ControlNetModel, StableDiffusionXLControlNetPipeline
        try:
            import torch as _torch
            from diffusers import (
                ControlNetModel as _CNetModel,
                StableDiffusionXLControlNetPipeline as _XLCPipe
            )
            torch = _torch
            ControlNetModel = _CNetModel
            StableDiffusionXLControlNetPipeline = _XLCPipe
        except ImportError as err:
            logger.error(f"Failed to load diffusers inside real generation: {err}")
            raise

        # Check if we need to reload pipeline with new active controlnets
        if self._loaded_controlnets != active_types or self._pipe is None:
            logger.info(f"Loading/rebuilding multi-ControlNet pipeline for active layers: {active_types}...")
            
            # Resolve repos
            MODEL_MAPPINGS = {
                "canny": "diffusers/controlnet-canny-sdxl-1.0",
                "sketch": "xinsir/controlnet-scribble-sdxl-1.0",
                "pose": "thibaud/controlnet-openpose-sdxl-1.0",
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
            }
            
            base_repo = "stabilityai/stable-diffusion-xl-base-1.0"
            if self.config:
                base_repo = self.config.model.base.repo_id or base_repo

            torch_dtype = torch.float16 if self.engine.dtype == "float16" else torch.float32

            # Load active controlnets
            controlnet_models = []
            for t in active_types:
                cnet_repo = MODEL_MAPPINGS[t]
                # Allow configs overrides
                if self.config and t == "canny" and self.config.controlnet.model_id:
                    cnet_repo = self.config.controlnet.model_id
                
                model = ControlNetModel.from_pretrained(
                    cnet_repo,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
                controlnet_models.append(model)

            # Load pipeline
            self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_repo,
                controlnet=controlnet_models,
                torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(self.engine.device)

            # Enable memory optimizations
            self._pipe.enable_attention_slicing()
            if self.engine.device == "cuda":
                self._pipe.enable_model_cpu_offload()

            self._loaded_controlnets = list(active_types)

        # Set generator seed
        generator = torch.Generator(device=self.engine.device).manual_seed(seed)

        # Execute
        try:
            logger.info("Executing multi-ControlNet forward pass...")
            images = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=active_images,
                controlnet_conditioning_scale=active_scales,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            ).images

            elapsed = round(time.perf_counter() - t0, 2)
            logger.success(f"Multi-Control design generated successfully in {elapsed}s.")

            return GenerationOutput(
                images=images,
                metadata={
                    "pipeline": "MultiControlFashionPipeline",
                    "device": self.engine.device,
                    "is_mock": False,
                    "timings": {"generation_s": elapsed}
                },
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                control_type="multi_control",
                success=True,
                elapsed_time_s=elapsed
            )

        except Exception as exc:
            logger.error(f"Multi-ControlNet SDXL inference failed: {exc}")
            return GenerationOutput(
                success=False,
                error=str(exc),
                control_type="multi_control"
            )
