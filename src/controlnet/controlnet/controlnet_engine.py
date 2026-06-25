"""
week3/controlnet/controlnet_engine.py
=====================================
Reusable ControlNet SDXL Generation Engine.
AI-Powered Fashion Design Assistant — Week 3.

Exposes the FashionControlNetEngine class which manages loading and executing
Canny, Sketch, Pose, and Depth ControlNet models for Stable Diffusion XL.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image, ImageDraw

# ── Lazy imports to allow importing without CUDA/dependencies ────────────────
torch = None
ControlNetModel = None
StableDiffusionXLControlNetPipeline = None
EulerDiscreteScheduler = None
DPMSolverMultistepScheduler = None


# =============================================================================
# ── GenerationOutput Dataclass
# =============================================================================

@dataclass
class GenerationOutput:
    """
    Structured response returned by all generation methods in FashionControlNetEngine.
    """
    images: List[Image.Image] = field(default_factory=list)
    image_paths: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    control_type: str = "canny"
    success: bool = True
    error: Optional[str] = None
    elapsed_time_s: float = 0.0
    device: str = "cpu"
    is_mock: bool = False

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "✓ SUCCESS" if self.success else f"✗ ERROR({self.error})"
        return (
            f"GenerationOutput({status} | "
            f"type={self.control_type} | "
            f"n={len(self.images)} | "
            f"time={self.elapsed_time_s:.1f}s | "
            f"mock={self.is_mock})"
        )


# =============================================================================
# ── FashionControlNetEngine Class
# =============================================================================

class FashionControlNetEngine:
    """
    ControlNet-based image generation engine using Stable Diffusion XL.
    Supports Canny edges, Scribble sketch, Openpose, and Depth map conditioning.
    """

    # Default Hugging Face weights repository IDs
    MODEL_MAPPINGS = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "sketch": "xinsir/controlnet-scribble-sdxl-1.0",
        "pose": "thibaud/controlnet-openpose-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
    }

    def __init__(self, config=None, mock: bool = False) -> None:
        """
        Initialize the ControlNet engine.

        Parameters
        ----------
        config : Week3Config, optional
            Typed Pydantic configuration settings.
        mock : bool
            Force mock/simulated execution mode (skips heavy model loading).
        """
        # Load config if not provided
        if config is None:
            try:
                from src.utils.config_manager import get_config
                self.config = get_config()
            except ImportError:
                self.config = None
        else:
            self.config = config

        self.mock = mock
        self.device = "cpu"
        self.dtype = "float32"
        self._current_type: Optional[str] = None
        self._pipe: Any = None
        self._controlnet: Any = None

        # Resolve hardware settings
        self._resolve_hardware()
        logger.info(
            "FashionControlNetEngine initialized | device={} | dtype={} | mock={}",
            self.device, self.dtype, self.mock
        )

    # ── Public APIs: Models Loading ───────────────────────────────────────────

    def load_models(self, controlnet_type: str = "canny") -> bool:
        """
        Lazy load SDXL pipeline and ControlNet checkpoints for a specific control type.
        """
        controlnet_type = controlnet_type.lower()
        if controlnet_type not in self.MODEL_MAPPINGS:
            raise ValueError(
                f"Unsupported controlnet_type: '{controlnet_type}'. "
                f"Must be one of: {list(self.MODEL_MAPPINGS.keys())}"
            )

        if self.mock:
            self._current_type = controlnet_type
            logger.info("Mock mode active. Models loaded virtually.")
            return True

        if self._current_type == controlnet_type and self._pipe is not None:
            logger.info(f"ControlNet model for '{controlnet_type}' already loaded.")
            return True

        t0 = time.perf_counter()
        logger.info(f"Loading SDXL base pipeline with ControlNet: '{controlnet_type}'...")

        # Unload any existing model to reclaim VRAM
        self.unload()

        # Import heavy deep learning modules lazily
        global torch, ControlNetModel, StableDiffusionXLControlNetPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
        try:
            import torch as _torch
            from diffusers import (
                ControlNetModel as _CNetModel,
                StableDiffusionXLControlNetPipeline as _XLCPipe,
                EulerDiscreteScheduler as _EulerSched,
                DPMSolverMultistepScheduler as _DPMSched
            )
            torch = _torch
            ControlNetModel = _CNetModel
            StableDiffusionXLControlNetPipeline = _XLCPipe
            EulerDiscreteScheduler = _EulerSched
            DPMSolverMultistepScheduler = _DPMSched
        except ImportError as err:
            logger.error(f"Failed to import PyTorch/Diffusers. Falling back to mock: {err}")
            self.mock = True
            self._current_type = controlnet_type
            return True

        try:
            # Resolve model repo IDs from config overrides if present
            cnet_repo = self.MODEL_MAPPINGS[controlnet_type]
            base_repo = "stabilityai/stable-diffusion-xl-base-1.0"
            vae_repo = "madebyollin/sdxl-vae-fp16-fix"

            if self.config:
                base_repo = self.config.model.base.repo_id or base_repo
                vae_repo = self.config.model.vae.repo_id or vae_repo
                if self.config.controlnet.enabled and self.config.controlnet.model_id:
                    # Only override default repo if same type
                    if controlnet_type == "canny":
                        cnet_repo = self.config.controlnet.model_id

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            # 1. Load ControlNet model weight
            logger.debug(f"Loading ControlNet weights from {cnet_repo}...")
            self._controlnet = ControlNetModel.from_pretrained(
                cnet_repo,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )

            # 2. Load SDXL base pipeline
            logger.debug(f"Loading SDXL base from {base_repo}...")
            self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_repo,
                controlnet=self._controlnet,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )

            # 3. Apply VAE fix
            if vae_repo:
                from diffusers import AutoencoderKL
                logger.debug(f"Loading custom VAE: {vae_repo}...")
                self._pipe.vae = AutoencoderKL.from_pretrained(
                    vae_repo,
                    torch_dtype=torch_dtype
                )

            # 4. Configure Scheduler
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config
            )

            # 5. Apply memory optimizations
            self._apply_optimizations()

            self._current_type = controlnet_type
            logger.success(
                "ControlNet SDXL pipeline loaded successfully in {:.2f}s",
                time.perf_counter() - t0
            )
            return True

        except Exception as exc:
            logger.error("Failed to load ControlNet models: {}", exc)
            self.unload()
            raise RuntimeError(f"Failed to load diffusion pipeline: {exc}") from exc

    def unload(self) -> None:
        """Clear GPU/CPU model instances from memory."""
        self._pipe = None
        self._controlnet = None
        self._current_type = None

        if not self.mock and torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Diffusion pipeline memory reclaimed.")

    # ── Public APIs: Generation Methods ───────────────────────────────────────

    def generate_from_sketch(
        self,
        prompt: str,
        sketch_image: Image.Image,
        negative_prompt: str = "",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate image conditioned on sketch contours/edges.
        """
        return self._generate(
            prompt=prompt,
            control_image=sketch_image,
            control_type="sketch",
            negative_prompt=negative_prompt,
            conditioning_scale=conditioning_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

    def generate_from_pose(
        self,
        prompt: str,
        pose_image: Image.Image,
        negative_prompt: str = "",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate image conditioned on pose skeleton structure.
        """
        return self._generate(
            prompt=prompt,
            control_image=pose_image,
            control_type="pose",
            negative_prompt=negative_prompt,
            conditioning_scale=conditioning_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

    def generate_from_depth(
        self,
        prompt: str,
        depth_image: Image.Image,
        negative_prompt: str = "",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate image conditioned on depth maps.
        """
        return self._generate(
            prompt=prompt,
            control_image=depth_image,
            control_type="depth",
            negative_prompt=negative_prompt,
            conditioning_scale=conditioning_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

    def save_output(
        self,
        output: GenerationOutput,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Save generated images and JSON metadata sidecar.
        """
        if not output.success or not output.images:
            logger.warning("save_output skipped: output is empty or failed.")
            return []

        out_path = Path(output_dir) if output_dir else Path("week3/outputs")
        if self.config and not output_dir:
            out_path = self.config.output_dir
            
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        # Generate unique prefix name
        run_id = str(uuid.uuid4())[:8]
        ts = int(time.time())

        for idx, img in enumerate(output.images):
            # Save Image file
            filename = f"cnet_{output.control_type}_{ts}_{run_id}_{idx+1}.png"
            filepath = out_path / filename
            img.save(filepath, format="PNG")
            saved_paths.append(filepath)
            output.image_paths.append(filepath)
            logger.info(f"Saved generated image to {filepath}")

        # Save JSON metadata sidecar
        meta_filename = f"metadata_cnet_{output.control_type}_{ts}_{run_id}.json"
        meta_filepath = out_path / meta_filename
        
        meta_filepath.write_text(
            json.dumps(output.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Saved metadata sidecar to {meta_filepath}")

        return saved_paths

    # ── Private Utility functions ─────────────────────────────────────────────

    def _resolve_hardware(self) -> None:
        """Detect and set optimal device and dtype configurations."""
        if self.mock:
            self.device = "cpu"
            self.dtype = "float32"
            return

        device_override = "auto"
        dtype_override = "float16"

        if self.config:
            device_override = self.config.model.runtime.device
            dtype_override = self.config.model.runtime.torch_dtype

        # Resolve Device
        if device_override == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device_override

        # Resolve Dtype
        if self.device == "cpu":
            # float16 operations are not supported natively on CPU by PyTorch
            self.dtype = "float32"
        else:
            self.dtype = dtype_override

    def _apply_optimizations(self) -> None:
        """Apply memory-efficiency optimizations based on config/runtimes."""
        if self._pipe is None:
            return

        enable_xformers = True
        enable_cpu_offload = False
        enable_seq_offload = False
        enable_slicing = False

        if self.config:
            r = self.config.model.runtime
            enable_xformers = r.enable_xformers
            enable_cpu_offload = r.enable_cpu_offload
            enable_seq_offload = r.enable_sequential_offload
            enable_slicing = r.enable_attention_slicing

        # 1. GPU offloads
        if self.device == "cuda":
            if enable_seq_offload:
                logger.debug("Enabling sequential CPU offloading...")
                self._pipe.enable_sequential_cpu_offload()
            elif enable_cpu_offload:
                logger.debug("Enabling model CPU offloading...")
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe.to("cuda")

            if enable_xformers:
                try:
                    logger.debug("Enabling xformers memory efficient attention...")
                    self._pipe.enable_xformers_memory_efficient_attention()
                except Exception as err:
                    logger.warning(f"Could not enable xformers: {err}")

            if enable_slicing:
                logger.debug("Enabling attention slicing...")
                self._pipe.enable_attention_slicing()
        else:
            # CPU or MPS
            self._pipe.to(self.device)

    def _generate(
        self,
        prompt: str,
        control_image: Image.Image,
        control_type: str,
        negative_prompt: str = "",
        conditioning_scale: Optional[float] = None,
        seed: int = -1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> GenerationOutput:
        """Internal worker function to load, preprocess, and execute diffusion."""
        t0 = time.perf_counter()
        control_type = control_type.lower()
        logger.info(f"Generating from ControlNet '{control_type}' | prompt: {prompt[:50]}...")

        # 1. Load correct model first
        try:
            self.load_models(control_type)
        except Exception as err:
            logger.error(f"Failed to auto-load models: {err}")
            return GenerationOutput(success=False, error=str(err), control_type=control_type)

        # 2. Resolve parameters
        resolved_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        scale = conditioning_scale
        if scale is None:
            scale = self.config.controlnet.conditioning_scale if self.config else 0.8

        if self.mock:
            # Simulated Mock image generation
            time.sleep(0.8) # Simulate generation latency
            mock_img = self._create_mock_image(prompt, control_image, control_type, resolved_seed)
            elapsed = time.perf_counter() - t0
            
            meta = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "generation": {
                    "seed": resolved_seed,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_type": control_type,
                    "conditioning_scale": scale,
                    "mock": True
                },
                "timings": {
                    "inference_s": round(elapsed, 2)
                }
            }
            return GenerationOutput(
                images=[mock_img],
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=resolved_seed,
                control_type=control_type,
                success=True,
                elapsed_time_s=elapsed,
                is_mock=True,
                metadata=meta
            )

        # ── Real Execution loop ───────────────────────────────────────────────
        try:
            generator = torch.Generator(device=self.device).manual_seed(resolved_seed)
            
            # Prepare image size
            w, h = control_image.size
            
            # Setup prompt and negative prompt
            logger.info("Executing StableDiffusionXLControlNetPipeline inference...")
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image.convert("RGB"),
                controlnet_conditioning_scale=float(scale),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                **kwargs
            )

            images = result.images
            elapsed = time.perf_counter() - t0

            # Build metadata sidecar dict
            meta = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "generation": {
                    "seed": resolved_seed,
                    "width": w,
                    "height": h,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_type": control_type,
                    "conditioning_scale": scale,
                    "scheduler": self._pipe.scheduler.__class__.__name__,
                    "device": self.device,
                    "dtype": self.dtype,
                },
                "timings": {
                    "inference_s": round(elapsed, 2)
                }
            }

            logger.success(f"Generation completed successfully in {elapsed:.2f}s")
            return GenerationOutput(
                images=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=resolved_seed,
                control_type=control_type,
                success=True,
                elapsed_time_s=elapsed,
                metadata=meta,
                device=self.device
            )

        except Exception as exc:
            logger.error("Inference run failed: {}", exc)
            return GenerationOutput(
                success=False,
                error=str(exc),
                control_type=control_type,
                elapsed_time_s=time.perf_counter() - t0
            )

    def _create_mock_image(
        self,
        prompt: str,
        control_image: Image.Image,
        control_type: str,
        seed: int
    ) -> Image.Image:
        """Creates a mock output overlaying the control image representation on background."""
        w, h = control_image.size
        # Pick background color based on control type
        colors = {
            "canny": (40, 40, 50),
            "sketch": (60, 40, 40),
            "pose": (40, 60, 40),
            "depth": (40, 40, 60),
        }
        bg_color = colors.get(control_type, (50, 50, 50))
        img = Image.new("RGB", (w, h), color=bg_color)
        
        # Overlay a thumbnail of the control image in the bottom-right corner
        thumb_w, thumb_h = w // 4, h // 4
        thumbnail = control_image.convert("RGB").resize((thumb_w, thumb_h))
        img.paste(thumbnail, (w - thumb_w - 10, h - thumb_h - 10))
        
        # Draw framing lines and identifiers
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, w - 10, h - 10], outline=(200, 200, 200), width=1)
        
        # Add labels
        text_lines = [
            "FASHION CONTROLNET ENGINE",
            f"Control Mode: {control_type.upper()}",
            f"Prompt: {prompt[:35]}...",
            f"Seed: {seed}",
            "[MOCK GENERATION SUCCESS]"
        ]
        
        y = h // 4
        for line in text_lines:
            draw.text((w // 2, y), line, fill=(255, 255, 255), anchor="mm")
            y += 35
            
        return img
