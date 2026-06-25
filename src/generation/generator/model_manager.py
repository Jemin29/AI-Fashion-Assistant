"""
week2/generator/model_manager.py
=================================
Responsible for loading, caching, and releasing SDXL models.

Design
------
- Lazy loading: models are loaded on first use, not at import time
- Memory-aware: applies xformers / CPU offload based on config
- Safe teardown: explicit unload frees VRAM immediately
- Singleton pool: one ModelManager per process
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


# =============================================================================
# ── Load Result Dataclass
# =============================================================================

@dataclass
class LoadedModels:
    """Container for all loaded SDXL pipeline objects."""
    base:     Optional[object]  = None    # StableDiffusionXLPipeline
    refiner:  Optional[object]  = None    # StableDiffusionXLImg2ImgPipeline
    is_ready: bool              = False
    device:   str               = "cpu"
    dtype_str: str              = "float32"

    @property
    def has_refiner(self) -> bool:
        return self.refiner is not None


# =============================================================================
# ── Model Manager
# =============================================================================

class ModelManager:
    """
    Manages loading, memory optimisation, and teardown of SDXL pipelines.

    Parameters
    ----------
    config : Week2Config
        Loaded Week 2 configuration (model section used).

    Example
    -------
        from src.utils.config_manager import get_config
        from src.generation.generator.model_manager import ModelManager

        mgr = ModelManager(config=get_config())
        models = mgr.load()
        pipe = models.base
    """

    def __init__(self, config) -> None:
        self._cfg    = config
        self._models: Optional[LoadedModels] = None
        self._device: Optional[str] = None
        self._dtype                 = None

    # ── Public API ────────────────────────────────────────────────────────

    def load(self, load_refiner: bool = False) -> LoadedModels:
        """
        Load and return the SDXL pipeline(s).

        Parameters
        ----------
        load_refiner : bool
            Whether to also load the SDXL refiner pipeline.

        Returns
        -------
        LoadedModels
        """
        if self._models and self._models.is_ready:
            logger.debug("ModelManager: returning cached models")
            return self._models

        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            AutoencoderKL,
        )

        rt  = self._cfg.model.runtime
        mspec = self._cfg.model

        # ── Resolve device ────────────────────────────────────────────────
        device = self._resolve_device(rt.device)
        dtype  = self._resolve_dtype(rt.torch_dtype)

        logger.info(
            "Loading SDXL base model | repo={} | device={} | dtype={}",
            mspec.base.repo_id,
            device,
            rt.torch_dtype,
        )

        # ── Set HF cache ──────────────────────────────────────────────────
        os.environ.setdefault(
            "HF_HOME",
            str(Path(mspec.cache.huggingface_home).resolve())
        )

        # ── Load VAE ──────────────────────────────────────────────────────
        vae = None
        if mspec.vae.repo_id:
            logger.info("Loading VAE | repo={}", mspec.vae.repo_id)
            try:
                vae = AutoencoderKL.from_pretrained(
                    mspec.vae.local_path or mspec.vae.repo_id,
                    torch_dtype=dtype,
                )
            except Exception as exc:
                logger.warning("VAE load failed ({}), using bundled VAE", exc)
                vae = None

        # ── Load base pipeline ────────────────────────────────────────────
        load_kwargs = dict(
            torch_dtype     = dtype,
            use_safetensors = mspec.base.use_safetensors,
        )
        if mspec.base.variant:
            load_kwargs["variant"] = mspec.base.variant
        if vae is not None:
            load_kwargs["vae"] = vae

        base_source = mspec.base.local_path or mspec.base.repo_id
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_source,
            **load_kwargs,
        )

        # ── Apply memory optimisations ────────────────────────────────────
        base_pipe = self._apply_optimisations(base_pipe, rt, device)
        logger.success("SDXL base loaded successfully | device={}", device)

        # ── Optionally load refiner ────────────────────────────────────────
        refiner_pipe = None
        if load_refiner and mspec.refiner.repo_id:
            logger.info("Loading SDXL refiner | repo={}", mspec.refiner.repo_id)
            try:
                refiner_kwargs = dict(
                    text_encoder_2  = base_pipe.text_encoder_2,
                    vae             = base_pipe.vae,
                    torch_dtype     = dtype,
                    use_safetensors = mspec.refiner.use_safetensors,
                )
                if mspec.refiner.variant:
                    refiner_kwargs["variant"] = mspec.refiner.variant

                refiner_source = mspec.refiner.local_path or mspec.refiner.repo_id
                refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_source,
                    **refiner_kwargs,
                )
                refiner_pipe = self._apply_optimisations(refiner_pipe, rt, device)
                logger.success("SDXL refiner loaded | device={}", device)
            except Exception as exc:
                logger.error("Refiner load failed: {} — skipping refiner", exc)
                refiner_pipe = None

        self._models = LoadedModels(
            base      = base_pipe,
            refiner   = refiner_pipe,
            is_ready  = True,
            device    = device,
            dtype_str = rt.torch_dtype,
        )
        return self._models

    def unload(self) -> None:
        """Release all models from memory and clear VRAM."""
        if self._models:
            logger.info("Unloading SDXL models…")
            self._models.base    = None
            self._models.refiner = None
            self._models.is_ready= False
            self._models         = None
        self._cleanup_memory()
        logger.success("Models unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._models is not None and self._models.is_ready

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype_str: str):
        import torch
        mapping = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        return mapping.get(dtype_str, torch.float16)

    @staticmethod
    def _apply_optimisations(pipe, rt, device: str):
        """Apply memory-saving optimisations to a pipeline."""
        # xformers
        if rt.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.debug("xformers attention enabled")
            except Exception:
                logger.debug("xformers not available — skipping")

        # Offloads (mutually exclusive — pick most aggressive)
        if rt.enable_sequential_offload:
            pipe.enable_sequential_cpu_offload()
            logger.debug("Sequential CPU offload enabled")
        elif rt.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.debug("Model CPU offload enabled")
        else:
            pipe = pipe.to(device)

        # Optional slicing / tiling
        if rt.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if rt.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if rt.enable_vae_tiling:
            pipe.enable_vae_tiling()

        return pipe

    @staticmethod
    def _cleanup_memory() -> None:
        """Free Python objects and clear CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
