"""
week2/generator/scheduler_factory.py
======================================
Registry of all supported noise schedulers for SDXL.

Usage
-----
    from src.generation.generator.scheduler_factory import SchedulerFactory

    scheduler = SchedulerFactory.create("euler", pipeline)
    # or
    pipeline.scheduler = SchedulerFactory.from_name("dpm++", pipeline.scheduler.config)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger


# =============================================================================
# ── Scheduler Registry
# =============================================================================

# Maps friendly names → (diffusers class name, import path)
_SCHEDULER_REGISTRY: Dict[str, Dict[str, str]] = {
    "euler": {
        "class": "EulerDiscreteScheduler",
        "module": "diffusers",
    },
    "euler_a": {
        "class": "EulerAncestralDiscreteScheduler",
        "module": "diffusers",
    },
    "dpm++": {
        "class": "DPMSolverMultistepScheduler",
        "module": "diffusers",
    },
    "dpm++_sde": {
        "class": "DPMSolverSDEScheduler",
        "module": "diffusers",
    },
    "dpm_2": {
        "class": "KDPM2DiscreteScheduler",
        "module": "diffusers",
    },
    "dpm_2_a": {
        "class": "KDPM2AncestralDiscreteScheduler",
        "module": "diffusers",
    },
    "pndm": {
        "class": "PNDMScheduler",
        "module": "diffusers",
    },
    "ddim": {
        "class": "DDIMScheduler",
        "module": "diffusers",
    },
    "lms": {
        "class": "LMSDiscreteScheduler",
        "module": "diffusers",
    },
    "heun": {
        "class": "HeunDiscreteScheduler",
        "module": "diffusers",
    },
    "unipc": {
        "class": "UniPCMultistepScheduler",
        "module": "diffusers",
    },
}


class SchedulerFactory:
    """
    Factory for creating and swapping SDXL noise schedulers.

    All schedulers are created by copying the pipeline's current
    scheduler config — ensuring compatibility with the loaded model.
    """

    @classmethod
    def available(cls) -> List[str]:
        """Return list of all registered scheduler names."""
        return list(_SCHEDULER_REGISTRY.keys())

    @classmethod
    def from_name(
        cls,
        name: str,
        config: Any,
        karras_sigmas: bool = False,
    ) -> Any:
        """
        Instantiate a scheduler from a registered name and config.

        Parameters
        ----------
        name : str
            Scheduler name (e.g. "euler", "dpm++", "ddim").
        config : SchedulerConfig
            The ``pipeline.scheduler.config`` dict from the loaded model.
        karras_sigmas : bool
            Use Karras-style sigma schedule (supported by some schedulers).

        Returns
        -------
        Scheduler instance.
        """
        if name not in _SCHEDULER_REGISTRY:
            logger.warning(
                "Unknown scheduler {!r}, falling back to 'euler'. "
                "Available: {}",
                name,
                cls.available(),
            )
            name = "euler"

        entry = _SCHEDULER_REGISTRY[name]
        try:
            import importlib
            mod = importlib.import_module(entry["module"])
            cls_obj = getattr(mod, entry["class"])

            kwargs: Dict[str, Any] = {}
            if karras_sigmas and hasattr(cls_obj, "use_karras_sigmas"):
                kwargs["use_karras_sigmas"] = True

            scheduler = cls_obj.from_config(config, **kwargs)
            logger.debug("Scheduler created: {} ({})", name, entry["class"])
            return scheduler

        except Exception as exc:
            logger.error(
                "Failed to create scheduler {} ({}): {}",
                name, entry["class"], exc
            )
            raise

    @classmethod
    def apply_to_pipeline(
        cls,
        pipeline: Any,
        scheduler_name: str,
        karras_sigmas: bool = False,
    ) -> Any:
        """
        Swap the scheduler on an already-loaded pipeline in-place.

        Parameters
        ----------
        pipeline
            Loaded diffusers pipeline.
        scheduler_name : str
            Target scheduler name.
        karras_sigmas : bool
            Use Karras sigmas.

        Returns
        -------
        The same pipeline with scheduler replaced.
        """
        new_scheduler = cls.from_name(
            scheduler_name,
            pipeline.scheduler.config,
            karras_sigmas=karras_sigmas,
        )
        pipeline.scheduler = new_scheduler
        logger.info("Pipeline scheduler swapped to {!r}", scheduler_name)
        return pipeline

    @classmethod
    def get_info(cls, name: str) -> Dict[str, str]:
        """Return registry entry for a given scheduler name."""
        if name not in _SCHEDULER_REGISTRY:
            raise KeyError(f"Unknown scheduler: {name!r}")
        return _SCHEDULER_REGISTRY[name]
