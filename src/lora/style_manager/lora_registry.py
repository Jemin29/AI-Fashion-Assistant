"""
week4/style_manager/lora_registry.py
====================================
LoRA Style Adapter Registry.
Manages the registration, load state, and active toggle states for brand LoRA adapters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class LoraRegistry:
    """
    Manages metadata, load statuses, and blending scales for fine-tuned brand style adapters.
    Supports Nike, Gucci, Zara, and H&M styles.
    """

    SUPPORTED_BRANDS = {"nike", "gucci", "zara", "h&m"}

    def __init__(
        self,
        config: Any = None,
        registry_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize LoraRegistry.

        Parameters
        ----------
        config : Week4Config, optional
        registry_path : Path or str, optional
            Path to the JSON database registry file (default: outputs/style_manager/lora_registry.json).
        """
        self.config = config
        
        # Resolve registry path
        if registry_path:
            self.registry_path = Path(registry_path).resolve()
        elif config and getattr(config, "output_root", None):
            self.registry_path = Path(config.output_root).resolve() / "style_manager" / "lora_registry.json"
        else:
            self.registry_path = Path("outputs/style_manager/lora_registry.json").resolve()

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
        logger.info(f"Initialized LoraRegistry | db_path={self.registry_path} | registered_count={len(self.models)}")

    def register_model(
        self,
        brand: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new style LoRA adapter model path.

        Parameters
        ----------
        brand : str
            Target brand (nike, gucci, zara, h&m).
        model_path : Path or str
            Path to the safetensors LoRA weights file.
        metadata : dict, optional
            Extra descriptors (e.g. style_description, rank).

        Returns
        -------
        dict
            The registered model registry record.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.SUPPORTED_BRANDS:
            raise ValueError(f"Brand '{brand}' not supported. Choose from: {self.SUPPORTED_BRANDS}")

        path_resolved = Path(model_path).resolve()
        if not path_resolved.exists():
            raise FileNotFoundError(f"LoRA weights file not found at: {path_resolved}")

        # Compute file parameters
        size_bytes = path_resolved.stat().st_size
        
        # Build entry
        entry = {
            "brand": brand_key,
            "model_path": str(path_resolved.as_posix()),
            "size_bytes": size_bytes,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "loaded": False,
            "active": False,
            "scale": 1.0,
            "metadata": metadata or {}
        }

        self.models[brand_key] = entry
        self._save_registry()
        logger.success(f"Registered brand style LoRA '{brand_key}' path: {path_resolved}")
        return entry

    def load_model(self, brand: str) -> Dict[str, Any]:
        """
        Mark the brand's model adapter as loaded in system memory.

        Parameters
        ----------
        brand : str

        Returns
        -------
        dict
            The updated registry record.
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.models:
            raise KeyError(f"No LoRA model registered for brand '{brand_key}'.")

        entry = self.models[brand_key]
        model_path = Path(entry["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Registered weights file missing on disk: {model_path}")

        entry["loaded"] = True
        self._save_registry()
        logger.info(f"Loaded brand style LoRA weights for '{brand_key}' into memory.")
        return entry

    def activate_model(self, brand: str, scale: float = 1.0) -> None:
        """
        Mark the adapter as active and set its blend scale.

        Parameters
        ----------
        brand : str
        scale : float
            Active blend scale weight (0.0 to 2.0).
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.models:
            raise KeyError(f"No LoRA model registered for brand '{brand_key}'.")

        if not 0.0 <= scale <= 2.0:
            raise ValueError("LoRA blending scale must be between 0.0 and 2.0.")

        entry = self.models[brand_key]
        entry["active"] = True
        entry["scale"] = scale
        self._save_registry()
        logger.success(f"Activated style LoRA '{brand_key}' | blend_scale={scale}")

    def deactivate_model(self, brand: str) -> None:
        """
        Deactivate the adapter.

        Parameters
        ----------
        brand : str
        """
        brand_key = brand.lower().strip()
        if brand_key not in self.models:
            raise KeyError(f"No LoRA model registered for brand '{brand_key}'.")

        entry = self.models[brand_key]
        entry["active"] = False
        entry["scale"] = 1.0
        self._save_registry()
        logger.info(f"Deactivated style LoRA '{brand_key}' adapter.")

    def list_models(self, filter_active: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List registered adapters.

        Parameters
        ----------
        filter_active : bool
            If True, only returns active adapters.

        Returns
        -------
        dict
            Registry dictionary mapping brand keys to model records.
        """
        if filter_active:
            return {b: record for b, record in self.models.items() if record["active"]}
        return self.models

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        """Load registrations database from JSON file."""
        if not self.registry_path.exists():
            self.models = {}
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.models = json.load(f)
        except Exception as err:
            logger.error(f"Failed to read registry database file {self.registry_path}: {err}. Starting empty.")
            self.models = {}

    def _save_registry(self) -> None:
        """Save registrations database to JSON file."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.models, f, indent=2, sort_keys=True)
        except Exception as err:
            logger.error(f"Failed to save registry database file {self.registry_path}: {err}")
