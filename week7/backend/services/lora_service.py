from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Dict, Optional

from src.lora.inference.lora_inference import LoraInferenceSystem
from src.lora.style_manager.lora_registry import LoraRegistry
from src.lora.style_manager.style_mixer import StyleMixer
from week7.backend.configs.config import get_settings
from week6.services.base import ServiceResult, ServiceStatus


class LoraService:
    """Business logic for LoRA adapter generation and style mixing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._lora_sys = None

    def _get_lora_sys(self) -> LoraInferenceSystem:
        if self._lora_sys is None:
            self._lora_sys = LoraInferenceSystem(
                device="cpu" if self.settings.model.global_mock else "cuda",
                dry_run=self.settings.model.global_mock
            )
            # Pre-populate registry with dummy adapter weights if empty
            if not self._lora_sys.registry.models:
                import tempfile
                from pathlib import Path
                temp_dir = Path(tempfile.gettempdir()) / "mock_loras"
                temp_dir.mkdir(parents=True, exist_ok=True)
                for brand in ["nike", "gucci", "zara", "h&m"]:
                    mock_file = temp_dir / f"{brand}_mock.safetensors"
                    if not mock_file.exists():
                        mock_file.write_bytes(b"mock safetensors content")
                    self._lora_sys.registry.register_model(
                        brand=brand,
                        model_path=mock_file,
                        metadata={"style_description": f"Mock adapter for {brand}"}
                    )
        return self._lora_sys

    def _serialize_to_base64(self, filepath: str) -> str:
        from week7.backend.api.dependencies import get_watermark_service
        from PIL import Image
        with Image.open(filepath) as img:
            img = get_watermark_service().apply_watermark(img)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _normalize_brand_key(self, brand: str) -> str:
        bk = brand.lower().strip()
        if bk in ("hm", "h and m"):
            return "h&m"
        return bk

    def generate_lora_style(
        self,
        prompt: str,
        brand: str,
        lora_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        brand_key = self._normalize_brand_key(brand)
        if brand_key not in LoraRegistry.SUPPORTED_BRANDS:
            raise ValueError(f"Brand '{brand}' is not supported. Supported: {list(LoraRegistry.SUPPORTED_BRANDS)}")

        lora_sys = self._get_lora_sys()
        if lora_sys.pipeline is None:
            lora_sys.load_pipeline()
            
        res = lora_sys.generate(
            prompt=prompt,
            brand=brand_key,
            scale=lora_scale,
            seed=seed
        )
        
        if not res.get("success", False):
            raise RuntimeError("LoRA generation failed.")
            
        img_b64 = self._serialize_to_base64(res["image_path"])
        
        with open(res["metadata_path"], "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            
        return {
            "image": img_b64,
            "metadata": meta,
            "saved_path": res["image_path"]
        }

    def generate_with_brand(
        self,
        prompt: str,
        brand: str,
        lora_scale: float = 1.0,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None
    ) -> ServiceResult[Dict[str, Any]]:
        raw_res = self.generate_lora_style(
            prompt=prompt,
            brand=brand,
            lora_scale=lora_scale,
            seed=seed
        )
        return ServiceResult(data=raw_res, status=ServiceStatus.OK, meta=raw_res["metadata"])

    def mix_styles(
        self,
        prompt: str,
        brand_weights: Dict[str, float],
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if not brand_weights:
            raise ValueError("Brand weights dictionary cannot be empty.")

        clean_weights = {}
        for brand, weight in brand_weights.items():
            bk = self._normalize_brand_key(brand)
            if bk not in LoraRegistry.SUPPORTED_BRANDS:
                raise ValueError(f"Brand '{brand}' is not supported. Supported: {list(LoraRegistry.SUPPORTED_BRANDS)}")
            if weight < 0.0:
                raise ValueError(f"Blend weight for '{brand}' cannot be negative ({weight}).")
            clean_weights[bk] = weight

        lora_sys = self._get_lora_sys()
        if lora_sys.pipeline is None:
            lora_sys.load_pipeline()
            
        mixer = StyleMixer(
            registry=lora_sys.registry,
            inference_pipeline=lora_sys.pipeline,
            output_dir=lora_sys.output_dir
        )
        
        res = mixer.generate_mixed_design(
            prompt=prompt,
            brand_weights=clean_weights,
            dry_run=lora_sys.dry_run
        )
        
        if not res.get("success", False):
            raise RuntimeError("Style mixing generation failed.")
            
        img_b64 = self._serialize_to_base64(res["image_path"])
        
        meta = {
            "prompt": prompt,
            "enriched_prompt": res["prompt"],
            "mixed_weights": res["weights"],
            "dry_run": res["dry_run"],
            "timestamp": int(time.time()),
            "resolution": "512x512" if res["dry_run"] else "1024x1024"
        }
        
        return {
            "image": img_b64,
            "metadata": meta,
            "saved_path": res["image_path"]
        }

    def mix_styles_for_router(
        self,
        prompt: str,
        brand_weights: Dict[str, float],
        seed: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5
    ) -> ServiceResult[Dict[str, Any]]:
        raw_res = self.mix_styles(prompt, brand_weights, seed=seed)
        return ServiceResult(data=raw_res, status=ServiceStatus.OK, meta=raw_res["metadata"])

    def health_check(self) -> Dict[str, Any]:
        """Verify the health status of the lora service."""
        return {
            "status": "ok" if self._get_lora_sys() is not None else "error",
            "name": "LoraService",
            "mode": "mock" if self.settings.model.global_mock else "production"
        }
