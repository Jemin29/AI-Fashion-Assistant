from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine
from week7.backend.configs.config import get_settings
from week6.services.base import ServiceResult, ServiceStatus


class ControlNetService:
    """Business logic for ControlNet-conditioned image generation (Sketch, Pose, Depth)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._engine = None

    def _get_engine(self) -> FashionControlNetEngine:
        if self._engine is None:
            self._engine = FashionControlNetEngine(mock=self.settings.model.global_mock)
        return self._engine

    def _serialize_to_base64(self, img: Image.Image) -> str:
        # Apply transparent watermark
        from week7.backend.api.dependencies import get_watermark_service
        img = get_watermark_service().apply_watermark(img)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def preprocess_image(self, image: Image.Image, mode: str) -> ServiceResult[Image.Image]:
        # Just return the image itself as the mock preprocessed guide
        return ServiceResult(data=image, status=ServiceStatus.OK, meta={"mode": mode})

    def generate_conditioned(
        self,
        prompt: str,
        control_image: Image.Image,
        mode: str,
        conditioning_scale: float = 0.7,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None
    ) -> ServiceResult[Dict[str, Any]]:
        s = seed if seed is not None else -1
        if mode == "sketch" or mode == "canny":
            raw_res = self.generate_from_sketch(
                prompt=prompt,
                sketch_image=control_image,
                negative_prompt=negative_prompt,
                control_strength=conditioning_scale,
                seed=s
            )
        elif mode == "pose":
            raw_res = self.generate_from_pose(
                prompt=prompt,
                pose_image=control_image,
                negative_prompt=negative_prompt,
                control_strength=conditioning_scale,
                seed=s
            )
        elif mode == "depth":
            raw_res = self.generate_from_depth(
                prompt=prompt,
                depth_image=control_image,
                negative_prompt=negative_prompt,
                control_strength=conditioning_scale,
                seed=s
            )
        else:
            raise ValueError(f"Unsupported conditioning mode: {mode}")

        # Save output image to file to populate image_path
        out_dir = Path(self.settings.database.chroma_db_dir).parent / "outputs" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(out_dir / f"controlnet_{mode}_{uuid.uuid4().hex}.png")
        
        img_data = base64.b64decode(raw_res["image"])
        with open(image_path, "wb") as f:
            f.write(img_data)

        data = {
            "image": raw_res["image"],
            "image_path": image_path,
            "metadata": raw_res["metadata"]
        }
        return ServiceResult(data=data, status=ServiceStatus.OK, meta=raw_res["metadata"])

    def generate_from_sketch(
        self,
        prompt: str,
        sketch_image: Image.Image,
        negative_prompt: str = "",
        control_strength: float = 0.7,
        seed: int = -1
    ) -> Dict[str, Any]:
        engine = self._get_engine()
        res = engine.generate_from_sketch(
            prompt=prompt,
            sketch_image=sketch_image,
            negative_prompt=negative_prompt,
            conditioning_scale=control_strength,
            seed=seed
        )
        if not res.success:
            raise RuntimeError(res.error or "Sketch generation failed.")
        return {
            "image": self._serialize_to_base64(res.images[0]),
            "metadata": res.metadata
        }

    def generate_from_pose(
        self,
        prompt: str,
        pose_image: Image.Image,
        negative_prompt: str = "",
        control_strength: float = 0.7,
        seed: int = -1
    ) -> Dict[str, Any]:
        engine = self._get_engine()
        res = engine.generate_from_pose(
            prompt=prompt,
            pose_image=pose_image,
            negative_prompt=negative_prompt,
            conditioning_scale=control_strength,
            seed=seed
        )
        if not res.success:
            raise RuntimeError(res.error or "Pose generation failed.")
        return {
            "image": self._serialize_to_base64(res.images[0]),
            "metadata": res.metadata
        }

    def generate_from_depth(
        self,
        prompt: str,
        depth_image: Image.Image,
        negative_prompt: str = "",
        control_strength: float = 0.7,
        seed: int = -1
    ) -> Dict[str, Any]:
        engine = self._get_engine()
        res = engine.generate_from_depth(
            prompt=prompt,
            depth_image=depth_image,
            negative_prompt=negative_prompt,
            conditioning_scale=control_strength,
            seed=seed
        )
        if not res.success:
            raise RuntimeError(res.error or "Depth generation failed.")
        return {
            "image": self._serialize_to_base64(res.images[0]),
            "metadata": res.metadata
        }

    def health_check(self) -> Dict[str, Any]:
        """Verify the health status of the controlnet service."""
        return {
            "status": "ok" if self._get_engine() is not None else "error",
            "name": "ControlNetService",
            "mode": "mock" if self.settings.model.global_mock else "production"
        }
