from __future__ import annotations

import time
import base64
import io
import random
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw

from src.generation.generator.sdxl_generator import FashionSDXLGenerator
from week7.backend.configs.config import get_settings
from week6.services.base import ServiceResult


class GenerationService:
    """Business logic for text-to-fashion design generation using SDXL."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._generator = None

    def _get_generator(self) -> FashionSDXLGenerator:
        if self._generator is None:
            self._generator = FashionSDXLGenerator(
                model_id=self.settings.model.sdxl_model_id,
                device="cpu" if self.settings.model.global_mock else "auto",
                global_mock=self.settings.model.global_mock,
            )
        return self._generator

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        cfg: float = 7.5,
        resolution: str = "1024x1024"
    ) -> Dict[str, Any]:
        try:
            res_str = resolution.strip().lower()
            from src.generation.generator.sdxl_generator import SIZE_PRESETS
            
            width = 1024
            height = 1024
            size_preset = None
            
            if res_str in SIZE_PRESETS:
                size_preset = res_str
                width, height = SIZE_PRESETS[res_str]
            elif "x" in res_str:
                parts = res_str.split("x")
                width = int(parts[0].strip())
                height = int(parts[1].strip())
            else:
                raise ValueError(f"Resolution '{resolution}' is not supported.")

            t_start = time.perf_counter()

            if self.settings.model.global_mock:
                # Mock mode generation
                img = self._make_mock_image(prompt, width, height, "SDXL Week 2 (Mock)")
                latency_s = time.perf_counter() - t_start
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                meta = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed if seed != -1 else random.randint(0, 1000000),
                    "width": width,
                    "height": height,
                    "guidance_scale": cfg,
                    "device_used": "cpu",
                    "model_id": self.settings.model.sdxl_model_id,
                    "run_mode": "mock",
                    "generation_time_s": latency_s
                }

                # Apply transparent watermark
                from week7.backend.api.dependencies import get_watermark_service
                watermarked_b64 = get_watermark_service().apply_watermark_to_b64(img_b64)

                if any(w in prompt.lower() for w in ["unsafe", "nsfw", "nudity", "nude"]):
                    watermarked_b64 = "unsafe_mock_content_" + watermarked_b64
                
                return ServiceResult(success=True, data={"image": watermarked_b64, "metadata": meta, "generation_time": latency_s}, metadata=meta)
            else:
                generator = self._get_generator()
                if not generator._is_loaded:
                    generator.load_model()
                
                res = generator.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=cfg,
                    seed=seed,
                    size_preset=size_preset
                )
                
                if not res.success:
                    raise RuntimeError(res.error or "SDXL Generation failed.")
                    
                latency_s = time.perf_counter() - t_start
                
                buffered = io.BytesIO()
                res.image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                meta = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": res.seed,
                    "width": width,
                    "height": height,
                    "guidance_scale": cfg,
                    "device_used": generator.device,
                    "model_id": self.settings.model.sdxl_model_id,
                    "run_mode": "production",
                    "generation_time_s": latency_s
                }
                
                # Apply transparent watermark
                from week7.backend.api.dependencies import get_watermark_service
                watermarked_b64 = get_watermark_service().apply_watermark_to_b64(img_b64)
                
                return ServiceResult(success=True, data={"image": watermarked_b64, "metadata": meta, "generation_time": latency_s}, metadata=meta)
        except Exception as exc:
            from week7.backend.logging_config import log_generation_failure
            log_generation_failure(prompt=prompt, error_msg=str(exc))
            raise exc

    def _make_mock_image(self, prompt: str, width: int, height: int, text: str) -> Image.Image:
        img = Image.new("RGB", (width, height), color=(30, 30, 40))
        d = ImageDraw.Draw(img)
        random.seed(hash(prompt) % 1234567)
        for _ in range(5):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            d.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=color, outline=None)
        d.text((10, 10), text, fill=(255, 255, 255))
        d.text((10, 30), prompt[:50] + "...", fill=(200, 200, 200))
        return img

    def health_check(self) -> ServiceResult:
        """Verify the health status of the generation service."""
        res = {
            "status": "ok",
            "name": "GenerationService",
            "mode": "mock" if self.settings.model.global_mock else "production"
        }
        return ServiceResult(success=True, data=res)
