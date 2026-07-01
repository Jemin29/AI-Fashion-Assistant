"""
Week 6 — ControlNet Service (Complete Rewrite).

Clean UI-to-backend interface for sketch/pose/depth-conditioned generation.

Interface
---------
``ControlNetService`` exposes:

- ``generate_conditioned(...)``  — primary conditioned image generation call
- ``preprocess_image(...)``      — apply conditioning pre-processor to input image
- ``get_modes()``                — supported conditioning mode names
- ``get_mode_info(mode)``        — metadata about a conditioning mode
- ``health_check()``             — lightweight backend status probe

All public methods return a typed ``ServiceResult``.

Validation
----------
- prompt:              non-empty, max 1 000 chars
- control_image:       must not be None
- mode:                one of {canny, sketch, pose, depth, normal, lineart}
- conditioning_scale:  float in [0.0, 2.0]
- steps:               int in [1, 100]
- guidance_scale:      float in [1.0, 30.0]

Error codes
-----------
``PROMPT_EMPTY``       — blank prompt
``IMAGE_MISSING``      — no control image provided
``INVALID_MODE``       — unsupported conditioning mode
``INVALID_PARAM``      — numeric parameter out of range
``BACKEND_ERROR``      — real engine raised an exception
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFilter, ImageOps

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError
from week6.services.generation_service import _make_mock_image
from week6.services.history_manager import HistoryManager

logger = get_logger(__name__)

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sketches"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Conditioning modes ─────────────────────────────────────────────────────────
CONDITIONING_MODES: Dict[str, Dict[str, str]] = {
    "canny": {
        "name":        "Canny Edge",
        "description": "Edge detection — best for sharp architectural / garment outlines.",
        "icon":        "🔲",
    },
    "sketch": {
        "name":        "Sketch / Scribble",
        "description": "Hand-drawn sketch input — ideal for designer rough sketches.",
        "icon":        "✏️",
    },
    "pose": {
        "name":        "OpenPose",
        "description": "Human skeleton / pose estimation — controls body position.",
        "icon":        "🦴",
    },
    "depth": {
        "name":        "Depth Map",
        "description": "Spatial depth conditioning — preserves 3-D composition.",
        "icon":        "🏔️",
    },
    "normal": {
        "name":        "Normal Map",
        "description": "Surface normal vectors — fine texture / material control.",
        "icon":        "🎯",
    },
    "lineart": {
        "name":        "Line Art",
        "description": "Clean line-art extraction — optimal for fashion illustration.",
        "icon":        "🖊️",
    },
}


class ControlNetService(BaseService):
    """
    Mock-safe UI adapter over ``FashionControlNetEngine``.

    Wraps the Week 4 ControlNet backend with input validation, structured
    logging, typed results, and graceful mock fallback.

    Args:
        mock_mode: If ``True`` (default), uses mock image generation.
                   If ``False``, attempts to load the real ControlNet engine.
    """

    _SERVICE_NAME = "ControlNetService"

    def __init__(self, mock_mode: bool = True) -> None:
        super().__init__(mock_mode)
        self._engine: Optional[Any] = None
        self.history_manager = HistoryManager()

        if not mock_mode:
            try:
                from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine
                self._engine = FashionControlNetEngine(mock=False)
                self.mock_mode = False
                logger.info("ControlNetService: real engine loaded")
            except Exception as exc:
                logger.warning("ControlNetService: real engine unavailable — %s", exc)
                self.mock_mode = True
        else:
            # Try to load mock engine for better preprocessing
            try:
                from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine
                self._engine = FashionControlNetEngine(mock=True)
                logger.info("ControlNetService: mock engine loaded")
            except Exception:
                self._engine = None
                logger.info("ControlNetService: pure mock mode (no engine)")

    # ── Primary API ────────────────────────────────────────────────────────────

    def generate_conditioned(
        self,
        prompt: str,
        control_image: Optional[Image.Image],
        *,
        mode: str = "canny",
        conditioning_scale: float = 0.7,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """Generate a fashion image conditioned on a control image.

        Args:
            prompt:              Text description of the desired output.
            control_image:       Input conditioning image (sketch, pose, etc.).
            mode:                Conditioning type.  One of the keys in
                                 ``CONDITIONING_MODES`` (default ``"canny"``).
            conditioning_scale:  ControlNet influence strength [0.0, 2.0].
            num_inference_steps: Denoising steps [1, 100].
            guidance_scale:      CFG scale [1.0, 30.0].
            negative_prompt:     Optional negative conditioning text.
            seed:                Optional random seed.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``image``, ``prompt``, ``mode``, ``saved_path``, ``model``.
        """
        self._call_count += 1
        warnings: List[str] = []

        # ── Validation ──────────────────────────────────────────────────────
        try:
            prompt = self._validate_str(prompt, "prompt", min_len=1, max_len=1000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("prompt", e.reason)

        if control_image is None:
            self._error_count += 1
            return ServiceResult.validation_fail(
                "control_image", "a control image is required for conditioned generation"
            )

        mode = mode.lower().strip()
        if mode not in CONDITIONING_MODES:
            warnings.append(
                f"Unknown mode '{mode}' — defaulting to 'canny'. "
                f"Valid modes: {list(CONDITIONING_MODES.keys())}"
            )
            mode = "canny"

        try:
            conditioning_scale = self._validate_range(
                conditioning_scale, "conditioning_scale", lo=0.0, hi=2.0
            )
        except ValidationError as e:
            warnings.append(f"conditioning_scale clamped: {e.reason}")
            conditioning_scale = max(0.0, min(2.0, float(conditioning_scale)))

        try:
            num_inference_steps = int(
                self._validate_range(num_inference_steps, "steps", lo=1, hi=100)
            )
        except ValidationError as e:
            warnings.append(f"Steps clamped: {e.reason}")
            num_inference_steps = max(1, min(100, int(num_inference_steps)))

        try:
            guidance_scale = self._validate_range(guidance_scale, "guidance_scale", lo=1.0, hi=30.0)
        except ValidationError as e:
            warnings.append(f"CFG clamped: {e.reason}")
            guidance_scale = max(1.0, min(30.0, float(guidance_scale)))

        # ── Generation ──────────────────────────────────────────────────────
        mode_info = CONDITIONING_MODES[mode]
        label = f"ControlNet [{mode_info['name']}]"

        with self._timer() as t:
            try:
                if self.mock_mode or self._engine is None:
                    img        = _make_mock_image(prompt, 512, 512, label)
                    model_name = f"ControlNet {mode_info['name']} (Mock)"
                    run_mode   = "mock"
                else:
                    raw = self._engine.generate(
                        prompt=prompt,
                        control_image=control_image,
                        mode=mode,
                        conditioning_scale=conditioning_scale,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                        seed=seed,
                    )
                    img        = raw.get("image")
                    model_name = f"ControlNet {mode_info['name']}"
                    run_mode   = "real"
            except Exception as exc:
                self._error_count += 1
                logger.error("ControlNetService.generate_conditioned error: %s", exc, exc_info=True)
                img        = _make_mock_image(f"[Error] {prompt}", 512, 512, f"{label} Failed")
                model_name = f"ControlNet {mode_info['name']} (Error)"
                run_mode   = "error"
                warnings.append(f"Backend error — showing mock fallback. Detail: {exc}")

        latency = t.elapsed_ms
        logger.info(
            "ControlNetService.generate_conditioned | mode=%s | scale=%.2f | latency=%.0fms",
            mode, conditioning_scale, latency,
        )

        # ── Save output ─────────────────────────────────────────────────────
        saved_path = ""
        if img is not None:
            try:
                fname = _OUTPUTS_DIR / f"cn_{mode}_{int(time.time())}.png"
                img.save(str(fname))
                saved_path = str(fname)
                meta_dict = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "mode": mode,
                    "conditioning_scale": conditioning_scale,
                    "steps": num_inference_steps,
                    "cfg": guidance_scale,
                    "seed": seed,
                    "model": model_name,
                    "run_mode": run_mode,
                    "saved_path": saved_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(fname.with_suffix(".json"), "w", encoding="utf-8") as jf:
                    json.dump(meta_dict, jf, indent=2)
                
                # Persist to central history index
                try:
                    self.history_manager.record(
                        prompt=prompt,
                        image_path=saved_path,
                        negative_prompt=negative_prompt,
                        model=model_name,
                        style=mode_info.get("name", mode),
                        mode=run_mode,
                        service="controlnet",
                        conditioning_mode=mode,
                        resolution="512×512", # controlnet output size is 512x512
                        seed=seed,
                        steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        latency_ms=latency,
                    )
                except Exception as hist_exc:
                    logger.warning("ControlNetService: failed to record in history manager — %s", hist_exc)
            except Exception as save_exc:
                logger.warning("ControlNetService: save failed — %s", save_exc)
                warnings.append(f"Image generated but not saved: {save_exc}")

        result_data = {
            "image":      img,
            "prompt":     prompt,
            "mode":       mode,
            "saved_path": saved_path,
            "model":      model_name,
        }
        meta = {
            "mode":               run_mode,
            "conditioning_mode":  mode,
            "conditioning_scale": conditioning_scale,
            "model":              model_name,
            "steps":              num_inference_steps,
            "cfg":                guidance_scale,
            "latency_ms":         round(latency, 1),
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        status = ServiceStatus.MOCK if run_mode == "mock" else (
            ServiceStatus.ERROR if run_mode == "error" else ServiceStatus.OK
        )
        return ServiceResult(
            data=result_data, meta=meta, status=status, warnings=warnings, latency_ms=latency,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def preprocess_image(
        self,
        image: Image.Image,
        mode: str = "canny",
    ) -> ServiceResult[Image.Image]:
        """Apply a conditioning pre-processor to an input image.

        The pre-processor converts the raw input into the conditioning signal
        that ControlNet expects (edge map, depth map, etc.).

        Args:
            image: Input PIL image.
            mode:  Conditioning mode determining which pre-processor to apply.

        Returns:
            ``ServiceResult`` whose ``.data`` is the pre-processed PIL image.
        """
        self._call_count += 1

        if image is None:
            return ServiceResult.validation_fail("image", "image must not be None")

        mode = mode.lower().strip()
        if mode not in CONDITIONING_MODES:
            mode = "canny"

        with self._timer() as t:
            try:
                # Try real engine first
                if self._engine and hasattr(self._engine, "preprocess"):
                    processed = self._engine.preprocess(image, mode=mode)
                else:
                    # Built-in PIL fallback pre-processors
                    processed = self._pil_preprocess(image, mode)
            except Exception as exc:
                logger.warning("ControlNetService.preprocess_image error: %s", exc)
                processed = self._pil_preprocess(image, mode)

        logger.debug(
            "ControlNetService.preprocess_image | mode=%s | latency=%.0fms", mode, t.elapsed_ms
        )
        return ServiceResult.ok(
            processed,
            meta={"mode": mode, "latency_ms": round(t.elapsed_ms, 1)},
            latency_ms=t.elapsed_ms,
        )

    def _pil_preprocess(self, image: Image.Image, mode: str) -> Image.Image:
        """Pure-PIL fallback preprocessing (no model required)."""
        if mode in ("canny", "lineart"):
            # Simple edge-like filter: convert to grayscale + find edges
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return ImageOps.invert(edges).convert("RGB")
        elif mode == "depth":
            # Grayscale as stand-in depth map
            return ImageOps.grayscale(image).convert("RGB")
        elif mode == "sketch":
            # High-contrast B&W
            gray = image.convert("L")
            return ImageOps.autocontrast(gray).convert("RGB")
        elif mode in ("pose", "normal"):
            # Fallback: return grayscale
            return image.convert("L").convert("RGB")
        return image.convert("RGB")

    # ── Metadata helpers ───────────────────────────────────────────────────────

    @staticmethod
    def get_modes() -> List[str]:
        """Return the list of supported conditioning mode keys."""
        return list(CONDITIONING_MODES.keys())

    @staticmethod
    def get_mode_info(mode: str) -> Optional[Dict[str, str]]:
        """Return metadata dict for a conditioning mode, or ``None``."""
        return CONDITIONING_MODES.get(mode.lower().strip())

    @staticmethod
    def get_mode_display_options() -> List[str]:
        """Return human-readable mode labels for Gradio dropdowns."""
        return [f"{v['icon']} {v['name']}" for v in CONDITIONING_MODES.values()]

    # ── Health ─────────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Return lightweight health status of the ControlNet backend."""
        if self.mock_mode or self._engine is None:
            return {
                "status":    "ok",
                "mock_mode": True,
                "backend":   "mock",
                "message":   "ControlNetService running in mock mode.",
                "modes":     list(CONDITIONING_MODES.keys()),
            }
        try:
            _ = self._engine
            return {
                "status":    "ok",
                "mock_mode": False,
                "backend":   "controlnet",
                "message":   "ControlNet engine loaded and ready.",
                "modes":     list(CONDITIONING_MODES.keys()),
            }
        except Exception as exc:
            return {
                "status":    "error",
                "mock_mode": self.mock_mode,
                "backend":   "controlnet",
                "message":   f"Engine probe failed: {exc}",
                "modes":     list(CONDITIONING_MODES.keys()),
            }
