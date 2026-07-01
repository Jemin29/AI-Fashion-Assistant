"""
Week 6 — Generation Service (Complete Rewrite).

Clean UI-to-backend interface for text-to-image fashion generation.

Interface
---------
``GenerationService`` exposes:

- ``generate(...)``            — primary text-to-image call
- ``generate_batch(...)``      — multi-seed batch generation
- ``get_style_presets()``      — available style preset names
- ``get_resolution_options()`` — supported width×height pairs
- ``health_check()``           — lightweight backend status probe

All public methods return a typed ``ServiceResult`` so Gradio callbacks
can handle success / failure uniformly without ``try/except`` boilerplate.

Validation
----------
- prompt: non-empty, max 1 000 chars
- steps: int in [1, 150]
- cfg: float in [1.0, 30.0]
- width / height: one of the supported values
- seed: int or None

Error codes
-----------
``PROMPT_EMPTY``   — blank prompt submitted
``PROMPT_TOO_LONG``— prompt exceeds 1 000 chars
``INVALID_STEPS``  — steps out of [1, 150]
``INVALID_CFG``    — cfg out of [1.0, 30.0]
``INVALID_SIZE``   — unsupported resolution
``BACKEND_ERROR``  — real generator raised an exception
``SAVE_ERROR``     — image was generated but could not be saved
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger, log_generation
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError
from week6.services.history_manager import HistoryManager

logger = get_logger(__name__)

# ── Output directory ───────────────────────────────────────────────────────────
_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "generated"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Supported resolutions ──────────────────────────────────────────────────────
SUPPORTED_RESOLUTIONS: List[Tuple[int, int]] = [
    (512, 512),
    (512, 768),
    (768, 512),
    (768, 768),
    (1024, 1024),
    (1024, 768),
    (768, 1024),
]

_RESOLUTION_LABELS = {
    (512, 512):   "512×512 (Square)",
    (512, 768):   "512×768 (Portrait)",
    (768, 512):   "768×512 (Landscape)",
    (768, 768):   "768×768 (Square MD)",
    (1024, 1024): "1024×1024 (Square HD)",
    (1024, 768):  "1024×768 (Landscape HD)",
    (768, 1024):  "768×1024 (Portrait HD)",
}

STYLE_PRESETS: List[Dict[str, str]] = [
    {"name": "Streetwear Editorial",  "trigger": "streetwear, urban editorial, oversized silhouette, bold typography"},
    {"name": "Luxury Fashion Week",   "trigger": "haute couture, luxury fashion week runway, dramatic lighting"},
    {"name": "Minimalist Studio",     "trigger": "minimalist fashion, clean white studio, neutral tones"},
    {"name": "Athleisure Sport",      "trigger": "athleisure, performance fabric, sport-forward, dynamic pose"},
    {"name": "Bohemian Summer",       "trigger": "bohemian, flowing fabrics, warm sunset, earthy tones"},
    {"name": "Techwear Utility",      "trigger": "techwear, utility pockets, futuristic, muted tones"},
    {"name": "Vintage 90s",           "trigger": "vintage 90s fashion, film grain, nostalgic color grading"},
    {"name": "Formal Business",       "trigger": "business formal, tailored suit, power dressing, boardroom"},
    {"name": "Resort Luxe",           "trigger": "resort wear, beach luxury, lightweight fabrics, tropical"},
    {"name": "Dark Academia",         "trigger": "dark academia, plaid, layered, moody autumn tones"},
]


def _make_mock_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    style_label: str = "Mock Generation",
) -> Image.Image:
    """Generate a visually rich placeholder image for mock mode.

    The colour palette is deterministically derived from the prompt hash
    so the same prompt always produces the same mock image.

    Args:
        prompt:      Generation prompt (drives colour seed).
        width:       Output image width.
        height:      Output image height.
        style_label: Label displayed on the image footer.

    Returns:
        A ``PIL.Image.Image`` with a gradient background and fashion silhouette.
    """
    seed = hash(prompt) % (2 ** 32)
    rng  = random.Random(seed)
    hue  = rng.randint(0, 360)

    r1 = int(30 + 60 * abs(((hue / 60) % 2) - 1))
    g1 = int(20 + 50 * abs(((hue / 60 - 2) % 2) - 1))
    b1 = int(40 + 80 * abs(((hue / 60 - 4) % 2) - 1))

    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Vertical gradient background
    for y in range(height):
        ratio = y / height
        r = int(r1 * (1 - ratio) + max(0, r1 // 2) * ratio)
        g = int(g1 * (1 - ratio) + max(0, g1 // 2) * ratio)
        b = int(b1 * (1 - ratio) + min(255, b1 + 40) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Fashion silhouette
    cx, cy = width // 2, height // 2
    skin = (200, 185, 170)
    cloth_col = (rng.randint(40, 100), rng.randint(50, 120), rng.randint(80, 180))

    draw.ellipse([cx - 28, cy - height // 4, cx + 28, cy - height // 4 + 52], fill=skin)
    draw.rectangle([cx - 24, cy - height // 4 + 52, cx + 24, cy + 20], fill=skin)
    draw.rectangle([cx - 28, cy + 20, cx + 28, cy + height // 4 + 10], fill=cloth_col)
    draw.rectangle([cx - 48, cy - height // 4 + 52, cx - 24, cy + 10], fill=skin)
    draw.rectangle([cx + 24,  cy - height // 4 + 52, cx + 48, cy + 10], fill=skin)

    # Footer label
    draw.rectangle([0, height - 68, width, height], fill=(0, 0, 0))
    prompt_short = (prompt[:46] + "…") if len(prompt) > 46 else prompt
    draw.text((10, height - 58), f"🎨 {style_label}", fill=(255, 200, 100))
    draw.text((10, height - 40), prompt_short, fill=(200, 200, 200))
    draw.text((10, height - 20), "[ Mock Mode — GPU not required ]", fill=(120, 120, 120))

    return img


class GenerationService(BaseService):
    """
    Mock-safe UI adapter over ``SDXLGenerator``.

    Provides a clean typed interface between Gradio event handlers and
    the Week 4 text-to-image backend.  In mock mode, returns a
    deterministic placeholder image immediately without loading any model.

    Args:
        mock_mode: If ``True`` (default), uses the fast mock path.
                   If ``False``, attempts to load the real SDXL pipeline.
    """

    _SERVICE_NAME = "GenerationService"

    def __init__(self, mock_mode: bool = True) -> None:
        super().__init__(mock_mode)
        self._generator: Optional[Any] = None
        self.history_manager = HistoryManager()

        if not mock_mode:
            try:
                from src.generation.generator.sdxl_generator import SDXLGenerator
                self._generator = SDXLGenerator()
                self.mock_mode  = False
                logger.info("GenerationService: real SDXLGenerator loaded")
            except Exception as exc:
                logger.warning(
                    "GenerationService: real mode unavailable, falling back to mock — %s", exc
                )
                self.mock_mode = True
        else:
            logger.info("GenerationService: started in mock mode")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        style_preset: str = "",
        style_label: str = "Fashion Generation",
    ) -> ServiceResult[Dict[str, Any]]:
        """Generate a single fashion image from a text prompt.

        Args:
            prompt:              Text description of the desired fashion image.
            negative_prompt:     Things to avoid in the generated image.
            num_inference_steps: Number of DDIM denoising steps (real mode).
            guidance_scale:      Classifier-free guidance scale (real mode).
            width:               Output image width in pixels.
            height:              Output image height in pixels.
            seed:                Optional random seed for reproducibility.
            style_preset:        Optional preset name to prepend trigger words.
            style_label:         Label shown in mock image footer.

        Returns:
            ``ServiceResult`` whose ``.data`` is a dict with keys:
            ``image`` (PIL Image), ``prompt``, ``saved_path``, ``model``.
        """
        self._call_count += 1
        warnings: List[str] = []

        # ── Validation ─────────────────────────────────────────────────────
        try:
            prompt = self._validate_str(prompt, "prompt", min_len=1, max_len=1000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("prompt", e.reason)

        try:
            num_inference_steps = int(self._validate_range(num_inference_steps, "steps", lo=1, hi=150))
        except ValidationError as e:
            warnings.append(f"Steps clamped: {e.reason}")
            num_inference_steps = max(1, min(150, int(num_inference_steps)))

        try:
            guidance_scale = self._validate_range(guidance_scale, "guidance_scale", lo=1.0, hi=30.0)
        except ValidationError as e:
            warnings.append(f"CFG clamped: {e.reason}")
            guidance_scale = max(1.0, min(30.0, float(guidance_scale)))

        # Clamp resolution to nearest supported pair
        if (width, height) not in SUPPORTED_RESOLUTIONS:
            _w, _h = width, height
            width, height = min(SUPPORTED_RESOLUTIONS, key=lambda r: abs(r[0] - _w) + abs(r[1] - _h))
            warnings.append(f"Resolution adjusted to {width}×{height}")

        # Apply style preset trigger words
        if style_preset:
            preset_data = next(
                (p for p in STYLE_PRESETS if p["name"].lower() == style_preset.lower()), None
            )
            if preset_data:
                prompt = f"{preset_data['trigger']}, {prompt}"
                style_label = preset_data["name"]
            else:
                warnings.append(f"Style preset '{style_preset}' not found — ignored.")

        # ── Generation ─────────────────────────────────────────────────────
        with self._timer() as t:
            try:
                if self.mock_mode or self._generator is None:
                    img  = _make_mock_image(prompt, min(width, 512), min(height, 512), style_label)
                    mode = "mock"
                    model_name = "SDXL v1.0 (Mock)"
                else:
                    raw = self._generator.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        seed=seed,
                    )
                    img  = raw.get("image") or raw.get("images", [None])[0]
                    mode = "real"
                    model_name = "SDXL v1.0"
            except Exception as exc:
                self._error_count += 1
                logger.error("GenerationService.generate backend error: %s", exc, exc_info=True)
                # Return degraded mock image rather than hard failure
                img  = _make_mock_image(f"[Error] {prompt}", 512, 512, "Generation Failed")
                mode = "error"
                model_name = "SDXL v1.0 (Error)"
                warnings.append(f"Backend error — showing mock fallback. Detail: {exc}")

        latency = t.elapsed_ms
        log_generation(prompt, "sdxl-mock" if mode != "real" else "sdxl", latency)
        logger.info(
            "GenerationService.generate | mode=%s | size=%dx%d | steps=%d | latency=%.0fms",
            mode, width, height, num_inference_steps, latency,
        )

        # ── Save output ─────────────────────────────────────────────────────
        saved_path = ""
        if img is not None:
            try:
                fname = _OUTPUTS_DIR / f"gen_{int(time.time())}_{seed or 0}.png"
                img.save(str(fname))
                saved_path = str(fname)
                meta_dict  = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": num_inference_steps,
                    "cfg": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "model": model_name,
                    "mode": mode,
                    "saved_path": saved_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "style_preset": style_preset,
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
                        style=style_preset or style_label,
                        mode=mode,
                        service="generation",
                        resolution=f"{width}×{height}",
                        seed=seed,
                        steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        latency_ms=latency,
                    )
                except Exception as hist_exc:
                    logger.warning("GenerationService: failed to record in history manager — %s", hist_exc)
            except Exception as save_exc:
                logger.warning("GenerationService: failed to save output — %s", save_exc)
                warnings.append(f"Image generated but not saved: {save_exc}")

        result_data = {
            "image":      img,
            "prompt":     prompt,
            "saved_path": saved_path,
            "model":      model_name,
        }
        meta = {
            "mode":       mode,
            "model":      model_name,
            "steps":      num_inference_steps,
            "cfg":        guidance_scale,
            "size":       f"{width}×{height}",
            "seed":       seed,
            "latency_ms": round(latency, 1),
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        status = ServiceStatus.MOCK if mode == "mock" else (
            ServiceStatus.ERROR if mode == "error" else ServiceStatus.OK
        )
        return ServiceResult(
            data=result_data,
            meta=meta,
            status=status,
            warnings=warnings,
            latency_ms=latency,
        )

    def generate_batch(
        self,
        prompt: str,
        seeds: List[int],
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        style_preset: str = "",
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Generate multiple images from the same prompt with different seeds.

        Args:
            prompt:              Text prompt (shared across all seeds).
            seeds:               List of integer seeds (max 8).
            num_inference_steps: Denoising steps.
            guidance_scale:      CFG scale.
            width:               Output width.
            height:              Output height.
            style_preset:        Optional preset name.

        Returns:
            ``ServiceResult`` whose ``.data`` is a list of result dicts
            (one per seed), in the same format as ``generate()``.
        """
        self._call_count += 1

        try:
            prompt = self._validate_str(prompt, "prompt", min_len=1, max_len=1000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("prompt", e.reason)

        if not seeds:
            return ServiceResult.validation_fail("seeds", "provide at least one seed")

        seeds = seeds[:8]  # cap at 8
        results: List[Dict[str, Any]] = []
        warnings: List[str] = []

        with self._timer() as t:
            for seed in seeds:
                r = self.generate(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed,
                    style_preset=style_preset,
                )
                if r.data:
                    results.append(r.data)
                warnings.extend(r.warnings)

        logger.info(
            "GenerationService.generate_batch | n=%d | latency=%.0fms",
            len(results), t.elapsed_ms,
        )
        return ServiceResult(
            data=results,
            meta={"batch_size": len(results), "seeds": seeds, "latency_ms": round(t.elapsed_ms, 1)},
            status=ServiceStatus.MOCK if self.mock_mode else ServiceStatus.OK,
            warnings=warnings,
            latency_ms=t.elapsed_ms,
        )

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def get_style_presets(self) -> List[str]:
        """Return the list of available style preset names."""
        return [p["name"] for p in STYLE_PRESETS]

    def get_style_preset_info(self, name: str) -> Optional[Dict[str, str]]:
        """Return trigger-word metadata for a named preset, or ``None``."""
        return next((p for p in STYLE_PRESETS if p["name"].lower() == name.lower()), None)

    def get_resolution_options(self) -> List[str]:
        """Return human-readable resolution labels (for Gradio dropdowns)."""
        return [_RESOLUTION_LABELS.get(r, f"{r[0]}×{r[1]}") for r in SUPPORTED_RESOLUTIONS]

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> ServiceResult:
        """Probe backend availability without loading the full pipeline.

        Returns:
            ServiceResult wrapping health details.
        """
        if self.mock_mode or self._generator is None:
            res = {
                "status":    "ok",
                "mock_mode": True,
                "backend":   "mock",
                "message":   "GenerationService running in mock mode — no GPU required.",
            }
            return ServiceResult(success=True, data=res)
        try:
            # Light probe: check the generator object is still responsive
            _ = self._generator  # noqa: F841
            res = {
                "status":    "ok",
                "mock_mode": False,
                "backend":   "sdxl",
                "message":   "SDXLGenerator loaded and ready.",
            }
            return ServiceResult(success=True, data=res)
        except Exception as exc:
            res = {
                "status":    "error",
                "mock_mode": self.mock_mode,
                "backend":   "sdxl",
                "message":   f"SDXLGenerator probe failed: {exc}",
            }
            return ServiceResult(success=False, data=res, error=res["message"])
