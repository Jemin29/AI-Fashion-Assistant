"""
Week 6 — LoRA Service (Complete Rewrite).

Clean UI-to-backend interface for brand-styled generation and style mixing.

Interface
---------
``LoRAService`` exposes:

- ``generate_with_brand(...)``  — single-brand LoRA-conditioned generation
- ``mix_styles(...)``           — weighted multi-brand style mixing
- ``get_brands()``              — list of registered brand keys
- ``get_brand_info(brand)``     — rich metadata for a brand
- ``get_brand_display_names()`` — pretty brand labels for UI dropdowns
- ``validate_brand_weights(w)`` — normalise and validate weight dict
- ``health_check()``            — lightweight backend status probe

All public methods return a typed ``ServiceResult``.

Validation
----------
- prompt:        non-empty, max 1 000 chars
- brand:         must be in ``BRAND_REGISTRY``
- lora_scale:    float in [0.0, 1.5]
- steps:         int in [1, 100]
- guidance_scale:float in [1.0, 30.0]
- brand_weights: at least one brand, all weights positive, auto-normalised

Error codes
-----------
``PROMPT_EMPTY``      — blank prompt
``UNKNOWN_BRAND``     — brand key not in registry
``EMPTY_WEIGHTS``     — no brands provided to mix_styles
``INVALID_WEIGHT``    — negative or zero weight value
``INVALID_PARAM``     — numeric parameter out of range
``BACKEND_ERROR``     — real engine raised an exception
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.logger import get_logger
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError
from week6.services.generation_service import _make_mock_image
from week6.services.history_manager import HistoryManager

logger = get_logger(__name__)

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "generated"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand registry ─────────────────────────────────────────────────────────────
BRAND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nike": {
        "name":          "Nike",
        "display":       "Nike — Performance Athletic",
        "description":   "Performance athletic wear — bold, dynamic, sport-forward innovation.",
        "aesthetic":     "High-energy, functional, iconic swoosh branding, performance fabrics.",
        "key_styles":    ["sportswear", "athleisure", "running", "training"],
        "color_palette": "Black, White, Safety Orange, Volt Green",
        "price_range":   "$$",
        "origin":        "USA",
        "trigger_words": "nike_style, athletic, sport-forward, dynamic",
    },
    "gucci": {
        "name":          "Gucci",
        "display":       "Gucci — Italian Luxury",
        "description":   "Opulent Italian luxury — maximalist, heritage-driven, high-craftsmanship.",
        "aesthetic":     "Floral motifs, interlocking GG, rich greens and reds, silk and leather.",
        "key_styles":    ["luxury", "maximalist", "formal", "resort"],
        "color_palette": "Forest Green, Deep Red, Gold, Ivory",
        "price_range":   "$$$$",
        "origin":        "Italy",
        "trigger_words": "gucci_style, opulent, maximalist, luxury Italian fashion",
    },
    "zara": {
        "name":          "Zara",
        "display":       "Zara — Contemporary European",
        "description":   "Contemporary European minimalism — clean, versatile, trend-responsive.",
        "aesthetic":     "Clean silhouettes, neutral palette, fast fashion with quality finishes.",
        "key_styles":    ["minimalist", "business_casual", "smart_casual"],
        "color_palette": "Camel, Black, White, Terracotta",
        "price_range":   "$$",
        "origin":        "Spain",
        "trigger_words": "zara_style, contemporary, minimalist European, versatile",
    },
    "hm": {
        "name":          "H&M",
        "display":       "H&M — Scandinavian Accessible",
        "description":   "Affordable Scandinavian fashion — trend-responsive, accessible, sustainable.",
        "aesthetic":     "Accessible basics, seasonal trend pieces, sustainability focus.",
        "key_styles":    ["casual", "streetwear", "bohemian", "sport"],
        "color_palette": "Pastels, Classic Navy, White, Earth tones",
        "price_range":   "$",
        "origin":        "Sweden",
        "trigger_words": "hm_style, Scandinavian, accessible fashion, minimalist basics",
    },
    "acne_studios": {
        "name":          "Acne Studios",
        "display":       "Acne Studios — Scandinavian Cool",
        "description":   "Swedish fashion house known for clean Scandinavian aesthetics and denim.",
        "aesthetic":     "Understated, intellectual, avant-garde minimalism.",
        "key_styles":    ["minimalist", "editorial", "smart_casual", "streetwear"],
        "color_palette": "Muted tones, Black, Off-White, Dusty Pink",
        "price_range":   "$$$",
        "origin":        "Sweden",
        "trigger_words": "acne_studios_style, Scandinavian cool, intellectual minimal",
    },
    "balenciaga": {
        "name":          "Balenciaga",
        "display":       "Balenciaga — Deconstructed Luxury",
        "description":   "Parisian luxury house pushing the boundaries of silhouette and proportion.",
        "aesthetic":     "Oversized, deconstructed, streetwear-luxury crossover, bold branding.",
        "key_styles":    ["streetwear", "luxury", "editorial", "techwear"],
        "color_palette": "Black, White, Neon Accents, Concrete Grey",
        "price_range":   "$$$$",
        "origin":        "France",
        "trigger_words": "balenciaga_style, deconstructed, oversized silhouette, luxury streetwear",
    },
    "off_white": {
        "name":          "Off-White",
        "display":       "Off-White — Streetwear Luxury",
        "description":   "Virgil Abloh's bridge between streetwear and high fashion.",
        "aesthetic":     "Industrial graphics, diagonal stripes, quotation marks, construction tape.",
        "key_styles":    ["streetwear", "luxury", "editorial", "urban"],
        "color_palette": "Black, White, Yellow, Orange",
        "price_range":   "$$$$",
        "origin":        "Italy/USA",
        "trigger_words": "off_white_style, streetwear luxury, industrial graphics, Abloh",
    },
    "toteme": {
        "name":          "Toteme",
        "display":       "Toteme — Quiet Luxury",
        "description":   "Swedish brand epitomising quiet luxury — refined, understated, impeccable.",
        "aesthetic":     "Tailored coats, neutral palette, premium fabrics, logo-free.",
        "key_styles":    ["quiet_luxury", "minimalist", "business_casual", "formal"],
        "color_palette": "Beige, Camel, Black, Cream",
        "price_range":   "$$$$",
        "origin":        "Sweden",
        "trigger_words": "toteme_style, quiet luxury, neutral minimalism, tailored Swedish",
    },
}


class LoRAService(BaseService):
    """
    Mock-safe UI adapter over ``LoRAInference`` and ``StyleMixer``.

    Provides brand-styled generation and multi-brand style mixing with
    full input validation, structured logging, and typed ``ServiceResult`` returns.

    Args:
        mock_mode: If ``True`` (default), uses mock image generation.
                   If ``False``, attempts to load the real LoRA pipeline.
    """

    _SERVICE_NAME = "LoRAService"

    def __init__(self, mock_mode: bool = True) -> None:
        super().__init__(mock_mode)
        self._inference: Optional[Any] = None
        self._mixer: Optional[Any] = None
        self.history_manager = HistoryManager()

        if not mock_mode:
            try:
                from src.lora.inference.lora_inference import LoRAInference
                from src.lora.style_manager.style_mixer import StyleMixer
                import torch
                dry_run = not torch.cuda.is_available()
                self._inference = LoRAInference(dry_run=dry_run)
                self._mixer     = StyleMixer(
                    registry=self._inference.registry,
                    inference_pipeline=self._inference.pipeline
                )
                self.mock_mode  = False
                logger.info(f"LoRAService: real inference engine loaded | dry_run={dry_run}")
            except Exception as exc:
                logger.warning("LoRAService: real engine unavailable — %s", exc)
                self.mock_mode = True
        else:
            logger.info("LoRAService: started in mock mode")

    # ── Brand-styled generation ────────────────────────────────────────────────

    def generate_with_brand(
        self,
        prompt: str,
        brand: str,
        *,
        lora_scale: float = 0.85,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """Generate a brand-styled fashion image using a LoRA adapter.

        Args:
            prompt:              Base text prompt.
            brand:               Brand key (from ``get_brands()``).
            lora_scale:          LoRA adapter influence strength [0.0, 1.5].
            num_inference_steps: Denoising steps [1, 100].
            guidance_scale:      CFG scale [1.0, 30.0].
            negative_prompt:     Optional negative conditioning text.
            seed:                Optional random seed.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``image``, ``prompt``, ``brand``, ``saved_path``, ``model``.
        """
        self._call_count += 1
        warnings: List[str] = []

        # ── Validation ──────────────────────────────────────────────────────
        try:
            prompt = self._validate_str(prompt, "prompt", min_len=1, max_len=1000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("prompt", e.reason)

        # ── Prompt token style switching ──────────────────────────────────────
        import re
        pattern = r"<(nike|gucci|zara|h&m|hm)>"
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            token = match.group(0)
            token_brand = match.group(1).lower()
            if token_brand in ("hm", "h&m"):
                token_brand = "hm"
            brand = token_brand
            prompt = prompt.replace(token, "").strip()
            prompt = re.sub(r"\s+", " ", prompt)

        brand = brand.lower().strip()
        if brand not in BRAND_REGISTRY:
            self._error_count += 1
            return ServiceResult.fail(
                f"Unknown brand '{brand}'. Valid brands: {list(BRAND_REGISTRY.keys())}",
                code="UNKNOWN_BRAND",
            )

        try:
            lora_scale = self._validate_range(lora_scale, "lora_scale", lo=0.0, hi=1.5)
        except ValidationError as e:
            warnings.append(f"lora_scale clamped: {e.reason}")
            lora_scale = max(0.0, min(1.5, float(lora_scale)))

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

        # ── Prompt enrichment ───────────────────────────────────────────────
        info = BRAND_REGISTRY[brand]
        enriched_prompt = f"{info['trigger_words']}, {prompt}, {info['aesthetic']}"

        # ── Generation ──────────────────────────────────────────────────────
        with self._timer() as t:
            try:
                if self.mock_mode or self._inference is None:
                    img        = _make_mock_image(enriched_prompt, 512, 512, f"LoRA [{info['name']}] (Mock)")
                    model_name = f"LoRA {info['name']} (Mock)"
                    run_mode   = "mock"
                else:
                    raw = self._inference.generate(
                        prompt=enriched_prompt,
                        brand=brand,
                        lora_scale=lora_scale,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                        seed=seed,
                    )
                    img        = raw.get("image")
                    model_name = f"LoRA {info['name']}"
                    run_mode   = "real"
            except Exception as exc:
                self._error_count += 1
                logger.error("LoRAService.generate_with_brand error: %s", exc, exc_info=True)
                img        = _make_mock_image(f"[Error] {prompt}", 512, 512, "LoRA Failed")
                model_name = f"LoRA {info['name']} (Error)"
                run_mode   = "error"
                warnings.append(f"Backend error — showing mock fallback. Detail: {exc}")

        latency = t.elapsed_ms
        logger.info(
            "LoRAService.generate_with_brand | brand=%s | scale=%.2f | latency=%.0fms",
            brand, lora_scale, latency,
        )

        # ── Save ────────────────────────────────────────────────────────────
        saved_path = ""
        if img is not None:
            try:
                fname = _OUTPUTS_DIR / f"lora_{brand}_{int(time.time())}.png"
                img.save(str(fname))
                saved_path = str(fname)
                with open(fname.with_suffix(".json"), "w", encoding="utf-8") as jf:
                    json.dump({
                        "prompt": prompt,
                        "enriched_prompt": enriched_prompt,
                        "brand": brand,
                        "lora_scale": lora_scale,
                        "steps": num_inference_steps,
                        "cfg": guidance_scale,
                        "seed": seed,
                        "model": model_name,
                        "run_mode": run_mode,
                        "saved_path": saved_path,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, jf, indent=2)
                
                # Persist to central history index
                try:
                    self.history_manager.record(
                        prompt=prompt,
                        image_path=saved_path,
                        model=model_name,
                        style=info["name"],
                        mode=run_mode,
                        service="lora",
                        brand=brand,
                        resolution="512×512", # lora output size is 512x512
                        seed=seed,
                        steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        lora_scale=lora_scale,
                        latency_ms=latency,
                    )
                except Exception as hist_exc:
                    logger.warning("LoRAService: failed to record in history manager — %s", hist_exc)
            except Exception as save_exc:
                logger.warning("LoRAService: save failed — %s", save_exc)
                warnings.append(f"Image generated but not saved: {save_exc}")

        status = ServiceStatus.MOCK if run_mode == "mock" else (
            ServiceStatus.ERROR if run_mode == "error" else ServiceStatus.OK
        )
        return ServiceResult(
            data={"image": img, "prompt": prompt, "brand": brand, "saved_path": saved_path, "model": model_name},
            meta={
                "mode": run_mode, "brand": info["name"], "lora_scale": lora_scale,
                "model": model_name, "latency_ms": round(latency, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "enriched_prompt": enriched_prompt,
            },
            status=status,
            warnings=warnings,
            latency_ms=latency,
        )

    # ── Style mixing ───────────────────────────────────────────────────────────

    def mix_styles(
        self,
        prompt: str,
        brand_weights: Dict[str, float],
        *,
        seed: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> ServiceResult[Dict[str, Any]]:
        """Mix multiple LoRA brand styles with specified weights.

        Args:
            prompt:        Base text prompt.
            brand_weights: Dict mapping brand key → weight (will be normalised).
                           At least one entry required.  All weights must be ≥ 0.
            seed:          Optional random seed.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.

        Returns:
            ``ServiceResult`` whose ``.data`` dict contains:
            ``image``, ``prompt``, ``brand_weights``, ``saved_path``, ``model``.
        """
        self._call_count += 1
        warnings: List[str] = []

        # ── Validation ──────────────────────────────────────────────────────
        try:
            prompt = self._validate_str(prompt, "prompt", min_len=1, max_len=1000)
        except ValidationError as e:
            self._error_count += 1
            return ServiceResult.validation_fail("prompt", e.reason)

        if not brand_weights:
            return ServiceResult.validation_fail("brand_weights", "at least one brand required")

        # Normalise weights: remove unknown brands, clamp negatives, normalise
        normalised: Dict[str, float] = {}
        for b, w in brand_weights.items():
            b_key = b.lower().strip()
            if b_key not in BRAND_REGISTRY:
                warnings.append(f"Unknown brand '{b}' ignored.")
                continue
            if w < 0:
                warnings.append(f"Negative weight for '{b}' set to 0.")
                w = 0.0
            normalised[b_key] = float(w)

        if not normalised or sum(normalised.values()) == 0:
            return ServiceResult.validation_fail(
                "brand_weights", "at least one valid brand with positive weight required"
            )

        total = sum(normalised.values())
        normalised = {b: w / total for b, w in normalised.items()}

        normalised, warnings = self.validate_brand_weights(brand_weights)
        if normalised is None:
            return ServiceResult.validation_fail("brand_weights", "; ".join(warnings))

        # Build descriptive label
        mix_label = "Style Mix: " + " + ".join(
            f"{BRAND_REGISTRY[b]['name']}({w:.0%})"
            for b, w in sorted(normalised.items(), key=lambda x: -x[1])
        )

        # ── Generation ──────────────────────────────────────────────────────
        with self._timer() as t:
            try:
                if self.mock_mode or self._mixer is None:
                    img        = _make_mock_image(prompt, 512, 512, mix_label)
                    model_name = "LoRA Style Mixer (Mock)"
                    run_mode   = "mock"
                else:
                    raw = self._mixer.mix(
                        prompt=prompt,
                        brand_weights=normalised,
                        seed=seed,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
                    img        = raw.get("image")
                    model_name = "LoRA Style Mixer"
                    run_mode   = "real"
            except Exception as exc:
                self._error_count += 1
                logger.error("LoRAService.mix_styles error: %s", exc, exc_info=True)
                img        = _make_mock_image(f"[Mix Error] {prompt}", 512, 512, "Mix Failed")
                model_name = "LoRA Style Mixer (Error)"
                run_mode   = "error"
                warnings.append(f"Backend error — showing mock fallback. Detail: {exc}")

        latency = t.elapsed_ms
        logger.info(
            "LoRAService.mix_styles | brands=%s | latency=%.0fms",
            list(normalised.keys()), latency,
        )

        # ── Save ────────────────────────────────────────────────────────────
        saved_path = ""
        if img is not None:
            try:
                fname = _OUTPUTS_DIR / f"lora_mix_{int(time.time())}.png"
                img.save(str(fname))
                saved_path = str(fname)
                with open(fname.with_suffix(".json"), "w", encoding="utf-8") as jf:
                    json.dump({
                        "prompt": prompt,
                        "brand_weights": normalised,
                        "seed": seed,
                        "model": model_name,
                        "run_mode": run_mode,
                        "saved_path": saved_path,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, jf, indent=2)
                
                # Persist to central history index
                try:
                    self.history_manager.record(
                        prompt=prompt,
                        image_path=saved_path,
                        model=model_name,
                        style=mix_label,
                        mode=run_mode,
                        service="lora",
                        resolution="512×512", # mix output size is 512x512
                        seed=seed,
                        steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        latency_ms=latency,
                    )
                except Exception as hist_exc:
                    logger.warning("LoRAService: failed to record mixed styles in history manager — %s", hist_exc)
            except Exception as save_exc:
                logger.warning("LoRAService: mix save failed — %s", save_exc)
                warnings.append(f"Image generated but not saved: {save_exc}")

        status = ServiceStatus.MOCK if run_mode == "mock" else (
            ServiceStatus.ERROR if run_mode == "error" else ServiceStatus.OK
        )
        return ServiceResult(
            data={
                "image": img, "prompt": prompt,
                "brand_weights": normalised, "saved_path": saved_path, "model": model_name,
            },
            meta={
                "mode": run_mode, "model": model_name, "brand_weights": normalised,
                "latency_ms": round(latency, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            status=status,
            warnings=warnings,
            latency_ms=latency,
        )

    # ── Metadata helpers ───────────────────────────────────────────────────────

    def get_brands(self) -> List[str]:
        """Return the list of registered brand keys."""
        return list(BRAND_REGISTRY.keys())

    def get_brand_info(self, brand: str) -> Optional[Dict[str, Any]]:
        """Return full metadata dict for *brand*, or ``None`` if unknown."""
        return BRAND_REGISTRY.get(brand.lower().strip())

    def get_brand_display_names(self) -> List[str]:
        """Return human-readable brand labels for Gradio dropdowns."""
        return [info["display"] for info in BRAND_REGISTRY.values()]

    def validate_brand_weights(
        self, brand_weights: Dict[str, float]
    ) -> Tuple[Optional[Dict[str, float]], List[str]]:
        """Validate and normalise a brand-weight dict.

        Args:
            brand_weights: Raw ``{brand_key: weight}`` dict from the UI.

        Returns:
            Tuple of ``(normalised_weights | None, warnings)``.
            If the dict is unrecoverable (e.g. all-zero), returns ``None``.
        """
        warnings: List[str] = []
        clean: Dict[str, float] = {}
        for b, w in brand_weights.items():
            b_key = b.lower().strip()
            if b_key not in BRAND_REGISTRY:
                warnings.append(f"Unknown brand '{b}' ignored.")
                continue
            if w < 0:
                warnings.append(f"Negative weight for '{b}' set to 0.")
                w = 0.0
            clean[b_key] = float(w)

        total = sum(clean.values())
        if not clean or total == 0:
            warnings.append("No valid brands with positive weights — cannot mix.")
            return None, warnings

        normalised = {b: w / total for b, w in clean.items()}
        return normalised, warnings

    # ── Health ─────────────────────────────────────────────────────────────────

    def health_check(self) -> ServiceResult:
        """Return lightweight health status of the LoRA backend."""
        res = {
            "status":        "ok" if not self.mock_mode else "mock",
            "mock_mode":     self.mock_mode,
            "backend":       "lora" if not self.mock_mode else "mock",
            "brands_loaded": len(BRAND_REGISTRY),
            "message": (
                f"LoRAService loaded with {len(BRAND_REGISTRY)} brand adapters."
                if not self.mock_mode
                else f"LoRAService running in mock mode with {len(BRAND_REGISTRY)} brand profiles."
            ),
        }
        return ServiceResult(success=True, data=res)
