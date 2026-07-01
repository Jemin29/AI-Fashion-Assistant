from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field, field_validator

from week7.backend.api.dependencies import get_generation_service
from week7.backend.configs.rate_limit import limiter
from week6.services.generation_service import GenerationService, SUPPORTED_RESOLUTIONS, STYLE_PRESETS

router = APIRouter(prefix="/generation", tags=["Text-to-Fashion"])


class GenerateRequest(BaseModel):
    """Validation schema for Text-to-Fashion generation requests."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Generative description text.")
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    style_preset: Optional[str] = Field(None, description="Pre-configured prompt style triggers.")
    width: int = Field(512, description="Output image width.")
    height: int = Field(512, description="Output image height.")
    steps: int = Field(30, ge=1, le=150, description="Denoising inference steps.")
    cfg: float = Field(7.5, ge=1.0, le=30.0, description="Classifier Free Guidance scale.")
    seed: Optional[int] = Field(None, description="Deterministic seed value.")
    session_id: str = Field("default-session", description="Target session identifier.")

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, val: int) -> int:
        # Dimension itself must be positive
        if val <= 0:
            raise ValueError("Dimensions must be positive integers.")
        return val


@router.get("/presets")
async def get_style_presets():
    """Retrieve all pre-configured prompt style triggers."""
    return {
        "success": True,
        "data": STYLE_PRESETS
    }


@router.get("/resolutions")
async def get_supported_resolutions():
    """Retrieve all supported aspect ratios and resolution options."""
    return {
        "success": True,
        "data": [
            {"width": w, "height": h, "label": f"{w}x{h}"}
            for w, h in SUPPORTED_RESOLUTIONS
        ]
    }


@router.post("/generate")
@limiter.limit("10/minute")
async def generate_fashion_image(
    request: Request,
    payload: GenerateRequest,
    gen_svc: GenerationService = Depends(get_generation_service)
):
    """Trigger Stable Diffusion XL to render a fashion concept based on parameters."""
    # Check if dimensions are in supported pairs
    resolution = (payload.width, payload.height)
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Resolution {payload.width}x{payload.height} is not supported. Choose from /resolutions."
        )

    res_str = f"{payload.width}x{payload.height}"
    final_prompt = payload.prompt
    if payload.style_preset:
        preset_trigger = next((p["trigger"] for p in STYLE_PRESETS if p["name"] == payload.style_preset), "")
        if preset_trigger:
            final_prompt = f"{final_prompt}, {preset_trigger}"

    try:
        res = gen_svc.generate(
            prompt=final_prompt,
            negative_prompt=payload.negative_prompt or "",
            seed=payload.seed if payload.seed is not None else -1,
            cfg=payload.cfg,
            resolution=res_str
        )
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Generation failed."
            )
        data = res.data or {}
        return {
            "success": True,
            "data": {
                "image": data.get("image", ""),
                "meta": data.get("metadata", {})
            }
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
