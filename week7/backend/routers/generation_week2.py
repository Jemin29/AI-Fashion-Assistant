from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_generation_service
from week7.backend.services.generation_service import GenerationService

router = APIRouter(tags=["Week 2 Model API"])


class Week2GenerateRequest(BaseModel):
    """Validation schema for Week 2 Stable Diffusion XL generation requests."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Positive prompt describing the desired design.")
    negative_prompt: Optional[str] = Field("", max_length=1000, description="Negative keywords to filter out.")
    seed: Optional[int] = Field(-1, description="Deterministic seed. Pass -1 for random seed.")
    cfg: Optional[float] = Field(7.5, ge=1.0, le=30.0, description="Classifier-free guidance scale.")
    resolution: Optional[str] = Field("1024x1024", description="Resolution preset (e.g. square_1024) or WxH string (e.g. 1024x1024).")


@router.post("/generate")
async def generate_image_week2(
    payload: Week2GenerateRequest,
    gen_svc: GenerationService = Depends(get_generation_service)
):
    """Direct REST endpoint wrapping the Week 2 SDXL generative engine."""
    try:
        res = gen_svc.generate(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt or "",
            seed=payload.seed if payload.seed is not None else -1,
            cfg=payload.cfg or 7.5,
            resolution=payload.resolution or "1024x1024"
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"],
            "generation_time": res["generation_time"]
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation Error: {str(exc)}"
        )
