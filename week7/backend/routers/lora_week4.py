from __future__ import annotations

from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_lora_service
from week7.backend.services.lora_service import LoraService

router = APIRouter(tags=["Week 4 LoRA & Style Mixer API"])


class LoRAWeek4Request(BaseModel):
    """Validation schema for single LoRA style generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Generative description text.")
    brand: str = Field(..., description="Brand adapter key (nike, gucci, zara, h&m).")
    lora_scale: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Adapter influence scaling factor.")
    seed: Optional[int] = Field(None, description="Deterministic seed value.")


class StyleMixWeek4Request(BaseModel):
    """Validation schema for multi-LoRA style mixing."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Generative description text.")
    brand_weights: Dict[str, float] = Field(..., description="Blending weights mapping (e.g. {'nike': 0.6, 'gucci': 0.4}).")
    seed: Optional[int] = Field(None, description="Deterministic seed value.")


@router.post("/lora")
async def generate_lora_style(
    payload: LoRAWeek4Request,
    lora_svc: LoraService = Depends(get_lora_service)
):
    """Generate a design using a single brand style adapter."""
    try:
        res = lora_svc.generate_lora_style(
            prompt=payload.prompt,
            brand=payload.brand,
            lora_scale=payload.lora_scale if payload.lora_scale is not None else 1.0,
            seed=payload.seed
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"]
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LoRA Inference Error: {str(exc)}"
        )


@router.post("/style-switch")
async def generate_style_switch(
    payload: LoRAWeek4Request,
    lora_svc: LoraService = Depends(get_lora_service)
):
    """Switch active brand style adapter and generate design output."""
    return await generate_lora_style(payload, lora_svc=lora_svc)


@router.post("/style-mix")
async def generate_style_mix(
    payload: StyleMixWeek4Request,
    lora_svc: LoraService = Depends(get_lora_service)
):
    """Mix multiple brand styles dynamically according to weights."""
    try:
        res = lora_svc.mix_styles(
            prompt=payload.prompt,
            brand_weights=payload.brand_weights,
            seed=payload.seed
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"]
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Style Mixing Error: {str(exc)}"
        )
