from __future__ import annotations

from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from week7.backend.configs.rate_limit import limiter

from week7.backend.api.dependencies import get_lora_service
from week6.services.lora_service import LoRAService, BRAND_REGISTRY

router = APIRouter(prefix="/lora", tags=["Brand Studio"])


class LoRAGenerateRequest(BaseModel):
    """Validation schema for single-brand generation requests."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Generative description text.")
    brand: str = Field(..., description="Brand code key.")
    lora_scale: float = Field(0.85, ge=0.0, le=1.5, description="Influence scaling factor.")
    steps: int = Field(25, ge=1, le=100, description="Denoising inference steps.")
    cfg: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale.")
    negative_prompt: Optional[str] = Field("", max_length=1000)
    seed: Optional[int] = Field(None)
    session_id: str = Field("default-session")


class StyleMixRequest(BaseModel):
    """Validation schema for multi-brand style mixing requests."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Generative description text.")
    brand_weights: Dict[str, float] = Field(..., description="Dict mapping brand keys to weights (e.g. {'nike': 0.6, 'gucci': 0.4}).")
    steps: int = Field(25, ge=1, le=100, description="Denoising inference steps.")
    cfg: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale.")
    seed: Optional[int] = Field(None)
    session_id: str = Field("default-session")


@router.get("/adapters")
async def get_brand_adapters():
    """Retrieve all supported brand adapters and style profiles (Nike, Gucci, Zara, H&M)."""
    return {
        "success": True,
        "data": [
            {
                "brand_key": k,
                "name": v["name"],
                "display": v["display"],
                "description": v["description"],
                "aesthetic": v["aesthetic"],
                "origin": v["origin"],
                "trigger_words": v["trigger_words"]
            }
            for k, v in BRAND_REGISTRY.items()
        ]
    }


@router.post("/generate")
@limiter.limit("10/minute")
async def generate_brand_style(
    request: Request,
    payload: LoRAGenerateRequest,
    lora_svc: LoRAService = Depends(get_lora_service)
):
    """Generate a design conditioned by a single brand's style adapter."""
    brand_key = payload.brand.lower().strip()
    if brand_key not in BRAND_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Brand key '{payload.brand}' is not supported. Use one of /adapters keys."
        )

    res = lora_svc.generate_with_brand(
        prompt=payload.prompt,
        brand=brand_key,
        lora_scale=payload.lora_scale,
        num_inference_steps=payload.steps,
        guidance_scale=payload.cfg,
        negative_prompt=payload.negative_prompt or "",
        seed=payload.seed
    )

    if not res.is_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Generation failed."
        )

    # Register temporary output
    from week7.backend.api.dependencies import get_state_manager
    state_mgr = get_state_manager()
    image_path = res.data.get("saved_path")
    if image_path:
        state_mgr.add_temporary_output(payload.session_id, image_path)

    # Clean in-memory PIL image from the json response
    data = dict(res.data)
    if "image" in data:
        del data["image"]

    return {
        "success": True,
        "data": data,
        "meta": res.meta
    }


@router.post("/mix")
@limiter.limit("10/minute")
async def mix_brand_styles(
    request: Request,
    payload: StyleMixRequest,
    lora_svc: LoRAService = Depends(get_lora_service)
):
    """Blend multiple brand style weights dynamically to generate a hybrid design."""
    if not payload.brand_weights:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one brand weight mapping must be provided."
        )

    # Validate brand keys
    for brand_key in payload.brand_weights.keys():
        bk = brand_key.lower().strip()
        if bk not in BRAND_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Brand key '{brand_key}' is not supported."
            )

    # Invoke service
    res = lora_svc.mix_styles_for_router(
        prompt=payload.prompt,
        brand_weights={k.lower().strip(): v for k, v in payload.brand_weights.items()},
        seed=payload.seed,
        num_inference_steps=payload.steps,
        guidance_scale=payload.cfg
    )

    if not res.is_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Style mixing failed."
        )

    # Register output
    from week7.backend.api.dependencies import get_state_manager
    state_mgr = get_state_manager()
    image_path = res.data.get("saved_path")
    if image_path:
        state_mgr.add_temporary_output(payload.session_id, image_path)

    # Clean in-memory PIL image
    data = dict(res.data)
    if "image" in data:
        del data["image"]

    return {
        "success": True,
        "data": data,
        "meta": res.meta
    }
