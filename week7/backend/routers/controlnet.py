from __future__ import annotations

import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from PIL import Image
from pydantic import BaseModel, Field

from week7.backend.configs.rate_limit import limiter

from week7.backend.api.dependencies import get_controlnet_service
from week6.services.controlnet_service import ControlNetService, CONDITIONING_MODES

router = APIRouter(prefix="/controlnet", tags=["ControlNet Studio"])


@router.get("/modes")
async def get_conditioning_modes():
    """Retrieve all supported conditioning/preprocessing modes (e.g. Canny, Depth, Pose)."""
    return {
        "success": True,
        "data": [
            {"mode": k, "name": v["name"], "description": v["description"], "icon": v["icon"]}
            for k, v in CONDITIONING_MODES.items()
        ]
    }


@router.post("/preprocess")
@limiter.limit("10/minute")
async def preprocess_control_image(
    request: Request,
    mode: str = Form("canny", description="Preprocessing conditioning mode to run"),
    file: UploadFile = File(..., description="Source image file to preprocess"),
    cn_svc: ControlNetService = Depends(get_controlnet_service)
):
    """Preprocess an uploaded sketch/image to preview its edge map/conditioning mask."""
    if mode not in CONDITIONING_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Conditioning mode '{mode}' is not supported."
        )

    # Read uploaded file as PIL Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(exc)}"
        )

    res = cn_svc.preprocess_image(image, mode=mode)
    if not res.is_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Preprocessing failed."
        )

    # Save output to temporary file and return path
    import uuid
    from week7.backend.configs.config import get_settings
    
    settings = get_settings()
    out_dir = Path(settings.database.chroma_db_dir).parent / "sketches"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"preprocess_{uuid.uuid4().hex}.png"
    out_path = out_dir / out_filename
    res.data.save(out_path, "PNG")

    return {
        "success": True,
        "data": {
            "preprocessed_image_path": str(out_path),
            "meta": res.meta
        }
    }


@router.post("/generate")
@limiter.limit("10/minute")
async def generate_conditioned_design(
    request: Request,
    prompt: str = Form(..., description="Text description of the desired design"),
    negative_prompt: str = Form("", description="Negative prompt controls"),
    mode: str = Form("canny", description="Conditioning mode to run"),
    conditioning_scale: float = Form(0.7, ge=0.0, le=2.0),
    steps: int = Form(25, ge=1, le=100),
    cfg: float = Form(7.5, ge=1.0, le=30.0),
    seed: Optional[int] = Form(None),
    session_id: str = Form("default-session"),
    file: UploadFile = File(..., description="Conditioning input sketch/image"),
    cn_svc: ControlNetService = Depends(get_controlnet_service)
):
    """Generate a fashion design using an uploaded sketch as layout conditioning."""
    if mode not in CONDITIONING_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Conditioning mode '{mode}' is not supported."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(exc)}"
        )

    res = cn_svc.generate_conditioned(
        prompt=prompt,
        control_image=image,
        mode=mode,
        conditioning_scale=conditioning_scale,
        num_inference_steps=steps,
        guidance_scale=cfg,
        negative_prompt=negative_prompt,
        seed=seed
    )

    if not res.is_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Generation failed."
        )

    # Register temporary outputs if session_id is active
    from week7.backend.api.dependencies import get_state_manager
    state_mgr = get_state_manager()
    image_path = res.data.get("image_path")
    if image_path:
        state_mgr.add_temporary_output(session_id, image_path)

    return {
        "success": True,
        "data": res.data
    }
