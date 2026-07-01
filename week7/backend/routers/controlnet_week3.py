from __future__ import annotations

import io
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image

from week7.backend.api.dependencies import get_controlnet_service
from week7.backend.services.controlnet_service import ControlNetService

router = APIRouter(tags=["Week 3 ControlNet API"])


async def _process_upload_to_pil(file: UploadFile) -> Image.Image:
    """Parse incoming multipart file stream into a PIL Image."""
    try:
        contents = await file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(exc)}"
        )


@router.post("/sketch")
async def generate_from_sketch(
    file: UploadFile = File(..., description="Uploaded sketch/contours layout image."),
    prompt: str = Form(..., description="Description of the target garments/sceneries."),
    control_strength: float = Form(0.7, ge=0.0, le=2.0, description="Conditioning control scale."),
    negative_prompt: str = Form("", description="Negative prompt instructions."),
    seed: int = Form(-1, description="Deterministic seed. Pass -1 for random seed."),
    cntl_svc: ControlNetService = Depends(get_controlnet_service)
):
    """Generate designs conditioned on sketch contours."""
    control_image = await _process_upload_to_pil(file)
    try:
        res = cntl_svc.generate_from_sketch(
            prompt=prompt,
            sketch_image=control_image,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.post("/pose")
async def generate_from_pose(
    file: UploadFile = File(..., description="Uploaded body pose/skeleton reference image."),
    prompt: str = Form(..., description="Description of the target garments/sceneries."),
    control_strength: float = Form(0.7, ge=0.0, le=2.0, description="Conditioning control scale."),
    negative_prompt: str = Form("", description="Negative prompt instructions."),
    seed: int = Form(-1, description="Deterministic seed. Pass -1 for random seed."),
    cntl_svc: ControlNetService = Depends(get_controlnet_service)
):
    """Generate designs conditioned on pose bones structures."""
    control_image = await _process_upload_to_pil(file)
    try:
        res = cntl_svc.generate_from_pose(
            prompt=prompt,
            pose_image=control_image,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.post("/depth")
async def generate_from_depth(
    file: UploadFile = File(..., description="Uploaded depth layout reference image."),
    prompt: str = Form(..., description="Description of the target garments/sceneries."),
    control_strength: float = Form(0.7, ge=0.0, le=2.0, description="Conditioning control scale."),
    negative_prompt: str = Form("", description="Negative prompt instructions."),
    seed: int = Form(-1, description="Deterministic seed. Pass -1 for random seed."),
    cntl_svc: ControlNetService = Depends(get_controlnet_service)
):
    """Generate designs conditioned on depth mapping layout properties."""
    control_image = await _process_upload_to_pil(file)
    try:
        res = cntl_svc.generate_from_depth(
            prompt=prompt,
            depth_image=control_image,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed
        )
        return {
            "success": True,
            "image": res["image"],
            "metadata": res["metadata"]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
