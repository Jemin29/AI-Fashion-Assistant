from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/liveness", tags=["Diagnostics"])

@router.get("")
async def get_liveness():
    """Simple check to verify the process is alive (returns HTTP 200)."""
    return {
        "success": True,
        "status": "alive"
    }
