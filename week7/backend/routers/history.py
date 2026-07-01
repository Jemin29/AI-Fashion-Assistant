from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_history_service
from week7.backend.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["Design Review & History Manager"])


class UpdateEntryRequest(BaseModel):
    """Validation schema for updating history entries."""
    rating: Optional[int] = Field(None, ge=0, le=5, description="Quality rating (1-5 stars).")
    notes: Optional[str] = Field(None, description="Reviewer notes.")
    tags: Optional[List[str]] = Field(None, description="Tag labels list.")


@router.get("/entries")
async def list_history_entries(
    query: str = Query("", description="Keyword search query across text fields."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort: str = Query("newest", description="Sort criteria (newest, oldest, rating, latency)."),
    model: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    has_image: Optional[bool] = Query(None),
    hist_svc: HistoryService = Depends(get_history_service)
):
    """Retrieve a paginated list of generated fashion designs with customizable search filters."""
    filters = {}
    if model:
        filters["model"] = model
    if style:
        filters["style"] = style
    if brand:
        filters["brand"] = brand
    if has_image is not None:
        filters["has_image"] = has_image

    # Use search method on history service
    entries, total = hist_svc.search(
        query=query,
        page=page,
        page_size=page_size,
        sort=sort,
        filters=filters
    )

    return {
        "success": True,
        "data": {
            "total_records": total,
            "page": page,
            "page_size": page_size,
            "entries": [e.to_dict() if hasattr(e, "to_dict") else vars(e) for e in entries]
        }
    }


@router.get("/entry/{entry_id}")
async def get_history_entry_details(
    entry_id: str,
    hist_svc: HistoryService = Depends(get_history_service)
):
    """Retrieve technical parameters and generation metadata for a single design."""
    entry = hist_svc.get_entry(entry_id.strip())
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with ID '{entry_id}' not found."
        )

    return {
        "success": True,
        "data": entry.to_dict() if hasattr(entry, "to_dict") else vars(entry)
    }


@router.put("/entry/{entry_id}")
async def update_history_entry(
    entry_id: str,
    payload: UpdateEntryRequest,
    hist_svc: HistoryService = Depends(get_history_service)
):
    """Update reviewer ratings, tag labels, and notes for a specific design."""
    updated = hist_svc.update_entry(
        entry_id=entry_id.strip(),
        rating=payload.rating,
        notes=payload.notes,
        tags=payload.tags
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with ID '{entry_id}' not found."
        )

    return {
        "success": True,
        "data": updated.to_dict() if hasattr(updated, "to_dict") else vars(updated)
    }


@router.delete("/entry/{entry_id}")
async def delete_history_entry(
    entry_id: str,
    delete_file: bool = Query(False, description="Physically remove the image file from local disk storage."),
    hist_svc: HistoryService = Depends(get_history_service)
):
    """Delete a generated design record from the history database index."""
    deleted = hist_svc.delete_entry(entry_id=entry_id.strip(), delete_file=delete_file)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with ID '{entry_id}' not found."
        )

    return {
        "success": True,
        "message": f"Successfully deleted history record '{entry_id}'."
    }
