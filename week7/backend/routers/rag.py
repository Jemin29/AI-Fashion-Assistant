from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_rag_service
from week7.backend.configs.rate_limit import limiter
from week7.backend.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["Fashion Assistant & RAG"])


class ChatRequest(BaseModel):
    """Validation schema for assistant conversational dialogue requests."""
    message: str = Field(..., min_length=1, max_length=2000, description="Dialogue text message.")
    session_id: str = Field("default-session", description="Target session identifier.")


class SearchRequest(BaseModel):
    """Validation schema for raw similarity vector searches."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query string.")
    limit: int = Field(5, ge=1, le=20, description="Maximum results to return.")


@router.post("/chat")
@limiter.limit("20/minute")
async def chat_with_assistant(
    request: Request,
    payload: ChatRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Submit dialogue turns to conversational RAG Fashion Assistant."""
    try:
        res = rag_svc.chat(message=payload.message, user_id=payload.session_id)
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Chat query failed."
            )
        data = res.data
        if isinstance(data, dict) and "response" in data and "answer" not in data:
            data["answer"] = data["response"]
        return {
            "success": True,
            "data": data
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.post("/search")
@limiter.limit("20/minute")
async def raw_semantic_search(
    request: Request,
    payload: SearchRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Execute vector similarity searches across the vector collections databases."""
    try:
        res = rag_svc.semantic_search(query=payload.query, n=payload.limit)
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Search query failed."
            )
        return {
            "success": True,
            "data": {
                "citations": res.data
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/stats")
async def get_collection_database_stats(
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Retrieve ChromaDB collection metadata and documents distribution statistics."""
    try:
        res = rag_svc.get_collection_stats()
        if not res.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.error or "Failed to retrieve statistics."
            )
        return {
            "success": True,
            "data": res.data
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
