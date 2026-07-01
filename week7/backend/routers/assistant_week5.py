from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from week7.backend.api.dependencies import get_rag_service
from week7.backend.services.rag_service import RAGService

router = APIRouter(tags=["Week 5 Fashion Assistant API"])


class AskRequest(BaseModel):
    """Validation schema for factual fashion Q&A."""
    question: str = Field(..., min_length=1, max_length=2000, description="Factual question about fashion.")


class RecommendRequest(BaseModel):
    """Validation schema for style recommendations."""
    style: Optional[str] = Field(None, description="Preferred style (e.g. casual, formal).")
    occasion: Optional[str] = Field(None, description="Target occasion (e.g. party, work).")
    fit: Optional[str] = Field(None, description="Preferred fit (e.g. slim, relaxed).")
    gender: Optional[str] = Field(None, description="Gender target.")
    color: Optional[str] = Field(None, description="Preferred color theme.")
    limit: int = Field(5, ge=1, le=20, description="Maximum recommendations to return.")


class SearchRequest(BaseModel):
    """Validation schema for raw similarity vector searches."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query string.")
    limit: int = Field(5, ge=1, le=20, description="Maximum results to return.")


class TrendRequest(BaseModel):
    """Validation schema for trend explanation."""
    trend_name: str = Field(..., min_length=1, max_length=200, description="Name of the trend to explain.")


@router.post("/ask")
async def ask_question(
    payload: AskRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Answer factual fashion questions with retrieved context."""
    try:
        data = rag_svc.answer_question(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    docs = data.get("source_documents", [])
    citations = []
    for doc in docs:
        if isinstance(doc, dict):
            citations.append(doc.get("document", ""))
        elif isinstance(doc, str):
            citations.append(doc)
            
    confidence_score = 0.90
    distances = []
    for doc in docs:
        if isinstance(doc, dict):
            d = doc.get("distance")
            if d is not None:
                try:
                    distances.append(float(d))
                except (ValueError, TypeError):
                    pass
    if distances:
        avg_dist = sum(distances) / len(distances)
        confidence_score = round(max(0.1, min(1.0, 1.0 - avg_dist)), 2)

    return {
        "success": True,
        "answer": data.get("response", ""),
        "recommendations": [],
        "confidence_score": confidence_score,
        "confidence score": confidence_score,
        "citations": citations
    }


@router.post("/recommend")
async def get_recommendations(
    payload: RecommendRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Fetch style recommendations matching the input preferences."""
    prefs = {}
    if payload.style:
        prefs["style"] = payload.style
    if payload.occasion:
        prefs["occasion"] = payload.occasion
    if payload.fit:
        prefs["fit"] = payload.fit
    if payload.gender:
        prefs["gender"] = payload.gender
    if payload.color:
        prefs["color"] = payload.color

    try:
        recs = rag_svc.recommend_styles(prefs, n=payload.limit)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    recommendations_list = []
    confidence_score = 0.85
    
    for item in recs:
        if isinstance(item, str):
            recommendations_list.append({
                "style": item,
                "description": f"Recommended {item} fashion style."
            })
        elif isinstance(item, dict):
            recommendations_list.append(item)
            if "score" in item:
                try:
                    confidence_score = round(float(item["score"]), 2)
                except (ValueError, TypeError):
                    pass
            elif "confidence" in item:
                try:
                    confidence_score = round(float(item["confidence"]), 2)
                except (ValueError, TypeError):
                    pass

    return {
        "success": True,
        "answer": f"Found {len(recommendations_list)} style recommendations matching your profile.",
        "recommendations": recommendations_list,
        "confidence_score": confidence_score,
        "confidence score": confidence_score,
        "citations": ["RAG Style Recommendation Database v1.0"]
    }


@router.post("/search")
async def semantic_search(
    payload: SearchRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Execute raw similarity vector searches."""
    try:
        docs = rag_svc.semantic_search(query=payload.query, n_results=payload.limit)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    citations = []
    for doc in docs:
        if isinstance(doc, dict):
            citations.append(f"[{doc.get('id', 'N/A')}] {doc.get('document', '')}")
        elif isinstance(doc, str):
            citations.append(doc)
            
    confidence_score = 0.88
    distances = []
    for doc in docs:
        if isinstance(doc, dict):
            d = doc.get("distance")
            if d is not None:
                try:
                    distances.append(float(d))
                except (ValueError, TypeError):
                    pass
    if distances:
        avg_dist = sum(distances) / len(distances)
        confidence_score = round(max(0.1, min(1.0, 1.0 - avg_dist)), 2)

    return {
        "success": True,
        "answer": f"Retrieved {len(docs)} matching documents for query '{payload.query}'.",
        "recommendations": [],
        "confidence_score": confidence_score,
        "confidence score": confidence_score,
        "citations": citations
    }


@router.post("/trend")
async def explain_trend(
    payload: TrendRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Fetch structured trend analysis and forecasting."""
    try:
        data = rag_svc.explain_trend(payload.trend_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    confidence_score = 0.75
    if "confidence" in data:
        try:
            confidence_score = round(float(data["confidence"]), 2)
        except (ValueError, TypeError):
            pass
            
    return {
        "success": True,
        "answer": data.get("explanation", ""),
        "recommendations": [],
        "confidence_score": confidence_score,
        "confidence score": confidence_score,
        "citations": [data.get("reasoning", "Trend forecast reasoning context")]
    }
