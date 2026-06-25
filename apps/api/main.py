from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.fashion_assistant import FashionAssistant

# Initialize application
app = FastAPI(
    title="AI-Powered Fashion Design Assistant API",
    description="Production-grade API backend providing context-aware fashion recommendations, RAG, and trend forecasting.",
    version="1.0.0"
)

# Instantiate assistant with mock mode for lightweight execution
assistant = FashionAssistant(force_mock_embeddings=True)


# =============================================================================
# ── Pydantic Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or fashion question.")
    user_id: Optional[str] = Field(None, description="Optional unique identifier for user profiling.")


class QueryResponse(BaseModel):
    response: str
    source_documents: List[Dict[str, Any]] = []


class StyleRecommendationRequest(BaseModel):
    gender: str = Field(default="unisex", description="Target gender: men, women, unisex.")
    style: str = Field(default="streetwear", description="Target style, e.g. streetwear, luxury.")
    occasion: str = Field(default="casual", description="Target occasion, e.g. casual, party.")
    fit: str = Field(default="regular_fit", description="Target fit, e.g. regular_fit, oversized.")
    user_id: Optional[str] = Field(None, description="Optional unique identifier for user profiling.")


class StyleRecommendationResponse(BaseModel):
    recommendations: List[str]


class BrandRecommendationRequest(BaseModel):
    preferred_styles: List[str] = Field(default_factory=list, description="List of preferred style adjectives.")
    target_aesthetic: str = Field(..., description="Target brand aesthetic/profile description.")
    user_id: Optional[str] = Field(None, description="Optional unique identifier for user profiling.")


class BrandRecommendationResponse(BaseModel):
    recommendations: List[str]


class TrendForecastResponse(BaseModel):
    forecasts: List[Dict[str, Any]]


# =============================================================================
# ── Endpoints
# =============================================================================

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "AI-Powered Fashion Design Assistant API",
        "version": "1.0.0"
    }


@app.post("/api/v1/query", response_model=QueryResponse)
def answer_question(payload: QueryRequest):
    """Answer general fashion questions using context-aware RAG pipeline."""
    try:
        res = assistant.answer_question(payload.query, user_id=payload.user_id)
        return QueryResponse(
            response=res.get("response", "No response generated."),
            source_documents=res.get("source_documents", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")


@app.post("/api/v1/recommendations/styles", response_model=StyleRecommendationResponse)
def recommend_styles(payload: StyleRecommendationRequest):
    """Generate personalized style recommendations based on fit, occasion, and preferences."""
    try:
        prefs = {
            "gender": payload.gender,
            "style": payload.style,
            "occasion": payload.occasion,
            "fit": payload.fit
        }
        recs = assistant.recommend_styles(prefs, user_id=payload.user_id)
        return StyleRecommendationResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style recommendation failed: {str(e)}")


@app.post("/api/v1/recommendations/brands", response_model=BrandRecommendationResponse)
def recommend_brands(payload: BrandRecommendationRequest):
    """Identify apparel brands matching target style coordinates."""
    try:
        profile = {
            "styles": payload.preferred_styles,
            "target_aesthetic": payload.target_aesthetic
        }
        recs = assistant.recommend_brands(profile, user_id=payload.user_id)
        return BrandRecommendationResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brand recommendation failed: {str(e)}")


@app.get("/api/v1/trends/explain")
def explain_trend(trend_name: str = Query(..., description="Name of the trend to analyze.")):
    """Retrieve detailed analysis, explanation, and confidence scores for a fashion trend."""
    try:
        res = assistant.explain_trend(trend_name)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis query failed: {str(e)}")


@app.get("/api/v1/trends/forecast", response_model=TrendForecastResponse)
def forecast_trends(season: str = Query(..., description="Target season: spring_summer or autumn_winter.")):
    """Forecast future design element trends for a targeted season."""
    try:
        recs = assistant.get_trend_forecast(season)
        return TrendForecastResponse(forecasts=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend forecasting failed: {str(e)}")


@app.get("/api/v1/search")
def semantic_search(query: str = Query(..., description="Query string to match in ChromaDB index.")):
    """Query high-dimensional vector search matches directly over ChromaDB index collections."""
    try:
        results = assistant.chroma_retriever.retrieve(query, n_results=5)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
