from __future__ import annotations

from typing import Any, Dict, List, Optional
from week6.services.rag_service import RAGService as Week6RAGService


class RAGService:
    """Business logic for Fashion RAG and Assistant Q&A."""

    def __init__(self, mock_mode: bool = True) -> None:
        self._svc = Week6RAGService(mock_mode=mock_mode)

    def chat(self, message: str, user_id: str = "default_user") -> Dict[str, Any]:
        res = self._svc.chat(message=message, user_id=user_id)
        if not res.is_ok:
            raise ValueError(res.error or "Chat query failed.")
        return res.data

    def answer_question(self, question: str) -> Dict[str, Any]:
        res = self._svc.answer_question(question=question)
        if not res.is_ok:
            raise ValueError(res.error or "Q&A failed.")
        return res.data

    def semantic_search(self, query: str, n_results: int = 5, n: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = n if n is not None else n_results
        res = self._svc.semantic_search(query=query, n_results=limit)
        if not res.is_ok:
            raise ValueError(res.error or "Semantic search failed.")
        return res.data

    def recommend_styles(self, preferences: Dict[str, Any], n: int = 5) -> Dict[str, Any]:
        res = self._svc.recommend_styles(preferences=preferences, n=n)
        if not res.is_ok:
            raise ValueError(res.error or "Style recommendations failed.")
        return res.data

    def explain_trend(self, trend_name: str) -> Dict[str, Any]:
        res = self._svc.explain_trend(trend_name=trend_name)
        if not res.is_ok:
            raise ValueError(res.error or "Trend explanation failed.")
        return res.data

    def get_collection_stats(self) -> Dict[str, Any]:
        res = self._svc.get_collection_stats()
        if not res.is_ok:
            raise ValueError(res.error or "Failed to retrieve statistics.")
        return res.data

    def health_check(self) -> Dict[str, Any]:
        """Verify the health status of the RAG service."""
        res = self._svc.health_check()
        if hasattr(res, "data"):
            return res.data
        return res
