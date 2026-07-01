from __future__ import annotations

from typing import Any, Dict, List, Optional
from week6.services.rag_service import RAGService as Week6RAGService
from week6.services.base import ServiceResult


class RAGService:
    """Business logic for Fashion RAG and Assistant Q&A."""

    def __init__(self, mock_mode: bool = True) -> None:
        self._svc = Week6RAGService(mock_mode=mock_mode)

    def chat(self, message: str, user_id: str = "default_user") -> ServiceResult:
        res = self._svc.chat(message=message, user_id=user_id)
        return res

    def answer_question(self, question: str) -> ServiceResult:
        res = self._svc.answer_question(question=question)
        return res

    def semantic_search(self, query: str, n_results: int = 5, n: Optional[int] = None) -> ServiceResult:
        limit = n if n is not None else n_results
        res = self._svc.semantic_search(query=query, n_results=limit)
        return res

    def recommend_styles(self, preferences: Dict[str, Any], n: int = 5) -> ServiceResult:
        res = self._svc.recommend_styles(preferences=preferences, n=n)
        return res

    def explain_trend(self, trend_name: str) -> ServiceResult:
        res = self._svc.explain_trend(trend_name=trend_name)
        return res

    def get_collection_stats(self) -> ServiceResult:
        res = self._svc.get_collection_stats()
        return res

    def health_check(self) -> ServiceResult:
        """Verify the health status of the RAG service."""
        res = self._svc.health_check()
        return res
