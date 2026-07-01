from __future__ import annotations

from typing import Any, Dict, List
from week7.backend.workers.celery_app import celery_app
from week7.backend.services.rag_service import RAGService
from week7.backend.api.dependencies import get_task_tracker


@celery_app.task(bind=True, name="tasks.answer_question_task")
def answer_question_task(self, question: str) -> Dict[str, Any]:
    """Background task for conversational Q&A retrieval."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = RAGService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.answer_question(question=question)
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.answer_question_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.recommend_styles_task")
def recommend_styles_task(self, preferences: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    """Background task for user style recommendations."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = RAGService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.recommend_styles(preferences=preferences, n=n)
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.recommend_styles_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.semantic_search_task")
def semantic_search_task(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Background task for raw similarity vector space searches."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = RAGService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.semantic_search(query=query, n_results=n_results)
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.semantic_search_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc


@celery_app.task(bind=True, name="tasks.explain_trend_task")
def explain_trend_task(self, trend_name: str) -> Dict[str, Any]:
    """Background task for trend analysis and explanation."""
    tracker = get_task_tracker()
    task_id = self.request.id
    tracker.update_progress(task_id, 10, "RUNNING")
    try:
        svc = RAGService()
        tracker.update_progress(task_id, 50, "RUNNING")
        res = svc.explain_trend(trend_name=trend_name)
        tracker.complete_task(task_id, res)
        return res
    except Exception as exc:
        from week7.backend.logging_config import log_celery_failure
        log_celery_failure("tasks.explain_trend_task", task_id, str(exc))
        tracker.fail_task(task_id, str(exc))
        raise exc
