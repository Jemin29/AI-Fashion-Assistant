from __future__ import annotations

import time
from typing import Any, Dict, Optional
from week7.backend.api.dependencies import get_redis_manager


class TaskTracker:
    """Helper to track asynchronous task metadata (status, progress, times) in Redis."""

    def __init__(self) -> None:
        self.redis = get_redis_manager()

    def init_task(self, task_id: str, task_type: str) -> None:
        """Initialize the task metadata record in Redis."""
        meta = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "PENDING",
            "progress": 0,
            "start_time": time.time(),
            "end_time": None,
            "execution_time_s": 0.0,
            "error": None,
            "result": None,
        }
        self.redis.cache_set(f"task_tracker:{task_id}", meta, ttl=86400)  # Keep record for 24 hours

    def update_progress(self, task_id: str, progress: int, status: str = "RUNNING") -> None:
        """Update the progress (0-100) and status of an active task."""
        meta = self.redis.cache_get(f"task_tracker:{task_id}")
        if meta:
            meta["status"] = status
            meta["progress"] = progress
            start = meta.get("start_time", time.time())
            meta["execution_time_s"] = round(time.time() - start, 2)
            self.redis.cache_set(f"task_tracker:{task_id}", meta, ttl=86400)

    def complete_task(self, task_id: str, result: Any) -> None:
        """Mark task as successfully completed and store its output."""
        meta = self.redis.cache_get(f"task_tracker:{task_id}")
        if meta:
            meta["status"] = "SUCCESS"
            meta["progress"] = 100
            now = time.time()
            meta["end_time"] = now
            start = meta.get("start_time", now)
            meta["execution_time_s"] = round(now - start, 2)
            meta["result"] = result
            self.redis.cache_set(f"task_tracker:{task_id}", meta, ttl=86400)

    def fail_task(self, task_id: str, error_msg: str) -> None:
        """Mark task as failed and record the traceback/error reason."""
        meta = self.redis.cache_get(f"task_tracker:{task_id}")
        if meta:
            meta["status"] = "FAILURE"
            now = time.time()
            meta["end_time"] = now
            start = meta.get("start_time", now)
            meta["execution_time_s"] = round(now - start, 2)
            meta["error"] = error_msg
            self.redis.cache_set(f"task_tracker:{task_id}", meta, ttl=86400)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task tracking metadata from Redis, calculating live execution time."""
        meta = self.redis.cache_get(f"task_tracker:{task_id}")
        if not meta:
            return None
        # Recalculate execution time if the task is still running
        if meta["status"] in ("PENDING", "RUNNING"):
            start = meta.get("start_time", time.time())
            meta["execution_time_s"] = round(time.time() - start, 2)
        return meta

    def delete_task(self, task_id: str) -> bool:
        """Delete task metadata record from Redis cache."""
        return self.redis.cache_delete(f"task_tracker:{task_id}")
