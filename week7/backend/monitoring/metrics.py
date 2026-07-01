from __future__ import annotations

import os
import time
import psutil
from typing import Dict, Any, Optional

class SystemMetricsCollector:
    """Collects system-level resource utilization metrics (CPU, Memory, Process details)."""
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Retrieve overall system CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
            
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Retrieve virtual memory stats."""
        try:
            mem = psutil.virtual_memory()
            return {
                "total_bytes": mem.total,
                "available_bytes": mem.available,
                "used_bytes": mem.used,
                "percent": mem.percent
            }
        except Exception:
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "used_bytes": 0,
                "percent": 0.0
            }

    @staticmethod
    def get_process_memory() -> float:
        """Memory usage of the current API process in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


class ApplicationMetricsCollector:
    """Collects application performance metrics, task execution speeds, and queue lengths."""
    
    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self.redis_client = redis_client

    def get_queue_length(self, queue_name: str = "celery") -> int:
        """Query active queue length in Redis."""
        if not self.redis_client:
            return 0
        try:
            return self.redis_client.llen(queue_name)
        except Exception:
            return 0
