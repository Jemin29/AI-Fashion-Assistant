from __future__ import annotations

import os
from typing import Dict, Any, Optional
from pathlib import Path

class HealthChecker:
    """Performs deep operational and dependency diagnostics for readiness/liveness checks."""
    
    def __init__(self, redis_client: Optional[Any] = None, db_dir: Optional[str] = None) -> None:
        self.redis_client = redis_client
        self.db_dir = db_dir

    def check_redis(self) -> Dict[str, Any]:
        """Verify Redis connectivity."""
        if not self.redis_client:
            return {"status": "unconfigured", "connected": False}
        try:
            self.redis_client.ping()
            return {"status": "healthy", "connected": True}
        except Exception as exc:
            return {"status": "unhealthy", "connected": False, "error": str(exc)}

    def check_database(self) -> Dict[str, Any]:
        """Verify that the database/storage paths are accessible and writeable."""
        if not self.db_dir:
            return {"status": "unconfigured", "writable": False}
        try:
            db_path = Path(self.db_dir)
            db_path.mkdir(parents=True, exist_ok=True)
            # Verify write access by writing and deleting a temporary indicator file
            temp_file = db_path / ".health_check_temp"
            temp_file.write_text("health_check_ok")
            temp_file.unlink()
            return {"status": "healthy", "writable": True, "path": str(db_path)}
        except Exception as exc:
            return {"status": "unhealthy", "writable": False, "error": str(exc)}

    def check_system(self) -> Dict[str, Any]:
        """Verify basic CPU and memory levels for availability degradation."""
        import psutil
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            status = "healthy"
            if cpu > 90.0 or mem > 90.0:
                status = "degraded"
            return {
                "status": status,
                "cpu_percent": cpu,
                "memory_percent": mem
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}
