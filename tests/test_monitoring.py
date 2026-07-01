from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest import mock
import pytest
from loguru import logger

from monitoring.metrics import MetricsRegistry, MetricsCollector, registry, track_api_latency, track_async_api_latency, GenerationTimer
from monitoring.logging import configure_logging, log_api_request, log_generation, log_redis_failure, log_celery_failure


class FakeRedisClient:
    """Mock Redis Client for unittest isolation."""
    def __init__(self, queue_len: int = 5, used_memory: int = 4096, clients: int = 2) -> None:
        self._queue_len = queue_len
        self._used_memory = used_memory
        self._clients = clients
        self.ping_called = False

    def ping(self) -> bool:
        self.ping_called = True
        return True

    def info(self) -> dict:
        return {
            "used_memory": self._used_memory,
            "connected_clients": self._clients
        }

    def llen(self, name: str) -> int:
        return self._queue_len


def test_metrics_registry():
    """Verify registry creation, metrics setting, and Prometheus format exposition."""
    test_reg = MetricsRegistry(prefix="test_app")
    test_reg.register("cpu_usage", "gauge", "CPU Usage")
    test_reg.register("requests_total", "counter", "Total requests")
    test_reg.register("op_latency", "histogram", "Operation latency")

    test_reg.set_gauge("cpu_usage", 42.5)
    test_reg.inc_counter("requests_total", 10)
    test_reg.observe_histogram("op_latency", 0.5)
    test_reg.observe_histogram("op_latency", 1.5)

    expo = test_reg.get_prometheus_exposition()
    
    assert "# HELP test_app_cpu_usage CPU Usage" in expo
    assert "# TYPE test_app_cpu_usage gauge" in expo
    assert "test_app_cpu_usage 42.5" in expo

    assert "# HELP test_app_requests_total Total requests" in expo
    assert "# TYPE test_app_requests_total counter" in expo
    assert "test_app_requests_total 10.0" in expo

    assert "# HELP test_app_op_latency_seconds Duration of operations" in expo
    assert "test_app_op_latency_count 2" in expo
    assert "test_app_op_latency_sum 2.0" in expo
    assert "test_app_op_latency_avg 1.0" in expo
    assert "test_app_op_latency_min 0.5" in expo
    assert "test_app_op_latency_max 1.5" in expo


def test_metrics_collector():
    """Verify collector queries system, GPU, Redis, and Celery telemetry."""
    fake_redis = FakeRedisClient(queue_len=12, used_memory=1048576, clients=4)
    collector = MetricsCollector(redis_client=fake_redis)

    # Test system resources collection
    sys_metrics = collector.collect_system_metrics()
    assert "cpu_percent" in sys_metrics
    assert "memory_percent" in sys_metrics
    assert "process_memory_mb" in sys_metrics

    # Test GPU metrics
    gpu_metrics = collector.collect_gpu_metrics()
    assert "gpu_available" in gpu_metrics
    assert "gpu_percent" in gpu_metrics
    assert "gpu_memory_used_mb" in gpu_metrics

    # Test Redis
    redis_metrics = collector.collect_redis_metrics()
    assert redis_metrics["status"] == 1
    assert redis_metrics["used_memory_bytes"] == 1048576
    assert redis_metrics["connected_clients"] == 4

    # Test Celery queue
    celery_len = collector.collect_celery_metrics("celery")
    assert celery_len == 12

    # Test aggregate collector summary
    all_metrics = collector.collect_all()
    assert "system" in all_metrics
    assert "gpu" in all_metrics
    assert "redis" in all_metrics
    assert all_metrics["redis"]["connected_clients"] == 4
    assert all_metrics["celery"]["queue_length"] == 12


def test_metrics_decorators():
    """Verify request hooks and generation context managers track telemetry."""
    # Reset specific keys to ensure zero contamination
    registry.set_gauge("api_requests_total", 0.0)
    registry._counters["fashion_assistant_api_requests_total"] = 0.0
    registry._counters["fashion_assistant_generation_requests_total"] = 0.0
    registry._histograms["fashion_assistant_api_latency"] = []
    registry._histograms["fashion_assistant_generation_duration"] = []

    # Sync latency decorator
    @track_api_latency("test_sync")
    def sync_endpoint():
        time.sleep(0.01)
        return "sync_ok"

    res = sync_endpoint()
    assert res == "sync_ok"
    assert registry._counters["fashion_assistant_api_requests_total"] == 1.0
    assert len(registry._histograms["fashion_assistant_api_latency"]) == 1

    # Async latency decorator
    @track_async_api_latency("test_async")
    async def async_endpoint():
        time.sleep(0.01)
        return "async_ok"

    import asyncio
    res_async = asyncio.run(async_endpoint())
    assert res_async == "async_ok"
    assert registry._counters["fashion_assistant_api_requests_total"] == 2.0
    assert len(registry._histograms["fashion_assistant_api_latency"]) == 2

    # GenerationTimer context manager
    with GenerationTimer():
        time.sleep(0.01)

    assert registry._counters["fashion_assistant_generation_requests_total"] == 1.0
    assert len(registry._histograms["fashion_assistant_generation_duration"]) == 1


def test_logging_configuration():
    """Verify log setup directory creations and structured handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Override logs folder in logging config
        with mock.patch("monitoring.logging.LOGS_DIR", tmp_path):
            configure_logging(log_level="DEBUG", log_to_file=True)
            
            # Emit logs
            logger.info("Operational logs setup completed.")
            log_api_request("GET", "/api/v1/design", 200, 0.150, "127.0.0.1")
            log_generation("a stylish trenchcoat", 5.2, True)
            log_redis_failure("ping", "Connection refused")
            log_celery_failure("compile_showcase", "task_abc", "Timeout")

            # Release file handles before validation
            logger.remove()

            log_file = tmp_path / "monitoring.log"
            assert log_file.exists()

            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            assert "Operational logs setup completed." in content
            assert "API Request: GET /api/v1/design" in content
            assert "Generation Success | Prompt: 'a stylish trenchcoat'" in content
            assert "Redis Connection Failure | Operation: ping" in content
            assert "Celery Task Failure | Task: compile_showcase" in content
