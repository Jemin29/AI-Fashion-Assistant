from __future__ import annotations

import os
import time
import subprocess
import shutil
import psutil
from typing import Dict, Any, Optional, List, Callable
from functools import wraps

# Try to import torch to support GPU metric collection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricsRegistry:
    """A lightweight, dependency-free metrics registry that exposes metrics in Prometheus text format."""

    def __init__(self, prefix: str = "fashion_assistant") -> None:
        self.prefix = prefix
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._descriptions: Dict[str, str] = {}
        self._types: Dict[str, str] = {}

    def _fullname(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    def register(self, name: str, metric_type: str, description: str) -> None:
        fullname = self._fullname(name)
        self._types[fullname] = metric_type
        self._descriptions[fullname] = description

    def set_gauge(self, name: str, value: float) -> None:
        fullname = self._fullname(name)
        self._gauges[fullname] = float(value)

    def inc_counter(self, name: str, amount: float = 1.0) -> None:
        fullname = self._fullname(name)
        self._counters[fullname] = self._counters.get(fullname, 0.0) + float(amount)

    def observe_histogram(self, name: str, value: float) -> None:
        fullname = self._fullname(name)
        if fullname not in self._histograms:
            self._histograms[fullname] = []
        self._histograms[fullname].append(float(value))

    def get_prometheus_exposition(self) -> str:
        """Render all registered metrics in the standard Prometheus exposition format."""
        lines = []

        # 1. Render Gauges
        for name, value in self._gauges.items():
            desc = self._descriptions.get(name, "Gauge metric")
            lines.append(f"# HELP {name} {desc}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # 2. Render Counters
        for name, value in self._counters.items():
            desc = self._descriptions.get(name, "Counter metric")
            lines.append(f"# HELP {name} {desc}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # 3. Render Histograms (custom lightweight representation)
        for name, observations in self._histograms.items():
            desc = self._descriptions.get(name, "Histogram metric")
            lines.append(f"# HELP {name}_seconds Duration of operations")
            lines.append(f"# TYPE {name}_seconds summary")
            
            count = len(observations)
            sum_val = sum(observations)
            lines.append(f"{name}_count {count}")
            lines.append(f"{name}_sum {sum_val}")
            
            if count > 0:
                avg_val = sum_val / count
                min_val = min(observations)
                max_val = max(observations)
                lines.append(f"{name}_avg {avg_val}")
                lines.append(f"{name}_min {min_val}")
                lines.append(f"{name}_max {max_val}")

        return "\n".join(lines) + "\n"


# Initialize the global metrics registry
registry = MetricsRegistry()

# System Metrics registrations
registry.register("system_resources_cpu_usage_percent", "gauge", "System CPU utilization percentage")
registry.register("system_resources_cpu_cores", "gauge", "Number of logical CPU cores")
registry.register("system_resources_memory_usage_percent", "gauge", "System RAM utilization percentage")
registry.register("system_resources_memory_total_bytes", "gauge", "Total system RAM in bytes")
registry.register("system_resources_memory_used_bytes", "gauge", "Used system RAM in bytes")
registry.register("system_resources_api_process_memory_mb", "gauge", "API process Resident Set Size (RSS) memory in MB")

# GPU Metrics registrations
registry.register("system_resources_gpu_available", "gauge", "Binary indicator (1=available, 0=not available) of CUDA GPU")
registry.register("system_resources_gpu_usage_percent", "gauge", "GPU engine utilization percentage")
registry.register("system_resources_gpu_memory_used_mb", "gauge", "GPU memory usage in MB")
registry.register("system_resources_gpu_memory_total_mb", "gauge", "Total GPU memory in MB")

# Cache & Queue registrations
registry.register("redis_status", "gauge", "Redis connectivity indicator (1=healthy, 0=offline)")
registry.register("redis_memory_usage_bytes", "gauge", "Redis memory usage in bytes")
registry.register("redis_connected_clients", "gauge", "Number of clients connected to Redis")
registry.register("queues_celery_queue_length", "gauge", "Length of Celery task queue")

# Application registrations
registry.register("api_requests_total", "counter", "Total count of API requests completed")
registry.register("api_latency", "histogram", "API response latency in seconds")
registry.register("generation_requests_total", "counter", "Total design generation runs initiated")
registry.register("generation_duration", "histogram", "Time elapsed during fashion design generations in seconds")


class MetricsCollector:
    """Core class containing collection handlers for all system, database, and pipeline metrics."""

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self.redis_client = redis_client

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect CPU, memory, and process stats using psutil."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_cores = psutil.cpu_count()
            mem = psutil.virtual_memory()
            
            # API Process details
            process = psutil.Process(os.getpid())
            proc_mem_mb = process.memory_info().rss / (1024 * 1024)
            
            registry.set_gauge("system_resources_cpu_usage_percent", cpu_percent)
            registry.set_gauge("system_resources_cpu_cores", cpu_cores)
            registry.set_gauge("system_resources_memory_usage_percent", mem.percent)
            registry.set_gauge("system_resources_memory_total_bytes", mem.total)
            registry.set_gauge("system_resources_memory_used_bytes", mem.used)
            registry.set_gauge("system_resources_api_process_memory_mb", proc_mem_mb)
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_cores": cpu_cores,
                "memory_percent": mem.percent,
                "memory_total_bytes": mem.total,
                "memory_used_bytes": mem.used,
                "process_memory_mb": proc_mem_mb
            }
        except Exception:
            return {}

    def collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU load and memory allocation using torch.cuda or nvidia-smi command execution."""
        gpu_stats = {
            "gpu_available": 0,
            "gpu_percent": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0
        }

        # Check torch CUDA availability first
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_stats["gpu_available"] = 1
            try:
                device = torch.cuda.current_device()
                gpu_stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
                gpu_stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
            except Exception:
                pass

        # Try to obtain actual compute load from nvidia-smi if available on Windows/Linux
        if shutil.which("nvidia-smi"):
            gpu_stats["gpu_available"] = 1
            try:
                cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
                parts = [p.strip() for p in result.stdout.split(",")]
                if len(parts) >= 3:
                    gpu_stats["gpu_percent"] = float(parts[0])
                    gpu_stats["gpu_memory_used_mb"] = float(parts[1])
                    gpu_stats["gpu_memory_total_mb"] = float(parts[2])
            except Exception:
                pass

        registry.set_gauge("system_resources_gpu_available", gpu_stats["gpu_available"])
        registry.set_gauge("system_resources_gpu_usage_percent", gpu_stats["gpu_percent"])
        registry.set_gauge("system_resources_gpu_memory_used_mb", gpu_stats["gpu_memory_used_mb"])
        registry.set_gauge("system_resources_gpu_memory_total_mb", gpu_stats["gpu_memory_total_mb"])

        return gpu_stats

    def collect_redis_metrics(self) -> Dict[str, Any]:
        """Collect Redis metrics like active connections and memory usages."""
        redis_stats = {
            "status": 0,
            "used_memory_bytes": 0,
            "connected_clients": 0
        }
        if not self.redis_client:
            registry.set_gauge("redis_status", 0)
            return redis_stats

        try:
            info = self.redis_client.info()
            redis_stats["status"] = 1
            redis_stats["used_memory_bytes"] = int(info.get("used_memory", 0))
            redis_stats["connected_clients"] = int(info.get("connected_clients", 0))
        except Exception:
            redis_stats["status"] = 0

        registry.set_gauge("redis_status", redis_stats["status"])
        registry.set_gauge("redis_memory_usage_bytes", redis_stats["used_memory_bytes"])
        registry.set_gauge("redis_connected_clients", redis_stats["connected_clients"])
        return redis_stats

    def collect_celery_metrics(self, queue_name: str = "celery") -> int:
        """Collect the size of Celery's task queues from Redis storage."""
        if not self.redis_client:
            registry.set_gauge("queues_celery_queue_length", 0)
            return 0
        try:
            # redis.llen checks list length (Celery broker list size)
            length = self.redis_client.llen(queue_name)
            registry.set_gauge("queues_celery_queue_length", length)
            return length
        except Exception:
            registry.set_gauge("queues_celery_queue_length", 0)
            return 0

    def collect_all(self) -> Dict[str, Any]:
        """Trigger collection on all gauges and return a compiled state summary."""
        return {
            "timestamp": time.time(),
            "system": self.collect_system_metrics(),
            "gpu": self.collect_gpu_metrics(),
            "redis": self.collect_redis_metrics(),
            "celery": {
                "queue_length": self.collect_celery_metrics()
            }
        }


# ── Latency & Generation Tracking Decorators ───────────────────────────────────

def track_api_latency(endpoint_name: str) -> Callable:
    """Decorator to measure API request latencies and update histogram metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                registry.inc_counter("api_requests_total")
                registry.observe_histogram("api_latency", duration)
        return wrapper
    return decorator


def track_async_api_latency(endpoint_name: str) -> Callable:
    """Async decorator to measure API request latencies and update histogram metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                registry.inc_counter("api_requests_total")
                registry.observe_histogram("api_latency", duration)
        return wrapper
    return decorator


class GenerationTimer:
    """A context manager to track design generation times and log completions."""
    
    def __enter__(self) -> GenerationTimer:
        self.start_time = time.perf_counter()
        registry.inc_counter("generation_requests_total")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.perf_counter() - self.start_time
        registry.observe_histogram("generation_duration", duration)
