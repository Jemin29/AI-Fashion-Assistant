"""
Celery production tuning overrides.

Optimized for high-performance Stable Diffusion/ControlNet execution.
"""

from __future__ import annotations

# Task broker and result backend URLs
broker_url = 'redis://fashion-redis-service:6379/0'
result_backend = 'redis://fashion-redis-service:6379/0'

# Safe JSON serialization formats
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Worker Performance Tuning
# Limit to 1 heavy SDXL/ControlNet process per worker container to avoid OOM
worker_concurrency = 1

# Disable task prefetching to keep queue allocation fair across dynamic pods
worker_prefetch_multiplier = 1

# Resiliency settings
task_acks_late = True
task_reject_on_worker_lost = True
task_time_limit = 600  # Kill task if execution hangs longer than 10 minutes
task_soft_time_limit = 500
