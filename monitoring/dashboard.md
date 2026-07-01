# 📊 AI Fashion Assistant Diagnostics Dashboard

This dashboard guides administrators on tracking, query monitoring, and alert configurations for system resource health and pipeline latency.

---

## 🔍 Telemetry & Metric Definition reference

The `monitoring/metrics.py` module exposes system utilization and application states under the `fashion_assistant_` prefix. Below is a detailed listing of all metrics scraped or polled:

| Metric Name | Type | Description | Alert Threshold |
| :--- | :--- | :--- | :--- |
| `fashion_assistant_system_resources_cpu_usage_percent` | Gauge | System-wide CPU utilization percentage | `> 85.0%` (Warning), `> 95.0%` (Critical) |
| `fashion_assistant_system_resources_memory_usage_percent` | Gauge | Virtual memory (RAM) usage percentage | `> 90.0%` (Critical) |
| `fashion_assistant_system_resources_api_process_memory_mb` | Gauge | Resident Set Size memory of API process in MB | `> 2048 MB` (Warning) |
| `fashion_assistant_system_resources_gpu_usage_percent` | Gauge | GPU compute engine utilization percentage | `> 90.0%` (Warning) |
| `fashion_assistant_system_resources_gpu_memory_used_mb` | Gauge | GPU physical memory usage in MB | `> 90.0%` of VRAM |
| `fashion_assistant_redis_status` | Gauge | Redis backend connection state (`1` = OK, `0` = Down) | `== 0` (Critical) |
| `fashion_assistant_redis_memory_usage_bytes` | Gauge | Memory footprint utilized by Redis instances | `> 1 GB` (Warning) |
| `fashion_assistant_queues_celery_queue_length` | Gauge | Current length of the `celery` worker queue | `> 20` (Warning), `> 100` (Critical) |
| `fashion_assistant_api_latency_seconds_avg` | Summary | Average duration of HTTP API requests | `> 2.0s` (Warning) |
| `fashion_assistant_generation_duration_seconds_avg` | Summary | Average design compilation & model inference duration | `> 30.0s` (Warning) |

---

## 📈 PromQL Visualizations (Grafana Query Reference)

Use the following Prometheus Query Language (PromQL) expressions to populate Grafana dashboards:

### 1. CPU & Memory Utilization
```promql
# CPU Utilization
fashion_assistant_system_resources_cpu_usage_percent

# Memory Utilization
fashion_assistant_system_resources_memory_usage_percent
```

### 2. GPU Performance
```promql
# GPU Core Utilization
fashion_assistant_system_resources_gpu_usage_percent

# GPU Memory Utilization Ratio
fashion_assistant_system_resources_gpu_memory_used_mb / fashion_assistant_system_resources_gpu_memory_total_mb * 100
```

### 3. API & Generation Latencies
```promql
# API Average Latency (last 5 minutes)
rate(fashion_assistant_api_latency_seconds_sum[5m]) / rate(fashion_assistant_api_latency_seconds_count[5m])

# Model Generation Latency (last 5 minutes)
rate(fashion_assistant_generation_duration_seconds_sum[5m]) / rate(fashion_assistant_generation_duration_seconds_count[5m])
```

### 4. Celery Queue & Redis Connectivity
```promql
# Celery Backlog Size
fashion_assistant_queues_celery_queue_length

# Redis Health Status
fashion_assistant_redis_status
```

---

## 🛠️ CLI Live Console Dashboard

To review live metrics without deploying Grafana, run the built-in terminal dashboard:

```powershell
python monitoring/dashboard_cli.py
```

### Dashboard CLI Preview:
```
======================================================================
         📊 AI FASHION ASSISTANT LIVE DIAGNOSTICS CONSOLE
======================================================================
[2026-07-01 09:30:15]
----------------------------------------------------------------------
SYSTEM RESOURCES:
  • CPU Utilization       : [██████░░░░] 60.5% (8 Cores)
  • RAM Utilization       : [████████░░] 82.1% (Used: 13.1 GB / 16.0 GB)
  • API Process RAM       : 245.2 MB
  • GPU Utilization       : [████░░░░░░] 42.0% (NVIDIA RTX 4070)
  • GPU VRAM Usage        : 3120 MB / 8192 MB (38.1%)

REDIS & QUEUES:
  • Redis Status          : ONLINE (Connected Clients: 5)
  • Redis Memory          : 4.2 MB
  • Celery Queue Length   : 0 pending tasks

PIPELINE TELEMETRY:
  • Total API Requests    : 142
  • API Latency           : Avg: 0.125s | Min: 0.005s | Max: 0.420s
  • Total Generations     : 12 runs
  • Generation Time       : Avg: 8.420s | Min: 5.120s | Max: 12.110s
----------------------------------------------------------------------
Press Ctrl+C to exit.
```
