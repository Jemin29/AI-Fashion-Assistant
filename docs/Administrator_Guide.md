# Administrator & DevOps Guide

This document provides system configuration details, database setup instructions, and troubleshooting guides for administrators maintaining the **AI Fashion Assistant** platform.

---

## ⚙️ 1. Environment Configurations

Platform settings are loaded dynamically via Pydantic settings from `week7/backend/configs/config.py`. Set variables in `.env` or as container environment variables:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `ENV` | `development` | Environment mode (`development`, `testing`, `production`). |
| `SERVER__DEBUG` | `True` | Toggle verbose debugging logs. |
| `MODEL__GLOBAL_MOCK` | `True` | Toggle mock stubs for diffusion and embedding engines. |
| `REDIS__MOCK_FALLBACK` | `True` | Toggle mock in-memory client if Redis is unavailable. |
| `REDIS_HOST` | `localhost` | Redis server hostname. |
| `REDIS_PORT` | `6379` | Redis server port. |
| `JWT_SECRET_KEY` | `dev_secret_key` | Secret key for signing JWT tokens. |
| `JWT_ALGORITHM` | `HS256` | JWT hash signature algorithm. |
| `JWT_EXPIRE_MINUTES` | `60` | Token expiration duration. |

---

## 🔒 2. Security & Gatekeeping Setup

### A. Rate Limiting (slowapi)
* Default endpoint limit: `10/minute` per client IP.
* Configured using `slowapi` decorators in routers (e.g. `/generate`).
* Override limit threshold: update config settings or decorate functions with custom limits.

### B. NSFW safety verification
* Intercepts Base64 image payload responses returned by the background worker.
* Uses lightweight safety checkers. If flagged, the payload is replaced with the fallback placeholder image `/assets/nsfw_fallback.png`.

### C. CORS Middleware
* Cross-Origin requests are configured in `main.py`. Update authorized origins list to restrict API usage.

---

## 🥦 3. Celery Asynchronous Workers

To verify worker connectivity and status, execute the following commands from the root directory:

### Check Active Worker Status
```bash
celery -A week7.backend.workers.celery_app status
```

### Inspect Tasks Queue Depth
```bash
celery -A week7.backend.workers.celery_app inspect active
```

*Note: Worker instances require access to the Redis instance defined by the environment `REDIS_HOST`.*

---

## 📊 4. Telemetry Metrics & Prometheus

FastAPI registers `/health` exposing system diagnostics:
1. **CPU/RAM Telemetry**: CPU load and RAM RSS footprints are returned in the JSON payload response.
2. **Prometheus Scraping**: Scrape configuration target is set in `prometheus.yml` under `week8/monitoring/`. Verify logs are incoming to Grafana visual dashboard interfaces.
