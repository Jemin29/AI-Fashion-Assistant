# System Architecture Blueprint

This document details the production-grade deployment layout and operational sequence of the **AI-Powered Fashion Design Assistant** platform.

---

## 1. Deploy Subsystem Layout

```mermaid
graph TB
    subgraph Ingress
        RuleUI[fashion.example.com]
        RuleAPI[api.fashion.example.com]
    end

    subgraph "Kubernetes Namespace: fashion-ai"
        UI[Gradio UI Service]
        API[FastAPI Backend Service]
        Worker[Celery Inference Worker]
        Redis[(Redis Cache & Broker)]
        Chroma[(ChromaDB Vector Store)]
    end

    RuleUI --> UI
    RuleAPI --> API
    
    UI --> API
    API --> Redis
    Worker --> Redis
    API --> Chroma
    Worker --> Chroma
```

* **Gradio UI**: Represents the web front portal where designers interact with text-to-fashion, sketch-to-design, style switcher, and lookbook dashboards.
* **FastAPI REST API**: Serving as the centralized business controller, handling user authorization, RAG chat routing, data history caching, and task queue registration.
* **Redis Broker**: Holds asynchronous Celery task payloads, rate-limit counters, and temporary session keys.
* **Celery Worker**: Hardened single-concurrency worker pod responsible for executing heavy Stable Diffusion / ControlNet layout generation.
* **ChromaDB / FAISS**: Provides vectorized indexes of materials, silhouettes, brand aesthetics, and trend histories.

---

## 2. Asynchronous Generation Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Designer as User / Gradio Portal
    participant API as FastAPI Backend
    participant Redis as Redis Queue
    participant Worker as Celery Worker
    participant Filter as NSFW Filter
    participant Watermark as Watermark Service

    Designer->>API: POST /api/v1/generation/generate
    Note over API: Verify JWT Token & Scopes
    API->>Redis: Publish generation task parameters
    API-->>Designer: Return Task ID (202 Accepted)
    
    activate Worker
    Redis->>Worker: Consume generation task
    Worker->>Worker: Run Stable Diffusion Inference
    Worker->>Watermark: Overlay transparent branding
    Worker->>Filter: Scan final output for NSFW compliance
    alt Safe Output
        Worker->>Redis: Save result path and status (SUCCESS)
    else Unsafe Output
        Worker->>Redis: Set status (FAILED - SAFETY_VIOLATION)
    end
    deactivate Worker

    loop Status Polling
        Designer->>API: GET /api/v1/tasks/status/{id}
        API->>Redis: Read task status details
        API-->>Designer: Return current state (SUCCESS / PENDING)
    end
```
