# Week 8: Production Platform Deployment

Welcome to the **Week 8 Deployment Framework** for the AI Fashion Assistant platform. This module converts the application stack into a production-grade, enterprise-orchestrated AI design suite.

---

## 📂 Module Architecture & Roadmap

This directory structure contains all resources required for isolated platform deployments:

* **[deployment/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/deployment/)**: Infrastructure container configurations and cluster manifests:
  * **[docker/Dockerfile](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/deployment/docker/Dockerfile)**: Multi-stage, size-optimized Dockerfile.
  * **[docker/docker-compose.prod.yml](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/deployment/docker/docker-compose.prod.yml)**: Multi-container orchestration (API, UI, Celery Workers, Redis).
  * **[kubernetes/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/deployment/kubernetes/)**: Manifests managing ConfigMaps, Secrets, Services, Deployments, Auto-scalers (HPA), and Ingress route mappings.
* **[monitoring/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/monitoring/)**: Telemetry scrapers and dashboards:
  * **[prometheus.yml](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/monitoring/prometheus.yml)**: Prometheus scrape configs querying backend `/health`.
  * **[grafana-dashboard.json](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/monitoring/grafana-dashboard.json)**: Real-time resource metrics dashboard panels.
* **[portfolio/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/portfolio/)**: Automation and showcasing:
  * **[portfolio_runner.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/portfolio/portfolio_runner.py)**: Trigger for building glassmorphic evaluation showcases.
* **[scripts/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/scripts/)**: Automated bootstrap scripts:
  * **[deploy.sh](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/scripts/deploy.sh)**: Linux Bash bootstrapper.
  * **[deploy.ps1](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/scripts/deploy.ps1)**: Windows PowerShell bootstrapper.
* **[docs/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/docs/)**: Onboarding references:
  * **[Deployment_Guide.md](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/docs/Deployment_Guide.md)**: Detailed step-by-step setup guide.
  * **[Architecture_Diagram.md](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/docs/Architecture_Diagram.md)**: Mermaid flow and structural layouts.
* **[configs/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/configs/)**: Production overrides:
  * **[.env.production](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/configs/.env.production)**: Environment variable overrides.
  * **[celery_config.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/configs/celery_config.py)**: Production worker performance parameters.

---

## 🚀 Getting Started

Deploy the platform instantly using the bootstrap scripts.

### On Windows (PowerShell):
```powershell
.\week8\scripts\deploy.ps1 -Mode docker
```

### On Linux/macOS (Bash):
```bash
./week8/scripts/deploy.sh docker
```
