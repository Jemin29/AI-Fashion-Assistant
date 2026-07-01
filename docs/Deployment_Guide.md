# Week 8 Production Deployment Guide

This guide provides step-by-step instructions for deploying the **AI-Powered Fashion Design Assistant** platform in both containerized (Docker Compose) and orchestrated (Kubernetes) environments.

---

## 1. Prerequisites

Ensure the following tools are installed and configured on the deployment host:
* **Docker Engine** (v20.10+) & **Docker Compose** (v2.0+)
* **Minikube** (for local Kubernetes testing) or an active production cluster (EKS, GKE, AKS)
* **kubectl** command-line tool configured to target the active cluster context
* **Python 3.11** (for local scripting and manual verifications)

---

## 2. Docker Compose Local Production Setup

For running a production-like environment on a single machine or testing server, use the optimized Docker Compose configuration:

### Steps:
1. Navigate to the project root directory.
2. Build and start the services in background (detached) mode:
   ```bash
   docker compose -f week8/deployment/docker/docker-compose.prod.yml up -d --build
   ```
3. Verify that all 4 containers (`fashion-redis-prod`, `fashion-api-prod`, `fashion-ui-prod`, and `fashion-worker-prod`) are healthy:
   ```bash
   docker compose -f week8/deployment/docker/docker-compose.prod.yml ps
   ```

### Exposed Endpoints:
* **Gradio Interactive Studio**: [http://localhost:7860](http://localhost:7860)
* **FastAPI Backend Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Diagnostics API**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 3. Kubernetes Multi-Node Cluster Deployment

For production deployments scaling to multiple nodes, follow the Kubernetes manifest registration order:

### Step 1: Create Namespace
Initialize the dedicated namespace `fashion-ai`:
```bash
kubectl create namespace fashion-ai
```

### Step 2: Configure Environment Settings & Secrets
Apply the ConfigMaps and secret keys:
```bash
kubectl apply -f week8/deployment/kubernetes/configmap.yaml
kubectl apply -f week8/deployment/kubernetes/secret.yaml
```

### Step 3: Register Internal Services
Configure the ClusterIPs and port forward definitions:
```bash
kubectl apply -f week8/deployment/kubernetes/service.yaml
```

### Step 4: Launch Deployments
Apply the multi-pod deployments containing liveness/readiness health probes and resource limits:
```bash
kubectl apply -f week8/deployment/kubernetes/deployment.yaml
```

### Step 5: Configure Ingress Rules
Expose the services externally using the NGINX Ingress controller:
```bash
kubectl apply -f week8/deployment/kubernetes/ingress.yaml
```

### Step 6: Enable Auto-Scaling
Register the Horizontal Pod Autoscalers (HPA) to scale pods under resource load:
```bash
kubectl apply -f week8/deployment/kubernetes/hpa.yaml
```

---

## 4. Telemetry Scrapes & Monitoring Integration

The REST API exposes system hardware usage and queue statistics on the `/api/v1/health` endpoint.

### Scrape Configuration (Prometheus):
Mount the custom [prometheus.yml](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/monitoring/prometheus.yml) file to scrape API stats every 15 seconds.

### Dashboard Setup (Grafana):
Import the [grafana-dashboard.json](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week8/monitoring/grafana-dashboard.json) schema into your Grafana instance to render real-time visualizations for:
* **System CPU Usage (%)**
* **Memory Headroom (%)**
* **Inference Celery Queue Size**
* **API Process Resident Set Size (RSS in MB)**

---

## 5. Automated Showcase Generation

Compile design portfolios and evaluation statistics into an interactive, glassmorphic showcase dashboard:

```bash
python week8/portfolio/portfolio_runner.py
```
This triggers the evaluation pipeline, generating mock evaluation graphs, card visuals, and compiles the showcase dashboard at `outputs/portfolio/index.html`.
