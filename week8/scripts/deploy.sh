#!/bin/bash
# =============================================================================
# Linux Bash deployment bootstrapper for the AI Fashion Assistant platform.
# =============================================================================
set -e

echo "============================================================================="
echo " Starting AI Fashion Assistant Platform Deployment (Bash Bootstrap)          "
echo "============================================================================="

# Ensure necessary runtimes exist
command -v docker >/dev/null 2>&1 || { echo "[!] Docker cli is required but not installed. Aborting." >&2; exit 1; }

# Parse deployment target mode (defaulting to docker compose)
MODE=${1:-docker}

if [ "$MODE" == "docker" ]; then
    echo "[*] Launching multi-container services with Docker Compose..."
    # Build and start services in detached mode
    docker compose -f week8/deployment/docker/docker-compose.prod.yml up -d --build
    echo "[+] Services started successfully."
    echo "[*] Gradio Portal: http://localhost:7860"
    echo "[*] FastAPI Server: http://localhost:8000"
    echo "[*] Health Gateway: http://localhost:8000/api/v1/health"

elif [ "$MODE" == "k8s" ]; then
    command -v kubectl >/dev/null 2>&1 || { echo "[!] kubectl is required for Kubernetes deployment. Aborting." >&2; exit 1; }
    
    echo "[*] Applying manifests to target Kubernetes cluster..."
    
    # Initialize namespace
    kubectl create namespace fashion-ai --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy ConfigMaps, Secrets, Services, Deployments, and Auto-Scalers
    kubectl apply -f week8/deployment/kubernetes/configmap.yaml
    kubectl apply -f week8/deployment/kubernetes/secret.yaml
    kubectl apply -f week8/deployment/kubernetes/service.yaml
    kubectl apply -f week8/deployment/kubernetes/deployment.yaml
    kubectl apply -f week8/deployment/kubernetes/ingress.yaml
    kubectl apply -f week8/deployment/kubernetes/hpa.yaml
    
    echo "[+] Kubernetes manifests applied successfully."
    echo "[*] Verify deployment status using: kubectl get all -n fashion-ai"

else
    echo "[!] Invalid deployment mode: '$MODE'."
    echo "[*] Usage: ./deploy.sh [docker | k8s]"
    exit 1
fi
