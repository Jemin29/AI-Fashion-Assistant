#!/usr/bin/env bash
# =============================================================================
# deploy.sh
# =========
# Master deployment bootstrapper for the AI Fashion Assistant.
# Supports local Docker Compose and production Kubernetes deployment modes.
# Logs all activities to logs/deploy.log.
# =============================================================================

set -eo pipefail

# Configurations
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/deploy.log"
DEPLOY_TARGET="${DEPLOY_TARGET:-docker}" # docker | k8s

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

log_info() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo -e "\033[0;32m$msg\033[0m"
    echo "$msg" >> "$LOG_FILE"
}

log_warn() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $1"
    echo -e "\033[0;33m$msg\033[0m"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo -e "\033[0;31m$msg\033[0m" >&2
    echo "$msg" >> "$LOG_FILE"
}

# Verification helpers
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "docker command is missing. Please install Docker."
        exit 1
    fi
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "docker-compose command is missing. Please install docker-compose."
        exit 1
    fi
}

check_kubernetes() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl command is missing. Please install kubectl."
        exit 1
    fi
}

# Main deployment pipeline
main() {
    log_info "Starting deployment automation (Target: ${DEPLOY_TARGET})..."

    if [ "$DEPLOY_TARGET" = "docker" ]; then
        check_docker
        
        log_info "Building container images..."
        if command -v docker-compose &> /dev/null; then
            docker-compose build 2>&1 | tee -a "$LOG_FILE"
        else
            docker compose build 2>&1 | tee -a "$LOG_FILE"
        fi

        log_info "Spinning up services in detached mode..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d 2>&1 | tee -a "$LOG_FILE"
        else
            docker compose up -d 2>&1 | tee -a "$LOG_FILE"
        fi

        # Service verification loop
        log_info "Waiting for FastAPI backend service (http://localhost:8001) to pass health checks..."
        local healthy=false
        for i in {1..20}; do
            if curl -s -f http://localhost:8001/api/v1/health > /dev/null; then
                healthy=true
                break
            fi
            log_warn "Backend not ready yet. Retrying in 3 seconds ($i/20)..."
            sleep 3
        done

        if [ "$healthy" = true ]; then
            log_info "REST API backend is healthy!"
        else
            log_error "FastAPI backend health check timed out! Check docker logs."
            exit 1
        fi

        log_info "Verifying running containers status:"
        if command -v docker-compose &> /dev/null; then
            docker-compose ps 2>&1 | tee -a "$LOG_FILE"
        else
            docker compose ps 2>&1 | tee -a "$LOG_FILE"
        fi

    elif [ "$DEPLOY_TARGET" = "k8s" ]; then
        check_kubernetes
        
        log_info "Applying Kubernetes manifests from deployment/kubernetes/..."
        kubectl apply -f "${WORKSPACE_ROOT}/deployment/kubernetes" 2>&1 | tee -a "$LOG_FILE"

        log_info "Waiting for fashion-api deployment to rollout..."
        if kubectl rollout status deployment/fashion-api --timeout=90s 2>&1 | tee -a "$LOG_FILE"; then
            log_info "Kubernetes deployment completed successfully."
        else
            log_error "Kubernetes deployment failed or timed out."
            exit 1
        fi
        
        log_info "Current Kubernetes resources:"
        kubectl get all -n default 2>&1 | tee -a "$LOG_FILE"

    else
        log_error "Unknown DEPLOY_TARGET: ${DEPLOY_TARGET}. Supported: docker | k8s"
        exit 1
    fi

    log_info "Deployment completed successfully!"
}

main "$@"
