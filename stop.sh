#!/usr/bin/env bash
# =============================================================================
# stop.sh
# =======
# Shutdown script for the AI Fashion Assistant.
# Gracefully stops running containers/manifests.
# Logs all activities to logs/deploy.log.
# =============================================================================

set -eo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/deploy.log"
DEPLOY_TARGET="${DEPLOY_TARGET:-docker}"

mkdir -p "$LOG_DIR"

log_info() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo -e "\033[0;32m$msg\033[0m"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo -e "\033[0;31m$msg\033[0m" >&2
    echo "$msg" >> "$LOG_FILE"
}

main() {
    log_info "Stopping platform services (Target: ${DEPLOY_TARGET})..."

    if [ "$DEPLOY_TARGET" = "docker" ]; then
        log_info "Tearing down Docker containers..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down 2>&1 | tee -a "$LOG_FILE"
        else
            docker compose down 2>&1 | tee -a "$LOG_FILE"
        fi

    elif [ "$DEPLOY_TARGET" = "k8s" ]; then
        log_info "Scaling Kubernetes deployments down to 0 replicas..."
        kubectl scale deployment --replicas=0 --all 2>&1 | tee -a "$LOG_FILE"
        
    else
        log_error "Unknown DEPLOY_TARGET: ${DEPLOY_TARGET}."
        exit 1
    fi

    log_info "Platform services stopped successfully."
}

main "$@"
