#!/usr/bin/env bash
# =============================================================================
# restart.sh
# ==========
# Restart script for the AI Fashion Assistant.
# Executes a clean stop and start sequence.
# Logs all activities to logs/deploy.log.
# =============================================================================

set -eo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/deploy.log"

mkdir -p "$LOG_DIR"

log_info() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo -e "\033[0;32m$msg\033[0m"
    echo "$msg" >> "$LOG_FILE"
}

main() {
    log_info "Restart sequence triggered..."

    # Call stop script
    if [ -f "${WORKSPACE_ROOT}/stop.sh" ]; then
        bash "${WORKSPACE_ROOT}/stop.sh" "$@"
    else
        log_info "stop.sh not found. Skipping clean teardown."
    fi

    # Give the system 2 seconds to settle
    sleep 2

    # Call start script
    if [ -f "${WORKSPACE_ROOT}/start.sh" ]; then
        bash "${WORKSPACE_ROOT}/start.sh" "$@"
    else
        log_info "start.sh not found. Skipping platform start."
    fi

    log_info "Platform restarted successfully!"
}

main "$@"
