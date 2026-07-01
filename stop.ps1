# =============================================================================
# stop.ps1
# ========
# Windows PowerShell shutdown script for the AI Fashion Assistant.
# Gracefully stops running containers.
# Logs all activities to logs/deploy.log.
# =============================================================================

$ErrorActionPreference = "Stop"

$WorkspaceRoot = $PSScriptRoot
$LogDir = Join-Path $WorkspaceRoot "logs"
$LogFile = Join-Path $LogDir "deploy.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Log-Info($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formatted = "[$timestamp] [INFO] $msg"
    Write-Host $formatted -ForegroundColor Green
    Add-Content -Path $LogFile -Value $formatted
}

function Log-Error($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formatted = "[$timestamp] [ERROR] $msg"
    Write-Host $formatted -ForegroundColor Red
    Add-Content -Path $LogFile -Value $formatted
}

Log-Info "Stopping platform services on Windows..."

$ComposeCmd = "docker-compose"
if (-not (Get-Command $ComposeCmd -ErrorAction SilentlyContinue)) {
    $ComposeCmd = "docker"
}

Log-Info "Tearing down Docker containers..."
if ($ComposeCmd -eq "docker") {
    docker compose down *>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    docker-compose down *>&1 | Tee-Object -FilePath $LogFile -Append
}

Log-Info "Platform services stopped successfully."
