# =============================================================================
# start.ps1
# =========
# Windows PowerShell startup script for the AI Fashion Assistant.
# Starts stopped containers and validates health status.
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

function Log-Warn($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formatted = "[$timestamp] [WARN] $msg"
    Write-Host $formatted -ForegroundColor Yellow
    Add-Content -Path $LogFile -Value $formatted
}

function Log-Error($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formatted = "[$timestamp] [ERROR] $msg"
    Write-Host $formatted -ForegroundColor Red
    Add-Content -Path $LogFile -Value $formatted
}

Log-Info "Starting platform services on Windows..."

$ComposeCmd = "docker-compose"
if (-not (Get-Command $ComposeCmd -ErrorAction SilentlyContinue)) {
    $ComposeCmd = "docker"
}

Log-Info "Starting containers..."
if ($ComposeCmd -eq "docker") {
    docker compose up -d *>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    docker-compose up -d *>&1 | Tee-Object -FilePath $LogFile -Append
}

Log-Info "Waiting for FastAPI backend service (http://localhost:8001) to pass health checks..."
$healthy = $false
for ($i = 1; $i -le 15; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8001/api/v1/health" -Method Get -TimeoutSec 3
        if ($resp) {
            $healthy = $true
            break
        }
    } catch {
        Log-Warn "Backend not ready yet. Retrying in 3 seconds ($i/15)..."
        Start-Sleep -Seconds 3
    }
}

if ($healthy) {
    Log-Info "REST API backend is healthy!"
} else {
    Log-Error "FastAPI backend health check timed out! Check docker logs."
    exit 1
}

Log-Info "Platform services started successfully!"
