# =============================================================================
# deploy.ps1
# ==========
# Windows PowerShell deployment bootstrapper for the AI Fashion Assistant.
# Builds images, spins up containers, and runs service health validation.
# Logs all activities to logs/deploy.log.
# =============================================================================

$ErrorActionPreference = "Stop"

$WorkspaceRoot = $PSScriptRoot
$LogDir = Join-Path $WorkspaceRoot "logs"
$LogFile = Join-Path $LogDir "deploy.log"

# Ensure logs directory exists
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

Log-Info "Starting deployment automation on Windows..."

# Verify docker availability
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Log-Error "docker command is missing. Please install Docker."
    exit 1
}

# Determine docker compose executable
$ComposeCmd = "docker-compose"
if (-not (Get-Command $ComposeCmd -ErrorAction SilentlyContinue)) {
    $ComposeCmd = "docker"
}

Log-Info "Building container images..."
if ($ComposeCmd -eq "docker") {
    docker compose build *>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    docker-compose build *>&1 | Tee-Object -FilePath $LogFile -Append
}

Log-Info "Spinning up services in detached mode..."
if ($ComposeCmd -eq "docker") {
    docker compose up -d *>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    docker-compose up -d *>&1 | Tee-Object -FilePath $LogFile -Append
}

Log-Info "Waiting for FastAPI backend service (http://localhost:8001) to pass health checks..."
$healthy = $false
for ($i = 1; $i -le 20; $i++) {
    try {
        # Query health endpoint
        $resp = Invoke-RestMethod -Uri "http://localhost:8001/api/v1/health" -Method Get -TimeoutSec 3
        if ($resp) {
            $healthy = $true
            break
        }
    } catch {
        Log-Warn "Backend not ready yet. Retrying in 3 seconds ($i/20)..."
        Start-Sleep -Seconds 3
    }
}

if ($healthy) {
    Log-Info "REST API backend is healthy!"
} else {
    Log-Error "FastAPI backend health check timed out! Check docker logs."
    exit 1
}

Log-Info "Verifying running containers status:"
if ($ComposeCmd -eq "docker") {
    docker compose ps *>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    docker-compose ps *>&1 | Tee-Object -FilePath $LogFile -Append
}

Log-Info "Deployment completed successfully!"
