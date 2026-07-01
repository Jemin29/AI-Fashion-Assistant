# =============================================================================
# restart.ps1
# ===========
# Windows PowerShell restart script for the AI Fashion Assistant.
# Executes a clean stop and start sequence.
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

Log-Info "Restart sequence triggered on Windows..."

# Execute stop script
$StopScript = Join-Path $WorkspaceRoot "stop.ps1"
if (Test-Path $StopScript) {
    & $StopScript
} else {
    Log-Info "stop.ps1 not found. Skipping clean teardown."
}

# Settle
Start-Sleep -Seconds 2

# Execute start script
$StartScript = Join-Path $WorkspaceRoot "start.ps1"
if (Test-Path $StartScript) {
    & $StartScript
} else {
    Log-Info "start.ps1 not found. Skipping platform start."
}

Log-Info "Platform restarted successfully!"
