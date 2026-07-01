# =============================================================================
# Windows PowerShell deployment bootstrapper for the AI Fashion Assistant.
# =============================================================================
param (
    [ValidateSet("docker", "k8s")]
    [string]$Mode = "docker"
)

Clear-Host
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host " Starting AI Fashion Assistant Platform Deployment (PowerShell Bootstrap)     " -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan

# Verify Docker availability
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
    Write-Error "[!] Docker CLI is required but not installed on this system. Aborting."
    exit 1
}

if ($Mode -eq "docker") {
    Write-Host "[*] Launching multi-container services with Docker Compose..." -ForegroundColor Yellow
    docker compose -f week8/deployment/docker/docker-compose.prod.yml up -d --build
    
    Write-Host "[+] Services started successfully." -ForegroundColor Green
    Write-Host "[*] Gradio Portal: http://localhost:7860" -ForegroundColor Green
    Write-Host "[*] FastAPI Server: http://localhost:8000" -ForegroundColor Green
    Write-Host "[*] Health Gateway: http://localhost:8000/api/v1/health" -ForegroundColor Green

}
elseif ($Mode -eq "k8s") {
    # Verify Kubectl availability
    $kubectlCmd = Get-Command kubectl -ErrorAction SilentlyContinue
    if (-not $kubectlCmd) {
        Write-Error "[!] kubectl is required for Kubernetes deployment. Aborting."
        exit 1
    }
    
    Write-Host "[*] Applying manifests to target Kubernetes namespace..." -ForegroundColor Yellow
    
    # Initialize namespace
    kubectl create namespace fashion-ai --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy ConfigMaps, Secrets, Services, Deployments, and Auto-Scalers
    kubectl apply -f week8/deployment/kubernetes/configmap.yaml
    kubectl apply -f week8/deployment/kubernetes/secret.yaml
    kubectl apply -f week8/deployment/kubernetes/service.yaml
    kubectl apply -f week8/deployment/kubernetes/deployment.yaml
    kubectl apply -f week8/deployment/kubernetes/ingress.yaml
    kubectl apply -f week8/deployment/kubernetes/hpa.yaml
    
    Write-Host "[+] Kubernetes manifests applied successfully." -ForegroundColor Green
    Write-Host "[*] Verify deployment status using: kubectl get all -n fashion-ai" -ForegroundColor Green
}
