# One-time setup: Python venv + backend deps, frontend npm packages.
# Usage (from repo root or app\):  .\app\setup.ps1      add -Cuda for a CUDA 12.1 PyTorch build
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Push-Location "$root\backend"
try {
    if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
    .\.venv\Scripts\python -m pip install --upgrade pip
    if ($Cuda) {
        .\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    } else {
        .\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    }
    .\.venv\Scripts\python -m pip install -r requirements.txt
} finally { Pop-Location }

Push-Location "$root\frontend"
try { npm install } finally { Pop-Location }
Write-Host "`nSetup complete. Dev: .\app\run_dev.ps1   Prod: .\app\run_prod.ps1" -ForegroundColor Green
