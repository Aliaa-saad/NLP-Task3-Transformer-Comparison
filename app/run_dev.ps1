# Dev mode: FastAPI with --reload (new window) + Vite dev server on http://localhost:5173
# Usage: .\app\run_dev.ps1            (add -Demo to run with random weights if artifacts are missing)
param([switch]$Demo)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Backend venv not found. Run .\app\setup.ps1 first." }

$envLine = if ($Demo) { "`$env:DEMO_RANDOM_WEIGHTS='1'; " } else { "" }
$backendCmd = "${envLine}Set-Location '$root\backend'; & '$py' -m uvicorn app.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Write-Host "Backend starting on http://127.0.0.1:8000 (separate window)" -ForegroundColor Cyan

Push-Location "$root\frontend"
try {
    if (-not (Test-Path "node_modules")) { npm install }
    npm run dev
} finally { Pop-Location }
