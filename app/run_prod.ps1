# Production mode: build the React app, then serve API + UI from FastAPI on http://localhost:8000
# Usage: .\app\run_prod.ps1  [-SkipBuild] [-Port 8000]
param([switch]$SkipBuild, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Backend venv not found. Run .\app\setup.ps1 first." }

if (-not $SkipBuild) {
    Push-Location "$root\frontend"
    try {
        if (-not (Test-Path "node_modules")) { npm install }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
    } finally { Pop-Location }
}

Push-Location "$root\backend"
try {
    Write-Host "Serving app on http://localhost:$Port" -ForegroundColor Green
    & $py -m uvicorn app.main:app --host 127.0.0.1 --port $Port
} finally { Pop-Location }
