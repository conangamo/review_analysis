param(
    [string]$Category = "electronics",
    [int]$AnalysisLimit = 1000
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Open UI (Demo Ready Mode)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw ".venv not found. Run: python -m venv .venv"
}

. .\.venv\Scripts\Activate.ps1
Write-Host "Activated: $($env:VIRTUAL_ENV)" -ForegroundColor Green

python scripts/prepare_demo.py --category $Category --analysis-limit $AnalysisLimit --auto-fix

Write-Host ""
Write-Host "Starting Streamlit UI..." -ForegroundColor Yellow
streamlit run src/ui/app.py
