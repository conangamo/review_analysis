param(
    [string]$Category = "electronics",
    [int]$LimitReviews = 200,
    [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Week 2 Demo Pipeline (mini)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Category      : $Category"
Write-Host "Limit Reviews : $LimitReviews"
Write-Host ""

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw ".venv not found. Run: python -m venv .venv"
}

. .\.venv\Scripts\Activate.ps1
Write-Host "Activated venv: $($env:VIRTUAL_ENV)" -ForegroundColor Green

Write-Host ""
Write-Host "[1/4] Run unit + integration tests..." -ForegroundColor Yellow
pytest tests -q

Write-Host ""
Write-Host "[2/4] Parse/load sample-limited data..." -ForegroundColor Yellow
python scripts/parse_data.py --category $Category --limit-products 50 --limit-reviews $LimitReviews

if (-not $SkipAnalysis) {
    Write-Host ""
    Write-Host "[3/4] Run analysis + summaries..." -ForegroundColor Yellow
    python scripts/run_analysis.py --category $Category --limit $LimitReviews --checkpoint-name "${Category}_week2_demo"
    python scripts/generate_summaries.py --category $Category
} else {
    Write-Host ""
    Write-Host "[3/4] Skip analysis as requested (--SkipAnalysis)." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "[4/4] Launch UI..." -ForegroundColor Yellow
Write-Host "Run manually in same terminal: streamlit run src/ui/app.py" -ForegroundColor Green
Write-Host ""
Write-Host "Done. Week 2 demo flow completed." -ForegroundColor Cyan
