# Pixel Plan AI - full-stack launcher.
# Starts the three services, each in its own window so logs stay visible:
#   1) Python backend      :8765  (generation / validation / repair loop)  - required
#   2) hfagent service     :8801  (candidate drawing + tracing)            - optional; Image/Trace modes degrade without it
#   3) Studio frontend     :3000  (Next.js, serves the built app)
#
# Usage:  powershell -ExecutionPolicy Bypass -File start-all.ps1
$ErrorActionPreference = "Stop"
$PixelPlanRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PixelPlanRoot
$HfagentPython = Join-Path $RepoRoot "hfagent\.venv\Scripts\python.exe"

# --- pre-flight: kill stale instances -------------------------------------------------
# Windows lets several ThreadingHTTPServer processes bind the same port (SO_REUSEADDR),
# so a stale process can keep answering with old code. Clear them all first.
Write-Host "Stopping stale services..." -ForegroundColor DarkGray
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*backend\server.py*" -or $_.CommandLine -like "*trace_server*" } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -Confirm:$false } catch {} }
Get-CimInstance Win32_Process -Filter "Name='node.exe'" |
    Where-Object { $_.CommandLine -like "*next*start*" -or $_.CommandLine -like "*next*dev*" } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -Confirm:$false } catch {} }
Start-Sleep -Seconds 1

# --- frontend build check --------------------------------------------------------------
if (-not (Test-Path -LiteralPath (Join-Path $PixelPlanRoot "web\.next"))) {
    Write-Host "No web build found - running npm run build once (first launch)..." -ForegroundColor Yellow
    Push-Location (Join-Path $PixelPlanRoot "web")
    npm run build
    Pop-Location
}

# --- launch ----------------------------------------------------------------------------
Write-Host "Starting backend (:8765)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle = 'pixel-plan backend :8765'; Set-Location '$PixelPlanRoot'; python -B backend\server.py"

if (Test-Path -LiteralPath $HfagentPython) {
    Write-Host "Starting hfagent service (:8801)..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = 'hfagent trace+draw :8801'; Set-Location '$RepoRoot'; & '$HfagentPython' -m hfagent.tools.trace_server"
} else {
    Write-Host "hfagent venv not found ($HfagentPython) - skipping; Image/Trace modes will degrade." -ForegroundColor Yellow
}

Write-Host "Starting studio (:3000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle = 'pixel-plan studio :3000'; Set-Location '$(Join-Path $PixelPlanRoot "web")'; npx next start -p 3000"

# --- health probes ---------------------------------------------------------------------
function Wait-Healthy($Name, $Url, $TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            Write-Host ("  {0,-18} OK   {1}" -f $Name, $Url) -ForegroundColor Green
            return $true
        } catch { Start-Sleep -Milliseconds 700 }
    }
    Write-Host ("  {0,-18} DOWN {1}" -f $Name, $Url) -ForegroundColor Red
    return $false
}

Write-Host "`nWaiting for services..." -ForegroundColor DarkGray
$backendOk = Wait-Healthy "backend"  "http://127.0.0.1:8765/api/health" 20
$null      = Wait-Healthy "hfagent"  "http://127.0.0.1:8801/health"     20
$studioOk  = Wait-Healthy "studio"   "http://127.0.0.1:3000"            40

if ($backendOk -and $studioOk) {
    Write-Host "`nPixel Plan Studio ready -> http://127.0.0.1:3000" -ForegroundColor Green
    Start-Process "http://127.0.0.1:3000"
} else {
    Write-Host "`nSomething did not come up - check the service windows for errors." -ForegroundColor Red
    exit 1
}
