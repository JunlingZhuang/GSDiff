$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath "public\app.js")) {
    tsc -p tsconfig.json
}

python -B backend\server.py
