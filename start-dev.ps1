$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
if (-not $repoRoot) {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$backendPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$frontendDir = Join-Path $repoRoot 'dashboard'

if (-not (Test-Path $backendPython)) {
    throw "Missing virtual environment Python at '$backendPython'. Create .venv first."
}

if (-not (Test-Path $frontendDir)) {
    throw "Missing dashboard directory at '$frontendDir'."
}

$backendCommand = "Set-Location -LiteralPath '$repoRoot'; & '$backendPython' -m uvicorn src.dashboard.main:app --host 127.0.0.1 --port 8010"
$frontendCommand = "Set-Location -LiteralPath '$frontendDir'; npm run dev"

Start-Process -FilePath powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
    '-NoExit',
    '-Command',
    $backendCommand
) | Out-Null

Start-Process -FilePath powershell.exe -WorkingDirectory $frontendDir -ArgumentList @(
    '-NoExit',
    '-Command',
    $frontendCommand
) | Out-Null

Write-Host 'Started backend at http://127.0.0.1:8010 and frontend at http://127.0.0.1:5173.'
