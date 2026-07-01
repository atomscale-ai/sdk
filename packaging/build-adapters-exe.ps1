# Build the `atomscale-adapters` host executable with PyInstaller (Windows).
#
# Usage:   powershell -ExecutionPolicy Bypass -File packaging\build-adapters-exe.ps1
# Output:  dist\atomscale-adapters.exe
#
# Requires Python 3.10-3.12 and a Rust toolchain (https://rustup.rs) on PATH,
# since installing the SDK from source compiles the Rust extension.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Venv = if ($env:VENV) { $env:VENV } else { ".venv-pyinstaller" }
python -m venv $Venv
& "$Venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install . "pyinstaller>=6.0,<7"

pyinstaller --clean --noconfirm packaging\atomscale-adapters.spec

Write-Host "Built: $RepoRoot\dist\atomscale-adapters.exe"
& ".\dist\atomscale-adapters.exe" list | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Smoke test 'list' FAILED (exit $LASTEXITCODE)" }
Write-Host "Smoke test 'list' OK"
