# Rebuild all precomputed Campos products with the opendrift environment.
#
#   .\main\rebuild_all.ps1                  # show the plan, change nothing
#   .\main\rebuild_all.ps1 --fresh          # delete manifests + rebuild ALL (~3.5-4 h)
#   .\main\rebuild_all.ps1 --fresh --only scenarios
#   .\main\rebuild_all.ps1 --resume         # continue an interrupted rebuild
#
# Any arguments are passed straight through to main/scripts/rebuild_all.py.

$ErrorActionPreference = "Stop"

$py = "C:\Users\nfreitas\AppData\Local\miniforge3\envs\opendrift\python.exe"
if (-not (Test-Path $py)) {
    Write-Warning "opendrift env python not found at $py - falling back to 'python' on PATH"
    $py = "python"
}

# Repo root = parent of this script's folder (main/)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

& $py "main\scripts\rebuild_all.py" @args