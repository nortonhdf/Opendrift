# Rebuild all precomputed Campos products with the opendrift environment.
#
#   .\main\rebuild_all.ps1                  # show the plan, change nothing
#   .\main\rebuild_all.ps1 --fresh          # delete manifests + rebuild ALL (~3.5-4 h)
#   .\main\rebuild_all.ps1 --fresh --only scenarios
#   .\main\rebuild_all.ps1 --resume         # continue an interrupted rebuild
#   .\main\rebuild_all.ps1 --resume --only ensemble,risk,beaching
#
# Any arguments are passed straight through to main/scripts/rebuild_all.py.
# Portable: searches common miniforge/miniconda/anaconda locations automatically.

$ErrorActionPreference = "Stop"

# Search for python.exe in the opendrift conda env across common install bases
$condaBases = @(
    "$env:USERPROFILE\miniforge3",
    "$env:LOCALAPPDATA\miniforge3",
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\anaconda3",
    "C:\ProgramData\miniforge3",
    "C:\ProgramData\miniconda3"
)

$py = $null
foreach ($base in $condaBases) {
    $candidate = "$base\envs\opendrift\python.exe"
    if (Test-Path $candidate) { $py = $candidate; break }
}

if (-not $py) {
    # Fall back to conda run if python not found directly
    $condaBat = $null
    foreach ($base in $condaBases) {
        $c = "$base\condabin\conda.bat"
        if (Test-Path $c) { $condaBat = $c; break }
    }
    if ($condaBat) {
        Write-Host "opendrift python.exe not found directly; using 'conda run -n opendrift'."
        $root = Split-Path $PSScriptRoot -Parent
        Set-Location $root
        $env:PYTHONUNBUFFERED = "1"
        $env:PYTHONUTF8 = "1"
        & $condaBat run -n opendrift python "main\scripts\rebuild_all.py" @args
        exit $LASTEXITCODE
    }
    Write-Error "Could not locate the 'opendrift' conda env. Activate it manually and run: python main\scripts\rebuild_all.py $args"
    exit 1
}

# Repo root = parent of this script's folder (main/)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

& $py "main\scripts\rebuild_all.py" @args
