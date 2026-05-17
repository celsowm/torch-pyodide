param(
  [string]$Python = "python"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$pyodidePath = Join-Path $repoRoot "external/pyodide"

if (-not (Test-Path $pyodidePath)) {
  throw "Submodule external/pyodide not found."
}

Push-Location $pyodidePath
try {
  & $Python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -e pyodide-build
  make
} finally {
  Pop-Location
}

