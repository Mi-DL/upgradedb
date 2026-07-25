$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = $env:PYTHON
if (-not $Python) {
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
}

Push-Location $Root
try {
    & $Python "tools\release_smoke.py" @args
    if ($LASTEXITCODE -ne 0) {
        throw "release smoke failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
