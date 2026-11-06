$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is required for npm start."
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $python)) {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
}

$env:PYYAML_FORCE_LIBYAML = "0"
& $python -m pip install -r requirements.txt
$installResult = $LASTEXITCODE
Remove-Item Env:\PYYAML_FORCE_LIBYAML
if ($installResult -ne 0) { throw "Python dependency installation failed." }
$env:PATH = "$(Split-Path -Parent $python);$env:PATH"

if (-not (Test-Path (Join-Path $root ".dvc\config"))) {
    & $python -m dvc init
    if ($LASTEXITCODE -ne 0) { throw "DVC initialization failed." }
}

& $python -m dvc repro
if ($LASTEXITCODE -ne 0) { throw "MLOps pipeline execution failed." }

& $python -m playwright install chromium
if ($LASTEXITCODE -ne 0) { throw "Chromium installation failed." }

if (-not (Test-Path (Join-Path $root "node_modules"))) {
    npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw "Node dependency installation failed." }
}

npm.cmd start
exit $LASTEXITCODE
