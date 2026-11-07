# UFCash installer for Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SolvAI/ufcash/main/install.ps1 | iex"
# (or .\install.ps1 inside a clone). Installs uv if needed, Python 3.12 + dependencies in .\.venv, then downloads the data.
$ErrorActionPreference = "Stop"
$repo = "https://github.com/SolvAI/ufcash"
function Say($m) { Write-Host "> $m" -ForegroundColor Cyan }

# 1. the code (skip when run from inside a clone)
if (-not (Test-Path "ufcash.py")) {
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "ufcash.py"))) {
        Set-Location $PSScriptRoot
    } else {
        Say "Downloading UFCash into .\ufcash"
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git clone --depth 1 "$repo.git" ufcash
        } else {
            Invoke-WebRequest "$repo/archive/refs/heads/main.zip" -OutFile ufcash.zip
            Expand-Archive ufcash.zip -DestinationPath .
            Rename-Item ufcash-main ufcash
            Remove-Item ufcash.zip
        }
        Set-Location ufcash
    }
}

# 2. uv (fast Python package manager; brings its own Python)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 3. Python 3.12 + dependencies
Say "Installing Python 3.12 and the dependencies into .\.venv"
uv venv --quiet --allow-existing --python 3.12 .venv
if ($LASTEXITCODE) { throw "uv venv failed" }
uv pip install --quiet --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE) { throw "dependency install failed" }

# 4. one-time data setup
.\ufcash.cmd setup
Write-Host ""
Write-Host "Installed. In $(Get-Location), run:  .\ufcash card" -ForegroundColor Green
