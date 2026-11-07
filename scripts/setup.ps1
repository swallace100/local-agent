# Sets up the local agent. Safe to run again after changing the config.
# Prerequisites: Docker Desktop, Node.js, Python 3.11+ and Ollama (winget install Ollama.Ollama).
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) { throw "Ollama not found. Install it with: winget install Ollama.Ollama" }

if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    Write-Host "Installing OpenCode..."
    npm install -g opencode-ai
    if ($LASTEXITCODE -ne 0) { throw "OpenCode install failed" }
}

try {
    Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Starting Ollama..."
    Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 5
}

Write-Host "Downloading the model if needed (about 6.6 GB)..."
& $ollama pull qwen3.5:9b
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
& $ollama create agent -f ollama/Modelfile
if ($LASTEXITCODE -ne 0) { throw "Model creation failed" }

Write-Host "Installing the web tools..."
Push-Location tools/web-search
npm install --no-audit --no-fund
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0) { throw "npm install failed" }

Write-Host "Starting SearXNG..."
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "Could not start SearXNG. Is Docker Desktop running?" }

Write-Host "Installing the LangGraph agent..."
& $ollama pull nomic-embed-text
if ($LASTEXITCODE -ne 0) { throw "Embedding model download failed" }
docker pull python:3.13-slim
if ($LASTEXITCODE -ne 0) { throw "Sandbox image download failed" }
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment. Python 3.11+ is required." }
}
& .venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "Python install failed" }
& .venv\Scripts\agent.exe index
if ($LASTEXITCODE -ne 0) { throw "Indexing the knowledge folder failed" }

# Install the config globally so the agent works in any project folder.
$globalDir = Join-Path $HOME ".config\opencode"
$agentsDir = Join-Path $globalDir "agents"
New-Item -ItemType Directory -Force $agentsDir | Out-Null

$globalConfig = Join-Path $globalDir "opencode.jsonc"
if ((Test-Path $globalConfig) -and -not (Select-String -Path $globalConfig -SimpleMatch "tools/web-search/server.mjs" -Quiet)) {
    Copy-Item $globalConfig "$globalConfig.bak" -Force
    Write-Host "Backed up your existing config to $globalConfig.bak"
}

$serverPath = (Join-Path $repo "tools\web-search\server.mjs").Replace("\", "/")
$config = (Get-Content "opencode.jsonc" -Raw).Replace('"tools/web-search/server.mjs"', "`"$serverPath`"")
[IO.File]::WriteAllText($globalConfig, $config, [Text.UTF8Encoding]::new($false))
Copy-Item ".opencode\agents\*.md" $agentsDir -Force

Write-Host ""
Write-Host "Ready."
Write-Host "Unattended agent: .venv\Scripts\agent.exe run `"your task`""
Write-Host "Interactive agent: open any project folder and run opencode"
