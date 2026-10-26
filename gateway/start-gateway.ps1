# One-command launcher for the LiteLLM gateway.
# Fixes two verified Windows gotchas:
#   1) bare `litellm` is NOT on PATH (it lives in .venv\Scripts; venv must be
#      activated first) -- this script uses the absolute path instead.
#   2) litellm fetches a remote model-cost-map at startup; on bad proxies that
#      fetch stalls/retries with zero console output for a long time.
#      LITELLM_LOCAL_MODEL_COST_MAP=True skips it.
# NOTE: this file must stay pure ASCII. Windows PowerShell 5.1 parses .ps1
# files WITHOUT a BOM using the legacy ANSI codepage (GBK on zh-CN hosts);
# non-ASCII content causes random parse errors.
#
# Usage (run from anywhere, no cd needed):
#   powershell -ExecutionPolicy Bypass -File D:\projects\stock-agent\gateway\start-gateway.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Lite = Join-Path $Root ".venv\Scripts\litellm.exe"

if (-not (Test-Path $Lite)) {
    throw "litellm.exe not found. Run first: .venv\Scripts\pip.exe install -r gateway\requirements.txt"
}

# Upstream DashScope key: session env -> OS machine-level env
if (-not $env:DASHSCOPE_API_KEY) {
    $env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY', 'Machine')
}
if (-not $env:DASHSCOPE_API_KEY) {
    Write-Warning "DASHSCOPE_API_KEY not found (not even at machine level). The gateway will start, but upstream calls will 401."
}

# Business-side master key, persisted: generated once, reused forever.
# AGENT_API_KEY should copy this value. The file is git-ignored; never commit it.
$KeyFile = Join-Path $PSScriptRoot ".master_key"
if (-not (Test-Path $KeyFile)) {
    ('sk-gw' + [guid]::NewGuid().ToString('N').Substring(0, 28)) | Set-Content -NoNewline $KeyFile
}
$env:LITELLM_MASTER_KEY = (Get-Content $KeyFile -Raw).Trim()

# Skip the remote model-cost-map fetch at startup (raw.githubusercontent.com is
# unreachable through this machine's proxy: it RSTs every attempt).
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"

# DashScope mainland endpoint: bypass any ambient proxy, same default policy as
# the repo's own LLM/embedding clients (trust_env=False). System-proxy flakiness
# is the usual root cause of TLS handshake / connection errors on this machine.
if (-not $env:NO_PROXY) {
    $env:NO_PROXY = 'dashscope.aliyuncs.com'
    $env:no_proxy = 'dashscope.aliyuncs.com'
}

# Idempotency: if port 4000 is already serving, do not start a second instance.
try {
    $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 4000); $c.Close()
    Write-Host "Port 4000 already in use -- the gateway is probably already running."
    Write-Host "Business-side AGENT_API_KEY value: $KeyFile"
    exit 0
} catch { }

Write-Host "Starting gateway (ready in ~10s)..."
Write-Host "AGENT_API_KEY <-- value of: $KeyFile"
Write-Host "If outbound calls are killed by the system proxy, set `$env:NO_PROXY='dashscope.aliyuncs.com' first."
& $Lite --config (Join-Path $PSScriptRoot "config.yaml") --port 4000
