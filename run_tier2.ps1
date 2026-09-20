# Text-only Bedrock models on the three text tasks (allocation, fire danger, tool use), 2026-09-18. Nineteen
# models: seventeen with tool support run all three; the two that failed the tool probe (Llama 3.1 8B, DeepSeek R1)
# run allocation and fire danger only. Mixtral 8x7B is out because its endpoint rejects system messages. One
# detached chain per model, at most $MaxParallel chains at once. -Pilot runs a ten-item allocation pilot per model
# in the foreground instead (truncation and parse check). The three reasoning models run at AI4FIRE_MAX_OUT=8192,
# recorded in every row's usage.max_out. Credentials come from AWS_BEARER_TOKEN_BEDROCK in the environment.
#
# Usage: pwsh -File run_tier2.ps1 [-Pilot] [-MaxParallel 7]
#        pwsh -File run_tier2.ps1 -Chain <bedrock:id> -Tasks 3|2   (internal: one chain in the foreground)
param([string]$Chain = '', [int]$Tasks = 3, [switch]$Pilot, [int]$MaxParallel = 7)
$ErrorActionPreference = 'Continue'
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$root = $PSScriptRoot
# Standard cap (1,536 output tokens, the benchmark cap). Mixtral 8x7B is out: it rejects system messages.
$withTools = @(
    'amazon.nova-micro-v1:0', 'us.meta.llama3-3-70b-instruct-v1:0', 'us.meta.llama3-1-70b-instruct-v1:0',
    'mistral.mistral-small-2402-v1:0', 'mistral.devstral-2-123b', 'qwen.qwen3-32b-v1:0', 'qwen.qwen3-next-80b-a3b',
    'qwen.qwen3-coder-30b-a3b-v1:0', 'deepseek.v3.2', 'openai.gpt-oss-120b-1:0', 'openai.gpt-oss-20b-1:0',
    'zai.glm-5', 'zai.glm-4.7', 'zai.glm-4.7-flash', 'nvidia.nemotron-super-3-120b'
)
$noTools = @('us.meta.llama3-1-8b-instruct-v1:0')
# Reasoning models whose emitted reasoning exhausted the 1,536 cap on most bare allocation pilot items (MiniMax M2.5 and
# Kimi K2 Thinking parsed 2 of 10, DeepSeek R1 parsed 6): run at AI4FIRE_MAX_OUT=8192, recorded in each of their rows'
# usage.max_out and reported as a documented variant.
$reasoningTools = @('minimax.minimax-m2.5', 'moonshot.kimi-k2-thinking')
$reasoningNoTools = @('us.deepseek.r1-v1:0')
Set-Location $root

if ($Chain -ne '') {
    $m = $Chain
    Write-Output "=== chain $m start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    & $py run_allocation.py --models $m --workers 4 2>&1 | Select-Object -Last 6
    Write-Output "--- mesogeos $(Get-Date -Format 'HH:mm:ss')"
    & $py run_mesogeos.py --models $m --workers 4 2>&1 | Select-Object -Last 6
    if ($Tasks -ge 3) {
        Write-Output "--- tooluse $(Get-Date -Format 'HH:mm:ss')"
        & $py run_tooluse.py --models $m --workers 4 2>&1 | Select-Object -Last 12
    }
    Write-Output "=== chain $m DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    exit 0
}

if ($Pilot) {
    foreach ($id in ($withTools + $noTools)) {
        Write-Output "=== pilot bedrock:$id $(Get-Date -Format 'HH:mm:ss')"
        & $py run_allocation.py --models "bedrock:$id" --limit 10 --workers 4 2>&1 | Select-String -Pattern '"run"|Traceback|Error' | ForEach-Object { $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) }
    }
    Write-Output "PILOT DONE $(Get-Date -Format 'HH:mm:ss')"
    exit 0
}

function Running-Chains {
    (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_tier2.ps1 -Chain' }).Count
}
$queue = @()
foreach ($id in $withTools) { $queue += @{ id = $id; tasks = 3; cap = '' } }
foreach ($id in $noTools) { $queue += @{ id = $id; tasks = 2; cap = '' } }
foreach ($id in $reasoningTools) { $queue += @{ id = $id; tasks = 3; cap = '8192' } }
foreach ($id in $reasoningNoTools) { $queue += @{ id = $id; tasks = 2; cap = '8192' } }
foreach ($job in $queue) {
    while ((Running-Chains) -ge $MaxParallel) { Start-Sleep -Seconds 30 }
    if ($job.cap -ne '') { $env:AI4FIRE_MAX_OUT = $job.cap } else { Remove-Item Env:AI4FIRE_MAX_OUT -ErrorAction SilentlyContinue }
    $m = 'bedrock:' + $job.id
    $safe = ($m -replace '[^A-Za-z0-9._-]', '_')
    $log = Join-Path $root ("logs\tier2-" + $safe + "-2026-09-18.log")
    $err = Join-Path $root ("logs\tier2-" + $safe + "-2026-09-18.err")
    Start-Process -FilePath 'pwsh' -WorkingDirectory $root -WindowStyle Hidden `
        -ArgumentList @('-NoProfile', '-File', (Join-Path $root 'run_tier2.ps1'), '-Chain', $m, '-Tasks', $job.tasks) `
        -RedirectStandardOutput $log -RedirectStandardError $err
    Write-Output "launched $m (tasks=$($job.tasks) cap=$($job.cap)) $(Get-Date -Format 'HH:mm:ss')"
    Start-Sleep -Seconds 5
}
Remove-Item Env:AI4FIRE_MAX_OUT -ErrorAction SilentlyContinue
Write-Output "ALL LAUNCHED $(Get-Date -Format 'HH:mm:ss')"
