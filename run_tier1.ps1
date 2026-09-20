# Full five-task runs for the seven Bedrock models that passed the capability probe and the ten-item pilots on
# 2026-09-18. One detached chain per model, so the seven run in parallel (Bedrock throttles per model id);
# inside a chain the runners go one after another: allocation, fire danger, smoke (run_all.py), then tool use,
# then aerial question answering. Credentials come from AWS_BEARER_TOKEN_BEDROCK in the environment.
#
# Usage: pwsh -File run_tier1.ps1            (launches all seven and returns)
#        pwsh -File run_tier1.ps1 -Chain <bedrock:id>   (internal: runs one chain in the foreground)
param([string]$Chain = '')
$ErrorActionPreference = 'Continue'
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$root = $PSScriptRoot
$models = @(
    'bedrock:amazon.nova-lite-v1:0',
    'bedrock:amazon.nova-pro-v1:0',
    'bedrock:us.amazon.nova-2-lite-v1:0',
    'bedrock:us.meta.llama4-scout-17b-instruct-v1:0',
    'bedrock:mistral.mistral-large-3-675b-instruct',
    'bedrock:mistral.ministral-3-8b-instruct',
    'bedrock:moonshotai.kimi-k2.5'
)
Set-Location $root
if ($Chain -ne '') {
    $m = $Chain
    Write-Output "=== chain $m start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    & $py run_all.py --workers 4 $m 2>&1
    Write-Output "--- tooluse $(Get-Date -Format 'HH:mm:ss')"
    & $py run_tooluse.py --models $m --workers 4 2>&1 | Select-Object -Last 12
    Write-Output "--- wildfirevqa $(Get-Date -Format 'HH:mm:ss')"
    & $py run_wildfirevqa.py --models $m --workers 4 2>&1 | Select-Object -Last 12
    Write-Output "=== chain $m DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    exit 0
}
foreach ($m in $models) {
    $safe = ($m -replace '[^A-Za-z0-9._-]', '_')
    $log = Join-Path $root ("logs\tier1-" + $safe + "-2026-09-18.log")
    $err = Join-Path $root ("logs\tier1-" + $safe + "-2026-09-18.err")
    Start-Process -FilePath 'pwsh' -WorkingDirectory $root -WindowStyle Hidden `
        -ArgumentList @('-NoProfile', '-File', (Join-Path $root 'run_tier1.ps1'), '-Chain', $m) `
        -RedirectStandardOutput $log -RedirectStandardError $err
    Write-Output "launched $m -> $log"
}
