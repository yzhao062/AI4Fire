# Ten-item pilot of the seven Bedrock models that passed the text, image, and tool probes on 2026-09-18,
# on the four runners that take --limit (fire danger has no limit flag and is piloted by its full fold-0 run).
# Checks parse rates and truncation before the full runs. Credentials come from AWS_BEARER_TOKEN_BEDROCK.
$ErrorActionPreference = 'Continue'
$py = 'C:\Users\yuezh\miniforge3\envs\py312\python.exe'
$root = 'C:\Users\yuezh\PycharmProjects\fire-bench'
Set-Location $root
$models = @(
    'bedrock:amazon.nova-lite-v1:0',
    'bedrock:amazon.nova-pro-v1:0',
    'bedrock:us.amazon.nova-2-lite-v1:0',
    'bedrock:us.meta.llama4-scout-17b-instruct-v1:0',
    'bedrock:mistral.mistral-large-3-675b-instruct',
    'bedrock:mistral.ministral-3-8b-instruct',
    'bedrock:moonshotai.kimi-k2.5'
)
foreach ($m in $models) {
    Write-Output "=== $m $(Get-Date -Format 'HH:mm:ss')"
    & $py run_figlib.py --models $m --limit 10 --workers 4 2>&1 | Select-Object -Last 6
    & $py run_allocation.py --models $m --limit 10 --workers 4 2>&1 | Select-Object -Last 6
    & $py run_tooluse.py --models $m --limit 10 --workers 4 2>&1 | Select-Object -Last 6
    & $py run_wildfirevqa.py --models $m --limit 10 --workers 4 --suffix -pilot 2>&1 | Select-Object -Last 6
}
Write-Output "PILOT DONE $(Get-Date -Format 'HH:mm:ss')"
