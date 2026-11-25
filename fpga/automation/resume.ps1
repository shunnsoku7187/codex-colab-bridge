$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$build = Join-Path $repo 'fpga\vivado\build'
$pidFile = Join-Path $build 'boardless_pipeline_process.json'

$running = $false
if (Test-Path -LiteralPath $pidFile) {
    $record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($process -and [math]::Abs(($process.StartTime.ToUniversalTime() - ([datetime]$record.started_utc).ToUniversalTime()).TotalSeconds) -lt 1) {
        $running = $true
        Write-Output "PROCESS: running (PID $($record.pid))"
    } else {
        Write-Output 'PROCESS: stopped'
    }
} else {
    Write-Output 'PROCESS: none started through start.ps1'
}

$stateFile = Join-Path $build 'boardless_pipeline_state.json'
if (-not $running -and (Test-Path -LiteralPath $stateFile)) {
    $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    $interrupted = @($state.PSObject.Properties | Where-Object { $_.Value.status -eq 'running' } | ForEach-Object { $_.Name })
    if ($interrupted.Count -gt 0) {
        Write-Output "INCOMPLETE: $($interrupted -join ', ') was interrupted; inspect its saved log before explicitly restarting."
    }
}

& python (Join-Path $PSScriptRoot 'boardless_pipeline.py') status
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$fullState = Join-Path $build 'full_eval_20pc_state.json'
if (Test-Path -LiteralPath $fullState) {
    & python (Join-Path $PSScriptRoot 'overnight_full_eval.py') status
}
