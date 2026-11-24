param([int]$RefreshSeconds = 5)
$repo = Split-Path $PSScriptRoot -Parent
$build = Join-Path $repo 'fpga/sim/build'
$Host.UI.RawUI.WindowTitle = 'PatchCore FPGA preparation - live progress'
$stages = @{
    starting='Starting pipeline'; remote_launch='Launching GPU job';
    remote_running='GPU preparation / waiting for returned results';
    real_feature_simulation='Real-image feature RTL verification';
    collecting_implementation='Collecting Vivado reports';
    complete='Finished - check individual results'; failed='Pipeline stopped with an error'
}
function Tail-Log([string]$Path, [int]$Lines) {
    if (Test-Path $Path) { Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue }
    else { Write-Host '(No log yet)' -ForegroundColor DarkGray }
}
while ($true) {
    Clear-Host
    Write-Host 'PATCHCORE FPGA PREPARATION' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host 'Refresh: 5 seconds. Ctrl+C or close this window to stop viewing only.'
    Write-Host ''
    $statePath = Join-Path $build 'preparation_status.json'
    if (Test-Path $statePath) {
        try {
            $state = Get-Content $statePath -Raw | ConvertFrom-Json
            $description = $stages[$state.stage]
            Write-Host ("Stage: {0} ({1})" -f $description,$state.stage) -ForegroundColor Yellow
            Write-Host ('Stage recorded: ' + $state.at)
            if ($state.error) { Write-Host $state.error -ForegroundColor Red }
            if ($state.stage -eq 'complete') {
                Write-Host ("GPU={0}  RTL={1}  Vivado exit={2}" -f $state.gpu_job,$state.rtl,$state.vivado_returncode)
            }
        } catch { Write-Host 'Status file updating...' }
    }
    $pidFile = Join-Path $build 'preparation.pid'
    if (Test-Path $pidFile) {
        $taskPid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
        $taskProcess = Get-Process -Id $taskPid -ErrorAction SilentlyContinue
        Write-Host ("Controller PID={0}  Running={1}" -f $taskPid,[bool]$taskProcess)
    }
    $profiles = Join-Path $repo 'results/patchcore_int8_preparation_001/profiles_and_results.json'
    if (Test-Path $profiles) {
        try { $count = @(Get-Content $profiles -Raw | ConvertFrom-Json).Count; Write-Host "GPU categories returned: $count / 15" } catch {}
    }
    $rtlSummary = Join-Path $repo 'fpga/evidence/real_features_001/summary.json'
    if (Test-Path $rtlSummary) {
        try {
            $rtl = @(Get-Content $rtlSummary -Raw | ConvertFrom-Json)
            Write-Host ("RTL categories passed: {0} / 15  Latest: {1}" -f $rtl.Count,$rtl[-1].category) -ForegroundColor Green
        } catch {}
    }
    Write-Host "`n--- PIPELINE ---" -ForegroundColor Cyan
    Tail-Log (Join-Path $build 'preparation.log') 8
    Write-Host "`n--- VIVADO ---" -ForegroundColor Cyan
    $vivadoLog = Join-Path $repo 'fpga/evidence/implementation_001/console.log'
    $retryLog = Join-Path $repo 'fpga/evidence/implementation_retry_002/console.log'
    if (Test-Path $retryLog) { $vivadoLog = $retryLog }
    $errors = if (Test-Path $vivadoLog) { Get-Content $vivadoLog -Tail 30 | Select-String 'ERROR|EXCEPTION|Abnormal' }
    if ($errors) { $errors | ForEach-Object { Write-Host $_ -ForegroundColor Red } }
    else { Tail-Log $vivadoLog 4 }
    Write-Host "`n--- CONTROLLER STDERR (may include normal Git messages) ---" -ForegroundColor DarkGray
    Tail-Log (Join-Path $build 'preparation.stderr.log') 3
    Start-Sleep -Seconds ([Math]::Max(1,$RefreshSeconds))
}
