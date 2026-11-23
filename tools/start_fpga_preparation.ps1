param([string]$Python = 'C:/Users/shun7/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$build = Join-Path $repo 'fpga/sim/build'
New-Item -ItemType Directory -Force $build | Out-Null
$pidFile = Join-Path $build 'preparation.pid'
if (Test-Path $pidFile) {
    $previous = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
    if ($previous) { throw "Preparation process is already running: $($previous.Id)" }
}
$process = Start-Process -FilePath $Python -ArgumentList '-u',"$PSScriptRoot/run_fpga_preparation_pipeline.py" -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $build 'preparation.log') -RedirectStandardError (Join-Path $build 'preparation.stderr.log')
$process.Id | Set-Content $pidFile
Write-Output "Started preparation PID=$($process.Id). Logs: $build/preparation.log"
