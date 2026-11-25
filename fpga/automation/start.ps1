param(
    [ValidateSet('bd','axi_wiring','profile_coverage','synth','route','calib','quant_test','feature_compare','compile')]
    [string]$Only,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$build = Join-Path $repo 'fpga\vivado\build'
$pidFile = Join-Path $build 'boardless_pipeline_process.json'
$runner = Join-Path $PSScriptRoot 'boardless_pipeline.py'
New-Item -ItemType Directory -Force -Path $build | Out-Null

if (Test-Path -LiteralPath $pidFile) {
    $record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $old = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($old -and [math]::Abs(($old.StartTime.ToUniversalTime() - ([datetime]$record.started_utc).ToUniversalTime()).TotalSeconds) -lt 1) {
        throw "Pipeline is already running (PID $($record.pid)). Use resume.ps1 to check it."
    }
}

$python = & python -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0 -or -not $python) { throw 'Python executable was not found.' }
$arguments = '"' + $runner + '" run'
if ($Only) { $arguments += ' --only ' + $Only }
if ($Force) { $arguments += ' --force' }
$stdout = Join-Path $build 'boardless_pipeline_stdout.log'
$stderr = Join-Path $build 'boardless_pipeline_stderr.log'
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{
    pid = $process.Id
    started_utc = $process.StartTime.ToUniversalTime().ToString('o')
    stdout = $stdout
    stderr = $stderr
    only = $Only
} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
Write-Output "Started boardless pipeline (PID $($process.Id)). The chat can now be closed."
Write-Output "Resume with: & '$PSScriptRoot\resume.ps1'"
