param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [int]$Lines = 80,

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge",
    [int]$ConnectTimeout = 10
)

$ErrorActionPreference = "Stop"

function Invoke-Caviar9 {
    param([Parameter(Mandatory = $true)][string]$Command)
    ssh -A -o ConnectTimeout=$ConnectTimeout -J $JumpHost $KonbuHost "ssh -A -o ConnectTimeout=$ConnectTimeout $GpuHost '$Command'"
}

Write-Host "== remote status =="
Invoke-Caviar9 "cat $RemoteRepo/results/$Job.remote_status.json 2>/dev/null || echo 'remote status is not available yet'"

Write-Host ""
Write-Host "== gpu =="
Invoke-Caviar9 "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"

Write-Host ""
Write-Host "== processes =="
Invoke-Caviar9 "ps -u shunya -o pid,etime,cmd"

Write-Host ""
Write-Host "== stdout tail =="
Invoke-Caviar9 "tail -n $Lines $RemoteRepo/logs/$Job.stdout.log 2>/dev/null || echo 'stdout log is not available yet'"

Write-Host ""
Write-Host "== stderr tail =="
Invoke-Caviar9 "tail -n $Lines $RemoteRepo/logs/$Job.stderr.log 2>/dev/null || echo 'stderr log is not available yet'"
