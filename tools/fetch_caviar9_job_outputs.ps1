param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [string]$Destination = "remote_outputs",

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$GpuUser = "shunya",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge"
)

$ErrorActionPreference = "Stop"

$destPath = Join-Path $Destination $Job
New-Item -ItemType Directory -Force -Path $destPath | Out-Null

$remoteFiles = @(
    "results/$Job.remote_status.json",
    "results/$Job.json",
    "results/${Job}_summary.json",
    "results/${Job}_tradeoff.png",
    "docs/$Job.md",
    "logs/$Job.stdout.log",
    "logs/$Job.stderr.log",
    "logs/$Job.remote_runner.log"
)

foreach ($file in $remoteFiles) {
    $target = Join-Path $destPath (Split-Path $file -Leaf)
    $source = "${GpuUser}@${GpuHost}:$RemoteRepo/$file"
    try {
        scp -o "ProxyJump=$JumpHost,$KonbuHost" $source $target | Out-Null
    } catch {
        if (Test-Path $target) {
            Remove-Item -LiteralPath $target
        }
    }
}

Write-Host "Fetched available files to $destPath"
