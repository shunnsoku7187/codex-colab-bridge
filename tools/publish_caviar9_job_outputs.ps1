param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [string]$Message = "",

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge"
)

$ErrorActionPreference = "Stop"

.\tools\fetch_caviar9_job_outputs.ps1 `
    -Job $Job `
    -JumpHost $JumpHost `
    -KonbuHost $KonbuHost `
    -GpuHost $GpuHost `
    -RemoteRepo $RemoteRepo

$paths = @(
    "results/$Job.remote_status.json",
    "results/$Job.json",
    "results/${Job}_summary.json",
    "results/${Job}_tradeoff.png",
    "results/${Job}_scores",
    "docs/$Job.md",
    "logs/$Job.stdout.log",
    "logs/$Job.stderr.log",
    "logs/$Job.remote_runner.log"
) | Where-Object { Test-Path $_ }

if ($paths.Count -eq 0) {
    Write-Host "No fetched output files found."
    exit 1
}

git add @paths

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No fetched output changes to publish."
    exit 0
}

if ($Message -eq "") {
    $Message = "add caviar9 outputs for $Job"
}

git commit -m $Message
git push origin main
