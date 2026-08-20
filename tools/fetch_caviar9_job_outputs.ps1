param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [string]$Destination = "remote_outputs",

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge",
    [switch]$UseKonbu,
    [int]$ConnectTimeout = 10,
    [int]$RetryCount = 3
)

$ErrorActionPreference = "Stop"

$remoteCommand = "cd $RemoteRepo && find results docs logs -type f -print | grep '$Job' | tar -czf - -T - | base64"
$innerSshOptions = "-o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
if ($UseKonbu) {
    $remoteFetchCommand = "ssh -A $innerSshOptions $GpuHost ""sh -lc '$remoteCommand'"""
    $targetHost = $KonbuHost
} else {
    $remoteFetchCommand = "sh -lc '$remoteCommand'"
    $targetHost = "shunya@$GpuHost"
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$archive = Join-Path $Destination "$Job.tar.gz"
$base64 = $null
for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
    Write-Host "Fetch attempt $attempt/$RetryCount"
    $base64 = ssh -A -o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -J $JumpHost $targetHost $remoteFetchCommand
    if ($LASTEXITCODE -eq 0) {
        break
    }
    if ($attempt -eq $RetryCount) {
        throw "remote fetch failed with exit code $LASTEXITCODE"
    }
    Start-Sleep -Seconds ([Math]::Min(30, 5 * $attempt))
}
$bytes = [Convert]::FromBase64String(($base64 -join ""))
[IO.File]::WriteAllBytes((Resolve-Path $Destination).Path + [IO.Path]::DirectorySeparatorChar + "$Job.tar.gz", $bytes)
tar -xzf $archive -C .
Remove-Item -LiteralPath $archive

Write-Host "Fetched available job outputs into repository paths."
