param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [string]$Destination = "remote_outputs",

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge"
)

$ErrorActionPreference = "Stop"

$remoteCommand = "cd $RemoteRepo && find results docs logs -maxdepth 1 -type f -name '$Job*' -print | tar -czf - -T - | base64"
$konbuCommand = "ssh -A $GpuHost ""sh -lc '$remoteCommand'"""

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$archive = Join-Path $Destination "$Job.tar.gz"
$base64 = ssh -A -J $JumpHost $KonbuHost $konbuCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote fetch failed with exit code $LASTEXITCODE"
}
$bytes = [Convert]::FromBase64String(($base64 -join ""))
[IO.File]::WriteAllBytes((Resolve-Path $Destination).Path + [IO.Path]::DirectorySeparatorChar + "$Job.tar.gz", $bytes)
tar -xzf $archive -C .
Remove-Item -LiteralPath $archive

Write-Host "Fetched available job outputs into repository paths."
