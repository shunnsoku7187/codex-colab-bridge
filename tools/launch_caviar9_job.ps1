param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [switch]$Background,

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "~/codex-gpu-work/colab-github-bridge",
    [string]$PythonBin = "~/miniconda3/envs/cuda/bin/python",
    [switch]$UseKonbu,
    [int]$ConnectTimeout = 10
)

$ErrorActionPreference = "Stop"

$runnerLog = "logs/$Job.remote_runner.log"

if ($Background) {
    $remoteCommand = "cd $RemoteRepo && git pull --ff-only origin main && setsid -f $PythonBin tools/caviar9_run_job.py --job $Job > $runnerLog 2>&1 < /dev/null"
} else {
    $remoteCommand = "cd $RemoteRepo && git pull --ff-only origin main && $PythonBin tools/caviar9_run_job.py --job $Job"
}

$escapedRemoteCommand = $remoteCommand.Replace("'", "'\''")
$innerSshOptions = "-o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
if ($UseKonbu) {
    $konbuCommand = "ssh -A $innerSshOptions $GpuHost ""sh -lc '$escapedRemoteCommand'"""
    ssh -A -o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -J $JumpHost $KonbuHost $konbuCommand
} else {
    ssh -A -o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -J $JumpHost "shunya@$GpuHost" "sh -lc '$escapedRemoteCommand'"
}
