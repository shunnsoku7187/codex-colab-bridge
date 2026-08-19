param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [switch]$Background,

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "~/codex-gpu-work/colab-github-bridge",
    [string]$PythonBin = "~/miniconda3/envs/cuda/bin/python"
)

$ErrorActionPreference = "Stop"

$runnerLog = "logs/$Job.remote_runner.log"

if ($Background) {
    $remoteCommand = "cd $RemoteRepo && git pull --ff-only origin main && nohup $PythonBin tools/caviar9_run_job.py --job $Job > $runnerLog 2>&1 < /dev/null &"
} else {
    $remoteCommand = "cd $RemoteRepo && git pull --ff-only origin main && $PythonBin tools/caviar9_run_job.py --job $Job"
}

$escapedRemoteCommand = $remoteCommand.Replace("'", "'\''")
$konbuCommand = "ssh -A $GpuHost ""sh -lc '$escapedRemoteCommand'"""
ssh -A -J $JumpHost $KonbuHost $konbuCommand
