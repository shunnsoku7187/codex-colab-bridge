param(
    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [switch]$UseKonbu,
    [int]$ConnectTimeout = 8
)

$ErrorActionPreference = "Continue"

function Test-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "== $Name =="
    $started = Get-Date
    & $Command
    $code = $LASTEXITCODE
    $elapsed = ((Get-Date) - $started).TotalSeconds
    if ($code -eq 0) {
        Write-Host "OK ($([Math]::Round($elapsed, 1)) sec)"
    } else {
        Write-Host "FAILED exit=$code ($([Math]::Round($elapsed, 1)) sec)"
    }
}

$sshOptions = @(
    "-A",
    "-o", "ConnectTimeout=$ConnectTimeout",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3"
)

Test-Step "jump host" {
    ssh @sshOptions $JumpHost "date"
}

Test-Step "konbu via jump" {
    ssh @sshOptions -J $JumpHost $KonbuHost "date"
}

if ($UseKonbu) {
    Test-Step "caviar9 via konbu" {
        ssh @sshOptions -J $JumpHost $KonbuHost "ssh -A -o ConnectTimeout=$ConnectTimeout -o ServerAliveInterval=30 -o ServerAliveCountMax=3 $GpuHost 'date; hostname; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader'"
    }
} else {
    Test-Step "caviar9 via jump" {
        ssh @sshOptions -J $JumpHost "shunya@$GpuHost" "date; hostname; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"
    }
}
