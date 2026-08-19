param(
    [Parameter(Mandatory = $true)]
    [string]$Job,

    [string]$JumpHost = "shunya@ssh.arch.info.mie-u.ac.jp",
    [string]$KonbuHost = "shunya@konbu.arch.info.mie-u.ac.jp",
    [string]$GpuHost = "caviar9",
    [string]$RemoteRepo = "/home/shunya/codex-gpu-work/colab-github-bridge",
    [int]$ConnectTimeout = 10
)

$ErrorActionPreference = "Stop"

Write-Host "== checking caviar9 job =="
.\tools\check_caviar9_job.ps1 `
    -Job $Job `
    -JumpHost $JumpHost `
    -KonbuHost $KonbuHost `
    -GpuHost $GpuHost `
    -RemoteRepo $RemoteRepo `
    -ConnectTimeout $ConnectTimeout

Write-Host ""
Write-Host "== fetching and publishing outputs =="
.\tools\publish_caviar9_job_outputs.ps1 `
    -Job $Job `
    -Message "add caviar9 outputs for $Job" `
    -JumpHost $JumpHost `
    -KonbuHost $KonbuHost `
    -GpuHost $GpuHost `
    -RemoteRepo $RemoteRepo

Write-Host ""
Write-Host "== refreshing KSDD2 baseline comparison =="
python -m scripts.summarize_ksdd2_baselines

$scoreDir = "results/${Job}_scores"
$scoreSummaryJson = "results/${Job}_scores_ensemble_summary.json"
$scoreSummaryMd = "results/${Job}_scores_ensemble_summary.md"
if (Test-Path $scoreDir) {
    Write-Host ""
    Write-Host "== checking score ensembles =="
    python -m scripts.summarize_ksdd2_score_ensembles --scores-dir $scoreDir
}

$changed = git status --short
if ($changed) {
    git add docs/ksdd2_baseline_comparison.md results/ksdd2_baseline_comparison.json
    if (Test-Path $scoreDir) {
        git add $scoreSummaryJson $scoreSummaryMd
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "summarize KSDD2 baseline results for $Job"
        git push origin main
    }
}

Write-Host ""
Write-Host "KSDD2 baseline completion helper finished."
