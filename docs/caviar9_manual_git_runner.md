# caviar9 manual Git runner

SSH control from Codex can be unstable.  In that case, use caviar9 itself as a
run-once Git worker:

`Codex -> GitHub -> caviar9 manual run -> GitHub -> Codex`

## One-time setup on caviar9

Clone the repository if it does not already exist:

```bash
mkdir -p ~/codex-gpu-work
cd ~/codex-gpu-work
git clone --branch main https://github.com/shunnsoku7187/codex-colab-bridge.git
cd codex-colab-bridge
```

For private-repo push access, set a GitHub token in the shell before running:

```bash
export GITHUB_TOKEN='YOUR_GITHUB_TOKEN'
```

The runner will rewrite `origin` to use that token, but it masks the token in
printed commands.

## Run one pending GPU job

```bash
cd ~/codex-gpu-work/codex-colab-bridge
git pull --ff-only origin main
bash tools/caviar9_run_once.sh --job ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001
```

If `--job` is omitted, the first pending job whose backend is `gpu`, `cuda`, or
`caviar9` is executed.

## What it does

1. configures git user as `caviar9-runner`,
2. pulls `origin/main`,
3. marks the job as `running` and pushes that state,
4. runs the job with the caviar9 CUDA conda Python,
5. writes logs under `logs/`,
6. writes result/status files under `results/`,
7. commits and pushes the final outputs,
8. exits.

This is intended to be manually triggered, not a daemon.
