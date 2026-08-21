# GPU workflow state, 2026-08-21

## Current architecture

- Operation/instruction PC:
  - create experiment code and job JSON,
  - push them to GitHub,
  - read final results from GitHub.
- konbu:
  - pull/push GitHub repository,
  - talk to caviar9 over SSH,
  - coordinate execution and result collection.
- caviar9:
  - run GPU jobs,
  - store generated logs/results in the repository checkout.

## Repository locations

- Local desktop:
  - `C:\Users\shun7\Documents\Codex\2026-07-01\goo\work\colab-github-bridge`
- konbu:
  - `/home/shunya/codex-gpu-work/colab-github-bridge`
- caviar9:
  - `/home/shunya/codex-gpu-work/colab-github-bridge`

## Git state observed on konbu

On 2026-08-21, konbu repo existed and could read GitHub:

- repo: `/home/shunya/codex-gpu-work/colab-github-bridge`
- branch: `main`
- remote: `https://github.com/shunnsoku7187/codex-colab-bridge.git`
- `git ls-remote --heads origin main`: succeeded
- push previously failed because GitHub write authentication was missing.
- user reports konbu-side settings are now complete.

## caviar9 result state

`ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001` already
finished successfully on caviar9:

- status: `done`
- returncode: `0`
- started: `2026-08-19T18:22:07Z`
- finished: `2026-08-19T19:01:17Z`

Observed generated files on caviar9:

- `docs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.md`
- `logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_runner.log`
- `logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stderr.log`
- `logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stdout.log`
- `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_status.json`
- `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json`
- `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_tradeoff.png`
- `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_123_scores.npz`
- `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_456_scores.npz`

## Preliminary result interpretation

Model:

- `segmentation-models-pytorch`
- U-Net
- ResNet50 encoder
- ImageNet weights

Observed key metrics:

- seed 123 top-k test AUROC/AUPR: about `0.9901 / 0.9605`
- aggregate top-k, target false-pass <= 5%, good-pass >= 90%:
  - feasible seeds: `1/2`
  - mean good-pass: about `96.81%`
  - mean false-pass: about `6.36%`
  - worst false-pass: about `10.00%`

Interpretation:

- The final-only defect score is strong enough to be a serious baseline.
- Strict inspection operating points are still unstable across seeds.
- Next value-add checks should focus on calibration, seed averaging/ensemble,
  selective decision, and later early-exit/FPGA value on top of this baseline.
