# MVTec AD Hugging Face download audit

Repo: `TheoM55/mvtec_all_objects_split`
Download root: `/home/shunya/codex-gpu-work/data/mvtec_ad`
Dataset root: `/home/shunya/codex-gpu-work/data/mvtec_ad`
Total size: `4.907 GiB`

## Category structure

| category | present | train images | test images | ground truth images | defect types | size GiB |
|---|---:|---:|---:|---:|---:|---:|
| bottle | no | 0 | 0 | 0 | 0 | 0.000 |
| cable | no | 0 | 0 | 0 | 0 | 0.000 |
| capsule | no | 0 | 0 | 0 | 0 | 0.000 |
| carpet | no | 0 | 0 | 0 | 0 | 0.000 |
| grid | no | 0 | 0 | 0 | 0 | 0.000 |
| hazelnut | no | 0 | 0 | 0 | 0 | 0.000 |
| leather | no | 0 | 0 | 0 | 0 | 0.000 |
| metal_nut | no | 0 | 0 | 0 | 0 | 0.000 |
| pill | no | 0 | 0 | 0 | 0 | 0.000 |
| screw | no | 0 | 0 | 0 | 0 | 0.000 |
| tile | no | 0 | 0 | 0 | 0 | 0.000 |
| toothbrush | no | 0 | 0 | 0 | 0 | 0.000 |
| transistor | no | 0 | 0 | 0 | 0 | 0.000 |
| wood | no | 0 | 0 | 0 | 0 | 0.000 |
| zipper | no | 0 | 0 | 0 | 0 | 0.000 |

## Next action

- If all categories are present, run a 3-category anomaly baseline probe first.
- Keep this dataset under the persistent data root; do not commit the raw dataset to Git.
- Use category-level results rather than a single random split, because MVTec AD already defines train/test structure.
