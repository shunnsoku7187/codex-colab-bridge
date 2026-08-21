# MVTec AD Parquet anomaly baseline probe

Purpose: check whether MVTec AD can provide a stronger inspection baseline than the small KSDD splits.

## Materialization

- Dataset root: `/home/shunya/codex-gpu-work/data/mvtec_ad`
- Materialized root: `/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2`
- Parquet shards: `31`
- Written images this run: `5354`

## Category sample counts

| category | train | test | good rows | defect rows | masks |
|---|---:|---:|---:|---:|---:|
| bottle | 209 | 83 | 229 | 63 | 63 |
| cable | 224 | 150 | 282 | 92 | 92 |
| capsule | 219 | 132 | 242 | 109 | 109 |
| carpet | 280 | 117 | 308 | 89 | 89 |
| grid | 264 | 78 | 285 | 57 | 57 |
| hazelnut | 391 | 110 | 431 | 70 | 70 |
| leather | 245 | 124 | 277 | 92 | 92 |
| metal_nut | 220 | 115 | 242 | 93 | 93 |
| pill | 267 | 167 | 293 | 141 | 141 |
| screw | 320 | 160 | 361 | 119 | 119 |
| tile | 230 | 117 | 263 | 84 | 84 |
| toothbrush | 60 | 42 | 72 | 30 | 30 |
| transistor | 213 | 100 | 273 | 40 | 40 |
| wood | 247 | 79 | 266 | 60 | 60 |
| zipper | 240 | 151 | 272 | 119 | 119 |

## Image-level anomaly separation

| category | method | score | AUROC | AUPR |
|---|---|---|---:|---:|
| bottle | patchcore_lite | max_score | 1.0 | 1.0 |
| bottle | patchcore_lite | topk_score | 1.0 | 1.0 |
| bottle | padim_diag | max_score | 0.995238 | 0.998479 |
| bottle | padim_diag | topk_score | 0.996032 | 0.998743 |
| cable | patchcore_lite | max_score | 0.937594 | 0.966171 |
| cable | patchcore_lite | topk_score | 0.95521 | 0.975769 |
| cable | padim_diag | max_score | 0.750937 | 0.852304 |
| cable | padim_diag | topk_score | 0.757121 | 0.855688 |
| capsule | patchcore_lite | max_score | 0.925409 | 0.981788 |
| capsule | patchcore_lite | topk_score | 0.941763 | 0.986177 |
| capsule | padim_diag | max_score | 0.86797 | 0.968127 |
| capsule | padim_diag | topk_score | 0.878341 | 0.970133 |
| carpet | patchcore_lite | max_score | 0.989165 | 0.996699 |
| carpet | patchcore_lite | topk_score | 0.986758 | 0.995997 |
| carpet | padim_diag | max_score | 0.951445 | 0.985416 |
| carpet | padim_diag | topk_score | 0.957865 | 0.987407 |
| grid | patchcore_lite | max_score | 0.9599 | 0.985485 |
| grid | patchcore_lite | topk_score | 0.963241 | 0.987049 |
| grid | padim_diag | max_score | 0.53467 | 0.791685 |
| grid | padim_diag | topk_score | 0.533835 | 0.784566 |
| hazelnut | patchcore_lite | max_score | 0.9975 | 0.998528 |
| hazelnut | patchcore_lite | topk_score | 1.0 | 1.0 |
| hazelnut | padim_diag | max_score | 0.571786 | 0.680505 |
| hazelnut | padim_diag | topk_score | 0.579286 | 0.671979 |
| leather | patchcore_lite | max_score | 1.0 | 1.0 |
| leather | patchcore_lite | topk_score | 1.0 | 1.0 |
| leather | padim_diag | max_score | 0.944293 | 0.98193 |
| leather | padim_diag | topk_score | 0.950747 | 0.984164 |
| metal_nut | patchcore_lite | max_score | 0.991691 | 0.998221 |
| metal_nut | patchcore_lite | topk_score | 0.991691 | 0.998217 |
| metal_nut | padim_diag | max_score | 0.833333 | 0.960779 |
| metal_nut | padim_diag | topk_score | 0.836266 | 0.961563 |
| pill | patchcore_lite | max_score | 0.931806 | 0.986851 |
| pill | patchcore_lite | topk_score | 0.944081 | 0.989805 |
| pill | padim_diag | max_score | 0.806601 | 0.956207 |
| pill | padim_diag | topk_score | 0.81533 | 0.960004 |
| screw | patchcore_lite | max_score | 0.771879 | 0.903965 |
| screw | patchcore_lite | topk_score | 0.764296 | 0.897866 |
| screw | padim_diag | max_score | 0.656692 | 0.82956 |
| screw | padim_diag | topk_score | 0.668375 | 0.833624 |
| tile | patchcore_lite | max_score | 0.987013 | 0.995156 |
| tile | patchcore_lite | topk_score | 0.991703 | 0.996887 |
| tile | padim_diag | max_score | 0.973665 | 0.989774 |
| tile | padim_diag | topk_score | 0.977633 | 0.991411 |
| toothbrush | patchcore_lite | max_score | 0.947222 | 0.978068 |
| toothbrush | patchcore_lite | topk_score | 0.975 | 0.99027 |
| toothbrush | padim_diag | max_score | 0.861111 | 0.947291 |
| toothbrush | padim_diag | topk_score | 0.866667 | 0.948343 |
| transistor | patchcore_lite | max_score | 0.975833 | 0.971247 |
| transistor | patchcore_lite | topk_score | 0.960833 | 0.959527 |
| transistor | padim_diag | max_score | 0.877917 | 0.860807 |
| transistor | padim_diag | topk_score | 0.879583 | 0.860796 |
| wood | patchcore_lite | max_score | 0.986842 | 0.996007 |
| wood | patchcore_lite | topk_score | 0.991228 | 0.997342 |
| wood | padim_diag | max_score | 0.955263 | 0.986096 |
| wood | padim_diag | topk_score | 0.959649 | 0.987427 |
| zipper | patchcore_lite | max_score | 0.971113 | 0.991703 |
| zipper | patchcore_lite | topk_score | 0.973739 | 0.992272 |
| zipper | padim_diag | max_score | 0.961922 | 0.986093 |
| zipper | padim_diag | topk_score | 0.963761 | 0.985901 |

## Best good retention under false-pass constraints

| category | method | score | max defect false-pass | best good pass | good loss |
|---|---|---|---:|---:|---:|
| bottle | patchcore_lite | max_score | 0.0% | 100.00% | 0.00% |
| bottle | patchcore_lite | max_score | 1.0% | 100.00% | 0.00% |
| bottle | patchcore_lite | max_score | 5.0% | 100.00% | 0.00% |
| bottle | patchcore_lite | topk_score | 0.0% | 100.00% | 0.00% |
| bottle | patchcore_lite | topk_score | 1.0% | 100.00% | 0.00% |
| bottle | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| bottle | padim_diag | max_score | 0.0% | 90.00% | 10.00% |
| bottle | padim_diag | max_score | 1.0% | 90.00% | 10.00% |
| bottle | padim_diag | max_score | 5.0% | 95.00% | 5.00% |
| bottle | padim_diag | topk_score | 0.0% | 90.00% | 10.00% |
| bottle | padim_diag | topk_score | 1.0% | 90.00% | 10.00% |
| bottle | padim_diag | topk_score | 5.0% | 95.00% | 5.00% |
| cable | patchcore_lite | max_score | 0.0% | 24.14% | 75.86% |
| cable | patchcore_lite | max_score | 1.0% | 24.14% | 75.86% |
| cable | patchcore_lite | max_score | 5.0% | 50.00% | 50.00% |
| cable | patchcore_lite | topk_score | 0.0% | 32.76% | 67.24% |
| cable | patchcore_lite | topk_score | 1.0% | 32.76% | 67.24% |
| cable | patchcore_lite | topk_score | 5.0% | 60.34% | 39.66% |
| cable | padim_diag | max_score | 0.0% | 3.45% | 96.55% |
| cable | padim_diag | max_score | 1.0% | 3.45% | 96.55% |
| cable | padim_diag | max_score | 5.0% | 12.07% | 87.93% |
| cable | padim_diag | topk_score | 0.0% | 3.45% | 96.55% |
| cable | padim_diag | topk_score | 1.0% | 3.45% | 96.55% |
| cable | padim_diag | topk_score | 5.0% | 13.79% | 86.21% |
| capsule | patchcore_lite | max_score | 0.0% | 39.13% | 60.87% |
| capsule | patchcore_lite | max_score | 1.0% | 39.13% | 60.87% |
| capsule | patchcore_lite | max_score | 5.0% | 65.22% | 34.78% |
| capsule | patchcore_lite | topk_score | 0.0% | 43.48% | 56.52% |
| capsule | patchcore_lite | topk_score | 1.0% | 43.48% | 56.52% |
| capsule | patchcore_lite | topk_score | 5.0% | 73.91% | 26.09% |
| capsule | padim_diag | max_score | 0.0% | 17.39% | 82.61% |
| capsule | padim_diag | max_score | 1.0% | 26.09% | 73.91% |
| capsule | padim_diag | max_score | 5.0% | 47.83% | 52.17% |
| capsule | padim_diag | topk_score | 0.0% | 17.39% | 82.61% |
| capsule | padim_diag | topk_score | 1.0% | 30.43% | 69.57% |
| capsule | padim_diag | topk_score | 5.0% | 52.17% | 47.83% |
| carpet | patchcore_lite | max_score | 0.0% | 71.43% | 28.57% |
| carpet | patchcore_lite | max_score | 1.0% | 71.43% | 28.57% |
| carpet | patchcore_lite | max_score | 5.0% | 89.29% | 10.71% |
| carpet | patchcore_lite | topk_score | 0.0% | 64.29% | 35.71% |
| carpet | patchcore_lite | topk_score | 1.0% | 64.29% | 35.71% |
| carpet | patchcore_lite | topk_score | 5.0% | 89.29% | 10.71% |
| carpet | padim_diag | max_score | 0.0% | 25.00% | 75.00% |
| carpet | padim_diag | max_score | 1.0% | 25.00% | 75.00% |
| carpet | padim_diag | max_score | 5.0% | 64.29% | 35.71% |
| carpet | padim_diag | topk_score | 0.0% | 28.57% | 71.43% |
| carpet | padim_diag | topk_score | 1.0% | 28.57% | 71.43% |
| carpet | padim_diag | topk_score | 5.0% | 71.43% | 28.57% |
| grid | patchcore_lite | max_score | 0.0% | 61.90% | 38.10% |
| grid | patchcore_lite | max_score | 1.0% | 61.90% | 38.10% |
| grid | patchcore_lite | max_score | 5.0% | 66.67% | 33.33% |
| grid | patchcore_lite | topk_score | 0.0% | 52.38% | 47.62% |
| grid | patchcore_lite | topk_score | 1.0% | 52.38% | 47.62% |
| grid | patchcore_lite | topk_score | 5.0% | 76.19% | 23.81% |
| grid | padim_diag | max_score | 0.0% | 0.00% | 100.00% |
| grid | padim_diag | max_score | 1.0% | 0.00% | 100.00% |
| grid | padim_diag | max_score | 5.0% | 0.00% | 100.00% |
| grid | padim_diag | topk_score | 0.0% | 0.00% | 100.00% |
| grid | padim_diag | topk_score | 1.0% | 0.00% | 100.00% |
| grid | padim_diag | topk_score | 5.0% | 0.00% | 100.00% |
| hazelnut | patchcore_lite | max_score | 0.0% | 97.50% | 2.50% |
| hazelnut | patchcore_lite | max_score | 1.0% | 97.50% | 2.50% |
| hazelnut | patchcore_lite | max_score | 5.0% | 97.50% | 2.50% |
| hazelnut | patchcore_lite | topk_score | 0.0% | 100.00% | 0.00% |
| hazelnut | patchcore_lite | topk_score | 1.0% | 100.00% | 0.00% |
| hazelnut | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| hazelnut | padim_diag | max_score | 0.0% | 7.50% | 92.50% |
| hazelnut | padim_diag | max_score | 1.0% | 7.50% | 92.50% |
| hazelnut | padim_diag | max_score | 5.0% | 15.00% | 85.00% |
| hazelnut | padim_diag | topk_score | 0.0% | 10.00% | 90.00% |
| hazelnut | padim_diag | topk_score | 1.0% | 10.00% | 90.00% |
| hazelnut | padim_diag | topk_score | 5.0% | 17.50% | 82.50% |
| leather | patchcore_lite | max_score | 0.0% | 100.00% | 0.00% |
| leather | patchcore_lite | max_score | 1.0% | 100.00% | 0.00% |
| leather | patchcore_lite | max_score | 5.0% | 100.00% | 0.00% |
| leather | patchcore_lite | topk_score | 0.0% | 100.00% | 0.00% |
| leather | patchcore_lite | topk_score | 1.0% | 100.00% | 0.00% |
| leather | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| leather | padim_diag | max_score | 0.0% | 28.12% | 71.88% |
| leather | padim_diag | max_score | 1.0% | 28.12% | 71.88% |
| leather | padim_diag | max_score | 5.0% | 46.88% | 53.12% |
| leather | padim_diag | topk_score | 0.0% | 28.12% | 71.88% |
| leather | padim_diag | topk_score | 1.0% | 28.12% | 71.88% |
| leather | padim_diag | topk_score | 5.0% | 53.12% | 46.88% |
| metal_nut | patchcore_lite | max_score | 0.0% | 40.91% | 59.09% |
| metal_nut | patchcore_lite | max_score | 1.0% | 40.91% | 59.09% |
| metal_nut | patchcore_lite | max_score | 5.0% | 100.00% | 0.00% |
| metal_nut | patchcore_lite | topk_score | 0.0% | 40.91% | 59.09% |
| metal_nut | patchcore_lite | topk_score | 1.0% | 40.91% | 59.09% |
| metal_nut | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| metal_nut | padim_diag | max_score | 0.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | max_score | 1.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | max_score | 5.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 0.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 1.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 5.0% | 9.09% | 90.91% |
| pill | patchcore_lite | max_score | 0.0% | 0.00% | 100.00% |
| pill | patchcore_lite | max_score | 1.0% | 3.85% | 96.15% |
| pill | patchcore_lite | max_score | 5.0% | 73.08% | 26.92% |
| pill | patchcore_lite | topk_score | 0.0% | 0.00% | 100.00% |
| pill | patchcore_lite | topk_score | 1.0% | 7.69% | 92.31% |
| pill | patchcore_lite | topk_score | 5.0% | 53.85% | 46.15% |
| pill | padim_diag | max_score | 0.0% | 11.54% | 88.46% |
| pill | padim_diag | max_score | 1.0% | 11.54% | 88.46% |
| pill | padim_diag | max_score | 5.0% | 38.46% | 61.54% |
| pill | padim_diag | topk_score | 0.0% | 11.54% | 88.46% |
| pill | padim_diag | topk_score | 1.0% | 11.54% | 88.46% |
| pill | padim_diag | topk_score | 5.0% | 23.08% | 76.92% |
| screw | patchcore_lite | max_score | 0.0% | 7.32% | 92.68% |
| screw | patchcore_lite | max_score | 1.0% | 14.63% | 85.37% |
| screw | patchcore_lite | max_score | 5.0% | 31.71% | 68.29% |
| screw | patchcore_lite | topk_score | 0.0% | 7.32% | 92.68% |
| screw | patchcore_lite | topk_score | 1.0% | 14.63% | 85.37% |
| screw | patchcore_lite | topk_score | 5.0% | 34.15% | 65.85% |
| screw | padim_diag | max_score | 0.0% | 0.00% | 100.00% |
| screw | padim_diag | max_score | 1.0% | 0.00% | 100.00% |
| screw | padim_diag | max_score | 5.0% | 9.76% | 90.24% |
| screw | padim_diag | topk_score | 0.0% | 0.00% | 100.00% |
| screw | padim_diag | topk_score | 1.0% | 0.00% | 100.00% |
| screw | padim_diag | topk_score | 5.0% | 4.88% | 95.12% |
| tile | patchcore_lite | max_score | 0.0% | 63.64% | 36.36% |
| tile | patchcore_lite | max_score | 1.0% | 63.64% | 36.36% |
| tile | patchcore_lite | max_score | 5.0% | 93.94% | 6.06% |
| tile | patchcore_lite | topk_score | 0.0% | 72.73% | 27.27% |
| tile | patchcore_lite | topk_score | 1.0% | 72.73% | 27.27% |
| tile | patchcore_lite | topk_score | 5.0% | 93.94% | 6.06% |
| tile | padim_diag | max_score | 0.0% | 72.73% | 27.27% |
| tile | padim_diag | max_score | 1.0% | 72.73% | 27.27% |
| tile | padim_diag | max_score | 5.0% | 78.79% | 21.21% |
| tile | padim_diag | topk_score | 0.0% | 72.73% | 27.27% |
| tile | padim_diag | topk_score | 1.0% | 72.73% | 27.27% |
| tile | padim_diag | topk_score | 5.0% | 78.79% | 21.21% |
| toothbrush | patchcore_lite | max_score | 0.0% | 75.00% | 25.00% |
| toothbrush | patchcore_lite | max_score | 1.0% | 75.00% | 25.00% |
| toothbrush | patchcore_lite | max_score | 5.0% | 75.00% | 25.00% |
| toothbrush | patchcore_lite | topk_score | 0.0% | 75.00% | 25.00% |
| toothbrush | patchcore_lite | topk_score | 1.0% | 75.00% | 25.00% |
| toothbrush | patchcore_lite | topk_score | 5.0% | 75.00% | 25.00% |
| toothbrush | padim_diag | max_score | 0.0% | 33.33% | 66.67% |
| toothbrush | padim_diag | max_score | 1.0% | 33.33% | 66.67% |
| toothbrush | padim_diag | max_score | 5.0% | 33.33% | 66.67% |
| toothbrush | padim_diag | topk_score | 0.0% | 33.33% | 66.67% |
| toothbrush | padim_diag | topk_score | 1.0% | 33.33% | 66.67% |
| toothbrush | padim_diag | topk_score | 5.0% | 41.67% | 58.33% |
| transistor | patchcore_lite | max_score | 0.0% | 65.00% | 35.00% |
| transistor | patchcore_lite | max_score | 1.0% | 65.00% | 35.00% |
| transistor | patchcore_lite | max_score | 5.0% | 93.33% | 6.67% |
| transistor | patchcore_lite | topk_score | 0.0% | 25.00% | 75.00% |
| transistor | patchcore_lite | topk_score | 1.0% | 25.00% | 75.00% |
| transistor | patchcore_lite | topk_score | 5.0% | 81.67% | 18.33% |
| transistor | padim_diag | max_score | 0.0% | 18.33% | 81.67% |
| transistor | padim_diag | max_score | 1.0% | 18.33% | 81.67% |
| transistor | padim_diag | max_score | 5.0% | 43.33% | 56.67% |
| transistor | padim_diag | topk_score | 0.0% | 20.00% | 80.00% |
| transistor | padim_diag | topk_score | 1.0% | 20.00% | 80.00% |
| transistor | padim_diag | topk_score | 5.0% | 41.67% | 58.33% |
| wood | patchcore_lite | max_score | 0.0% | 73.68% | 26.32% |
| wood | patchcore_lite | max_score | 1.0% | 73.68% | 26.32% |
| wood | patchcore_lite | max_score | 5.0% | 84.21% | 15.79% |
| wood | patchcore_lite | topk_score | 0.0% | 73.68% | 26.32% |
| wood | patchcore_lite | topk_score | 1.0% | 73.68% | 26.32% |
| wood | patchcore_lite | topk_score | 5.0% | 94.74% | 5.26% |
| wood | padim_diag | max_score | 0.0% | 42.11% | 57.89% |
| wood | padim_diag | max_score | 1.0% | 42.11% | 57.89% |
| wood | padim_diag | max_score | 5.0% | 78.95% | 21.05% |
| wood | padim_diag | topk_score | 0.0% | 47.37% | 52.63% |
| wood | padim_diag | topk_score | 1.0% | 47.37% | 52.63% |
| wood | padim_diag | topk_score | 5.0% | 78.95% | 21.05% |
| zipper | patchcore_lite | max_score | 0.0% | 37.50% | 62.50% |
| zipper | patchcore_lite | max_score | 1.0% | 87.50% | 12.50% |
| zipper | patchcore_lite | max_score | 5.0% | 87.50% | 12.50% |
| zipper | patchcore_lite | topk_score | 0.0% | 43.75% | 56.25% |
| zipper | patchcore_lite | topk_score | 1.0% | 87.50% | 12.50% |
| zipper | patchcore_lite | topk_score | 5.0% | 90.62% | 9.38% |
| zipper | padim_diag | max_score | 0.0% | 34.38% | 65.62% |
| zipper | padim_diag | max_score | 1.0% | 62.50% | 37.50% |
| zipper | padim_diag | max_score | 5.0% | 93.75% | 6.25% |
| zipper | padim_diag | topk_score | 0.0% | 37.50% | 62.50% |
| zipper | padim_diag | topk_score | 1.0% | 68.75% | 31.25% |
| zipper | padim_diag | topk_score | 5.0% | 93.75% | 6.25% |

## Reading the result

- AUROC/AUPR checks whether normal and defect images separate at all.
- The tradeoff curve checks the inspection question: how much good product is lost when defect false-pass is restricted.
- If PatchCore is strong and PaDiM is close, PaDiM is the more FPGA-friendly baseline candidate.
- If both are strong, the next step is attaching early-exit style acceleration to this inspection baseline.

Curve image: `results/mvtec_ad_parquet_all15_probe_002_tradeoff.png`
