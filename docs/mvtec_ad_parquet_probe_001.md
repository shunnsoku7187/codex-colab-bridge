# MVTec AD Parquet anomaly baseline probe

Purpose: check whether MVTec AD can provide a stronger inspection baseline than the small KSDD splits.

## Materialization

- Dataset root: `/home/shunya/codex-gpu-work/data/mvtec_ad`
- Materialized root: `/home/shunya/codex-gpu-work/data/mvtec_ad_materialized`
- Parquet shards: `31`
- Written images this run: `1982`

## Category sample counts

| category | train | test | good rows | defect rows | masks |
|---|---:|---:|---:|---:|---:|
| bottle | 209 | 83 | 229 | 63 | 63 |
| cable | 224 | 150 | 282 | 92 | 92 |
| hazelnut | 391 | 110 | 431 | 70 | 70 |
| metal_nut | 220 | 115 | 242 | 93 | 93 |
| screw | 320 | 160 | 361 | 119 | 119 |

## Image-level anomaly separation

| category | method | score | AUROC | AUPR |
|---|---|---|---:|---:|
| bottle | patchcore_lite | max_score | 1.0 | 1.0 |
| bottle | patchcore_lite | topk_score | 1.0 | 1.0 |
| bottle | padim_diag | max_score | 0.995238 | 0.998479 |
| bottle | padim_diag | topk_score | 0.996032 | 0.998743 |
| cable | patchcore_lite | max_score | 0.944528 | 0.965824 |
| cable | patchcore_lite | topk_score | 0.960645 | 0.977851 |
| cable | padim_diag | max_score | 0.750937 | 0.852304 |
| cable | padim_diag | topk_score | 0.757121 | 0.855688 |
| hazelnut | patchcore_lite | max_score | 0.9975 | 0.998636 |
| hazelnut | patchcore_lite | topk_score | 0.999286 | 0.999603 |
| hazelnut | padim_diag | max_score | 0.571786 | 0.680505 |
| hazelnut | padim_diag | topk_score | 0.579286 | 0.671979 |
| metal_nut | patchcore_lite | max_score | 0.98827 | 0.997541 |
| metal_nut | patchcore_lite | topk_score | 0.988759 | 0.997621 |
| metal_nut | padim_diag | max_score | 0.833333 | 0.960779 |
| metal_nut | padim_diag | topk_score | 0.836266 | 0.961563 |
| screw | patchcore_lite | max_score | 0.774339 | 0.907544 |
| screw | patchcore_lite | topk_score | 0.82066 | 0.926145 |
| screw | padim_diag | max_score | 0.656692 | 0.82956 |
| screw | padim_diag | topk_score | 0.668375 | 0.833624 |

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
| cable | patchcore_lite | max_score | 0.0% | 46.55% | 53.45% |
| cable | patchcore_lite | max_score | 1.0% | 46.55% | 53.45% |
| cable | patchcore_lite | max_score | 5.0% | 56.90% | 43.10% |
| cable | patchcore_lite | topk_score | 0.0% | 41.38% | 58.62% |
| cable | patchcore_lite | topk_score | 1.0% | 41.38% | 58.62% |
| cable | patchcore_lite | topk_score | 5.0% | 72.41% | 27.59% |
| cable | padim_diag | max_score | 0.0% | 3.45% | 96.55% |
| cable | padim_diag | max_score | 1.0% | 3.45% | 96.55% |
| cable | padim_diag | max_score | 5.0% | 12.07% | 87.93% |
| cable | padim_diag | topk_score | 0.0% | 3.45% | 96.55% |
| cable | padim_diag | topk_score | 1.0% | 3.45% | 96.55% |
| cable | padim_diag | topk_score | 5.0% | 13.79% | 86.21% |
| hazelnut | patchcore_lite | max_score | 0.0% | 87.50% | 12.50% |
| hazelnut | patchcore_lite | max_score | 1.0% | 87.50% | 12.50% |
| hazelnut | patchcore_lite | max_score | 5.0% | 100.00% | 0.00% |
| hazelnut | patchcore_lite | topk_score | 0.0% | 95.00% | 5.00% |
| hazelnut | patchcore_lite | topk_score | 1.0% | 95.00% | 5.00% |
| hazelnut | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| hazelnut | padim_diag | max_score | 0.0% | 7.50% | 92.50% |
| hazelnut | padim_diag | max_score | 1.0% | 7.50% | 92.50% |
| hazelnut | padim_diag | max_score | 5.0% | 15.00% | 85.00% |
| hazelnut | padim_diag | topk_score | 0.0% | 10.00% | 90.00% |
| hazelnut | padim_diag | topk_score | 1.0% | 10.00% | 90.00% |
| hazelnut | padim_diag | topk_score | 5.0% | 17.50% | 82.50% |
| metal_nut | patchcore_lite | max_score | 0.0% | 22.73% | 77.27% |
| metal_nut | patchcore_lite | max_score | 1.0% | 22.73% | 77.27% |
| metal_nut | patchcore_lite | max_score | 5.0% | 100.00% | 0.00% |
| metal_nut | patchcore_lite | topk_score | 0.0% | 27.27% | 72.73% |
| metal_nut | patchcore_lite | topk_score | 1.0% | 27.27% | 72.73% |
| metal_nut | patchcore_lite | topk_score | 5.0% | 100.00% | 0.00% |
| metal_nut | padim_diag | max_score | 0.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | max_score | 1.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | max_score | 5.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 0.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 1.0% | 9.09% | 90.91% |
| metal_nut | padim_diag | topk_score | 5.0% | 9.09% | 90.91% |
| screw | patchcore_lite | max_score | 0.0% | 12.20% | 87.80% |
| screw | patchcore_lite | max_score | 1.0% | 14.63% | 85.37% |
| screw | patchcore_lite | max_score | 5.0% | 34.15% | 65.85% |
| screw | patchcore_lite | topk_score | 0.0% | 7.32% | 92.68% |
| screw | patchcore_lite | topk_score | 1.0% | 14.63% | 85.37% |
| screw | patchcore_lite | topk_score | 5.0% | 43.90% | 56.10% |
| screw | padim_diag | max_score | 0.0% | 0.00% | 100.00% |
| screw | padim_diag | max_score | 1.0% | 0.00% | 100.00% |
| screw | padim_diag | max_score | 5.0% | 9.76% | 90.24% |
| screw | padim_diag | topk_score | 0.0% | 0.00% | 100.00% |
| screw | padim_diag | topk_score | 1.0% | 0.00% | 100.00% |
| screw | padim_diag | topk_score | 5.0% | 4.88% | 95.12% |

## Reading the result

- AUROC/AUPR checks whether normal and defect images separate at all.
- The tradeoff curve checks the inspection question: how much good product is lost when defect false-pass is restricted.
- If PatchCore is strong and PaDiM is close, PaDiM is the more FPGA-friendly baseline candidate.
- If both are strong, the next step is attaching early-exit style acceleration to this inspection baseline.

Curve image: `results/mvtec_ad_parquet_probe_001_tradeoff.png`
