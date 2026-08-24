# PatchCore-lite reduction-limit probe

## Purpose

This experiment explains why a category-specific PatchCore-lite profile can or cannot be reduced.
It supports the profile-selection hypothesis with measurable quantities instead of only reporting that a sweep happened to work.

## Formulas used

- Nearest-neighbor cost proxy: `C_NN = P * B * D`
  - `P`: number of test patches per image
  - `B`: number of memory-bank patches
  - `D`: feature dimension per patch
- Normal-bank coverage radius: `R_q = quantile_q min_{b in Bank} ||z_normal - b||_2`
- Defect-safe threshold at false-pass target `alpha`: `tau_alpha = quantile_alpha(S_defect)`
- Good pass predicted by the score distributions: `GP_alpha = Pr[S_good < tau_alpha]`
- Margin for accepting 95% of good samples: `M_95 = tau_alpha - quantile_0.95(S_good)`
  - `M_95 > 0`: enough score-space margin remains.
  - `M_95 < 0`: normal and defect score distributions overlap under that constraint.

## Category summary

| category | selected profile | good pass | NN ops ratio | patch ratio | bank ratio | dim ratio | margin M95 | defect collision | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bottle | `effb0_l23_g10_b500_topk0p05` | 95.00% | 0.001772x | 0.5102x | 0.0417x | 0.0833x | 0.019948 | 84.83% | reducible: positive margin remains |
| cable | `res18_l23_g7_b1500_topk0p05` | 24.14% | 0.007812x | 0.2500x | 0.1250x | 0.2500x | -0.077431 | 89.02% | limit: normal/defect score overlap |
| capsule | `effb0_l23_g14_b500_topk0p02` | 4.35% | 0.003472x | 1.0000x | 0.0417x | 0.0833x | -0.184371 | 92.43% | limit: normal/defect score overlap |
| carpet | `res18_l23_g14_b750_topk0p01` | 71.43% | 0.015625x | 1.0000x | 0.0625x | 0.2500x | -0.016603 | 85.48% | limit: normal/defect score overlap |
| grid | `res34_l23_g10_b750_topk0p02` | 23.81% | 0.007972x | 0.5102x | 0.0625x | 0.2500x | -0.042774 | 91.12% | limit: normal/defect score overlap |
| hazelnut | `res34_l23_g14_b500_topk0p02` | 97.50% | 0.010417x | 1.0000x | 0.0417x | 0.2500x | 0.003589 | 89.88% | reducible: positive margin remains |
| leather | `effb0_l23_g14_b125_topk0p005` | 96.88% | 0.000868x | 1.0000x | 0.0104x | 0.0833x | 0.052533 | 88.34% | reducible: positive margin remains |
| metal_nut | `wrn_l3_g14_b1500_topk0p01` | 77.27% | 0.083333x | 1.0000x | 0.1250x | 0.6667x | -0.008579 | 83.55% | limit: normal/defect score overlap |
| pill | `mobv3l_l23_g7_b2000_topk0p05` | 30.77% | 0.003472x | 0.2500x | 0.1667x | 0.0833x | -0.042631 | 90.92% | limit: normal/defect score overlap |
| screw | `res34_l23_g14_b9000_topk0p05` | 17.07% | 0.187500x | 1.0000x | 0.7500x | 0.2500x | -0.039462 | 92.02% | limit: normal/defect score overlap |
| tile | `effb0_l23_g14_b250_topk0p02` | 87.88% | 0.001736x | 1.0000x | 0.0208x | 0.0833x | -0.011572 | 86.07% | reducible but margin is tight |
| toothbrush | `effb0_l23_g10_b750_topk0p05` | 75.00% | 0.002712x | 0.5102x | 0.0638x | 0.0833x | -0.028964 | 89.93% | limit: normal/defect score overlap |
| transistor | `res34_l23_g10_b6000_topk0p005` | 40.00% | 0.063776x | 0.5102x | 0.5000x | 0.2500x | -0.039393 | 84.33% | limit: normal/defect score overlap |
| wood | `mobv3s_l23_g7_b125_topk0p005` | 52.63% | 0.000136x | 0.2500x | 0.0104x | 0.0521x | -0.089841 | 84.49% | limit: normal/defect score overlap |
| zipper | `wrn_l3_g14_b1500_topk0p05` | 84.38% | 0.083333x | 1.0000x | 0.1250x | 0.6667x | -0.02277 | 81.60% | reducible but margin is tight |

## Reading

- A category is a strong reduction candidate when the reduced profile keeps good pass high and `M_95` remains positive while `C_NN` is much smaller.
- A category is near its reduction limit when `M_95` becomes negative: there is no threshold that both accepts most good images and rejects almost all defects under the reduced feature space.
- If `C_NN` remains large even in the selected profile, the category likely needs either stronger features, finer grids, or a larger bank.
