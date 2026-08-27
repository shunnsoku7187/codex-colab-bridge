# 固定coreset比率によるprofile切替の公平比較

## 目的

bank数のカテゴリ別手動最適化を禁止し，全方式に同じk-center coreset比率を適用する。候補bankの抽出とk-center初期点は決定的に固定し，seed探索による上振れを避ける。

## top_pairs

| subset | system | 最低良品通過 | 平均良品通過 | 平均NN計算量 | 標準比 |
|---|---|---:|---:|---:|---:|
| toothbrush + zipper | ① 共通標準profile + merged bank | 75.00% | 84.38% | 35825664 | 1.000000x |
| toothbrush + zipper | ② 共通標準profile + category bank切替 | 75.00% | 84.38% | 17912832 | 0.500000x |
| toothbrush + zipper | 固定profile=toothbrush + category bank切替 | 58.33% | 65.10% | 204000 | 0.005694x |
| toothbrush + zipper | 固定profile=zipper + category bank切替 | 25.00% | 31.25% | 145040 | 0.004048x |
| toothbrush + zipper | ★ category profile + category bank 両切替 | 53.12% | 55.73% | 175640 | 0.004903x |
| toothbrush + zipper | 主張用差分 |  |  | vs標準削減 99.51% | vs bank-only追加削減 99.02% |
| bottle + toothbrush | ① 共通標準profile + merged bank | 75.00% | 87.50% | 35825664 | 1.000000x |
| bottle + toothbrush | ② 共通標準profile + category bank切替 | 75.00% | 87.50% | 17912832 | 0.500000x |
| bottle + toothbrush | 固定profile=bottle + category bank切替 | 16.67% | 55.83% | 208544 | 0.005821x |
| bottle + toothbrush | 固定profile=toothbrush + category bank切替 | 58.33% | 79.17% | 187200 | 0.005225x |
| bottle + toothbrush | ★ category profile + category bank 両切替 | 58.33% | 79.17% | 221504 | 0.006183x |
| bottle + toothbrush | 主張用差分 |  |  | vs標準削減 99.38% | vs bank-only追加削減 98.76% |
| pill + toothbrush | ① 共通標準profile + merged bank | 53.85% | 64.42% | 35825664 | 1.000000x |
| pill + toothbrush | ② 共通標準profile + category bank切替 | 53.85% | 64.42% | 17912832 | 0.500000x |
| pill + toothbrush | 固定profile=pill + category bank切替 | 19.23% | 22.12% | 235200 | 0.006565x |
| pill + toothbrush | 固定profile=toothbrush + category bank切替 | 3.85% | 31.09% | 220800 | 0.006163x |
| pill + toothbrush | ★ category profile + category bank 両切替 | 19.23% | 38.78% | 248160 | 0.006927x |
| pill + toothbrush | 主張用差分 |  |  | vs標準削減 99.31% | vs bank-only追加削減 98.61% |
| screw + toothbrush | ① 共通標準profile + merged bank | 14.63% | 44.82% | 35825664 | 1.000000x |
| screw + toothbrush | ② 共通標準profile + category bank切替 | 14.63% | 44.82% | 17912832 | 0.500000x |
| screw + toothbrush | 固定profile=screw + category bank切替 | 12.20% | 18.60% | 235200 | 0.006565x |
| screw + toothbrush | 固定profile=toothbrush + category bank切替 | 9.76% | 34.04% | 252000 | 0.007034x |
| screw + toothbrush | ★ category profile + category bank 両切替 | 12.20% | 35.26% | 248160 | 0.006927x |
| screw + toothbrush | 主張用差分 |  |  | vs標準削減 99.31% | vs bank-only追加削減 98.61% |
| bottle + zipper | ① 共通標準profile + merged bank | 87.50% | 93.75% | 36126720 | 1.000000x |
| bottle + zipper | ② 共通標準profile + category bank切替 | 87.50% | 93.75% | 18063360 | 0.500000x |
| bottle + zipper | 固定profile=bottle + category bank切替 | 71.88% | 83.44% | 346528 | 0.009592x |
| bottle + zipper | 固定profile=zipper + category bank切替 | 37.50% | 68.75% | 216580 | 0.005995x |
| bottle + zipper | ★ category profile + category bank 両切替 | 53.12% | 76.56% | 277144 | 0.007671x |
| bottle + zipper | 主張用差分 |  |  | vs標準削減 99.23% | vs bank-only追加削減 98.47% |
| pill + zipper | ① 共通標準profile + merged bank | 53.85% | 70.67% | 36126720 | 1.000000x |
| pill + zipper | ② 共通標準profile + category bank切替 | 53.85% | 70.67% | 18063360 | 0.500000x |
| pill + zipper | 固定profile=pill + category bank切替 | 19.23% | 43.99% | 373184 | 0.010330x |
| pill + zipper | 固定profile=zipper + category bank切替 | 19.23% | 28.37% | 233240 | 0.006456x |
| pill + zipper | ★ category profile + category bank 両切替 | 19.23% | 36.18% | 303800 | 0.008409x |
| pill + zipper | 主張用差分 |  |  | vs標準削減 99.16% | vs bank-only追加削減 98.32% |
| screw + zipper | ① 共通標準profile + merged bank | 14.63% | 54.19% | 36126720 | 1.000000x |
| screw + zipper | ② 共通標準profile + category bank切替 | 14.63% | 54.19% | 18063360 | 0.500000x |
| screw + zipper | 固定profile=screw + category bank切替 | 12.20% | 40.47% | 373184 | 0.010330x |
| screw + zipper | 固定profile=zipper + category bank切替 | 4.88% | 21.19% | 233240 | 0.006456x |
| screw + zipper | ★ category profile + category bank 両切替 | 12.20% | 32.66% | 303800 | 0.008409x |
| screw + zipper | 主張用差分 |  |  | vs標準削減 99.16% | vs bank-only追加削減 98.32% |
| bottle + pill | ① 共通標準profile + merged bank | 53.85% | 76.92% | 36126720 | 1.000000x |
| bottle + pill | ② 共通標準profile + category bank切替 | 53.85% | 76.92% | 18063360 | 0.500000x |
| bottle + pill | 固定profile=bottle + category bank切替 | 34.62% | 64.81% | 349664 | 0.009679x |
| bottle + pill | 固定profile=pill + category bank切替 | 19.23% | 54.62% | 349664 | 0.009679x |
| bottle + pill | ★ category profile + category bank 両切替 | 19.23% | 59.62% | 349664 | 0.009679x |
| bottle + pill | 主張用差分 |  |  | vs標準削減 99.03% | vs bank-only追加削減 98.06% |

## top_triples

| subset | system | 最低良品通過 | 平均良品通過 | 平均NN計算量 | 標準比 |
|---|---|---:|---:|---:|---:|
| bottle + toothbrush + zipper | ① 共通標準profile + merged bank | 75.00% | 87.50% | 53889024 | 1.000000x |
| bottle + toothbrush + zipper | ② 共通標準profile + category bank切替 | 75.00% | 87.50% | 17963008 | 0.333333x |
| bottle + toothbrush + zipper | 固定profile=bottle + category bank切替 | 16.67% | 61.18% | 262378 | 0.004869x |
| bottle + toothbrush + zipper | 固定profile=toothbrush + category bank切替 | 58.33% | 76.74% | 220800 | 0.004097x |
| bottle + toothbrush + zipper | 固定profile=zipper + category bank切替 | 25.00% | 54.17% | 163986 | 0.003043x |
| bottle + toothbrush + zipper | ★ category profile + category bank 両切替 | 53.12% | 70.49% | 224762 | 0.004171x |
| bottle + toothbrush + zipper | 主張用差分 |  |  | vs標準削減 99.58% | vs bank-only追加削減 98.75% |
| pill + toothbrush + zipper | ① 共通標準profile + merged bank | 53.85% | 72.12% | 53889024 | 1.000000x |
| pill + toothbrush + zipper | ② 共通標準profile + category bank切替 | 53.85% | 72.12% | 17963008 | 0.333333x |
| pill + toothbrush + zipper | 固定profile=pill + category bank切替 | 19.23% | 37.66% | 280149 | 0.005199x |
| pill + toothbrush + zipper | 固定profile=toothbrush + category bank切替 | 3.85% | 44.68% | 243200 | 0.004513x |
| pill + toothbrush + zipper | 固定profile=zipper + category bank切替 | 19.23% | 27.24% | 175093 | 0.003249x |
| pill + toothbrush + zipper | ★ category profile + category bank 両切替 | 19.23% | 43.56% | 242533 | 0.004501x |
| pill + toothbrush + zipper | 主張用差分 |  |  | vs標準削減 99.55% | vs bank-only追加削減 98.65% |
| screw + toothbrush + zipper | ① 共通標準profile + merged bank | 14.63% | 61.13% | 53889024 | 1.000000x |
| screw + toothbrush + zipper | ② 共通標準profile + category bank切替 | 14.63% | 61.13% | 17963008 | 0.333333x |
| screw + toothbrush + zipper | 固定profile=screw + category bank切替 | 12.20% | 35.32% | 280149 | 0.005199x |
| screw + toothbrush + zipper | 固定profile=toothbrush + category bank切替 | 9.76% | 46.65% | 264000 | 0.004899x |
| screw + toothbrush + zipper | 固定profile=zipper + category bank切替 | 4.88% | 22.46% | 175093 | 0.003249x |
| screw + toothbrush + zipper | ★ category profile + category bank 両切替 | 12.20% | 41.22% | 242533 | 0.004501x |
| screw + toothbrush + zipper | 主張用差分 |  |  | vs標準削減 99.55% | vs bank-only追加削減 98.65% |
| bottle + pill + toothbrush | ① 共通標準profile + merged bank | 53.85% | 76.28% | 53889024 | 1.000000x |
| bottle + pill + toothbrush | ② 共通標準profile + category bank切替 | 53.85% | 76.28% | 17963008 | 0.333333x |
| bottle + pill + toothbrush | 固定profile=bottle + category bank切替 | 16.67% | 48.76% | 264469 | 0.004908x |
| bottle + pill + toothbrush | 固定profile=pill + category bank切替 | 19.23% | 44.74% | 264469 | 0.004908x |
| bottle + pill + toothbrush | 固定profile=toothbrush + category bank切替 | 3.85% | 54.06% | 232000 | 0.004305x |
| bottle + pill + toothbrush | ★ category profile + category bank 両切替 | 19.23% | 59.19% | 273109 | 0.005068x |
| bottle + pill + toothbrush | 主張用差分 |  |  | vs標準削減 99.49% | vs bank-only追加削減 98.48% |
| bottle + screw + toothbrush | ① 共通標準profile + merged bank | 19.51% | 64.84% | 53889024 | 1.000000x |
| bottle + screw + toothbrush | ② 共通標準profile + category bank切替 | 19.51% | 64.84% | 17963008 | 0.333333x |
| bottle + screw + toothbrush | 固定profile=bottle + category bank切替 | 12.20% | 41.29% | 264469 | 0.004908x |
| bottle + screw + toothbrush | 固定profile=screw + category bank切替 | 12.20% | 42.40% | 264469 | 0.004908x |
| bottle + screw + toothbrush | 固定profile=toothbrush + category bank切替 | 9.76% | 56.03% | 252800 | 0.004691x |
| bottle + screw + toothbrush | ★ category profile + category bank 両切替 | 12.20% | 56.84% | 273109 | 0.005068x |
| bottle + screw + toothbrush | 主張用差分 |  |  | vs標準削減 99.49% | vs bank-only追加削減 98.48% |
| pill + screw + toothbrush | ① 共通標準profile + merged bank | 19.51% | 52.02% | 53889024 | 1.000000x |
| pill + screw + toothbrush | ② 共通標準profile + category bank切替 | 19.51% | 49.45% | 17963008 | 0.333333x |
| pill + screw + toothbrush | 固定profile=pill + category bank切替 | 12.20% | 18.81% | 282240 | 0.005237x |
| pill + screw + toothbrush | 固定profile=screw + category bank切替 | 12.20% | 18.81% | 282240 | 0.005237x |
| pill + screw + toothbrush | 固定profile=toothbrush + category bank切替 | 3.85% | 23.98% | 275200 | 0.005107x |
| pill + screw + toothbrush | ★ category profile + category bank 両切替 | 12.20% | 29.92% | 290880 | 0.005398x |
| pill + screw + toothbrush | 主張用差分 |  |  | vs標準削減 99.46% | vs bank-only追加削減 98.38% |
| bottle + pill + zipper | ① 共通標準profile + merged bank | 53.85% | 80.45% | 54190080 | 1.000000x |
| bottle + pill + zipper | ② 共通標準profile + category bank切替 | 53.85% | 80.45% | 18063360 | 0.333333x |
| bottle + pill + zipper | 固定profile=bottle + category bank切替 | 34.62% | 67.16% | 356458 | 0.006578x |
| bottle + pill + zipper | 固定profile=pill + category bank切替 | 19.23% | 59.33% | 356458 | 0.006578x |
| bottle + pill + zipper | 固定profile=zipper + category bank切替 | 19.23% | 52.24% | 222786 | 0.004111x |
| bottle + pill + zipper | ★ category profile + category bank 両切替 | 19.23% | 57.45% | 310202 | 0.005724x |
| bottle + pill + zipper | 主張用差分 |  |  | vs標準削減 99.43% | vs bank-only追加削減 98.28% |
| bottle + screw + zipper | ① 共通標準profile + merged bank | 21.95% | 69.82% | 54190080 | 1.000000x |
| bottle + screw + zipper | ② 共通標準profile + category bank切替 | 19.51% | 69.00% | 18063360 | 0.333333x |
| bottle + screw + zipper | 固定profile=bottle + category bank切替 | 12.20% | 59.69% | 356458 | 0.006578x |
| bottle + screw + zipper | 固定profile=screw + category bank切替 | 12.20% | 56.98% | 356458 | 0.006578x |
| bottle + screw + zipper | 固定profile=zipper + category bank切替 | 4.88% | 47.46% | 222786 | 0.004111x |
| bottle + screw + zipper | ★ category profile + category bank 両切替 | 12.20% | 55.11% | 310202 | 0.005724x |
| bottle + screw + zipper | 主張用差分 |  |  | vs標準削減 99.43% | vs bank-only追加削減 98.28% |
