# k-centerを標準適用したprofile切替比較

## 目的

全方式でk-center coresetによるbank選択を標準適用し，bank数を最適化変数から外した状態で，特徴層・grid・top-k・閾値のカテゴリ別切替だけに効果が残るかを確認する。

## 条件

- backbone: `wide_resnet50_2` 固定
- 欠陥誤通過率上限: 3.00%
- profile選択時のbank数: 12000
- bank選択: 全方式でk-center coresetを使用する。
- ABでは `{A+B}` と `{A}+{B}` の総bank数を同じにする。
- ABCでは `{A+B+C}` と `{A}+{B}+{C}` の総bank数を同じにする。

## bank/category = 600

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + cable | standard profile + k-center common bank | 41.38% | 70.69% | 180633600.0 | 1.000000x | 1200 |
| bottle + cable | standard profile + k-center category-bank switch | 13.79% | 56.90% | 90316800.0 | 0.500000x | 1200 |
| bottle + cable | fixed profile=bottle + k-center category-bank switch | 43.10% | 71.55% | 60211200.0 | 0.333333x | 1200 |
| bottle + cable | fixed profile=cable + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| bottle + cable | proposed profile switch + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| bottle + cable | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + carpet | standard profile + k-center common bank | 21.43% | 49.51% | 180633600.0 | 1.000000x | 1200 |
| cable + carpet | standard profile + k-center category-bank switch | 13.79% | 31.90% | 90316800.0 | 0.500000x | 1200 |
| cable + carpet | fixed profile=cable + k-center category-bank switch | 44.83% | 45.63% | 60211200.0 | 0.333333x | 1200 |
| cable + carpet | fixed profile=carpet + k-center category-bank switch | 43.10% | 53.69% | 60211200.0 | 0.333333x | 1200 |
| cable + carpet | proposed profile switch + k-center category-bank switch | 44.83% | 54.56% | 60211200.0 | 0.333333x | 1200 |
| cable + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + hazelnut | standard profile + k-center common bank | 37.93% | 68.97% | 180633600.0 | 1.000000x | 1200 |
| cable + hazelnut | standard profile + k-center category-bank switch | 13.79% | 56.90% | 90316800.0 | 0.500000x | 1200 |
| cable + hazelnut | fixed profile=cable + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 43.10% | 71.55% | 60211200.0 | 0.333333x | 1200 |
| cable + hazelnut | proposed profile switch + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + leather | standard profile + k-center common bank | 34.48% | 67.24% | 180633600.0 | 1.000000x | 1200 |
| cable + leather | standard profile + k-center category-bank switch | 13.79% | 56.90% | 90316800.0 | 0.500000x | 1200 |
| cable + leather | fixed profile=cable + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + leather | fixed profile=leather + k-center category-bank switch | 43.10% | 71.55% | 60211200.0 | 0.333333x | 1200 |
| cable + leather | proposed profile switch + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + metal_nut | standard profile + k-center common bank | 32.76% | 64.11% | 180633600.0 | 1.000000x | 1200 |
| cable + metal_nut | standard profile + k-center category-bank switch | 13.79% | 56.90% | 90316800.0 | 0.500000x | 1200 |
| cable + metal_nut | fixed profile=cable + k-center category-bank switch | 44.83% | 70.14% | 60211200.0 | 0.333333x | 1200 |
| cable + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 43.10% | 71.55% | 60211200.0 | 0.333333x | 1200 |
| cable + metal_nut | proposed profile switch + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + tile | standard profile + k-center common bank | 55.17% | 77.59% | 180633600.0 | 1.000000x | 1200 |
| cable + tile | standard profile + k-center category-bank switch | 13.79% | 56.90% | 90316800.0 | 0.500000x | 1200 |
| cable + tile | fixed profile=cable + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + tile | fixed profile=tile + k-center category-bank switch | 43.10% | 71.55% | 60211200.0 | 0.333333x | 1200 |
| cable + tile | proposed profile switch + k-center category-bank switch | 44.83% | 72.41% | 60211200.0 | 0.333333x | 1200 |
| cable + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + transistor | standard profile + k-center common bank | 36.67% | 37.30% | 180633600.0 | 1.000000x | 1200 |
| cable + transistor | standard profile + k-center category-bank switch | 13.79% | 52.73% | 90316800.0 | 0.500000x | 1200 |
| cable + transistor | fixed profile=cable + k-center category-bank switch | 44.83% | 60.75% | 60211200.0 | 0.333333x | 1200 |
| cable + transistor | fixed profile=transistor + k-center category-bank switch | 32.76% | 59.71% | 60211200.0 | 0.333333x | 1200 |
| cable + transistor | proposed profile switch + k-center category-bank switch | 44.83% | 65.75% | 60211200.0 | 0.333333x | 1200 |
| cable + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| cable + zipper | standard profile + k-center common bank | 36.21% | 47.79% | 180633600.0 | 1.000000x | 1200 |
| cable + zipper | standard profile + k-center category-bank switch | 13.79% | 53.77% | 90316800.0 | 0.500000x | 1200 |
| cable + zipper | fixed profile=cable + k-center category-bank switch | 44.83% | 66.16% | 60211200.0 | 0.333333x | 1200 |
| cable + zipper | fixed profile=zipper + k-center category-bank switch | 43.10% | 62.18% | 60211200.0 | 0.333333x | 1200 |
| cable + zipper | proposed profile switch + k-center category-bank switch | 44.83% | 63.04% | 60211200.0 | 0.333333x | 1200 |
| cable + zipper | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| bottle + cable + carpet | standard profile + k-center common bank | 28.57% | 64.12% | 270950400.0 | 1.000000x | 1800 |
| bottle + cable + carpet | standard profile + k-center category-bank switch | 13.79% | 54.60% | 90316800.0 | 0.333333x | 1800 |
| bottle + cable + carpet | fixed profile=bottle + k-center category-bank switch | 43.10% | 69.13% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + carpet | fixed profile=cable + k-center category-bank switch | 44.83% | 63.75% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + carpet | fixed profile=carpet + k-center category-bank switch | 43.10% | 69.13% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + carpet | proposed profile switch + k-center category-bank switch | 44.83% | 69.70% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + carpet | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| bottle + cable + hazelnut | standard profile + k-center common bank | 60.34% | 86.78% | 270950400.0 | 1.000000x | 1800 |
| bottle + cable + hazelnut | standard profile + k-center category-bank switch | 13.79% | 71.26% | 90316800.0 | 0.333333x | 1800 |
| bottle + cable + hazelnut | fixed profile=bottle + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + hazelnut | fixed profile=cable + k-center category-bank switch | 44.83% | 81.61% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + hazelnut | proposed profile switch + k-center category-bank switch | 44.83% | 81.61% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| bottle + cable + leather | standard profile + k-center common bank | 46.55% | 82.18% | 270950400.0 | 1.000000x | 1800 |
| bottle + cable + leather | standard profile + k-center category-bank switch | 13.79% | 71.26% | 90316800.0 | 0.333333x | 1800 |
| bottle + cable + leather | fixed profile=bottle + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + leather | fixed profile=cable + k-center category-bank switch | 44.83% | 81.61% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + leather | fixed profile=leather + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + leather | proposed profile switch + k-center category-bank switch | 44.83% | 81.61% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
| bottle + cable + metal_nut | standard profile + k-center common bank | 34.48% | 78.16% | 270950400.0 | 1.000000x | 1800 |
| bottle + cable + metal_nut | standard profile + k-center category-bank switch | 13.79% | 71.26% | 90316800.0 | 0.333333x | 1800 |
| bottle + cable + metal_nut | fixed profile=bottle + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + metal_nut | fixed profile=cable + k-center category-bank switch | 44.83% | 80.09% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 43.10% | 81.03% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + metal_nut | proposed profile switch + k-center category-bank switch | 44.83% | 81.61% | 60211200.0 | 0.222222x | 1800 |
| bottle + cable + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3103 |  |
## bank/category = 1500

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + carpet | standard profile + k-center common bank | 67.86% | 83.93% | 451584000.0 | 1.000000x | 3000 |
| bottle + carpet | standard profile + k-center category-bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| bottle + carpet | fixed profile=bottle + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | proposed profile switch + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| cable + carpet | standard profile + k-center common bank | 46.43% | 58.56% | 451584000.0 | 1.000000x | 3000 |
| cable + carpet | standard profile + k-center category-bank switch | 50.00% | 50.86% | 225792000.0 | 0.500000x | 3000 |
| cable + carpet | fixed profile=cable + k-center category-bank switch | 53.57% | 69.89% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | fixed profile=carpet + k-center category-bank switch | 60.34% | 73.03% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | proposed profile switch + k-center category-bank switch | 85.71% | 85.96% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + hazelnut | standard profile + k-center common bank | 64.29% | 82.14% | 451584000.0 | 1.000000x | 3000 |
| carpet + hazelnut | standard profile + k-center category-bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + hazelnut | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | proposed profile switch + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + leather | standard profile + k-center common bank | 60.71% | 80.36% | 451584000.0 | 1.000000x | 3000 |
| carpet + leather | standard profile + k-center category-bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + leather | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | fixed profile=leather + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | proposed profile switch + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + metal_nut | standard profile + k-center common bank | 50.00% | 75.00% | 451584000.0 | 1.000000x | 3000 |
| carpet + metal_nut | standard profile + k-center category-bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + metal_nut | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | proposed profile switch + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + tile | standard profile + k-center common bank | 50.00% | 75.00% | 451584000.0 | 1.000000x | 3000 |
| carpet + tile | standard profile + k-center category-bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + tile | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | fixed profile=tile + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | proposed profile switch + k-center category-bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + transistor | standard profile + k-center common bank | 60.71% | 73.69% | 451584000.0 | 1.000000x | 3000 |
| carpet + transistor | standard profile + k-center category-bank switch | 50.00% | 72.50% | 225792000.0 | 0.500000x | 3000 |
| carpet + transistor | fixed profile=carpet + k-center category-bank switch | 85.71% | 92.02% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | fixed profile=transistor + k-center category-bank switch | 85.71% | 91.19% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | proposed profile switch + k-center category-bank switch | 85.71% | 91.19% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + cable + carpet | standard profile + k-center common bank | 50.00% | 66.67% | 677376000.0 | 1.000000x | 4500 |
| bottle + cable + carpet | standard profile + k-center category-bank switch | 50.00% | 67.24% | 225792000.0 | 0.333333x | 4500 |
| bottle + cable + carpet | fixed profile=bottle + k-center category-bank switch | 60.34% | 82.02% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | fixed profile=cable + k-center category-bank switch | 53.57% | 79.93% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | fixed profile=carpet + k-center category-bank switch | 60.34% | 82.02% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | proposed profile switch + k-center category-bank switch | 85.71% | 90.64% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + hazelnut | standard profile + k-center common bank | 35.71% | 78.57% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + hazelnut | standard profile + k-center category-bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + hazelnut | fixed profile=bottle + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | fixed profile=carpet + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | proposed profile switch + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + leather | standard profile + k-center common bank | 67.86% | 89.29% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + leather | standard profile + k-center category-bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + leather | fixed profile=bottle + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | fixed profile=carpet + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | fixed profile=leather + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | proposed profile switch + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + metal_nut | standard profile + k-center common bank | 60.71% | 86.90% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + metal_nut | standard profile + k-center category-bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + metal_nut | fixed profile=bottle + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | fixed profile=carpet + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | proposed profile switch + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + tile | standard profile + k-center common bank | 67.86% | 88.28% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + tile | standard profile + k-center category-bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + tile | fixed profile=bottle + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | fixed profile=carpet + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | fixed profile=tile + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | proposed profile switch + k-center category-bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
## bank/category = 3000

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + carpet | standard profile + k-center common bank | 71.43% | 85.71% | 903168000.0 | 1.000000x | 6000 |
| bottle + carpet | standard profile + k-center category-bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| bottle + carpet | fixed profile=bottle + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | proposed profile switch + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + hazelnut | standard profile + k-center common bank | 53.57% | 76.79% | 903168000.0 | 1.000000x | 6000 |
| carpet + hazelnut | standard profile + k-center category-bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + hazelnut | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | proposed profile switch + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + leather | standard profile + k-center common bank | 67.86% | 83.93% | 903168000.0 | 1.000000x | 6000 |
| carpet + leather | standard profile + k-center category-bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + leather | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | fixed profile=leather + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | proposed profile switch + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + metal_nut | standard profile + k-center common bank | 60.71% | 80.36% | 903168000.0 | 1.000000x | 6000 |
| carpet + metal_nut | standard profile + k-center category-bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + metal_nut | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | proposed profile switch + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + tile | standard profile + k-center common bank | 67.86% | 82.41% | 903168000.0 | 1.000000x | 6000 |
| carpet + tile | standard profile + k-center category-bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + tile | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | fixed profile=tile + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | proposed profile switch + k-center category-bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + transistor | standard profile + k-center common bank | 67.86% | 78.93% | 903168000.0 | 1.000000x | 6000 |
| carpet + transistor | standard profile + k-center category-bank switch | 60.71% | 74.52% | 451584000.0 | 0.500000x | 6000 |
| carpet + transistor | fixed profile=carpet + k-center category-bank switch | 89.29% | 92.14% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | fixed profile=transistor + k-center category-bank switch | 71.43% | 84.88% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | proposed profile switch + k-center category-bank switch | 89.29% | 93.81% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + hazelnut | standard profile + k-center common bank | 60.71% | 86.90% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + hazelnut | standard profile + k-center category-bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + hazelnut | fixed profile=bottle + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | fixed profile=carpet + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | fixed profile=hazelnut + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | proposed profile switch + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + leather | standard profile + k-center common bank | 71.43% | 90.48% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + leather | standard profile + k-center category-bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + leather | fixed profile=bottle + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | fixed profile=carpet + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | fixed profile=leather + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | proposed profile switch + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + metal_nut | standard profile + k-center common bank | 64.29% | 88.10% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + metal_nut | standard profile + k-center category-bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + metal_nut | fixed profile=bottle + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | fixed profile=carpet + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | fixed profile=metal_nut + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | proposed profile switch + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + tile | standard profile + k-center common bank | 53.57% | 83.51% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + tile | standard profile + k-center category-bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + tile | fixed profile=bottle + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | fixed profile=carpet + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | fixed profile=tile + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | proposed profile switch + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + transistor | standard profile + k-center common bank | 71.43% | 86.03% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + transistor | standard profile + k-center category-bank switch | 60.71% | 83.02% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + transistor | fixed profile=bottle + k-center category-bank switch | 89.29% | 94.76% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | fixed profile=carpet + k-center category-bank switch | 89.29% | 94.76% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | fixed profile=transistor + k-center category-bank switch | 71.43% | 89.92% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | proposed profile switch + k-center category-bank switch | 89.29% | 95.87% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + toothbrush | standard profile + k-center common bank | 75.00% | 84.52% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + toothbrush | standard profile + k-center category-bank switch | 60.71% | 78.57% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + toothbrush | fixed profile=bottle + k-center category-bank switch | 75.00% | 88.10% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + toothbrush | fixed profile=carpet + k-center category-bank switch | 75.00% | 88.10% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + toothbrush | fixed profile=toothbrush + k-center category-bank switch | 57.14% | 82.94% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + toothbrush | proposed profile switch + k-center category-bank switch | 89.29% | 93.65% | 351232000.0 | 0.259259x | 9000 |
| bottle + carpet + toothbrush | difference |  |  | vs common削減 74.07% / vs bank-only追加削減 22.22% | min good差 vs bank-only +0.2857 |  |

## 読み取り方

- `standard profile + k-center common bank` は，対象カテゴリ全体のk-center bankを毎回探索する実装baselineである。
- `standard profile + k-center category-bank switch` は，総保存bank数は同じだが，検品対象カテゴリのk-center bankだけを探索する方式である。
- `fixed profile=X` は，X向けprofileを他カテゴリにも流用した場合の劣化を確認する対照実験。
- `proposed` が `bank-only` よりさらに軽ければ，bank数ではなくprofile切替自体に効果がある。
- `proposed` が `fixed profile=X` より良品通過率で勝てば，単一軽量profileの使い回しではなくカテゴリ別profileが必要である。
