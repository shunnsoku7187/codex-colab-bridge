# 固定bank数・固定backboneでのprofile切替比較

## 目的

bank数を最適化変数から外し，総bank数を揃えた状態で，特徴層・grid・top-k・閾値のカテゴリ別切替だけに効果が残るかを確認する。

## 条件

- backbone: `wide_resnet50_2` 固定
- 欠陥誤通過率上限: 3.00%
- profile選択時のbank数: 12000
- ABでは `{A+B}` と `{A}+{B}` の総bank数を同じにする。
- ABCでは `{A+B+C}` と `{A}+{B}+{C}` の総bank数を同じにする。

## bank/category = 500

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + cable | common standard profile + common bank | 27.59% | 63.79% | 150528000.0 | 1.000000x | 1000 |
| bottle + cable | common standard profile + category bank switch | 25.86% | 62.93% | 75264000.0 | 0.500000x | 1000 |
| bottle + cable | fixed profile=bottle + category bank switch | 60.34% | 80.17% | 50176000.0 | 0.333333x | 1000 |
| bottle + cable | fixed profile=cable + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| bottle + cable | proposed profile switch + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| bottle + cable | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + capsule | common standard profile + common bank | 26.09% | 33.73% | 150528000.0 | 1.000000x | 1000 |
| cable + capsule | common standard profile + category bank switch | 25.86% | 43.37% | 75264000.0 | 0.500000x | 1000 |
| cable + capsule | fixed profile=cable + category bank switch | 60.87% | 62.33% | 50176000.0 | 0.333333x | 1000 |
| cable + capsule | fixed profile=capsule + category bank switch | 65.22% | 68.82% | 50176000.0 | 0.333333x | 1000 |
| cable + capsule | proposed profile switch + category bank switch | 63.79% | 64.51% | 50176000.0 | 0.333333x | 1000 |
| cable + capsule | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + hazelnut | common standard profile + common bank | 20.69% | 60.34% | 150528000.0 | 1.000000x | 1000 |
| cable + hazelnut | common standard profile + category bank switch | 25.86% | 62.93% | 75264000.0 | 0.500000x | 1000 |
| cable + hazelnut | fixed profile=cable + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + hazelnut | fixed profile=hazelnut + category bank switch | 60.34% | 80.17% | 50176000.0 | 0.333333x | 1000 |
| cable + hazelnut | proposed profile switch + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + leather | common standard profile + common bank | 29.31% | 64.66% | 150528000.0 | 1.000000x | 1000 |
| cable + leather | common standard profile + category bank switch | 25.86% | 62.93% | 75264000.0 | 0.500000x | 1000 |
| cable + leather | fixed profile=cable + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + leather | fixed profile=leather + category bank switch | 60.34% | 80.17% | 50176000.0 | 0.333333x | 1000 |
| cable + leather | proposed profile switch + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + metal_nut | common standard profile + common bank | 31.03% | 60.97% | 150528000.0 | 1.000000x | 1000 |
| cable + metal_nut | common standard profile + category bank switch | 25.86% | 62.93% | 75264000.0 | 0.500000x | 1000 |
| cable + metal_nut | fixed profile=cable + category bank switch | 63.79% | 79.62% | 50176000.0 | 0.333333x | 1000 |
| cable + metal_nut | fixed profile=metal_nut + category bank switch | 60.34% | 80.17% | 50176000.0 | 0.333333x | 1000 |
| cable + metal_nut | proposed profile switch + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + tile | common standard profile + common bank | 37.93% | 65.94% | 150528000.0 | 1.000000x | 1000 |
| cable + tile | common standard profile + category bank switch | 25.86% | 61.42% | 75264000.0 | 0.500000x | 1000 |
| cable + tile | fixed profile=cable + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + tile | fixed profile=tile + category bank switch | 60.34% | 80.17% | 50176000.0 | 0.333333x | 1000 |
| cable + tile | proposed profile switch + category bank switch | 63.79% | 81.90% | 50176000.0 | 0.333333x | 1000 |
| cable + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + transistor | common standard profile + common bank | 27.59% | 40.46% | 150528000.0 | 1.000000x | 1000 |
| cable + transistor | common standard profile + category bank switch | 25.86% | 56.26% | 75264000.0 | 0.500000x | 1000 |
| cable + transistor | fixed profile=cable + category bank switch | 55.00% | 59.40% | 50176000.0 | 0.333333x | 1000 |
| cable + transistor | fixed profile=transistor + category bank switch | 79.31% | 87.99% | 50176000.0 | 0.333333x | 1000 |
| cable + transistor | proposed profile switch + category bank switch | 63.79% | 80.23% | 50176000.0 | 0.333333x | 1000 |
| cable + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| cable + zipper | common standard profile + common bank | 46.55% | 48.28% | 150528000.0 | 1.000000x | 1000 |
| cable + zipper | common standard profile + category bank switch | 25.86% | 59.81% | 75264000.0 | 0.500000x | 1000 |
| cable + zipper | fixed profile=cable + category bank switch | 63.79% | 75.65% | 50176000.0 | 0.333333x | 1000 |
| cable + zipper | fixed profile=zipper + category bank switch | 60.34% | 70.80% | 50176000.0 | 0.333333x | 1000 |
| cable + zipper | proposed profile switch + category bank switch | 63.79% | 72.52% | 50176000.0 | 0.333333x | 1000 |
| cable + zipper | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| bottle + cable + capsule | common standard profile + common bank | 4.35% | 52.02% | 225792000.0 | 1.000000x | 1500 |
| bottle + cable + capsule | common standard profile + category bank switch | 25.86% | 62.24% | 75264000.0 | 0.333333x | 1500 |
| bottle + cable + capsule | fixed profile=bottle + category bank switch | 43.48% | 67.94% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + capsule | fixed profile=cable + category bank switch | 60.87% | 74.89% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + capsule | fixed profile=capsule + category bank switch | 65.22% | 79.21% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + capsule | proposed profile switch + category bank switch | 63.79% | 76.34% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + capsule | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| bottle + cable + hazelnut | common standard profile + common bank | 55.17% | 85.06% | 225792000.0 | 1.000000x | 1500 |
| bottle + cable + hazelnut | common standard profile + category bank switch | 25.86% | 75.29% | 75264000.0 | 0.333333x | 1500 |
| bottle + cable + hazelnut | fixed profile=bottle + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + hazelnut | fixed profile=cable + category bank switch | 63.79% | 87.93% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + hazelnut | fixed profile=hazelnut + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + hazelnut | proposed profile switch + category bank switch | 63.79% | 87.93% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| bottle + cable + leather | common standard profile + common bank | 50.00% | 83.33% | 225792000.0 | 1.000000x | 1500 |
| bottle + cable + leather | common standard profile + category bank switch | 25.86% | 75.29% | 75264000.0 | 0.333333x | 1500 |
| bottle + cable + leather | fixed profile=bottle + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + leather | fixed profile=cable + category bank switch | 63.79% | 87.93% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + leather | fixed profile=leather + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + leather | proposed profile switch + category bank switch | 63.79% | 87.93% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
| bottle + cable + metal_nut | common standard profile + common bank | 20.69% | 72.05% | 225792000.0 | 1.000000x | 1500 |
| bottle + cable + metal_nut | common standard profile + category bank switch | 25.86% | 75.29% | 75264000.0 | 0.333333x | 1500 |
| bottle + cable + metal_nut | fixed profile=bottle + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + metal_nut | fixed profile=cable + category bank switch | 63.79% | 86.42% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + metal_nut | fixed profile=metal_nut + category bank switch | 60.34% | 86.78% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + metal_nut | proposed profile switch + category bank switch | 63.79% | 87.93% | 50176000.0 | 0.222222x | 1500 |
| bottle + cable + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3793 |  |
## bank/category = 1500

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + carpet | common standard profile + common bank | 67.86% | 83.93% | 451584000.0 | 1.000000x | 3000 |
| bottle + carpet | common standard profile + category bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| bottle + carpet | fixed profile=bottle + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | fixed profile=carpet + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | proposed profile switch + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| bottle + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| cable + carpet | common standard profile + common bank | 46.43% | 58.56% | 451584000.0 | 1.000000x | 3000 |
| cable + carpet | common standard profile + category bank switch | 50.00% | 50.86% | 225792000.0 | 0.500000x | 3000 |
| cable + carpet | fixed profile=cable + category bank switch | 53.57% | 69.89% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | fixed profile=carpet + category bank switch | 60.34% | 73.03% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | proposed profile switch + category bank switch | 85.71% | 85.96% | 150528000.0 | 0.333333x | 3000 |
| cable + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + hazelnut | common standard profile + common bank | 64.29% | 82.14% | 451584000.0 | 1.000000x | 3000 |
| carpet + hazelnut | common standard profile + category bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + hazelnut | fixed profile=carpet + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | fixed profile=hazelnut + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | proposed profile switch + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + leather | common standard profile + common bank | 60.71% | 80.36% | 451584000.0 | 1.000000x | 3000 |
| carpet + leather | common standard profile + category bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + leather | fixed profile=carpet + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | fixed profile=leather + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | proposed profile switch + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + metal_nut | common standard profile + common bank | 50.00% | 75.00% | 451584000.0 | 1.000000x | 3000 |
| carpet + metal_nut | common standard profile + category bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + metal_nut | fixed profile=carpet + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | fixed profile=metal_nut + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | proposed profile switch + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + tile | common standard profile + common bank | 50.00% | 75.00% | 451584000.0 | 1.000000x | 3000 |
| carpet + tile | common standard profile + category bank switch | 50.00% | 75.00% | 225792000.0 | 0.500000x | 3000 |
| carpet + tile | fixed profile=carpet + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | fixed profile=tile + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | proposed profile switch + category bank switch | 85.71% | 92.86% | 150528000.0 | 0.333333x | 3000 |
| carpet + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| carpet + transistor | common standard profile + common bank | 60.71% | 73.69% | 451584000.0 | 1.000000x | 3000 |
| carpet + transistor | common standard profile + category bank switch | 50.00% | 72.50% | 225792000.0 | 0.500000x | 3000 |
| carpet + transistor | fixed profile=carpet + category bank switch | 85.71% | 92.02% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | fixed profile=transistor + category bank switch | 85.71% | 91.19% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | proposed profile switch + category bank switch | 85.71% | 91.19% | 150528000.0 | 0.333333x | 3000 |
| carpet + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + cable + carpet | common standard profile + common bank | 50.00% | 66.67% | 677376000.0 | 1.000000x | 4500 |
| bottle + cable + carpet | common standard profile + category bank switch | 50.00% | 67.24% | 225792000.0 | 0.333333x | 4500 |
| bottle + cable + carpet | fixed profile=bottle + category bank switch | 60.34% | 82.02% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | fixed profile=cable + category bank switch | 53.57% | 79.93% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | fixed profile=carpet + category bank switch | 60.34% | 82.02% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | proposed profile switch + category bank switch | 85.71% | 90.64% | 150528000.0 | 0.222222x | 4500 |
| bottle + cable + carpet | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + hazelnut | common standard profile + common bank | 35.71% | 78.57% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + hazelnut | common standard profile + category bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + hazelnut | fixed profile=bottle + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | fixed profile=carpet + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | fixed profile=hazelnut + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | proposed profile switch + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + leather | common standard profile + common bank | 67.86% | 89.29% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + leather | common standard profile + category bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + leather | fixed profile=bottle + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | fixed profile=carpet + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | fixed profile=leather + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | proposed profile switch + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + metal_nut | common standard profile + common bank | 60.71% | 86.90% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + metal_nut | common standard profile + category bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + metal_nut | fixed profile=bottle + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | fixed profile=carpet + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | fixed profile=metal_nut + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | proposed profile switch + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
| bottle + carpet + tile | common standard profile + common bank | 67.86% | 88.28% | 677376000.0 | 1.000000x | 4500 |
| bottle + carpet + tile | common standard profile + category bank switch | 50.00% | 83.33% | 225792000.0 | 0.333333x | 4500 |
| bottle + carpet + tile | fixed profile=bottle + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | fixed profile=carpet + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | fixed profile=tile + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | proposed profile switch + category bank switch | 85.71% | 95.24% | 150528000.0 | 0.222222x | 4500 |
| bottle + carpet + tile | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.3571 |  |
## bank/category = 3000

| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |
|---|---|---:|---:|---:|---:|---:|
| bottle + carpet | common standard profile + common bank | 71.43% | 85.71% | 903168000.0 | 1.000000x | 6000 |
| bottle + carpet | common standard profile + category bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| bottle + carpet | fixed profile=bottle + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | fixed profile=carpet + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | proposed profile switch + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| bottle + carpet | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + hazelnut | common standard profile + common bank | 53.57% | 76.79% | 903168000.0 | 1.000000x | 6000 |
| carpet + hazelnut | common standard profile + category bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + hazelnut | fixed profile=carpet + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | fixed profile=hazelnut + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | proposed profile switch + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + hazelnut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + leather | common standard profile + common bank | 67.86% | 83.93% | 903168000.0 | 1.000000x | 6000 |
| carpet + leather | common standard profile + category bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + leather | fixed profile=carpet + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | fixed profile=leather + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | proposed profile switch + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + leather | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + metal_nut | common standard profile + common bank | 60.71% | 80.36% | 903168000.0 | 1.000000x | 6000 |
| carpet + metal_nut | common standard profile + category bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + metal_nut | fixed profile=carpet + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | fixed profile=metal_nut + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | proposed profile switch + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + metal_nut | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + tile | common standard profile + common bank | 67.86% | 82.41% | 903168000.0 | 1.000000x | 6000 |
| carpet + tile | common standard profile + category bank switch | 60.71% | 80.36% | 451584000.0 | 0.500000x | 6000 |
| carpet + tile | fixed profile=carpet + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | fixed profile=tile + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | proposed profile switch + category bank switch | 89.29% | 94.64% | 301056000.0 | 0.333333x | 6000 |
| carpet + tile | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| carpet + transistor | common standard profile + common bank | 67.86% | 78.93% | 903168000.0 | 1.000000x | 6000 |
| carpet + transistor | common standard profile + category bank switch | 60.71% | 74.52% | 451584000.0 | 0.500000x | 6000 |
| carpet + transistor | fixed profile=carpet + category bank switch | 89.29% | 92.14% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | fixed profile=transistor + category bank switch | 71.43% | 84.88% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | proposed profile switch + category bank switch | 89.29% | 93.81% | 301056000.0 | 0.333333x | 6000 |
| carpet + transistor | difference |  |  | vs common削減 66.67% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + hazelnut | common standard profile + common bank | 60.71% | 86.90% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + hazelnut | common standard profile + category bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + hazelnut | fixed profile=bottle + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | fixed profile=carpet + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | fixed profile=hazelnut + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | proposed profile switch + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + hazelnut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + leather | common standard profile + common bank | 71.43% | 90.48% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + leather | common standard profile + category bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + leather | fixed profile=bottle + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | fixed profile=carpet + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | fixed profile=leather + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | proposed profile switch + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + metal_nut | common standard profile + common bank | 64.29% | 88.10% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + metal_nut | common standard profile + category bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + metal_nut | fixed profile=bottle + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | fixed profile=carpet + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | fixed profile=metal_nut + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | proposed profile switch + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + metal_nut | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + tile | common standard profile + common bank | 53.57% | 83.51% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + tile | common standard profile + category bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + tile | fixed profile=bottle + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | fixed profile=carpet + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | fixed profile=tile + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | proposed profile switch + category bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + tile | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + transistor | common standard profile + common bank | 71.43% | 86.03% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + transistor | common standard profile + category bank switch | 60.71% | 83.02% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + transistor | fixed profile=bottle + category bank switch | 89.29% | 94.76% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | fixed profile=carpet + category bank switch | 89.29% | 94.76% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | fixed profile=transistor + category bank switch | 71.43% | 89.92% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | proposed profile switch + category bank switch | 89.29% | 95.87% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + transistor | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |
| bottle + carpet + toothbrush | common standard profile + common bank | 75.00% | 84.52% | 1354752000.0 | 1.000000x | 9000 |
| bottle + carpet + toothbrush | common standard profile + category bank switch | 60.71% | 78.57% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + toothbrush | fixed profile=bottle + category bank switch | 75.00% | 88.10% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + toothbrush | fixed profile=carpet + category bank switch | 75.00% | 88.10% | 301056000.0 | 0.222222x | 9000 |
| bottle + carpet + toothbrush | fixed profile=toothbrush + category bank switch | 57.14% | 82.94% | 451584000.0 | 0.333333x | 9000 |
| bottle + carpet + toothbrush | proposed profile switch + category bank switch | 89.29% | 93.65% | 351232000.0 | 0.259259x | 9000 |
| bottle + carpet + toothbrush | difference |  |  | vs common削減 74.07% / vs bank-only追加削減 22.22% | min good差 vs bank-only +0.2857 |  |

## 読み取り方

- `common standard profile + common bank` は，対象カテゴリ全体のbankを毎回探索する基準方式。
- `common standard profile + category bank switch` は，総保存bank数は同じだが，検品対象カテゴリのbankだけを探索する方式。
- `fixed profile=X` は，X向けprofileを他カテゴリにも流用した場合の劣化を確認する対照実験。
- `proposed` が `bank-only` よりさらに軽ければ，bank数ではなくprofile切替自体に効果がある。
- `proposed` が `fixed profile=X` より良品通過率で勝てば，単一軽量profileの使い回しではなくカテゴリ別profileが必要である。
