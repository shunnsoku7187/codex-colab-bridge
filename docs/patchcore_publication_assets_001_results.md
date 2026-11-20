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
| carpet + hazelnut + leather | standard profile + k-center common bank | 67.86% | 89.29% | 1354752000.0 | 1.000000x | 9000 |
| carpet + hazelnut + leather | standard profile + k-center category-bank switch | 60.71% | 86.90% | 451584000.0 | 0.333333x | 9000 |
| carpet + hazelnut + leather | fixed profile=carpet + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| carpet + hazelnut + leather | fixed profile=hazelnut + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| carpet + hazelnut + leather | fixed profile=leather + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| carpet + hazelnut + leather | proposed profile switch + k-center category-bank switch | 89.29% | 96.43% | 301056000.0 | 0.222222x | 9000 |
| carpet + hazelnut + leather | difference |  |  | vs common削減 77.78% / vs bank-only追加削減 33.33% | min good差 vs bank-only +0.2857 |  |

## 読み取り方

- `standard profile + k-center common bank` は，対象カテゴリ全体のk-center bankを毎回探索する実装baselineである。
- `standard profile + k-center category-bank switch` は，総保存bank数は同じだが，検品対象カテゴリのk-center bankだけを探索する方式である。
- `fixed profile=X` は，X向けprofileを他カテゴリにも流用した場合の劣化を確認する対照実験。
- `proposed` が `bank-only` よりさらに軽ければ，bank数ではなくprofile切替自体に効果がある。
- `proposed` が `fixed profile=X` より良品通過率で勝てば，単一軽量profileの使い回しではなくカテゴリ別profileが必要である。
