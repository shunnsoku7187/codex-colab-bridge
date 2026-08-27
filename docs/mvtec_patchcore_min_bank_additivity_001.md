# 精度維持最小bank数とカテゴリ加法性

## 目的

bank数を固定比率で決めるのではなく，full bank時の良品通過率を保てる最小bank数を求める。さらに同一profileで `N(A+B)` が `N(A)+N(B)` に近いかを確認し，bank切替の意義を評価する。

## 単独カテゴリ

| profile | category | reference bank | reference good-pass | required bank | required/reference |
|---|---|---:|---:|---:|---:|
| standard | bottle | 12000 | 100.00% | 25 | 0.21% |
| standard | cable | 12000 | 72.41% | 250 | 2.08% |
| standard | capsule | 12000 | 78.26% | 500 | 4.17% |
| standard | carpet | 12000 | 96.43% | 9000 | 75.00% |
| standard | grid | 12000 | 71.43% | 500 | 4.17% |
| standard | hazelnut | 12000 | 100.00% | 25 | 0.21% |
| standard | leather | 12000 | 100.00% | 25 | 0.21% |
| standard | metal_nut | 12000 | 100.00% | 750 | 6.25% |
| standard | pill | 12000 | 61.54% | 125 | 1.04% |
| standard | screw | 12000 | 48.78% | 3000 | 25.00% |
| standard | tile | 12000 | 100.00% | 25 | 0.21% |
| standard | toothbrush | 11760 | 91.67% | 9000 | 76.53% |
| standard | transistor | 12000 | 95.00% | 750 | 6.25% |
| standard | wood | 12000 | 100.00% | 25 | 0.21% |
| standard | zipper | 12000 | 90.62% | 100 | 0.83% |
| selected | bottle | 10241 | 95.00% | 100 | 0.98% |
| selected | cable | 10976 | 55.17% | 750 | 6.83% |
| selected | capsule | 12000 | 69.57% | 1000 | 8.33% |
| selected | carpet | 12000 | 96.43% | 6000 | 50.00% |
| selected | grid | 12000 | 52.38% | 6000 | 50.00% |
| selected | hazelnut | 12000 | 100.00% | 100 | 0.83% |
| selected | leather | 12000 | 100.00% | 25 | 0.21% |
| selected | metal_nut | 12000 | 100.00% | 500 | 4.17% |
| selected | pill | 12000 | 34.62% | 50 | 0.42% |
| selected | screw | 12000 | 24.39% | 3000 | 25.00% |
| selected | tile | 12000 | 96.97% | 1000 | 8.33% |
| selected | toothbrush | 1500 | 50.00% | 25 | 1.67% |
| selected | transistor | 12000 | 96.67% | 500 | 4.17% |
| selected | wood | 12000 | 100.00% | 1500 | 12.50% |
| selected | zipper | 11760 | 87.50% | 4000 | 34.01% |

## 混合カテゴリでの加法性

| profile | subset | N(A)+N(B/...) | N(A+B/...) | merged/sum | reference min good-pass |
|---|---|---:|---:|---:|---:|
| standard | bottle + cable | 3025 | 500 | 0.165289 | 63.79% |
| selected:bottle | bottle + cable | 9100 | 1000 | 0.10989 | 25.86% |
| selected:cable | bottle + cable | 850 | 500 | 0.588235 | 50.00% |
| standard | bottle + cable + capsule | 6025 | 3000 | 0.497925 | 65.22% |
| selected:bottle | bottle + cable + capsule | 15100 | 12000 | 0.794702 | 25.86% |
| selected:cable | bottle + cable + capsule | 1600 | 12000 | 7.5 | 21.74% |
| selected:capsule | bottle + cable + capsule | 2525 | 6000 | 2.376238 | 41.38% |
| standard | bottle + cable + carpet | 6025 | 2000 | 0.33195 | 65.52% |
| selected:bottle | bottle + cable + carpet | 9600 | 100 | 0.010417 | 14.29% |
| selected:cable | bottle + cable + carpet | 975 | 12000 | 12.307692 | 35.71% |
| selected:carpet | bottle + cable + carpet | 7550 | 12000 | 1.589404 | 82.76% |
| standard | bottle + cable + grid | 3525 | 4000 | 1.134752 | 65.52% |
| selected:bottle | bottle + cable + grid | 9125 | 100 | 0.010959 | 4.76% |
| selected:cable | bottle + cable + grid | 2850 | 12000 | 4.210526 | 55.17% |
| selected:grid | bottle + cable + grid | 7750 | 1500 | 0.193548 | 33.33% |
| standard | bottle + cable + hazelnut | 3125 | 1500 | 0.48 | 60.34% |
| selected:bottle | bottle + cable + hazelnut | 9350 | 1000 | 0.106952 | 20.00% |
| selected:cable | bottle + cable + hazelnut | 2350 | 2000 | 0.851064 | 55.17% |
| selected:hazelnut | bottle + cable + hazelnut | 1625 | 2000 | 1.230769 | 65.52% |
| standard | bottle + cable + leather | 3050 | 1000 | 0.327869 | 65.52% |
| selected:bottle | bottle + cable + leather | 15100 | 1500 | 0.099338 | 12.50% |
| selected:cable | bottle + cable + leather | 1600 | 6000 | 3.75 | 55.17% |
| selected:leather | bottle + cable + leather | 9125 | 2000 | 0.219178 | 43.10% |
| standard | bottle + cable + metal_nut | 3775 | 1000 | 0.264901 | 60.34% |
| selected:bottle | bottle + cable + metal_nut | 9225 | 1000 | 0.108401 | 18.18% |
| selected:cable | bottle + cable + metal_nut | 1100 | 4000 | 3.636364 | 40.91% |
| selected:metal_nut | bottle + cable + metal_nut | 1300 | 1500 | 1.153846 | 74.14% |
| standard | bottle + cable + pill | 3150 | 3000 | 0.952381 | 51.72% |
| selected:bottle | bottle + cable + pill | 9225 | 4000 | 0.433604 | 22.41% |
| selected:cable | bottle + cable + pill | 1350 | 3000 | 2.222222 | 38.46% |
| selected:pill | bottle + cable + pill | 3300 | 1000 | 0.30303 | 12.07% |
| standard | bottle + cable + screw | 4525 | 1000 | 0.220994 | 34.15% |
| selected:bottle | bottle + cable + screw | 10600 | 9000 | 0.849057 | 24.39% |
| selected:cable | bottle + cable + screw | 2350 | 12000 | 5.106383 | 31.71% |
| selected:screw | bottle + cable + screw | 6250 | 12000 | 1.92 | 17.24% |
| standard | bottle + cable + tile | 3075 | 1500 | 0.487805 | 65.52% |
| selected:bottle | bottle + cable + tile | 10600 | 6000 | 0.566038 | 39.66% |
| selected:cable | bottle + cable + tile | 1350 | 1500 | 1.111111 | 55.17% |
| selected:tile | bottle + cable + tile | 7100 | 12000 | 1.690141 | 34.48% |
| standard | bottle + cable + toothbrush | 3150 | 1000 | 0.31746 | 48.28% |
| selected:bottle | bottle + cable + toothbrush | 9850 | 6000 | 0.609137 | 25.86% |
| selected:cable | bottle + cable + toothbrush | 900 | 1500 | 1.666667 | 50.00% |
| selected:toothbrush | bottle + cable + toothbrush | 325 | 2000 | 6.153846 | 29.31% |
| standard | bottle + cable + transistor | 3775 | 6000 | 1.589404 | 63.79% |
| selected:bottle | bottle + cable + transistor | 9600 | 3000 | 0.3125 | 25.86% |
| selected:cable | bottle + cable + transistor | 1350 | 3000 | 2.222222 | 50.00% |
| selected:transistor | bottle + cable + transistor | 1050 | 3000 | 2.857143 | 84.48% |
| standard | bottle + cable + wood | 3050 | 3000 | 0.983607 | 65.52% |
| selected:bottle | bottle + cable + wood | 9150 | 12000 | 1.311475 | 51.72% |
| selected:cable | bottle + cable + wood | 875 | 1000 | 1.142857 | 55.17% |
| selected:wood | bottle + cable + wood | 2350 | 500 | 0.212766 | 25.86% |
| standard | bottle + cable + zipper | 3125 | 1500 | 0.48 | 63.79% |
| selected:bottle | bottle + cable + zipper | 9850 | 3000 | 0.304569 | 25.86% |
| selected:cable | bottle + cable + zipper | 950 | 3000 | 3.157895 | 50.00% |
| selected:zipper | bottle + cable + zipper | 4350 | 125 | 0.028736 | 5.17% |
| standard | bottle + capsule | 3025 | 750 | 0.247934 | 69.57% |
| selected:bottle | bottle + capsule | 6100 | 4000 | 0.655738 | 34.78% |
| selected:capsule | bottle + capsule | 1025 | 3000 | 2.926829 | 69.57% |
| standard | bottle + capsule + carpet | 6025 | 750 | 0.124481 | 69.57% |
| selected:bottle | bottle + capsule + carpet | 6600 | 1000 | 0.151515 | 21.74% |
| selected:capsule | bottle + capsule + carpet | 1525 | 1500 | 0.983607 | 53.57% |
| selected:carpet | bottle + capsule + carpet | 7550 | 6000 | 0.794702 | 82.61% |
| standard | bottle + capsule + grid | 3525 | 1000 | 0.283688 | 69.57% |
| selected:bottle | bottle + capsule + grid | 6125 | 25 | 0.004082 | 4.76% |
| selected:capsule | bottle + capsule + grid | 1050 | 25 | 0.02381 | 0.00% |
| selected:grid | bottle + capsule + grid | 6750 | 6000 | 0.888889 | 52.38% |
| standard | bottle + capsule + hazelnut | 3125 | 3000 | 0.96 | 69.57% |
| selected:bottle | bottle + capsule + hazelnut | 6350 | 12000 | 1.889764 | 20.00% |
| selected:capsule | bottle + capsule + hazelnut | 1525 | 6000 | 3.934426 | 62.50% |
| selected:hazelnut | bottle + capsule + hazelnut | 625 | 6000 | 9.6 | 82.61% |
| standard | bottle + capsule + leather | 3050 | 1000 | 0.327869 | 65.22% |
| selected:bottle | bottle + capsule + leather | 12100 | 25 | 0.002066 | 0.00% |
| selected:capsule | bottle + capsule + leather | 1150 | 3000 | 2.608696 | 56.52% |
| selected:leather | bottle + capsule + leather | 4125 | 3000 | 0.727273 | 65.22% |
| standard | bottle + capsule + metal_nut | 3775 | 6000 | 1.589404 | 65.22% |
| selected:bottle | bottle + capsule + metal_nut | 6225 | 3000 | 0.481928 | 18.18% |
| selected:capsule | bottle + capsule + metal_nut | 1125 | 3000 | 2.666667 | 27.27% |
| selected:metal_nut | bottle + capsule + metal_nut | 1050 | 6000 | 5.714286 | 56.52% |
| standard | bottle + capsule + pill | 3150 | 3000 | 0.952381 | 61.54% |
| selected:bottle | bottle + capsule + pill | 6225 | 12000 | 1.927711 | 23.08% |
| selected:capsule | bottle + capsule + pill | 1150 | 3000 | 2.608696 | 50.00% |
| selected:pill | bottle + capsule + pill | 800 | 6000 | 7.5 | 34.62% |
| standard | bottle + capsule + screw | 4525 | 750 | 0.165746 | 31.71% |
| selected:bottle | bottle + capsule + screw | 7600 | 2000 | 0.263158 | 17.39% |
| selected:capsule | bottle + capsule + screw | 2025 | 3000 | 1.481481 | 24.39% |
| selected:screw | bottle + capsule + screw | 3750 | 6000 | 1.6 | 24.39% |
| standard | bottle + carpet | 3025 | 125 | 0.041322 | 89.29% |
| selected:bottle | bottle + carpet | 600 | 500 | 0.833333 | 28.57% |
| selected:carpet | bottle + carpet | 6050 | 9000 | 1.487603 | 96.43% |
| standard | bottle + grid | 525 | 1000 | 1.904762 | 76.19% |
| selected:bottle | bottle + grid | 125 | 25 | 0.2 | 0.00% |
| selected:grid | bottle + grid | 6250 | 1500 | 0.24 | 52.38% |
| standard | bottle + hazelnut | 125 | 750 | 6.0 | 100.00% |
| selected:bottle | bottle + hazelnut | 350 | 100 | 0.285714 | 20.00% |
| selected:hazelnut | bottle + hazelnut | 125 | 250 | 2.0 | 100.00% |
| standard | bottle + leather | 50 | 100 | 2.0 | 100.00% |
| selected:bottle | bottle + leather | 6100 | 12000 | 1.967213 | 46.88% |
| selected:leather | bottle + leather | 125 | 100 | 0.8 | 100.00% |
| standard | bottle + metal_nut | 775 | 750 | 0.967742 | 100.00% |
| selected:bottle | bottle + metal_nut | 225 | 500 | 2.222222 | 18.18% |
| selected:metal_nut | bottle + metal_nut | 550 | 500 | 0.909091 | 100.00% |
| standard | bottle + pill | 150 | 500 | 3.333333 | 61.54% |
| selected:bottle | bottle + pill | 225 | 125 | 0.555556 | 23.08% |
| selected:pill | bottle + pill | 300 | 100 | 0.333333 | 34.62% |
| standard | bottle + screw | 1525 | 2000 | 1.311475 | 31.71% |
| selected:bottle | bottle + screw | 1600 | 500 | 0.3125 | 14.63% |
| selected:screw | bottle + screw | 3250 | 750 | 0.230769 | 24.39% |
| standard | bottle + tile | 75 | 250 | 3.333333 | 96.97% |
| selected:bottle | bottle + tile | 1600 | 6000 | 3.75 | 95.00% |
| selected:tile | bottle + tile | 1100 | 2000 | 1.818182 | 96.97% |
| standard | bottle + toothbrush | 150 | 250 | 1.666667 | 75.00% |
| selected:bottle | bottle + toothbrush | 850 | 50 | 0.058824 | 58.33% |
| selected:toothbrush | bottle + toothbrush | 75 | 100 | 1.333333 | 50.00% |
| standard | bottle + transistor | 775 | 6000 | 7.741935 | 95.00% |
| selected:bottle | bottle + transistor | 600 | 250 | 0.416667 | 36.67% |
| selected:transistor | bottle + transistor | 550 | 750 | 1.363636 | 96.67% |
| standard | bottle + wood | 50 | 50 | 1.0 | 100.00% |
| selected:bottle | bottle + wood | 150 | 100 | 0.666667 | 73.68% |
| selected:wood | bottle + wood | 1600 | 100 | 0.0625 | 100.00% |
| standard | bottle + zipper | 125 | 250 | 2.0 | 87.50% |
| selected:bottle | bottle + zipper | 850 | 3000 | 3.529412 | 75.00% |
| selected:zipper | bottle + zipper | 4100 | 4000 | 0.97561 | 87.50% |
| standard | cable + capsule | 750 | 6000 | 8.0 | 72.41% |
| selected:cable | cable + capsule | 1500 | 12000 | 8.0 | 34.78% |
| selected:capsule | cable + capsule | 2500 | 9000 | 3.6 | 46.55% |
| standard | cable + carpet | 9250 | 2000 | 0.216216 | 57.14% |
| selected:cable | cable + carpet | 875 | 12000 | 13.714286 | 32.14% |
| selected:carpet | cable + carpet | 7500 | 3000 | 0.4 | 82.76% |
| standard | cable + grid | 750 | 6000 | 8.0 | 57.14% |
| selected:cable | cable + grid | 2750 | 9000 | 3.272727 | 47.62% |
| selected:grid | cable + grid | 7500 | 750 | 0.1 | 23.81% |
| standard | cable + hazelnut | 275 | 750 | 2.727273 | 72.41% |
| selected:cable | cable + hazelnut | 2250 | 1000 | 0.444444 | 55.17% |
| selected:hazelnut | cable + hazelnut | 1600 | 2000 | 1.25 | 74.14% |
| standard | cable + leather | 275 | 1500 | 5.454545 | 72.41% |
| selected:cable | cable + leather | 1500 | 4000 | 2.666667 | 55.17% |
| selected:leather | cable + leather | 9025 | 12000 | 1.32964 | 60.34% |
| standard | cable + metal_nut | 375 | 3000 | 8.0 | 67.24% |
| selected:cable | cable + metal_nut | 1000 | 1000 | 1.0 | 40.91% |
| selected:metal_nut | cable + metal_nut | 1250 | 4000 | 3.2 | 74.14% |

## 読み取り

- `merged/sum` が1に近い場合，混合bankはカテゴリごとの必要bank数をほぼ足し合わせる必要があり，カテゴリ別bank切替の意義が強い。
- `merged/sum` が大きく1を下回る場合，カテゴリ間で代表点を共有でき，bank切替だけの寄与は弱くなる。
- 提案の中心は，bank削減ではなく，この最小bank数を含むprofileをカテゴリごとに切り替えることである。
