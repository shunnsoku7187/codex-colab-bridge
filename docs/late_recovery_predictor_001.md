# 後段改善見込み判定器の予備実験

## 目的

早い出口の低信頼度だけで棄却するのではなく、早い出口の情報から「finalまで進めても信頼ある通過にならないか」を直接予測できるかを調べた。
棄却しなかった画像はすべてfinalまで進める前提にし、下側出口専用判定器そのものの省計算効果を測る。

## 指標

- 信頼ある通過: finalが正解し、final信頼度が要求通過精度を満たす閾値以上
- 良品ロス: finalなら信頼ある通過になった画像を早期棄却した割合
- 早期棄却率: finalまで進めずに下側出口で止めた割合
- 平均計算量: finalのみを1.0とした相対計算量

## 最良結果

| 条件 | 要求通過精度 | 良品ロス上限 | 出口 | 特徴 | 判定器 | 早期棄却率 | final実行率 | 平均計算量 | 速度換算 | 実測良品ロス |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| clean40_mixed_low_quality60 | 99.0% | 1.0% | exit1 | class_aware | tree_depth2 | 39.96% | 60.04% | 0.7530 | 1.33x | 1.28% |
| clean75_jpeg10_25 | 99.0% | 1.0% | exit1 | class_aware | mlp_8 | 19.62% | 80.38% | 0.8787 | 1.14x | 1.26% |
| clean75_occlude16_25 | 99.0% | 1.0% | exit1 | class_aware | logistic_l2 | 23.72% | 76.28% | 0.8534 | 1.17x | 1.70% |
| clean80_mild_quality20 | 99.0% | 1.0% | exit1 | trace | mlp_8 | 12.60% | 87.40% | 0.9221 | 1.08x | 1.74% |
| clean90_mild_quality10 | 99.0% | 1.0% | exit1 | trace | mlp_8 | 9.92% | 90.08% | 0.9387 | 1.07x | 1.58% |
| clean_only | 99.0% | 1.0% | exit1 | class_aware | mlp_8 | 6.52% | 93.48% | 0.9597 | 1.04x | 1.16% |
| clean40_mixed_low_quality60 | 99.0% | 2.0% | exit1 | class_aware | mlp_8 | 52.32% | 47.68% | 0.6766 | 1.48x | 2.24% |
| clean75_jpeg10_25 | 99.0% | 2.0% | exit1 | class_aware | mlp_8 | 27.56% | 72.44% | 0.8296 | 1.21x | 2.10% |
| clean75_occlude16_25 | 99.0% | 2.0% | exit1 | scalar | logistic_l2 | 30.12% | 69.88% | 0.8138 | 1.23x | 2.42% |
| clean80_mild_quality20 | 99.0% | 2.0% | exit1 | class_aware | mlp_8 | 18.80% | 81.20% | 0.8838 | 1.13x | 2.68% |
| clean90_mild_quality10 | 99.0% | 2.0% | exit1 | trace | mlp_8 | 17.38% | 82.62% | 0.8926 | 1.12x | 3.04% |
| clean_only | 99.0% | 2.0% | exit1 | trace | mlp_8 | 11.78% | 88.22% | 0.9272 | 1.08x | 2.74% |
| clean40_mixed_low_quality60 | 99.0% | 5.0% | exit0 | class_aware | mlp_8 | 52.46% | 47.54% | 0.5040 | 1.98x | 4.90% |
| clean75_jpeg10_25 | 99.0% | 5.0% | exit1 | class_aware | mlp_8 | 43.96% | 56.04% | 0.7282 | 1.37x | 5.34% |
| clean75_occlude16_25 | 99.0% | 5.0% | exit0 | class_aware | logistic_l2 | 31.02% | 68.98% | 0.7067 | 1.41x | 5.12% |
| clean80_mild_quality20 | 99.0% | 5.0% | exit1 | class_aware | mlp_8 | 33.26% | 66.74% | 0.7944 | 1.26x | 5.90% |
| clean90_mild_quality10 | 99.0% | 5.0% | exit1 | class_aware | mlp_8 | 30.28% | 69.72% | 0.8128 | 1.23x | 6.18% |
| clean_only | 99.0% | 5.0% | exit1 | scalar | mlp_8 | 21.70% | 78.30% | 0.8659 | 1.15x | 5.78% |
| clean40_mixed_low_quality60 | 99.5% | 1.0% | exit1 | class_aware | tree_depth3 | 40.44% | 59.56% | 0.7500 | 1.33x | 1.04% |
| clean75_jpeg10_25 | 99.5% | 1.0% | exit1 | class_aware | mlp_8 | 25.32% | 74.68% | 0.8435 | 1.19x | 1.22% |
| clean75_occlude16_25 | 99.5% | 1.0% | exit1 | trace | logistic_l2 | 30.38% | 69.62% | 0.8122 | 1.23x | 1.44% |
| clean80_mild_quality20 | 99.5% | 1.0% | exit1 | trace | mlp_8 | 17.86% | 82.14% | 0.8896 | 1.12x | 1.46% |
| clean90_mild_quality10 | 99.5% | 1.0% | exit1 | class_aware | mlp_8 | 14.96% | 85.04% | 0.9075 | 1.10x | 1.32% |
| clean_only | 99.5% | 1.0% | exit1 | class_aware | mlp_8 | 8.12% | 91.88% | 0.9498 | 1.05x | 0.82% |
| clean40_mixed_low_quality60 | 99.5% | 2.0% | exit0 | class_aware | mlp_8 | 39.96% | 60.04% | 0.6222 | 1.61x | 2.56% |
| clean75_jpeg10_25 | 99.5% | 2.0% | exit1 | class_aware | mlp_8 | 35.34% | 64.66% | 0.7815 | 1.28x | 2.22% |
| clean75_occlude16_25 | 99.5% | 2.0% | exit1 | class_aware | mlp_8 | 39.82% | 60.18% | 0.7538 | 1.33x | 2.40% |
| clean80_mild_quality20 | 99.5% | 2.0% | exit1 | class_aware | mlp_8 | 25.48% | 74.52% | 0.8425 | 1.19x | 2.36% |
| clean90_mild_quality10 | 99.5% | 2.0% | exit1 | trace | mlp_8 | 22.14% | 77.86% | 0.8631 | 1.16x | 2.44% |
| clean_only | 99.5% | 2.0% | exit1 | scalar | mlp_8 | 13.26% | 86.74% | 0.9180 | 1.09x | 2.06% |
| clean40_mixed_low_quality60 | 99.5% | 5.0% | exit0 | class_aware | logistic_l2 | 66.66% | 33.34% | 0.3698 | 2.70x | 5.96% |
| clean75_jpeg10_25 | 99.5% | 5.0% | exit0 | class_aware | mlp_8 | 38.46% | 61.54% | 0.6364 | 1.57x | 5.34% |
| clean75_occlude16_25 | 99.5% | 5.0% | exit0 | scalar | logistic_l2 | 39.88% | 60.12% | 0.6230 | 1.61x | 5.52% |
| clean80_mild_quality20 | 99.5% | 5.0% | exit1 | class_aware | logistic_l2 | 44.52% | 55.48% | 0.7248 | 1.38x | 6.10% |
| clean90_mild_quality10 | 99.5% | 5.0% | exit1 | class_aware | mlp_8 | 41.12% | 58.88% | 0.7458 | 1.34x | 6.08% |
| clean_only | 99.5% | 5.0% | exit1 | class_aware | mlp_8 | 27.84% | 72.16% | 0.8279 | 1.21x | 4.98% |

## 読み方

raw_lower_confidenceが勝つなら、単純な低信頼度閾値で十分という意味になる。
treeやlinearが勝つなら、出口信頼度だけでなく、エントロピー・出口間の変化・予測クラスなどを使う小型判定器に価値がある。
特にtree_depth2/3で効果が出る場合、FPGAでは比較器とLUTに近い小回路として下側出口へ置ける可能性がある。
