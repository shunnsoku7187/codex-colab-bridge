# PatchCore標準設定の再現診断

## 目的

これまでのPatchCore-lite結果が先行研究水準より低く見える原因を切り分ける。
特に，特徴層が浅すぎた可能性，入力前処理の違い，bank数の扱いを確認する。

## 比較profile

| profile | backbone | out indices | resize/crop | grid | bank policy | top-k |
|---|---|---:|---:|---:|---|---:|
| current_lite_out12_g14_b6000 | wide_resnet50_2 | [1, 2] | 224/224 | 14 | fixed:6000.0 | 0.01 |
| patchcore_l23_g14_b6000 | wide_resnet50_2 | [2, 3] | 256/224 | 14 | fixed:6000.0 | 0.01 |
| patchcore_l23_g28_b6000 | wide_resnet50_2 | [2, 3] | 256/224 | 28 | fixed:6000.0 | 0.01 |
| patchcore_l23_g28_core10 | wide_resnet50_2 | [2, 3] | 256/224 | 28 | ratio:0.1 | 0.01 |

## 集計結果: 欠陥誤通過率 <= 3.00%

| profile | 平均良品通過率 | 最低良品通過率 | 平均AUROC | 平均bank数 | 相対NN演算量 |
|---|---:|---:|---:|---:|---:|
| current_lite_out12_g14_b6000 | 83.26% | 60.98% | 0.971786 | 6000.0 | 1.0000x |
| patchcore_l23_g14_b6000 | 77.79% | 29.27% | 0.952331 | 6000.0 | 2.0000x |
| patchcore_l23_g28_b6000 | 79.50% | 36.59% | 0.967199 | 6000.0 | 8.0000x |
| patchcore_l23_g28_core10 | 81.17% | 46.34% | 0.964579 | 12000.0 | 16.0000x |

## カテゴリ別AUROC

| category | profile | AUROC | 良品通過率 | bank数 | patch数 | 特徴次元 |
|---|---|---:|---:|---:|---:|---:|
| bottle | current_lite_out12_g14_b6000 | 1.0 | 100.00% | 6000 | 196 | 768 |
| cable | current_lite_out12_g14_b6000 | 0.980322 | 75.86% | 6000 | 196 | 768 |
| capsule | current_lite_out12_g14_b6000 | 0.974073 | 82.61% | 6000 | 196 | 768 |
| screw | current_lite_out12_g14_b6000 | 0.917401 | 60.98% | 6000 | 196 | 768 |
| zipper | current_lite_out12_g14_b6000 | 0.987132 | 96.88% | 6000 | 196 | 768 |
| bottle | patchcore_l23_g14_b6000 | 1.0 | 100.00% | 6000 | 196 | 1536 |
| cable | patchcore_l23_g14_b6000 | 0.996064 | 98.28% | 6000 | 196 | 1536 |
| capsule | patchcore_l23_g14_b6000 | 0.937774 | 73.91% | 6000 | 196 | 1536 |
| screw | patchcore_l23_g14_b6000 | 0.84751 | 29.27% | 6000 | 196 | 1536 |
| zipper | patchcore_l23_g14_b6000 | 0.980305 | 87.50% | 6000 | 196 | 1536 |
| bottle | patchcore_l23_g28_b6000 | 1.0 | 100.00% | 6000 | 784 | 1536 |
| cable | patchcore_l23_g28_b6000 | 0.995877 | 98.28% | 6000 | 784 | 1536 |
| capsule | patchcore_l23_g28_b6000 | 0.934184 | 78.26% | 6000 | 784 | 1536 |
| screw | patchcore_l23_g28_b6000 | 0.926419 | 36.59% | 6000 | 784 | 1536 |
| zipper | patchcore_l23_g28_b6000 | 0.979517 | 84.38% | 6000 | 784 | 1536 |
| bottle | patchcore_l23_g28_core10 | 1.0 | 100.00% | 12000 | 784 | 1536 |
| cable | patchcore_l23_g28_core10 | 0.996814 | 100.00% | 12000 | 784 | 1536 |
| capsule | patchcore_l23_g28_core10 | 0.926606 | 78.26% | 12000 | 784 | 1536 |
| screw | patchcore_l23_g28_core10 | 0.92232 | 46.34% | 12000 | 784 | 1536 |
| zipper | patchcore_l23_g28_core10 | 0.977153 | 81.25% | 12000 | 784 | 1536 |

## 判断

- 原論文寄りprofileでAUROCが大きく上がるなら，これまでの低スコアは提案手法の問題ではなくbaseline設定の問題である。
- 原論文寄りprofileでもAUROCが低いなら，MVTec mirrorの読み込み・前処理・score計算・PatchCore reweighting欠落をさらに疑う。
- AUROCは高いが良品通過率が低い場合，先行研究指標と検品動作点指標の違いが原因である。

図: `results/mvtec_patchcore_faithful_reproduction_001.png`
