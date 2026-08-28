# PatchCore既存削減技術の再現実験

## 目的

PatchCoreをFPGAへ載せる先行研究で前提になりうる既存技術を，こちらのMVTec AD条件で再現する。
ここでは提案手法の優位性ではなく，標準PatchCoreに対して次の削減がどれだけ成立するかを確認する。

- k-center coreset: 正常特徴分布を覆う代表点を意図的に選んでbankを削る既存手法。
- random: 同じbank数でも，選び方がランダムだとどれだけ不安定かを見る対照群。
- INT8/INT4: FPGA実装を想定した特徴量・bank量子化の近似評価。

## 評価条件

- backbone: `wide_resnet50_2`
- 中間特徴層: `[1, 2]`
- patch grid: `14 x 14`
- full bank上限: `6000`
- score: `topk_score`, top-k fraction `0.01`

## 集計結果

| variant | method | 欠陥誤通過上限 | 平均良品通過率 | 最低良品通過率 | 平均AUROC | fullとの順位相関 | NN演算量 | bankメモリ量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_full_bank | full | 3.00% | 73.23% | 17.07% | 0.947972 | 1.0 | 1.0000x | 1.0000x |
| int4_full_bank | full_quantized | 3.00% | 72.14% | 9.76% | 0.949517 | 0.992084 | 1.0000x | 0.1250x |
| int8_full_bank | full_quantized | 3.00% | 73.39% | 19.51% | 0.949416 | 0.993209 | 1.0000x | 0.2500x |
| kcenter_b600 | kcenter_coreset | 3.00% | 74.32% | 9.76% | 0.951609 | 0.971139 | 0.1000x | 0.1000x |
| kcenter_b750 | kcenter_coreset | 3.00% | 70.44% | 9.76% | 0.949454 | 0.974912 | 0.1250x | 0.1250x |
| kcenter_b1500 | kcenter_coreset | 3.00% | 75.06% | 7.32% | 0.951082 | 0.989792 | 0.2500x | 0.2500x |
| kcenter_b3000 | kcenter_coreset | 3.00% | 74.23% | 7.32% | 0.950196 | 0.996843 | 0.5000x | 0.5000x |
| kcenter_b600_int4 | kcenter_coreset_quantized | 3.00% | 72.11% | 9.76% | 0.951518 | 0.960088 | 0.1000x | 0.0125x |
| kcenter_b600_int8 | kcenter_coreset_quantized | 3.00% | 70.84% | 9.76% | 0.95228 | 0.961222 | 0.1000x | 0.0250x |
| kcenter_b750_int4 | kcenter_coreset_quantized | 3.00% | 71.77% | 4.88% | 0.950828 | 0.965181 | 0.1250x | 0.0156x |
| kcenter_b750_int8 | kcenter_coreset_quantized | 3.00% | 72.25% | 9.76% | 0.951091 | 0.966259 | 0.1250x | 0.0312x |
| kcenter_b1500_int4 | kcenter_coreset_quantized | 3.00% | 73.39% | 7.32% | 0.951638 | 0.982272 | 0.2500x | 0.0312x |
| kcenter_b1500_int8 | kcenter_coreset_quantized | 3.00% | 75.20% | 7.32% | 0.952161 | 0.982456 | 0.2500x | 0.0625x |
| kcenter_b3000_int4 | kcenter_coreset_quantized | 3.00% | 73.66% | 9.76% | 0.951401 | 0.989299 | 0.5000x | 0.0625x |
| kcenter_b3000_int8 | kcenter_coreset_quantized | 3.00% | 75.03% | 9.76% | 0.951351 | 0.989826 | 0.5000x | 0.1250x |
| random_b600_r0 | random | 3.00% | 52.83% | 0.00% | 0.885906 | 0.871408 | 0.1000x | 0.1000x |
| random_b600_r1 | random | 3.00% | 52.36% | 0.00% | 0.872083 | 0.85581 | 0.1000x | 0.1000x |
| random_b750_r0 | random | 3.00% | 52.33% | 0.00% | 0.893408 | 0.887573 | 0.1250x | 0.1250x |
| random_b750_r1 | random | 3.00% | 57.62% | 7.32% | 0.898202 | 0.888052 | 0.1250x | 0.1250x |
| random_b1500_r0 | random | 3.00% | 62.28% | 2.44% | 0.918132 | 0.925625 | 0.2500x | 0.2500x |
| random_b1500_r1 | random | 3.00% | 60.98% | 0.00% | 0.915163 | 0.917773 | 0.2500x | 0.2500x |
| random_b3000_r0 | random | 3.00% | 67.81% | 7.32% | 0.936984 | 0.966002 | 0.5000x | 0.5000x |
| random_b3000_r1 | random | 3.00% | 66.97% | 2.44% | 0.936266 | 0.96105 | 0.5000x | 0.5000x |

## 読み取り方

- `良品通過率` は，欠陥誤通過率を指定上限以下に抑えたうえで，正常品を正常として通せた割合である。
- `fullとの順位相関` は，full bank fp32 PatchCoreの異常スコア順位をどれだけ保てたかを示す。1に近いほど削減後も同じ判断順序を保っている。
- k-centerがrandomより安定すれば，bank削減は「たまたま当たった」ではなく，正常特徴空間の代表点選択として扱える。
- INT4まで落として順位相関や良品通過率が保てる場合，MAD-Flow型の低bit KNN実装を比較対象として採用しやすい。

図: `results/mvtec_patchcore_prior_art_reproduction_001.png`
