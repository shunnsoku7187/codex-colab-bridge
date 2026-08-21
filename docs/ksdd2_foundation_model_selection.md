# KSDD2土台モデルの性能再確認と評価軸

## 結論

現時点で一番マシな土台モデルは `unet/resnet50` です。
根拠ファイルは `ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json`、採用スコアは `max_score` です。

欠陥誤通過を5%以下に抑える条件では、良品通過率は平均 94.13%、最悪seedでも 92.17% でした。
また、良品通過率90%付近では欠陥誤通過が平均 3.18% です。

ただし完全な土台モデルではありません。検証データで選んだ「欠陥誤通過5%以下・良品通過90%以上」の閾値がtestでも成功したのは 1/2 seed です。つまり、スコア分離能力は最も高いが、閾値安定性にはまだ改善または再確認が必要です。

## なぜAUROCだけで決めないか

検品タスクで一番避けたい失敗は、欠陥品を良品として通してしまうことです。AUROCが高くても、欠陥誤通過を低く抑えようとした瞬間に良品まで大量に捨てるモデルでは、実運用の土台として弱いです。

そのため、土台モデル選定では「欠陥スコアの順位付け性能」だけでなく、「実際に閾値を置いたときの良品通過率と欠陥誤通過率」を重視します。

## 候補モデルの再確認

| 順位 | モデル | スコア | 元結果 | 平均AUROC | 平均AUPR | 欠陥誤通過5%以下での良品通過 | 良品通過90%付近の欠陥誤通過 | 検証閾値の成功数 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | unet/resnet50 | max_score | ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json | 0.988301 | 0.950604 | 94.13% | 3.18% | 1/2 |
| 2 | unet/resnet50 | topk_score | ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json | 0.988199 | 0.957 | 94.02% | 2.27% | 1/2 |
| 3 | unetplusplus/resnet34 | topk_score | ksdd2_smp_final_inspection_baseline_caviar9_001_summary.json | 0.977934 | 0.937412 | 91.67% | 5.00% | 0/2 |
| 4 | unetplusplus/resnet34 | max_score | ksdd2_smp_final_inspection_baseline_caviar9_001_summary.json | 0.978854 | 0.937003 | 91.39% | 5.45% | 0/2 |
| 5 | fpn/resnet50 | topk_score | ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_summary.json | 0.98737 | 0.950233 | 91.22% | 4.55% | 0/2 |
| 6 | small_unet/base24 | topk_score | ksdd2_unet_inspection_baseline_001_summary.json | 0.976113 | 0.940315 | 88.81% | 5.00% | 0/2 |
| 7 | small_unet/base24 | max_score | ksdd2_unet_inspection_baseline_001_summary.json | 0.974588 | 0.940527 | 88.03% | 4.55% | 0/2 |
| 8 | unet/resnet50 | topk_score | ksdd2_unet_resnet50_foundation_recheck_caviar9_001_summary.json | 0.982713 | 0.947557 | 87.73% | 4.85% | 0/3 |
| 9 | fpn/resnet50 | max_score | ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_summary.json | 0.983745 | 0.928622 | 86.91% | 4.55% | 0/2 |
| 10 | unet/resnet50 | max_score | ksdd2_unet_resnet50_foundation_recheck_caviar9_001_summary.json | 0.982771 | 0.945066 | 86.88% | 4.24% | 0/3 |
| 11 | patchcore_lite | topk_score | ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json | 0.938951 | 0.850823 | 58.50% | 15.00% | 0/0 |
| 12 | patchcore_lite | max_score | ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json | 0.934843 | 0.833829 | 55.15% | 15.00% | 0/0 |
| 13 | padim_diag | topk_score | ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json | 0.883903 | 0.592429 | 46.59% | 30.91% | 0/0 |
| 14 | padim_diag | max_score | ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json | 0.875331 | 0.575167 | 43.23% | 32.27% | 0/0 |

## 今後の評価軸

### 1. 土台モデル単体の評価

- 欠陥誤通過率: 欠陥品を良品として通した割合。安全性の最重要制約。
- 良品通過率: 良品を良品として通した割合。歩留まり・生産性の指標。
- 良品棄却率: 良品を止めた割合。欠陥誤通過よりは許容されるが、コストとして扱う。
- AUROC/AUPR: 欠陥スコアそのものの順位付け性能。補助指標。
- 閾値安定性: validationで選んだ閾値がtestでも崩れないか。

### 2. 提案手法を載せた後の評価

- 欠陥誤通過率は土台モデルと同じ安全予算内に維持する。
- その条件で良品通過率・良品棄却率がどう変わるかを見る。
- 精度だけでなく、平均実行段数・平均計算量・平均消費電力を比較する。
- 平均レイテンシと最悪レイテンシを分けて報告する。
- FPGA化では、使用LUT/DSP/BRAM、メモリアクセス、推定電力、パイプライン化、分岐先回路の停止可能性を評価する。

## 現時点の読み

現時点では、PatchCore-liteやPaDiM-diagonalよりも、U-Net/ResNet50系のセグメンテーションモデルを土台にするのが妥当です。

PatchCoreは保険テーマとしては残せますが、今回のKSDD2実験では土台モデルとして最有力とは言えません。もしPatchCore FPGA化を主題にするなら、まずPatchCore本体をより忠実に再現し、性能面でU-Net/ResNet50に近づくか上回ることを確認する必要があります。

再確認の結果、`unet/resnet50` は暫定土台として残すが、validation閾値のtest安定性は不十分でした。次にやるべきことは、土台モデルをむやみに替えることではなく、閾値校正・校正セット設計・保守的な安全閾値選択を整理することです。その後、この固定土台に対して両側早期終了またはFPGA化による計算量削減を評価します。
