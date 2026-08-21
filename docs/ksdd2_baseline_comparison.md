# KSDD2 baseline value-add evaluation

目的は、既存の欠陥検出モデルを否定することではない。既存技術を土台にしたとき、提案手法やFPGA化でどこに価値を足せるかを確認する。

読み分け:

- `mean test AUROC/AUPR`: 既存技術として欠陥スコアが妥当に出ているかを見る再現・土台確認の指標。
- `worst false-pass/good-pass`: 検品ライン風の運用制約に置いたとき、どこが不足しやすいかを見る指標。
- 低いfalse-pass条件は、既存技術の合否判定ではなく、提案手法/FPGA化が改善すべき運用上の負荷を見つけるための条件。

| result | model | score | mean test AUROC | mean test AUPR | operating false-pass | operating good-pass | feasible seeds | worst false-pass | worst good-pass | value-add reading |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ksdd2_smp_final_inspection_baseline_caviar9_001_summary.json | unetplusplus/resnet34 | topk_score | 0.9779 | 0.9374 | 0.00% | 90.00% | 0/1 | 4.55% | 88.70% | 欠陥スコアは有効だが、運用閾値には改善余地がある。校正・選択的判定・FPGA化の効果を見る。 |
| ksdd2_smp_final_inspection_baseline_001_summary.json | unetplusplus/resnet34 | max_score | 0.9740 | 0.9420 | 0.00% | 95.00% | 0/1 | 5.45% | 98.32% | 欠陥スコアは有効だが、運用閾値には改善余地がある。校正・選択的判定・FPGA化の効果を見る。 |
| ksdd2_unet_inspection_baseline_001_summary.json | ksdd2_unet_inspection_baseline_001 | topk_score | 0.9761 | 0.9403 | 0.00% | 90.00% | 0/2 | 5.45% | 93.40% | 欠陥スコアは有効だが、運用閾値には改善余地がある。校正・選択的判定・FPGA化の効果を見る。 |
| ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json | unet/resnet50 | max_score | 0.9883 | 0.9506 | 5.00% | 95.00% | 0/1 | 9.09% | 98.21% | 欠陥スコアは有効だが、運用閾値には改善余地がある。校正・選択的判定・FPGA化の効果を見る。 |

次に見るべきこと:

- 既存技術の再現はAUROC/AUPRや先行研究の評価条件で見る。
- 提案手法の価値は、同じ土台モデルに対して計算量・消費電力・レイテンシ・棄却率・閾値安定性がどれだけ改善するかで見る。
- FPGA化の価値は、分岐後に動かさない回路を物理的に止める、複数段をパイプライン化する、固定小数点化で電力と資源を見積もる、という追加指標で測る。
