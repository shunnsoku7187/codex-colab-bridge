# 一度の起動で実施するFPGA準備

起動（Windows、研究用SSH鍵がssh-agentで利用可能な状態）：

```powershell
.\tools\start_fpga_preparation.ps1
```

事前に変更をGitHubへpushする。Python（numpy）、WSLのiverilog/vvp、Vivado、SSH経路が必要。標準のPython/Vivadoパスはローカル設定のため他PCでは引数・環境変数を調整する。

## 順序と保存物

1. caviar9で `patchcore_int8_preparation_001` を起動。保存済みfloat特徴から15カテゴリの選択profileと各3,000点bankを取り出す。CNN再抽出はしない。
2. 正常学習特徴の最大絶対値だけで `scale=maxabs/127` を決める。対称signed int8、zero point=0、np.rint（偶数丸め）、[-127,127]飽和。同じカテゴリ/profileのqueryとbankに同じscaleを適用。評価画像でscaleを調整しない。
3. 全評価画像でfloat/int8スコアを比較。保存済みfloatしきい値固定での良品通過・欠陥誤通過と、3%条件でしきい値を再選択した参考値を別々に記録。後者は独立検証の品質保証ではない。
4. 各カテゴリから正常・欠陥各1画像を選び、float異常スコアが最大のパッチを1つずつ取り出す。完全な3,000点int8 bankと2本のquery、整数ソフトの正解距離をnpzに保存してGitへ返送。
5. このPCが結果返送を3分間隔で確認。完了後に全15カテゴリの実特徴でIcarusシミュレーションし、整数ソフトとRTLの距離を照合。失敗時はログを保存する。
6. ローカルVivadoのスカラーNN合成・配置配線を並行実行し、原レポートを保存。最後に結果をGitへpush。

実行ログ・段階状態は `fpga/sim/build/preparation.log` と `preparation_status.json`。非表示のバックグラウンド処理なので、このPCの電源・ネット接続は維持する。caviar9側はSSH切断後も継続する。途中完了状態は各カテゴリの結果JSONに保存するが、GPUジョブ自体の自動再開は未対応。完了済みジョブの無断再実行はしない。

## 数値仕様と評価範囲

NN RTLはint8差分をsigned9bit、二乗を18bit、累積・最小距離を32bit unsignedで保持。最大は `768*254^2=49,548,288 < 2^26` なので32bitに収まる。RTLは二乗L2距離を返す。ソフト画像スコアは `sqrt(min_squared_distance)*scale` のパッチ距離を既存top-k集約へ渡す。画像判定は `score >= threshold` が欠陥。

再生成floatと同じ集約を使うが、量子化の影響は実測結果で判断する。RTLテスト対象は実画像由来パッチのNNで、画像スコア集約・しきい値回路やCNNを含む全システムの完成ではない。Vivadoもスカラー単体のOOC評価であり64並列の実装結果ではない。

GPU出力は `results/patchcore_int8_preparation_001/`。RTLログは `fpga/evidence/real_features_001/`、合成原レポートは `fpga/evidence/implementation_001/`。最終状態は `results/fpga_preparation_pipeline_001.json`。
