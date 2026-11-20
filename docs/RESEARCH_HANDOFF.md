# 研究引き継ぎ（2026-11-18整理）

2026-11-20追記：その後のFPGA試作と保存不足の確認は [最新の不足監査](ARCHIVE_GAP_AUDIT.md) と `fpga/BOARDLESS_GAPS.md` を参照。スカラーNNの合成入力シミュレーション証拠を追加済み。以下の主研究の数値はソフト評価時のもの。

## 読む順序と現在地

1. この文書：研究の主張、根拠、限界、再開方法。
2. `output/sympo/patchcore_profile_switch_summary_en.pdf`：英語予稿。日本語確認版は同名 `_ja.pdf`。対応する `.tex` と `figures/` も収録。
3. `output/poster/patchcore_profile_switch_poster_en_A0_color_v25.pptx`：提出済み英語A0ポスター。日本語確認版は `_ja_A0_color_v25.pptx`。提出版は保存し、改訂は別名で行う。
4. `output/poster/patchcore_poster_talk_en_final.md`：最終口頭原稿。日本語訳、カタカナ補助版、想定問答も同ディレクトリ。印刷版は `output/pdf/`。
5. `jobs/mvtec_patchcore_kcenter_default_system_compare_002.json` と同名 `results/*_summary.json`、`docs/*.md`：最終比較の実行条件と結果。

題目：**FPGA Implementation of a PatchCore-lite Anomaly Detection System with Category-wise Profile Switching**。
著者：山田峻也（Shunya Yamada）、指導教員：高木一義（Kazuyoshi Takagi）。三重大学大学院工学研究科情報工学専攻、コンピュータアーキテクチャ研究室。

現時点でソフトウェア評価・NN計算量の理論見積もり・発表資料の作成まで完了。FPGAへの合成、配置配線、実機での速度・電力測定は未完了。題目だけを見て実装済みと解釈しないこと。

## 何を提案しているか

PatchCoreは事前学習済みCNNから正常画像のパッチ特徴を抽出し、代表的な特徴をmemory bankに保存する。検査画像とbankの最近傍距離を異常スコアにし、しきい値で判定する。CNN自体は固定する。

coresetによるbank削減、特徴設定の調整は既存技術。本研究では、カテゴリが検品前に既知の複数製品ラインを想定し、製品の切り替え時にそのカテゴリに適したprofileとbankを同時に選ぶ。単一CNNとNN探索回路を共有し、profileレジスタとbankアドレス制御で切り替える構想である。画像からカテゴリを自動識別する研究ではない。

profileは特徴層・特徴次元、パッチgrid、スコア集約、判定しきい値。最終比較ではbackboneはWideResNet50-2に固定し、CNN全体の交換は行わない。bank点数のカテゴリ別最小化を主張の根拠から外している。

## 最終比較と主要結果

MVTec ADの15カテゴリについて全105ペア（AB）と455トリプル（ABC）を評価。k-center coresetを全構成に使い、各カテゴリ3,000点に統一。ABの総点数は6,000、ABCは9,000で全構成同じ。

| 構成 | profile | bank |
|---|---|---|
| I | デフォルト | 対象カテゴリのみの共通bank |
| II | デフォルト | カテゴリ別bank切り替え |
| III | 対象カテゴリ専用候補のうち最良の固定profile | カテゴリ別bank切り替え |
| IV（提案） | カテゴリ別profile切り替え | カテゴリ別bank切り替え |

IIIは全対象に同じ設定を強制する。最良候補を採用した有利な固定比較であり、全設計空間の最適解を証明するものではない。

profileは先行する単一カテゴリ実験（bank 12,000点）で、デフォルトに対する良品通過率低下が2ポイント以内の候補からパッチ数×次元数を小さくする設定を選んだ。そのprofileを3,000点条件で比較した。

| 対象 | 指標 | I | II | III | IV |
|---|---|---:|---:|---:|---:|
| AB / 105通り | 平均良品通過率 | 80.27% | 81.95% | 85.01% | 85.53% |
| AB / 105通り | 平均相対NN探索量（I=1） | 1.000 | 0.500 | 0.400 | 0.378 |
| ABC / 455通り | 平均良品通過率 | 79.27% | 81.95% | 84.49% | 85.53% |
| ABC / 455通り | 平均相対NN探索量（I=1） | 1.000 | 0.333 | 0.271 | 0.252 |

欠陥誤通過率3%以下となる評価上のしきい値のうち、良品通過率が高い点を採用。画像を都合よく除いたわけではない。良品通過率は正常画像を正常として受け入れた割合で、分類全体の正解率ではない。

IIに対しIVは平均良品通過率+3.58ポイント、NN探索量約24%削減。ABでは1−0.378/0.500=24.4%。グラフの正規化基準Iと、24%の比較基準IIを混同しない。IIIに対する良品通過率の改善はAB+0.52、ABC+1.04ポイント。平均の改善であり、すべての組み合わせで勝つとは主張しない。

探索量モデルは `W = P * B * D`（画像のパッチ数、探索するbank点数、特徴次元）。Iは対象全bankを探索し、II以降は対応カテゴリのみ探索する。II→IVはbank点数そのものを減らした効果ではない。

## FPGA見積もりと限界

`docs/mvtec_patchcore_fpga_cost_model_001.md` と対応する結果、予稿7章・ポスター5を参照。
参照ワークロードは105ペア＋120トリプルで、検知性能の全455トリプルとは異なる。各カテゴリ3,000点、bank特徴8bit、64次元/サイクル、メモリ待ちなしのモデル。

| 構成 | NNサイクル/画像 | bank容量（次元数に合わせて詰めて保存） |
|---|---:|---:|
| I | 17.88M | 5.57 MiB |
| IV | 5.23M | 4.11 MiB |

I比の推定NNサイクル削減は70.7%。CNN計算・メモリ待ち・切り替え負荷を除くため、全体速度が70.7%改善したという結果ではない。bank点数は同じでも次元数で容量が変わる。固定幅で保存するなら両方5.57 MiBである。

切り替えそのものはCPUでも可能。FPGAの価値は共有回路、並列化、パイプライン、設定制御を実装し、実際の資源・遅延・電力で評価すること。現状でCPUより高速、省電力、面積削減済みとは断言できない。bank常駐ならアドレス更新で済むが、転送が必要ならその負荷も数える。

3%は比較用条件であり実際の品質保証ではない。評価データに基づくしきい値選択を含むため、未知データでの上限保証ではない。全組み合わせ評価も独立した560データセットによる汎化証明ではない。coresetにも精度無劣化の保証はない。

## 研究の経緯（旧結果の扱い）

- 固定画像統計ルーター：`docs/feasibility_findings.md`、`information_location_findings.md`、`model_process_trace_findings.md`。試した表現・条件での見込みが薄い。あらゆる画像統計による分離の不可能性を証明したわけではない。
- deadline路線：`docs/deadline_aware_broad_feasibility_findings.md`。同じ時間制約の小型モデルとの比較が必要となり、主題から外した。
- 両側早期終了：`docs/dual_exit_speed_tradeoff_001_findings.md` 等。低信頼例を早期排出して計算を省くが、良品ロスとのトレードオフ、データセット依存、土台モデルの品質が課題。現在のPatchCore主題とは別の履歴。
- PatchCore軽量化・bank最小化：`docs/mvtec_patchcore_*`。旧来の約98%削減やbank最小数探索の値を、現在の公平な固定点数比較と混ぜない。初期の比較基準やbank選定・対象カテゴリ数が異なる。
- 先行研究：`docs/patchcore_fpga_prior_art_survey.md`、`patchcore_fpga_open_theme_survey.md`、`mad_flow_deep_reading_sheet.md`。既存FPGA実装があるため「初めて載せた」を主張しない。調査時点の記録であり、投稿・新規性判断前に更新が必要。
- 過去の進捗スライド最新版は `output/history/progress27_dual_exit_added_v24.pptx`。現在の主成果を説明する資料とは区別する。

## 実行・結果確認の引き継ぎ

ローカルPCは編集・Git操作・軽い集計、konbuは中継、caviar9はGPU処理。起動後にCodexを常時監視させない。結果・ログはrunnerがGitへ返す。

`docs/konbu_gpu_start_button.md`、`caviar9_gpu_execution_setup.md`、`caviar9_manual_git_runner.md`、READMEの基盤説明を参照。認証鍵・パスワードはGitに保存しない。

```bash
# konbu上。ジョブは複製して新IDにし、実行条件を確認してから使う。
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --list
bash tools/konbu_gpu_start_button.sh --job NEW_JOB_ID --no-terminal
# 後で状態を確認
bash tools/konbu_gpu_start_button.sh --check NEW_JOB_ID
```

最終比較ジョブ `_002` はdone、returncode=0、実行時間約21,221秒（5.9時間）。完了済みジョブのstatusを無断でpendingに戻さない。この整理時点ではリモート計算機へ接続していないため、現在の稼働状態は未確認。

生データ・重み・特徴キャッシュはGit外。ジョブで参照しているデータは `/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2`。取得・実体化方法は `docs/mvtec_ad_hf_download_001.md`、`mvtec_ad_parquet_all15_probe_002.md` と対応jobs/scriptsを参照。Gitのみで研究内容と再実行手順を引き継げるが、重い再実行にはデータ取得・GPU環境・SSH権限が必要。

資料の生成元は `tools/build_sympo_patchcore_summary.py`（予稿）、`build_patchcore_profile_switch_poster.py`、`prepare_poster_pptx_content.py`、`build_poster_pptx.mjs`、`repair_poster_bullets.py`（ポスター）、`build_talk_print_pdf.py`、`build_talk_kana_pdf.py`、`build_qa_cheatsheet.py`（原稿等）。WindowsフォントやCodex専用ランタイムへの依存があり、他環境ではパス調整が必要。既存PDF/PPTXを読むだけなら不要。ポスターの最終内容JSONとPowerPoint出力画像は `output/poster/source/` に保存。

## 次の作業

1. 対象FPGAとメモリ構成を確定し、共有CNN・特徴選択・NNパイプライン・profileレジスタのRTL/HLS仕様を作る。
2. 8bit化による検知品質と、profile/bankの切り替え機能を検証する。
3. 外部メモリ帯域、待ち、転送、ロット長を含む全体サイクルモデルで損益分岐点を求める。
4. 合成・配置配線でLUT/DSP/BRAM、周波数を評価し、実機で遅延・スループット・電力をI〜IVと同条件で測る。
5. 独立データでprofileとしきい値を検証し、先行研究との差分を更新する。

アップロード対象の索引とSHA-256は `docs/research_archive_manifest.json`。古い資料の途中版、音声、作業用画像、認証情報、生データ、重みは対象外。既存の研究コード・結果・ログは履歴として保持する。
