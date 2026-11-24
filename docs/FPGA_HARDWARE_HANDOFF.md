# FPGA回路改善・配置配線担当者への引き継ぎ

作成日：2026-11-24。基準回路・結果はコミット `34cb53d`。研究全体は `RESEARCH_HANDOFF.md`、一括実行は `FPGA_PREPARATION_PIPELINE.md` を参照。

## 依頼したいこと

カテゴリ別profile・bank切り替えを備えるPatchCore-liteのうち、まずNN探索部を改善し、実装可能なメモリ接続・パイプライン構成へ進めたい。現在のRTLは正しさを確認するためのスカラー回路である。既存の合格テストを維持しながら、速度・回路資源・メモリ帯域・切り替えコストを比較できる改善を依頼する。

「最適」の基準は未確定。目標画像/秒、許容レイテンシ、DSP/BRAM予算、CNNとの資源配分、bankの常駐先、許容精度低下を担当者と研究担当者で決める必要がある。単に最高周波数だけを最大化せず、1画像の総サイクルと転送を含めて評価してほしい。

## ファイルの入口

| 内容 | パス |
|---|---|
| 改善対象RTL | `fpga/rtl/profiled_nn_scalar.v` |
| 合成入力テスト | `fpga/sim/tb_profiled_nn_scalar.v`、`fpga/sim/run_scalar_sim.sh` |
| 最大値・丸め・飽和の事前確認 | `tools/check_int8_preparation_contract.py` |
| 実特徴RTLテスト | `fpga/sim/tb_real_features.v`、`tools/run_real_feature_sim.py` |
| 実特徴入力・bank・整数正解距離 | `results/patchcore_int8_preparation_001/*_rtl.npz` |
| profile、scale、しきい値、画像の出所・ハッシュ | 同ディレクトリの `profiles_and_results.json` |
| Vivado再現Tcl | `fpga/vivado/implement_profiled_nn.tcl` |
| 配置配線原レポート | `fpga/evidence/implementation_retry_002/` |
| 実特徴RTLの合格ログ | `fpga/evidence/real_features_001/` |
| 未完成の全体構成 | `fpga/BOARDLESS_GAPS.md`（古い不足記述は最新監査と合わせて読む） |

`*_rtl.npz` は実画像由来の正常・欠陥各1パッチと、切り詰めていない3,000点のカテゴリbank。全15カテゴリで30パッチ。各配列は `bank`、`queries`、`expected`、`labels`、`image_indices`、`patch_indices`。画像全体・全パッチをRTLで検証した結果ではない。元画像そのものはZIP/Gitに含めていない。

20.3GBのfloat特徴・bank・スコアの全体はcaviar9の `/home/shunya/codex-gpu-work/data/patchcore_publication_assets_001`。その索引は `results/patchcore_publication_assets_001_manifest.json`（Git参照）。大規模な検証が必要なら研究担当者を通してアクセスする。SSH鍵・認証情報は共有資料に含めない。

## 現在の構成と範囲

```text
検品前に既知のカテゴリ
    ↓ profile/bank選択（上位制御は未完成）
共有CNN → 特徴選択・grid → queryパッチ
（未実装）                  ↓ query_weで内部配列へロード
                        profiled_nn_scalar
外部bank（未実装） ← bank_addr → bank_data（同一サイクル）
                              ↓ min_dist = 最小二乗L2距離
                sqrt・scale復元・top-k集約・しきい値判定
                              （RTL未実装）
```

FPGA候補はUltra96-V2のXCZU3EG。今回のフローは `xczu3eg-sbva484-1-e` のOOC実装。ボードピン・PS/PL・DDR・AXI・実クロックは未設計。NN用のbank容量をBRAM 0で実現したのではなく、bank自体を現回路に含めていない。

## RTLのインターフェースとタイミング

| 信号 | 意味・現状の契約 |
|---|---|
| `clk`, `rst` | 立ち上がり動作、同期active-high reset |
| `cfg_valid` | `!busy` 時だけbase/count/dimを取り込む |
| `cfg_bank_base[31:0]` | bankの先頭アドレス。1アドレス=1個のint8値 |
| `cfg_bank_points[15:0]` | bank点数。実特徴テストは3,000 |
| `cfg_dim[9:0]` | 特徴次元。現在の入力は512または768、MAX_DIM=768 |
| `query_we`, `query_addr`, `query_data` | `!busy` かつ範囲内でqueryを書き込み。未書込み要素は未定義 |
| `start` | `!busy` 時に開始。設定取り込みと別サイクルで送ること |
| `bank_addr[31:0]` | `base + point_idx*dim + dim_idx`。組合せ出力 |
| `bank_data` | 上記アドレスに対応するsigned int8値。同一計算サイクルで有効という前提 |
| `busy` | 演算中。現在はstall/ready入力がない |
| `done` | 正常または不正設定終了で1サイクルのパルス |
| `error` | points=0、dim=0、dim>MAX_DIMの場合。正常な設定取り込みまたは開始時にクリア |
| `min_dist[31:0]` | bank全点に対する最小二乗L2距離。正常done時に有効 |

queryロードと設定はbusy中は無視される。`cfg_valid`と`start`を同時に出すとnonblocking代入のためstartは以前の設定を見る。query最終書込みとstartも別サイクルにする。resetはquery RAMを初期化しない。アドレス上限・bank実容量チェックは上位側の責務で未実装。

**同期BRAM/DDRをそのまま接続できない。** 現テストのbankは組合せreadモデルで、要求と応答の遅延を扱っていない。同期RAM化ではvalid/ready、応答遅延、stall時のアドレス・accum・index保持を追加する必要がある。

## 数値仕様と量子化の結果

- float正常学習特徴からカテゴリ/profileごとのスカラー `s=max(abs(train))/127` を決定。queryとbankで同一s、zero point=0。
- `q=clip(rint(x/s), -127, 127)`。rintは最近接・偶数丸め。検証画像からscaleを選んでいない。
- 差分はsigned9bit、二乗は18bit、累積・最小距離はunsigned32bit。
- 量子化データの最大累積は `768*254^2=49,548,288`。32bitで収まる。演算を削る場合もsigned/unsignedと中間幅を維持して検証すること。
- 現RTLは二乗距離。ソフトはパッチごとに `sqrt(min_dist)*s` を復元し、profile指定のtop-k平均を求める。二乗距離の平均は距離の平均と一般に同じではないため、sqrtを消すなら同じアルゴリズムとして扱わず再評価する。
- `image_score >= threshold` が欠陥、未満が良品通過。
- 15カテゴリの単純平均で、floatしきい値を固定した良品通過率は85.5269%→int8で83.9925%。この評価データ上ではint8の欠陥誤通過率3%超のカテゴリは0。約1.53ポイントの良品ロス増を実装都合で隠さない。
- しきい値再選択の参考値はJSONに別保存。テスト側でしきい値を選んだ値は未知データの品質保証にはならない。

## 現在の配置配線結果とボトルネック

Vivado 2025.2、10ns test_clk、OOC配置配線。LUT 390（logic270、distributed RAM120）、FF151、DSP2、BRAM0。setup WNS +3.165ns、hold worst slack +0.059ns、setup/hold違反endpointは0。

最長setupパスは `dim_idx_reg[0]/C → min_dist_reg[31]/D`。データパス6.813ns、logic2.583ns（37.9%）、route4.230ns（62.1%）、logic level19。queryの非同期readアドレスから距離更新・最小値出力までつながっている。原レポートの該当パスを読んで改善すること。別実装でこのパスが最長のままとは限らない。

1次元/サイクルなので、queryロード等を除くNNコアの処理サイクルは概ね `B*D`。B=3,000、100MHz仮定ならD512で15.36ms/パッチ、D768で23.04ms/パッチ。196パッチ/画像では3.01056秒・4.51584秒相当。これはstallなしの計算で、実機速度ではない。現在のスカラー回路は高速化を示す完成品ではない。

OOCではHD.CLK_SRC未設定警告と外部I/O制約の不足がある。WNSから全システムの最高周波数を断定せず、実際のクロック・入出力遅延を設定して再評価する。

## 改善候補と確認すること

| 候補 | 期待する改善 | 変更に伴う確認 |
|---|---|---|
| read・差分・二乗・加算・最小値更新のパイプライン化 | 長い組合せパスの分割 | 最後の次元/点のflush、valid整列、accum依存、doneのタイミング |
| アドレスの逐次加算 | 毎サイクルの乗算アドレス生成を削減する可能性 | base/count/dim切り替え、最終値、overflow。現DSP2が何に使われるか合成結果も確認 |
| queryのBRAM/レジスタ/複製の比較 | LUT RAMの遅延・fanout改善 | 同期read遅延、複製分の資源、並列readポート |
| 次元方向の並列化 | B*ceil(D/L)方向のサイクル削減 | L本分のメモリ帯域、banking、加算木、Dの端数、DSP増 |
| bank点方向の並列化 | 複数候補を同時探索 | query配信fanout、bank帯域、最小値reduceの遅延 |
| category bankの常駐/転送とdouble buffering | 切り替え負荷・待ちの抑制 | BRAM/DDR容量、ロード時間、旧bank混入、ロット長との損益 |

64次元/サイクルは論文の見積もり条件で、既に実現済みの仕様ではない。上記候補の効果も未実測。全候補を一度に入れず、同一条件で差分を比較してほしい。

## 再実行と合格条件

WindowsのPythonはnumpyが必要（その他の研究全体の依存は `requirements-colab.txt`）。WSLにiverilog/vvp、配置配線にはVivadoが必要。下記はリポジトリルートから実行する。

```powershell
python -m tools.check_int8_preparation_contract
python tools/run_real_feature_sim.py
# Vivadoは作業用ディレクトリで実行。相対パスはリポジトリから解決する。
$repo = (Get-Location).Path
New-Item -ItemType Directory -Force fpga/vivado/build | Out-Null
Set-Location fpga/vivado/build
& 'C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat' -mode batch -source "$repo/fpga/vivado/implement_profiled_nn.tcl"
```

通常の合成入力テストはLinux/WSLで `bash fpga/sim/run_scalar_sim.sh`。変更前の実特徴テストは2パッチ×15カテゴリで全件PASS。テストベンチはスカラー回路のインターフェース・遅延を前提にしているので、並列化後はアダプタやtimeoutを調整し、その変更を記録する。

必須の返却物：

1. RTL/Tcl/制約ファイルと変更理由。別branch・別top名などで基準回路を残す。
2. 同じ実特徴bank/queryに対する整数正解距離一致。追加のランダム入力、不正設定、busy中操作、reset、stall、端数次元の結果。
3. 合成・配置配線原レポート、tool版、part、clock、制約、commit、警告と未制約パス。
4. 1パッチ/1画像のサイクル数、動作周波数、LUT/FF/DSP/BRAM、必要帯域・容量、bankロードと切り替え時間。合成推定と実測を区別。
5. I〜IVを比較するなら精度、クロック、メモリ条件、対象カテゴリ、入出力範囲を揃える。CNNやbankを除外している場合は必ず明示。

Windows Smart App ControlによりVivado DLLがブロックされた履歴がある。ユーザーがWindows設定を変更後に再実行して成功した。別PCでも同じ障害が出る場合は証拠ログを確認し、そのPCの管理方針に従うこと。
