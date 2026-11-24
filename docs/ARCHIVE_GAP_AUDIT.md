# Git保存内容の不足監査（2026-11-20）

2026-11-24追加：GPUでの全15カテゴリint8評価と実特徴RTL照合が完了（`results/patchcore_int8_preparation_001/`、`fpga/evidence/real_features_001/`）。WindowsのSmart App ControlによるDLLブロックはユーザーの設定変更後に解消し、スカラーNNの配置配線まで正常終了。原レポートは `fpga/evidence/implementation_retry_002/` に保存済み。以下の原レポート未回収という旧不足は、新規再実行の証拠保存によって解消した。全CNN・メモリ接続などの未完成項目は引き続き残る。

## 2026-11-24更新：実特徴・bankの再生成完了

`patchcore_publication_assets_001` は2026-11-21 01:44 JSTに正常終了（returncode 0、14,256秒、約3時間58分）。各カテゴリ3,000点条件で全105ペア・455トリプルの結果行が、発表時 `_002` の同条件と完全一致した。選択profileも一致。これは数値結果の一致であり、未保存だった当時のbank配列とのビット単位比較ではない。

保存先 `/home/shunya/codex-gpu-work/data/patchcore_publication_assets_001` に特徴120ファイル、bank 590ファイル、パッチスコア1,605ファイル、計2,315配列・20,263,534,864 bytesを保存。9,240評価の設定・スコア対応もmanifestに記録した。

2026-11-24にSSHで全2,315配列の存在とサイズ一致を確認。欠落0、サイズ不一致0。特徴・bank・スコアから各1ファイルのSHA-256を抜き取り検証し、3件とも一致した（全ファイルのハッシュ再検証は未実施）。

Git内の索引：`results/patchcore_publication_assets_001_manifest.json`。再生成結果：同名 `_summary.json`。大きな配列の実体はcaviar9側であり、Gitには入れていない。以前の「実特徴bankが未保存」という不足は今回の再生成で解消した。int8校正・量子化検証はまだ別途必要。

基準はorigin/mainの83d70ad。既存成果を上書きせず、再開に必要な不足を確認した。

| 項目 | 確認結果・今回の対応 |
|---|---|
| 最終予稿、ポスターv25、原稿、想定問答 | Gitに保存済み。未登録の旧版を最新版として追加しない |
| 最終比較の結果・ジョブ | `_002` summary、完了ジョブ、ログ、コードはGitに保存済み |
| ポスター生成の入力JSON | 保存済みsourceに対するfallbackを追加。clone直後にtmpがなくても入力を読める |
| ポスター生成のパス | リポジトリパスを自動取得。POSTER_RUNTIME / POSTER_PRESENTATIONS_SKILLで外部ツールパスを指定可能にした。専用ツール・フォント自体は別途必要 |
| NN試作のシミュレーション証拠 | Icarusで再実行し、コマンド・ツール版・出力を `fpga/evidence/scalar_sim_2026-11-20.log` に保存。起動スクリプトも追加 |
| FPGAの索引 | manifestの対象にfpga/が入っていなかったため修正して再作成 |
| 390 LUT / 151 FF / 2 DSP / +3.165nsの原レポート | 既存fpga/READMEに記載はあるが、対応する原レポートは今回のローカル検索範囲で未発見。Git保存済みの原レポートで検証できた値として扱わない |
| ローカルTemp内のVivadoレポート | `codex_vivado_synth_check`で発見したがDesign=synth_check、69 LUTの別回路。NN試作の証拠として流用しない |
| 画像・重み・実特徴bank | 実画像から特徴とbankを生成して最終実験で使用した。FeatureCacheはメモリ内dictであり、最終比較コードには特徴・bank配列をファイル保存する処理がない。Gitには数値結果・profile・条件があるがbankそのものはない。元画像・重みのリモート現存確認は未実施 |
| int8校正・実画像由来の数値検証 | 未作成。合成入力のPASSで検知精度や量子化精度を証明しない |
| 共有CNN、64-lane NN、メモリ接続、全画像制御 | 未完成。`fpga/BOARDLESS_GAPS.md`参照 |

## 取得または作成を要するもの

1. リモートGPU機から実際に使用したweights、profile別bank、特徴抽出・前処理の設定とハッシュを取得。大きなバイナリはデータ保管場所を用意し、Gitにはmanifestと小さな正常/欠陥検証ベクトルを保存する。
2. CNN出力の量子化規則・スコア集約・しきい値の数値仕様を定義し、float/int8/RTLの同一入力比較を作る。
3. NN試作のVivado合成・配置配線を再実行するか、対応する原レポートを回収する。対象RTLのコミット、part、クロック条件と合わせて保存する。

以前の研究引き継ぎは2026-11-18のソフト成果整理を基準にしている。その後に追加されたスカラーNN試作は実装の一部であり、PatchCore全体のFPGA完成を意味しない。

## 実験で使ったbankの保存状況の訂正

最終比較 `scripts/mvtec_patchcore_fixed_bank_profile_switch.py` は、実画像由来の正常特徴を `FeatureCache.get()` で取得し、`build_bank()` でk-center選択し、`evaluate_category()`で実画像の検知評価に使っている。FeatureCacheの実体は `scripts/mvtec_patchcore_fixed_coreset_profile_switch.py` のメモリ内辞書で、永続キャッシュではない。最終出力はJSON集計・Markdown・図であり、bank配列・選択インデックス・個々の特徴は保存していない。したがって「未使用」ではなく「使用した中間データの永続保存が不足」が正確な状態である。

元画像の設定パスは `/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2`。元画像・使用重みが残っていれば条件とseedを合わせた再生成を試せるが、当時の配列ハッシュがないため、元bankとのビット単位の一致は証明できない。別実験で保存されたbankの有無はリモート保管先を調べるまで未確認。

## caviar9実在確認（2026-11-20追記）

SSH経由で読み取り専用の監査を実施。記録は `results/remote_patchcore_asset_audit_2026-11-20.json`、再確認用コードは `tools/audit_remote_patchcore_assets.py`。

- `mvtec_ad_materialized_v2` に15カテゴリすべての画像ファイルが存在した。件数はJSONを参照（マスク画像があれば件数に含まれる）。
- WideResNet50-2重みはHugging Faceキャッシュに存在。`timm/wide_resnet50_2.racm_in1k` snapshot `30f73aceaaa1911830a9795b83ab1908dba18719`、275,835,296 bytes、SHA-256 `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000`。
- 既知のwork配下とtorch/Hugging Faceキャッシュを調査した範囲ではPatchCoreの保存済みbank配列・特徴配列は見つからなかった。results/artifactsにある配列・モデルはJSONに列挙。これはディスク全域の不存在証明ではない。
- 元画像とbackbone重みが残っているため再生成の主要入力は確保されている。ただし再生成はまだ実行しておらず、当時のbankとの同一性も未確認。今回の確認でGPU実験は起動していない。
