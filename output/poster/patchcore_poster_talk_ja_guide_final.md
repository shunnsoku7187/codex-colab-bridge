# ポスター口頭発表：日本語確認・練習用（最終版）

提出済みポスターと英語原稿 `patchcore_poster_talk_en_final.md` に対応する。ポスターは変更せず、原稿の用語・語順を合わせた。
英語原稿の太字はポスターから拾える表現。通常の文字も読むが、日本語の案内と見出しは読まない。
左列を1→2→3と下り、右上の4へ移って5へ進む。

## 冒頭

三重大学の山田峻也です。一つの共通FPGAシステムで異なる製品カテゴリを検品することが目標です。検品前に設定を切り替えます。

## 1. PatchCore anomaly detection

**拾う言葉：** image features → stored normal examples → memory bank → nearest-neighbor search → high anomaly score → threshold

まずPatchCoreについて説明します。PatchCoreは画像特徴を保存済みの正常例と比較して欠陥を検知します。
学習時には正常な製品画像だけを使います。事前学習済みCNNが、パッチと呼ばれる画像の小領域から特徴を抽出します。CNNの重みは固定したまま、代表的な正常特徴をメモリバンクに保存します。つまり、欠陥の種類を一つずつ学習するのではなく、特徴を保存することで正常な製品の見た目を覚えます。

全特徴を保存するとbankが大きくなります。k-center coresetは、正常特徴を代表する小さな集合を選ぶ既存手法です。検知性能を維持しながら、保存量と探索量を減らすことを目指します。

検品時は同じCNNで新しい画像のパッチ特徴を抽出します。各パッチについて、近傍探索によりbank内で最も近い特徴を探します。bankからの距離が大きいほど異常スコアが高くなります。パッチスコアを画像全体のスコアに集約し、しきい値で正常画像と欠陥画像を分けます。

## 2. Category-wise profile and bank switching

では、提案手法を説明します。

**拾う言葉：** Before inspection → feature profile and normal memory bank → shares one CNN and NN engine → Profile → address range

profileは設定値の組で、特徴層・次元数、パッチの配置、スコア集約方法、しきい値を含みます。coresetでbankを減らすことや特徴設定の調整は既存のアプローチです。ただし、あるカテゴリに適した設定が別のカテゴリにも適するとは限りません。

提案は、複数カテゴリを扱う共通システムでprofileとbankを両方切り替えることです。検品前に既知のカテゴリに対応するprofileと正常bankを選びます。例えば製品Aから製品Bへ生産が変わるとき、設定とbankの両方を切り替えます。本研究では、bank切り替えのみに対するprofile切り替えの追加効果を評価します。FPGAではCNN・NN回路を共用する構成を計画しています。bank切り替えでは探索先のアドレス範囲を変えます。

## 3. Experiment and comparison

次に、実験について説明し、4つのシステムを比較します。

**拾う言葉：** all 15 categories → all 105 pairs / all 455 triples → 3,000 vectors/category → Profile selection → 比較表I～IV

MVTec ADの全15カテゴリを使い、2カテゴリ105通り・3カテゴリ455通りを評価しました。CNNを共通にし、k-center coresetで各カテゴリのbankを3,000点に統一しました。

profileは事前に各カテゴリを単独で実験し、今回の評価基準の下で最良のものをカテゴリごとに設定しました。

| 読み上げ | 表示 | 説明 |
|---|---|---|
| system one | I: Default / Pooled | 標準profile、対象カテゴリを統合したbank |
| system two | II: Default / Category switching | 標準profile、bankのみ切り替え |
| system three | III: Best fixed profile / Category switching | 最良の専用profileに固定、bank切り替え |
| system four | IV: Proposed | profileとbankの両方を切り替え |

全構成でbankの総点数をそろえています。

## 4. Detection performance and search cost

続いて、検知性能と探索コストを見ていきます。

**拾う言葉：** at most 3% of defects pass → Normal pass → NN work → 緑の棒IV → Versus bank switching alone (II)

欠陥誤通過率3%以下を評価条件にしました。良品通過率は正常画像を受け入れた割合で、高いほどよい指標です。NN探索量はパッチ数×探索bank点数×特徴次元数で、小さいほどよい指標です。

上段は2カテゴリ、下段は3カテゴリです。左は良品通過率、右はNN探索量です。緑の棒が提案です。

bankのみ切り替えるIIと比べ、良品通過率は平均3.58ポイント向上し、NN探索量は両方のカテゴリ数で約24%減りました。グラフの倍率はI基準ですが、この24%削減はIIからの追加削減です。

## 5. FPGA estimates and next steps

最後に、FPGAの見積もりと今後の取り組みを説明します。

**拾う言葉：** Estimated NN cycles: 70.7% fewer than I → excluded → register updates → 8-bit accuracy → end-to-end latency and power

参照条件ではNN処理サイクルをI比70.7%削減できる見積もりです。ハードウェア実測ではなく理論値で、CNN処理・メモリ待ち・切り替え負荷を除きます。したがって、装置全体の高速化率ではありません。

新しいビットストリームではなくレジスタ更新で切り替える計画です。一つの共通CNN・NN回路をカテゴリ間で使い回します。profile設定レジスタとbankアドレス制御によって、使う特徴とbankを選びます。これにより、各カテゴリで不要なbank読み出しと距離計算を減らします。遅延とエネルギーの低減を期待していますが、ハードウェアでの測定が必要です。今後、8bit化の精度、切り替え負荷、メモリ待ちを検証し、全体の遅延と消費電力を測定します。

まとめると、profileとbankの両切り替えで平均良品通過率を改善しながらNN探索量を削減しました。一つのCNN・NN回路を共用し、profileとbankを切り替えられるFPGAシステムの構築を目指します。カテゴリごとに別の回路を用意せず、異なる製品を効率よく検品することが目標です。

## 数字・用語の練習

| ポスター表示 | 読み上げ |
|---|---|
| 15 | fifteen |
| 105 | one hundred and five |
| 455 | four hundred and fifty-five |
| 3,000 | three thousand |
| 3% | three percent |
| +3.58 points | three point five eight percentage points |
| 24% | twenty-four percent |
| 70.7% | seventy point seven percent |
| NN | エヌ・エヌ |
| FPGA / CNN | 略語を一文字ずつ読む |
| MVTec AD | エム・ヴィー・テック・エー・ディー |
| profile | プロウファイル |
| threshold | スレッシュホウルド |
| nearest-neighbor | ニアレスト・ネイバー |
| k-center coreset | ケイ・センター・コアセット |

## 質問されたときの注意

- 24%はII基準のNN探索量削減。70.7%はI基準のNNサイクル見積もり。全体の速度・電力の削減率ではない。
- 項目4は105ペア・455トリプル。項目5の既存コストモデルは別の参照条件で105ペア・120トリプル。原稿では reference workload と区別している。
- profile選択の事前評価はbank 12,000点。今回の比較は各カテゴリ3,000点。
- IIIは各組み合わせで最も良品通過率が高い固定profile。Iのbankは対象カテゴリのみを統合している。
- 結果は組み合わせの平均。全組み合わせで優れる、欠陥を一切通さない、FPGA実装が完成した、とは主張しない。
- `audio_final` の読み上げ音声は、節間の導入文を含む今回の原稿に更新済み。

## 本番直前の確認

- ポスターの黒丸・太字を見てから一文ずつ話す。数字の前後では一拍置く。
- 3.58は percentage points。24%はIIとの比較。70.7%はIとの比較。
- 70.7%の説明後は、not a hardware measurement と not the speedup of the whole system を省略しない。
- profile選定基準の詳細は質問されたときだけ説明する。口頭本文では、単一カテゴリの事前実験で最良の設定を選んだことを伝える。
- 読み間違いでは文全体をやり直さず、該当する単語や数字だけ言い直して進む。
