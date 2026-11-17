Hello, my name is Shunya Yamada, from Mie University.

ハロー、マイ ネイム イズ シュンヤ ヤマダ、フロム ミエ ユニヴァーシティ。



Our goal is to inspect different product categories with one shared FPGA system.

アワー ゴウル イズ トゥ インスペクト ディファレント プロダクト カテゴリーズ ウィズ ワン シェアド エフ・ピー・ジー・エー システム。



We switch the settings before inspection.

ウィー スウィッチ ザ セティングズ ビフォー インスペクション。



First, let me explain PatchCore.

ファースト、レット ミー イクスプレイン パッチコア。



**PatchCore detects defects by comparing image features with stored normal examples.**

パッチコア ディテクツ ディフェクツ バイ コンペアリング イミジ フィーチャーズ ウィズ ストアド ノーマル イグザンプルズ。



During training, we use only normal product images.

デュアリング トレイニング、ウィー ユーズ オウンリー ノーマル プロダクト イミジズ。



A pretrained CNN extracts features from small image areas called patches.

ア プリートレインド シー・エヌ・エヌ イクストラクツ フィーチャーズ フロム スモール イミジ エリアズ コールド パッチズ。



We keep the CNN fixed and store representative normal features in a **memory bank**.

ウィー キープ ザ シー・エヌ・エヌ フィクスト アンド ストア レプリゼンタティヴ ノーマル フィーチャーズ イン ア メモリー バンク。



So, PatchCore learns what normal products look like by storing their features, rather than learning every defect type.

ソウ、パッチコア ラーンズ ワット ノーマル プロダクツ ルック ライク バイ ストアリング ゼア フィーチャーズ、ラザー ザン ラーニング エヴリ ディフェクト タイプ。



Storing all features would require a large bank.

ストアリング オール フィーチャーズ ウッド リクワイア ア ラージ バンク。



**K-center coreset selection** is an existing method for selecting a smaller set that represents the normal features.

ケイ・センター コアセット セレクション イズ アン イグズィスティング メソッド フォー セレクティング ア スモーラー セット ザット レプリゼンツ ザ ノーマル フィーチャーズ。



It aims to reduce storage and search work while preserving detection performance.

イット エイムズ トゥ リデュース ストーリジ アンド サーチ ワーク ワイル プリザーヴィング ディテクション パフォーマンス。



During inspection, the same CNN extracts patch features from a new image.

デュアリング インスペクション、ザ セイム シー・エヌ・エヌ イクストラクツ パッチ フィーチャーズ フロム ア ニュー イミジ。



For each patch, we find the closest feature in the bank using **nearest-neighbor search**, or N N search.

フォー イーチ パッチ、ウィー ファインド ザ クロウセスト フィーチャー イン ザ バンク ユージング ニアレスト・ネイバー サーチ、オア エヌ・エヌ サーチ。



A **large distance from the bank** gives a **high anomaly score**.

ア ラージ ディスタンス フロム ザ バンク ギヴズ ア ハイ アナマリー スコア。



We combine the patch scores into one image score.

ウィー コンバイン ザ パッチ スコアズ イントゥ ワン イミジ スコア。



**A threshold separates normal and defective images.**

ア スレッシュホウルド セパレイツ ノーマル アンド ディフェクティヴ イミジズ。



Now, let me explain our proposal.

ナウ、レット ミー イクスプレイン アワー プロポウザル。



A profile is a set of settings.

ア プロウファイル イズ ア セット オヴ セティングズ。



These include the **feature layer and dimension**, the **patch grid**, and **score aggregation and threshold**.

ジーズ インクルード ザ フィーチャー レイヤー アンド ディメンション、ザ パッチ グリッド、アンド スコア アグリゲイション アンド スレッシュホウルド。



Reducing the bank with a coreset and tuning feature settings are existing approaches.

リデューシング ザ バンク ウィズ ア コアセット アンド チューニング フィーチャー セティングズ アー イグズィスティング アプロウチズ。



But settings that work well for one category may not work well for another.

バット セティングズ ザット ワーク ウェル フォー ワン キャタゴリー メイ ノット ワーク ウェル フォー アナザー。



Our proposal is **category-wise profile and bank switching** for a shared multi-category system.

アワー プロポウザル イズ キャタゴリー・ワイズ プロウファイル アンド バンク スウィッチング フォー ア シェアド マルティ・キャタゴリー システム。



**Before inspection**, we know the product category and select its matching profile and normal bank.

ビフォー インスペクション、ウィー ノウ ザ プロダクト キャタゴリー アンド セレクト イッツ マッチング プロウファイル アンド ノーマル バンク。



For example, when production changes from product A to product B, we switch both settings and bank.

フォー イグザンプル、ウェン プロダクション チェインジズ フロム プロダクト エー トゥ プロダクト ビー、ウィー スウィッチ ボウス セティングズ アンド バンク。



Our contribution is to evaluate the added benefit of profile switching, beyond bank switching alone.

アワー コントリビューション イズ トゥ イヴァリュエイト ジ アディド ベネフィット オヴ プロウファイル スウィッチング、ビヨンド バンク スウィッチング アロウン。



**The proposed FPGA design shares one CNN and NN engine.**

ザ プロポウズド エフ・ピー・ジー・エー ディザイン シェアズ ワン シー・エヌ・エヌ アンド エヌ・エヌ エンジン。



**Bank selection changes the address range searched.**

バンク セレクション チェインジズ ジ アドレス レインジ サーチト。



Next, let me explain our experiment and compare the four systems.

ネクスト、レット ミー イクスプレイン アワー イクスペリメント アンド コンペア ザ フォー システムズ。



We used M V Tec A D, with **all fifteen categories**.

ウィー ユーズド エム・ヴィー・テック・エー・ディー、ウィズ オール フィフティーン カテゴリーズ。



We tested **all one hundred and five pairs** and **all four hundred and fifty-five triples**.

ウィー テスティド オール ワン ハンドレッド アンド ファイヴ ペアズ アンド オール フォー ハンドレッド アンド フィフティ・ファイヴ トリプルズ。



We used a shared CNN.

ウィー ユーズド ア シェアド シー・エヌ・エヌ。



We used **k-center coreset** to set every category's bank size to **three thousand vectors**.

ウィー ユーズド ケイ・センター コアセット トゥ セット エヴリ キャタゴリーズ バンク サイズ トゥ スリー サウザンド ヴェクターズ。



For **profile selection**, we first tested each category separately.

フォー プロウファイル セレクション、ウィー ファースト テスティド イーチ キャタゴリー セパレットリー。



From these single-category experiments, we chose the best profile for each category under our evaluation criteria.

フロム ジーズ シングル・キャタゴリー イクスペリメンツ、ウィー チョウズ ザ ベスト プロウファイル フォー イーチ キャタゴリー アンダー アワー イヴァリュエイション クライテリア。



We compared four systems.

ウィー コンペアド フォー システムズ。



System one uses the **default** profile and a **pooled** bank for the target categories.

システム ワン ユーズィズ ザ ディフォルト プロウファイル アンド ア プールド バンク フォー ザ ターゲット カテゴリーズ。



System two keeps the default profile and switches only the bank.

システム トゥー キープス ザ ディフォルト プロウファイル アンド スウィッチズ オウンリー ザ バンク。



System three switches the bank but uses the **best fixed profile** for each pair or triple.

システム スリー スウィッチズ ザ バンク バット ユーズィズ ザ ベスト フィクスト プロウファイル フォー イーチ ペア オア トリプル。



System four is our proposal.

システム フォー イズ アワー プロポウザル。



It switches both the profile and the bank for each category.

イット スウィッチズ ボウス ザ プロウファイル アンド ザ バンク フォー イーチ キャタゴリー。



**All systems retain the same total bank-entry budget.**

オール システムズ リテイン ザ セイム トウタル バンク・エントリー バジット。



Now, let us look at detection performance and search cost.

ナウ、レット アス ルック アット ディテクション パフォーマンス アンド サーチ コスト。



For our **operating point**, we set thresholds so that **at most three percent of defects pass** in the evaluation.

フォー アワー オペレイティング ポイント、ウィー セット スレッシュホウルズ ソウ ザット アット モウスト スリー パーセント オヴ ディフェクツ パス イン ジ イヴァリュエイション。



**Normal pass** is the **fraction of normal images accepted**.

ノーマル パス イズ ザ フラクション オヴ ノーマル イミジズ アクセプティド。



Higher is better.

ハイアー イズ ベター。



**N N work** is **patches times searched bank entries times dimensions**.

エヌ・エヌ ワーク イズ パッチズ タイムズ サーチト バンク エントリーズ タイムズ ディメンションズ。



Lower is better.

ロウアー イズ ベター。



The top charts show two categories, and the bottom charts show three categories.

ザ トップ チャーツ ショウ トゥー カテゴリーズ、アンド ザ ボトム チャーツ ショウ スリー カテゴリーズ。



The left charts show normal pass, and the right charts show N N work.

ザ レフト チャーツ ショウ ノーマル パス、アンド ザ ライト チャーツ ショウ エヌ・エヌ ワーク。



The green bars show our proposal.

ザ グリーン バーズ ショウ アワー プロポウザル。



**Versus bank switching alone**, or system two, our mean normal-pass rate increases by **three point five eight percentage points**.

ヴァーサス バンク スウィッチング アロウン、オア システム トゥー、アワー ミーン ノーマル・パス レイト インクリースィズ バイ スリー ポイント ファイヴ エイト パーセンティジ ポインツ。



We also use **about twenty-four percent less N N work for both task sizes**, on average.

ウィー オールソウ ユーズ アバウト トウェンティ・フォー パーセント レス エヌ・エヌ ワーク フォー ボウス タスク サイズィズ、オン アヴェリジ。



The N N charts use system one as the reference.

ジ エヌ・エヌ チャーツ ユーズ システム ワン アズ ザ レファレンス。



But this twenty-four percent reduction is compared with system two.

バット ディス トウェンティ・フォー パーセント リダクション イズ コンペアド ウィズ システム トゥー。



Finally, let me explain our FPGA estimates and next steps.

ファイナリー、レット ミー イクスプレイン アワー エフ・ピー・ジー・エー エスティメッツ アンド ネクスト ステップス。



Our model estimates **seventy point seven percent fewer** **N N cycles** than system one.

アワー モデル エスティメイツ セヴンティ ポイント セヴン パーセント フューアー エヌ・エヌ サイクルズ ザン システム ワン。



This is a theoretical estimate for the reference workload, not a hardware measurement.

ディス イズ ア シアレティカル エスティメット フォー ザ レファレンス ワークロウド、ノット ア ハードウェア メジャーメント。



**CNN, memory stalls and switching overhead** are **excluded**.

シー・エヌ・エヌ、メモリー ストールズ アンド スウィッチング オウヴァーヘッド アー イクスクルーディド。



So this is not the speedup of the whole system.

ソウ ディス イズ ノット ザ スピードアップ オヴ ザ ホウル システム。



In our planned design, **switching requires register updates rather than a new bitstream**.

イン アワー プランド ディザイン、スウィッチング リクワイアズ レジスター アップデイツ ラザー ザン ア ニュー ビットストリーム。



We reuse **one shared CNN and N N engine** across categories.

ウィー リユーズ ワン シェアド シー・エヌ・エヌ アンド エヌ・エヌ エンジン アクロス カテゴリーズ。



**Profile registers and bank address control** select the features and bank to use.

プロウファイル レジスターズ アンド バンク アドレス コントロウル セレクト ザ フィーチャーズ アンド バンク トゥ ユーズ。



This reduces unnecessary **bank reads and distance calculations** for each category.

ディス リデュースィズ アンネセサリー バンク リーズ アンド ディスタンス キャルキュレイションズ フォー イーチ キャタゴリー。



We expect this to reduce **latency and energy**, but we still need hardware measurements.

ウィー イクスペクト ディス トゥ リデュース レイテンシー アンド エナジー、バット ウィー スティル ニード ハードウェア メジャーメンツ。



Next, we will check **eight-bit accuracy, switching overhead and memory stalls**.

ネクスト、ウィー ウィル チェック エイト・ビット アキュラシー、スウィッチング オウヴァーヘッド アンド メモリー ストールズ。



We will also measure **end-to-end latency and power**.

ウィー ウィル オールソウ メジャー エンド・トゥ・エンド レイテンシー アンド パワー。



In summary, switching both the profile and the bank improves average normal pass while reducing N N work.

イン サマリー、スウィッチング ボウス ザ プロウファイル アンド ザ バンク インプルーヴズ アヴェリジ ノーマル パス ワイル リデューシング エヌ・エヌ ワーク。



We aim to build a configurable FPGA system that switches profiles and banks while sharing one CNN and N N engine.

ウィー エイム トゥ ビルド ア コンフィギュラブル エフ・ピー・ジー・エー システム ザット スウィッチズ プロウファイルズ アンド バンクス ワイル シェアリング ワン シー・エヌ・エヌ アンド エヌ・エヌ エンジン。



Our goal is to inspect different products efficiently without a separate circuit for each category.

アワー ゴウル イズ トゥ インスペクト ディファレント プロダクツ イフィシェントリー ウィズアウト ア セパレット サーキット フォー イーチ キャタゴリー。



Thank you for listening.

サンキュー フォー リスニング。
