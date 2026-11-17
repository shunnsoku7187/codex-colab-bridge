# English Talk: Final Script

読むのは英語本文のみ。日本語の案内と見出しは読まない。
**太字はポスターから拾える表現**。太字以外も読む。一行を一文として、空行で一呼吸置く。
左列の1→2→3、次に右列の4→5へ進む。I～IVは system one～four、NNはエヌ・エヌと読む。
提出済みポスターの文言・数値に対応。太字には、数字を読み上げ用に綴った表現も含む。

## Opening

> 見る場所：タイトル → 左上の項目1

Hello, my name is Shunya Yamada, from Mie University.
Our goal is to inspect different product categories with one shared FPGA system.
We switch the settings before inspection.

## 1. PatchCore anomaly detection

> 見る場所：最初の2文 → CNNからNNへの矢印 → 下の2文

First, let me explain PatchCore.
**PatchCore detects defects by comparing image features with stored normal examples.**
During training, we use only normal product images.
A pretrained CNN extracts features from small image areas called patches.
We keep the CNN fixed and store representative normal features in a **memory bank**.
So, PatchCore learns what normal products look like by storing their features, rather than learning every defect type.

Storing all features would require a large bank.
**K-center coreset selection** is an existing method for selecting a smaller set that represents the normal features.
It aims to reduce storage and search work while preserving detection performance.

During inspection, the same CNN extracts patch features from a new image.
For each patch, we find the closest feature in the bank using **nearest-neighbor search**, or N N search.
A **large distance from the bank** gives a **high anomaly score**.
We combine the patch scores into one image score.
**A threshold separates normal and defective images.**

## 2. Category-wise profile and bank switching

> 見る場所：最初の文 → Profile + bank selection → 下のProfileの説明

Now, let me explain our proposal.
A profile is a set of settings.
These include the **feature layer and dimension**, the **patch grid**, and **score aggregation and threshold**.
Reducing the bank with a coreset and tuning feature settings are existing approaches.
But settings that work well for one category may not work well for another.

Our proposal is **category-wise profile and bank switching** for a shared multi-category system.
**Before inspection**, we know the product category and select its matching profile and normal bank.
For example, when production changes from product A to product B, we switch both settings and bank.
Our contribution is to evaluate the added benefit of profile switching, beyond bank switching alone.
**The proposed FPGA design shares one CNN and NN engine.**
**Bank selection changes the address range searched.**

## 3. Experiment and comparison

> 見る場所：実験条件 → Profile selection → 比較表をIからIVへ

Next, let me explain our experiment and compare the four systems.
We used M V Tec A D, with **all fifteen categories**.
We tested **all one hundred and five pairs** and **all four hundred and fifty-five triples**.
We used a shared CNN.
We used **k-center coreset** to set every category's bank size to **three thousand vectors**.

For **profile selection**, we first tested each category separately.
From these single-category experiments, we chose the best profile for each category under our evaluation criteria.

We compared four systems.
System one uses the **default** profile and a **pooled** bank for the target categories.
System two keeps the default profile and switches only the bank.
System three switches the bank but uses the **best fixed profile** for each pair or triple.
System four is our proposal.
It switches both the profile and the bank for each category.
**All systems retain the same total bank-entry budget.**

## 4. Detection performance and search cost

> 見る場所：評価条件 → 左右の棒グラフ → 下の緑色の文

Now, let us look at detection performance and search cost.
For our **operating point**, we set thresholds so that **at most three percent of defects pass** in the evaluation.
**Normal pass** is the **fraction of normal images accepted**.
Higher is better.
**N N work** is **patches times searched bank entries times dimensions**.
Lower is better.

The top charts show two categories, and the bottom charts show three categories.
The left charts show normal pass, and the right charts show N N work.
The green bars show our proposal.

**Versus bank switching alone**, or system two, our mean normal-pass rate increases by **three point five eight percentage points**.
We also use **about twenty-four percent less N N work for both task sizes**, on average.
The N N charts use system one as the reference.
But this twenty-four percent reduction is compared with system two.

## 5. FPGA estimates and next steps

> 見る場所：緑色の2文 → 下のregister updates → next steps

Finally, let me explain our FPGA estimates and next steps.
Our model estimates **seventy point seven percent fewer** **N N cycles** than system one.
This is a theoretical estimate for the reference workload, not a hardware measurement.
**CNN, memory stalls and switching overhead** are **excluded**.
So this is not the speedup of the whole system.

In our planned design, **switching requires register updates rather than a new bitstream**.
We reuse **one shared CNN and N N engine** across categories.
**Profile registers and bank address control** select the features and bank to use.
This reduces unnecessary **bank reads and distance calculations** for each category.
We expect this to reduce **latency and energy**, but we still need hardware measurements.
Next, we will check **eight-bit accuracy, switching overhead and memory stalls**.
We will also measure **end-to-end latency and power**.

In summary, switching both the profile and the bank improves average normal pass while reducing N N work.
We aim to build a configurable FPGA system that switches profiles and banks while sharing one CNN and N N engine.
Our goal is to inspect different products efficiently without a separate circuit for each category.
Thank you for listening.
