# PatchCore bank削減のcoresetカバー率検証

## 目的

bank削減をランダムな当たり外れではなく，既存のk-center coreset選択で「正常分布を覆う代表点数」として評価する。
また，A単独，B単独，A+B/ABC混合集合で必要bank数が単純加算になるかを確認する。

## 定義

- 代表点選択: farthest-first k-center greedy。
- カバー半径: 全正常パッチから最も近いbank代表点までの距離。
- 必要bank数: 最大bank候補での95%カバー半径に対し，指定slack以内に入る最小bank数。

| subset | 標準A単独+B単独の必要数合計 | 標準merged必要数 | merged/sum | 提案側カテゴリ別bank数 | 読み取り |
|---|---:|---:|---:|---:|---|
| toothbrush + wood | 3500 | 1500 | 0.428571 | 750 / 125 | mergedが単純和より小さい |
| leather + wood | 3000 | 1500 | 0.5 | 125 / 125 | mergedが単純和より小さい |
| tile + wood | 2500 | 1500 | 0.6 | 250 / 125 | mergedが単純和より小さい |
| bottle + toothbrush + wood | 5500 | 1500 | 0.272727 | 500 / 750 / 125 | mergedが単純和より小さい |
| leather + tile + wood | 4000 | 1500 | 0.375 | 125 / 250 / 125 | mergedが単純和より小さい |
| bottle + leather + wood | 5000 | 1500 | 0.3 | 500 / 125 / 125 | mergedが単純和より小さい |

## 解釈

- bankのみ切替の0.5x/0.333xは，対象カテゴリごとに標準bankを分ける効果である。
- profile切替でさらに小さくなるかは，そのカテゴリの正常特徴分布を少数代表点で覆えるかに依存する。
- merged/sumが1に近い場合，A+Bの正常分布を覆う代表点数はほぼ加法的であり，カテゴリ別bank切替の意義が出やすい。
- merged/sumが大きく1を下回る場合，カテゴリ間で正常特徴の共通部分があり，混合bankでも代表点を共有できる可能性がある。
