# Kolektor dual-exit significance 001 check

## 結果

ジョブ自体は正常終了したが、比較表の行数が0だった。

これは「提案手法が負けた」という意味ではなく、validationで制約を満たす設定を探索できなかった扱いになっている。

## なぜ再実験が必要か

ログ上では、学習中のfinal 0.5しきい値でvalidationが次を満たしている。

- 正常通過率: 100%
- 欠陥誤通過率: 9.09%

したがって、少なくとも `max false pass 10% / min good pass 95%` の条件では、final-onlyの行が出るべきである。

行が0になった原因は、しきい値候補が粗く、実測値と実測値の間にある有効なしきい値を取り逃した可能性が高い。

## 修正

`scripts/kolektor_dual_exit_significance.py` のしきい値候補を修正した。

候補に含めるもの:

- 実測された確率値
- 隣接する実測値の中間値
- 分位点
- 0.0 / 0.5 / 1.0

修正版は `kolektor_dual_exit_significance_002` として再実験する。
