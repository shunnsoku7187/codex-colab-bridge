# Kolektor dual-exit significance 002 check

## 結果

`kolektor_dual_exit_significance_002` は完了していない。
Colabランナーの上限時間9000秒に到達し、`returncode -9` で強制終了した。

## 原因

修正版では、しきい値候補に実測値と中間値を多く含めた。
その結果、両側早期終了の探索が5次元の組み合わせ探索になり、候補数が爆発した。

これは提案手法の性能結果ではなく、探索実装の失敗である。

## 修正

しきい値候補数に上限を設ける。

`kolektor_dual_exit_significance_003` では、各出口の候補数を13に制限して再実験する。
これにより、制約を満たす設定を探しつつ、Colabの実行時間内に収める。
