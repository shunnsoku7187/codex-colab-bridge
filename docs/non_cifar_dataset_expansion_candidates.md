# CIFAR以外の追加データセット候補

## 目的

KolektorSDDだけで良い結果が出ても、「特定データセットにだけ合った」可能性が残る。
そこで、性質の違う検品データセットを複数用意し、提案手法の有効条件を切り分ける。

## すぐ回す候補

### KolektorSDD split違い

同じデータセットでも、欠陥が少ないため分割の影響が大きい。
まずはseed違いでfinal性能が安定するかを確認する。

目的:

- 1回目の高性能が偶然の分割でないか確認する
- 後続の両側早期終了評価に使える土台モデルを複数確保する

### KolektorSDD モデル違い

同じsplitで、ResNet系、EfficientNet系、MobileNet/ShuffleNet系を比較する。

目的:

- 精度重視モデルと軽量モデルの差を見る
- FPGA実装候補として、重いモデルだけに依存しない候補を残す
- 早期出口を付けるならどのbackboneが扱いやすいかを判断する

## 次に広げる候補

### KolektorSDD2

KolektorSDDより規模が大きい表面欠陥データセット。
ModelScope/OpenDataLabでは約922MB、3000枚以上、欠陥356枚、正常2979枚とされている。

利点:

- KolektorSDDよりサンプル数が多い
- 欠陥種類が増え、実験の信頼性が上がる

注意:

- ダウンロード方法がGit LFSや外部ミラーに依存する
- Colabジョブへ組み込む前に、Driveへ一度置く方が安定する可能性が高い

### MVTec AD

産業異常検知の代表的データセット。
カテゴリが複数あり、bottle、capsule、metal nutなど検品に近い対象を選べる。

利点:

- 研究で参照されやすい
- カテゴリ別に評価できる
- 正常のみ学習、異常検出という検品らしい設定に寄せやすい

注意:

- 公式ダウンロードはログインが必要になる場合がある
- 自動DLジョブにするより、Driveに配置してから使う方がよい

## 今回追加したジョブ

- `kolektor_strong_final_accuracy_002`
  - ResNet34 / EfficientNet-B0
  - 精度重視

- `kolektor_strong_final_lightweight_003`
  - MobileNetV3-Small / ShuffleNetV2 / ResNet18
  - 軽量・FPGA候補

- `kolektor_strong_final_split_recheck_004`
  - seed 456
  - 分割違いでの再現性確認

## 判断基準

まず見るべきはaccuracyではなく、次の3点である。

1. 欠陥誤通過を0%または十分低くできるか
2. そのとき良品ロスが許容範囲か
3. 同じ傾向がモデル違い・分割違いでも残るか

この土台が確認できた後で、両側早期終了を載せる。
