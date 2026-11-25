from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

here = Path(__file__).resolve().parent
font_path = r"C:\Windows\Fonts\meiryo.ttc"
img = Image.new("RGB", (1120, 630), "#f8fafc")
d = ImageDraw.Draw(img)


def font(size, bold=False):
    return ImageFont.truetype(font_path, size, index=1 if bold else 0)


def text(x, y, value, size=18, color="#244534", bold=False):
    d.text((x, y), value, font=font(size, bold), fill=color)


def box(x0, y0, x1, y1, fill, outline):
    d.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=fill, outline=outline, width=2)


text(50, 42, "Ultra96-V2：実機なしでどこまで確認できたか", 32, "#152238", True)
text(50, 87, "2026-11-26 時点  ・  FPGAは小さな計算機、PLは書き換え可能な回路部分", 18, "#526176")

green = ("#e3f5ea", "#42a56c")
yellow = ("#fff3d9", "#dfaa38")
for x in (50, 405, 760):
    box(x, 140, x + 310, 284, *green)
box(50, 320, 540, 452, *yellow)
box(580, 320, 1070, 452, *yellow)
box(50, 486, 1070, 574, "#e9edf3", "#a2acb8")

for x, title, line1, line2 in (
    (76, "✓ 開発ツール", "Vivado / Vitis 2023.1", "Vitis AI 3.5 起動確認"),
    (431, "✓ ボードの土台", "Ultra96-V2の定義を認識", "PS設計を生成・検証"),
    (786, "✓ NN計算の試作品", "実データ由来30点で一致", "2023.1で配置配線合格"),
):
    text(x, 163, title, 23, "#19663e", True)
    text(x, 206, line1)
    text(x, 235, line2)

text(76, 342, "△ CNNとDPUの接続", 23, "#8b5b09", True)
text(76, 386, "共有CNNのDPU向け変換は成功", 18, "#6f541d")
text(76, 416, "配置配線合格・警告は要検証", 18, "#6f541d")
text(606, 342, "△ 画像全体の判定", 23, "#8b5b09", True)
text(606, 386, "CNN → 特徴 → NN → 判定の連結", 18, "#6f541d")
text(606, 416, "全体シミュレーションは未完成", 18, "#6f541d")
text(76, 504, "○ 実機が届いてから", 23, "#40526a", True)
text(338, 508, "起動・接続・速度・消費電力の実測", 18, "#40526a")
text(76, 543, "緑＝確認済み　 黄＝未完成　 灰＝ボードが必要　 ※ NN試作品の合格は全体の合格ではありません", 16, "#526176")

img.save(here / "boardless_status.png")
