"""Create explanatory figures for the professor-facing PatchCore FPGA report."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/figures")
FONT = "C:/Windows/Fonts/NotoSansJP-VF.ttf"
BOLD = "C:/Windows/Fonts/YuGothB.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def save(img: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / name, quality=95)


def rounded_box(draw: ImageDraw.ImageDraw, xy, fill, outline="#334155", radius=18, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill="#111827", width=18, line_gap=6, anchor=None):
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill, anchor=anchor)
        y += fnt.size + line_gap


def arrow(draw: ImageDraw.ImageDraw, start, end, fill="#334155", width=4):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - sign * 16, y2 - 8), (x2 - sign * 16, y2 + 8)]
    else:
        sign = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 8, y2 - sign * 16), (x2 + 8, y2 - sign * 16)]
    draw.polygon(pts, fill=fill)


def patchcore_flow() -> None:
    img = Image.new("RGB", (2000, 900), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "PatchCoreの基本動作", font=font(42, True), fill="#0f172a")
    d.text((70, 105), "正常画像だけから特徴メモリバンクを作り、推論時は各パッチ特徴の最近傍距離で異常度を決める。", font=font(25), fill="#334155")

    train_boxes = [
        ((80, 230, 400, 370), "正常画像\n欠陥画像は使わない", "#dbeafe"),
        ((520, 230, 840, 370), "CNN特徴抽出\nImageNet事前学習特徴", "#e0f2fe"),
        ((960, 230, 1280, 370), "パッチ特徴\n画像内の局所特徴集合", "#dcfce7"),
        ((1400, 230, 1720, 370), "メモリバンク\n正常パッチ特徴を保存", "#fef3c7"),
    ]
    for xy, text, color in train_boxes:
        rounded_box(d, xy, color)
        draw_wrapped(d, (xy[0] + 28, xy[1] + 28), text, font(25, True), width=16)
    for a, b in [((400, 300), (520, 300)), ((840, 300), (960, 300)), ((1280, 300), (1400, 300))]:
        arrow(d, a, b)
    d.text((80, 190), "学習/準備", font=font(26, True), fill="#1d4ed8")

    infer_boxes = [
        ((80, 590, 400, 730), "検査画像\n良品か欠陥か未知", "#fee2e2"),
        ((520, 590, 840, 730), "CNN特徴抽出\n同じ特徴空間へ写す", "#e0f2fe"),
        ((960, 590, 1280, 730), "KNN探索\n近い正常特徴を探す", "#fce7f3"),
        ((1400, 590, 1720, 730), "異常スコア\n距離が大きいほど異常", "#ede9fe"),
        ((1810, 590, 1950, 730), "判定\n通過/排出", "#f3f4f6"),
    ]
    for xy, text, color in infer_boxes:
        rounded_box(d, xy, color)
        draw_wrapped(d, (xy[0] + 28, xy[1] + 24), text, font(24, True), width=15)
    for a, b in [((400, 660), (520, 660)), ((840, 660), (960, 660)), ((1280, 660), (1400, 660)), ((1720, 660), (1810, 660))]:
        arrow(d, a, b)
    arrow(d, (1560, 370), (1120, 590), fill="#9333ea", width=5)
    d.text((80, 545), "推論/検品", font=font(26, True), fill="#b91c1c")
    d.text((1320, 435), "重い部分: メモリバンク容量とKNN探索", font=font(28, True), fill="#7e22ce")
    save(img, "patchcore_flow.png")


def proposed_system() -> None:
    img = Image.new("RGB", (1800, 900), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "提案システム: カテゴリ別プロファイル型 PatchCore-lite FPGA", font=font(40, True), fill="#0f172a")
    d.text((70, 105), "検品対象カテゴリが事前に分かることを利用し、そのカテゴリに必要な特徴層・グリッド・バンクだけをFPGAに載せる。", font=font(24), fill="#334155")

    left = [
        ((80, 230, 390, 370), "事前プロファイル\nカテゴリごとに探索", "#e0f2fe"),
        ((80, 470, 390, 610), "設定表\n特徴層 / grid / bank / 閾値", "#dbeafe"),
    ]
    right = [
        ((560, 190, 830, 320), "カテゴリ選択\n例: hazelnut", "#dcfce7"),
        ((970, 190, 1250, 320), "設定レジスタ\n特徴層・閾値を切替", "#fef3c7"),
        ((560, 430, 830, 570), "特徴抽出器\n選択層だけ使用", "#e0f2fe"),
        ((970, 430, 1250, 570), "カテゴリ別\nメモリバンク", "#fde68a"),
        ((1390, 430, 1690, 570), "並列KNN距離計算\nFPGAでパイプライン化", "#fce7f3"),
        ((970, 690, 1250, 810), "異常スコア", "#ede9fe"),
        ((1390, 690, 1690, 810), "通過/排出判定", "#f3f4f6"),
    ]
    for xy, text, color in left + right:
        rounded_box(d, xy, color)
        draw_wrapped(d, (xy[0] + 25, xy[1] + 24), text, font(25, True), width=15)
    arrow(d, (235, 370), (235, 470))
    arrow(d, (390, 540), (560, 255))
    arrow(d, (830, 255), (970, 255))
    arrow(d, (1110, 320), (700, 430))
    arrow(d, (1110, 320), (1110, 430))
    arrow(d, (830, 500), (970, 500))
    arrow(d, (1250, 500), (1390, 500))
    arrow(d, (1540, 570), (1110, 690))
    arrow(d, (1250, 750), (1390, 750))

    d.text((80, 675), "狙い", font=font(30, True), fill="#0f172a")
    notes = [
        "全カテゴリ共通の重い構成を避ける",
        "カテゴリごとに必要なバンクだけを保持",
        "KNN探索を小さくし、FPGAで並列化する",
        "品質は欠陥誤通過と良品通過で評価する",
    ]
    y = 720
    for note in notes:
        d.ellipse((85, y + 8, 100, y + 23), fill="#2563eb")
        d.text((115, y), note, font=font(23), fill="#1f2937")
        y += 38
    save(img, "proposed_patchcore_fpga.png")


def result_bars() -> None:
    closure = json.loads(Path("results/patchcore_fpga_preimplementation_closure_001_summary.json").read_text(encoding="utf-8"))
    mode = closure["mode_switch_summary"]
    cost = closure["cost_summary"]
    values = [
        ("KNN演算\n平均", cost["mean_relative_nn_ops"], "#2563eb"),
        ("KNN演算\n中央値", cost["median_relative_nn_ops"], "#60a5fa"),
        ("総コスト\n平均", cost["mean_relative_total_proxy_ops"], "#16a34a"),
        ("総コスト\n中央値", cost["median_relative_total_proxy_ops"], "#86efac"),
        ("全カテゴリ\nバンク", mode["all_category_bank_ratio"], "#f97316"),
    ]
    img = Image.new("RGB", (1600, 930), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "カテゴリ別プロファイルによる削減効果", font=font(42, True), fill="#0f172a")
    d.text((70, 105), "基準構成を1.0xとした相対値。小さいほどFPGA実装時のメモリ/探索負荷が軽い。", font=font(25), fill="#334155")
    chart = (150, 210, 1480, 650)
    d.line([(chart[0], chart[3]), (chart[2], chart[3])], fill="#334155", width=3)
    d.line([(chart[0], chart[1]), (chart[0], chart[3])], fill="#334155", width=3)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = chart[3] - int((chart[3] - chart[1]) * t)
        d.line([(chart[0] - 8, y), (chart[2], y)], fill="#e5e7eb", width=1)
        d.text((70, y - 15), f"{t:.2f}x", font=font(20), fill="#475569")
    bar_w = 160
    gap = 80
    x = 240
    for label, val, color in values:
        h = int((chart[3] - chart[1]) * val)
        y0 = chart[3] - h
        d.rounded_rectangle((x, y0, x + bar_w, chart[3]), radius=14, fill=color)
        d.text((x + bar_w / 2, y0 - 38), f"{val:.4f}x", font=font(24, True), fill="#111827", anchor="mm")
        for i, line in enumerate(label.split("\n")):
            d.text((x + bar_w / 2, chart[3] + 38 + 32 * i), line, font=font(23, True), fill="#111827", anchor="mm")
        x += bar_w + gap
    d.text((150, 820), "注: 総コストはCNN MACとKNN演算の近似合算。実際の速度・電力はFPGA実装後に測定する。", font=font(24), fill="#475569")
    save(img, "profiled_reduction_bars.png")


def main() -> None:
    patchcore_flow()
    proposed_system()
    result_bars()
    print("created report figures in docs/figures")


if __name__ == "__main__":
    main()
