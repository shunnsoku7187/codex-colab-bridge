"""Create slide-ready PNG diagrams for the proposed dual-sided early-exit system."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("results")
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


COLORS = {
    "bg": "#f8fafc",
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#334155",
    "stage": "#dbeafe",
    "head": "#ede9fe",
    "judge": "#fef3c7",
    "pass": "#dcfce7",
    "reject": "#fee2e2",
    "gate": "#cffafe",
    "fifo": "#e2e8f0",
    "accent": "#2563eb",
    "green": "#16a34a",
    "red": "#dc2626",
}


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)
    return int(box[2] - box[0]), int(box[3] - box[1])


def draw_centered(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    size: int = 28,
    bold: bool = False,
    fill: str = COLORS["ink"],
) -> None:
    fnt = font(size, bold)
    w, h = text_size(draw, text, fnt)
    x1, y1, x2, y2 = rect
    draw.multiline_text(
        ((x1 + x2 - w) / 2, (y1 + y2 - h) / 2 - 2),
        text,
        font=fnt,
        fill=fill,
        align="center",
        spacing=6,
    )


def box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str = COLORS["line"],
    radius: int = 18,
    size: int = 27,
    bold: bool = False,
) -> None:
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=3)
    draw_centered(draw, rect, text, size=size, bold=bold)


def diamond(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    w: int,
    h: int,
    text: str,
    fill: str = COLORS["judge"],
) -> tuple[int, int, int, int]:
    points = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(points, fill=fill, outline=COLORS["line"])
    draw.line(points + [points[0]], fill=COLORS["line"], width=3)
    rect = (cx - w // 2 + 22, cy - h // 2 + 28, cx + w // 2 - 22, cy + h // 2 - 28)
    draw_centered(draw, rect, text, size=25, bold=True)
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = COLORS["line"],
    width: int = 4,
    label: str | None = None,
    label_pos: tuple[int, int] | None = None,
) -> None:
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        head = [(x2, y2), (x2 - 16 * direction, y2 - 10), (x2 - 16 * direction, y2 + 10)]
    else:
        direction = 1 if y2 >= y1 else -1
        head = [(x2, y2), (x2 - 10, y2 - 16 * direction), (x2 + 10, y2 - 16 * direction)]
    draw.polygon(head, fill=color)
    if label:
        fnt = font(21)
        lx, ly = label_pos if label_pos else ((x1 + x2) // 2, (y1 + y2) // 2)
        tw, th = text_size(draw, label, fnt)
        pad = 7
        draw.rounded_rectangle((lx - tw // 2 - pad, ly - th // 2 - pad, lx + tw // 2 + pad, ly + th // 2 + pad), radius=8, fill=COLORS["bg"])
        draw.text((lx - tw // 2, ly - th // 2), label, font=fnt, fill=COLORS["muted"])


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: str = COLORS["line"],
    width: int = 4,
    label: str | None = None,
    label_pos: tuple[int, int] | None = None,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    arrow(draw, points[-2], points[-1], color=color, width=width)
    if label:
        fnt = font(21)
        lx, ly = label_pos if label_pos else points[len(points) // 2]
        tw, th = text_size(draw, label, fnt)
        draw.rounded_rectangle((lx - tw // 2 - 7, ly - th // 2 - 7, lx + tw // 2 + 7, ly + th // 2 + 7), radius=8, fill=COLORS["bg"])
        draw.text((lx - tw // 2, ly - th // 2), label, font=fnt, fill=COLORS["muted"])


def title(draw: ImageDraw.ImageDraw, text: str, subtitle: str, width: int) -> None:
    draw.text((54, 34), text, font=font(40, True), fill=COLORS["ink"])
    draw.text((56, 91), subtitle, font=font(24), fill=COLORS["muted"])
    draw.line((54, 132, width - 54, 132), fill="#cbd5e1", width=2)


def create_flow_diagram() -> None:
    w, h = 1600, 1180
    img = Image.new("RGB", (w, h), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    title(
        draw,
        "提案システムの処理フロー",
        "高信頼は早期通過、後段でも信頼回復しにくい画像は早期棄却、中間だけを後段へ送る",
        w,
    )

    x = 560
    bw, bh = 480, 82
    input_r = (x, 170, x + bw, 252)
    s0_r = (x, 305, x + bw, 387)
    j0 = (800, 475)
    s1_r = (x, 610, x + bw, 692)
    j1 = (800, 780)
    sf_r = (x, 915, x + bw, 997)
    jf = (800, 1080)

    box(draw, input_r, "入力画像", "#ffffff", size=30, bold=True)
    box(draw, s0_r, "浅いCNN層\nStage 0 / exit0まで実行", COLORS["stage"])
    diamond(draw, *j0, 460, 140, "exit0判定")
    box(draw, s1_r, "中間CNN層\nStage 1 / exit1まで実行", COLORS["stage"])
    diamond(draw, *j1, 460, 140, "exit1判定")
    box(draw, sf_r, "重い後段CNN層\nfinalまで実行", COLORS["stage"])
    diamond(draw, *jf, 440, 130, "final判定")

    pass0 = (1145, 382, 1510, 464)
    rej0 = (90, 382, 455, 464)
    pass1 = (1145, 687, 1510, 769)
    rej1 = (90, 687, 455, 769)
    passf = (1145, 1032, 1510, 1114)
    rejf = (90, 1032, 455, 1114)
    box(draw, pass0, "早期通過\n分類結果を採用", COLORS["pass"], outline=COLORS["green"])
    box(draw, rej0, "早期棄却\n不良・再検査扱い", COLORS["reject"], outline=COLORS["red"])
    box(draw, pass1, "早期通過\n分類結果を採用", COLORS["pass"], outline=COLORS["green"])
    box(draw, rej1, "早期棄却\n不良・再検査扱い", COLORS["reject"], outline=COLORS["red"])
    box(draw, passf, "通過\n分類結果を採用", COLORS["pass"], outline=COLORS["green"])
    box(draw, rejf, "棄却\n不良・再検査扱い", COLORS["reject"], outline=COLORS["red"])

    arrow(draw, (800, 252), (800, 305))
    arrow(draw, (800, 387), (800, 405))
    arrow(draw, (800, 545), (800, 610), label="中間", label_pos=(855, 578))
    arrow(draw, (800, 692), (800, 710))
    arrow(draw, (800, 850), (800, 915), label="中間", label_pos=(855, 882))
    arrow(draw, (800, 997), (800, 1015))

    poly_arrow(draw, [(1030, 475), (1120, 475), (1120, 423), (1145, 423)], color=COLORS["green"], label="高信頼", label_pos=(1110, 451))
    poly_arrow(draw, [(570, 475), (480, 475), (480, 423), (455, 423)], color=COLORS["red"], label="低信頼 + 改善見込みなし", label_pos=(345, 498))
    poly_arrow(draw, [(1030, 780), (1120, 780), (1120, 728), (1145, 728)], color=COLORS["green"], label="高信頼", label_pos=(1110, 756))
    poly_arrow(draw, [(570, 780), (480, 780), (480, 728), (455, 728)], color=COLORS["red"], label="低信頼 + 改善見込みなし", label_pos=(345, 803))
    poly_arrow(draw, [(1020, 1080), (1120, 1080), (1145, 1073)], color=COLORS["green"], label="高信頼", label_pos=(1090, 1053))
    poly_arrow(draw, [(580, 1080), (480, 1080), (455, 1073)], color=COLORS["red"], label="低信頼", label_pos=(500, 1053))

    note = "下側出口の役割\n分類ではなく\n「finalまで進める価値」\nを早期に判断"
    box(draw, (90, 165, 500, 288), note, "#fff7ed", outline="#fb923c", size=20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "proposal_system_flow.png", quality=95)


def create_fpga_architecture_diagram() -> None:
    w, h = 1900, 1080
    img = Image.new("RGB", (w, h), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    title(
        draw,
        "FPGA向けアーキテクチャ案",
        "CNNパイプライン途中に小型出口判定器を置き、valid/gate信号で後段回路の起動を抑える",
        w,
    )

    top_y = 330
    bot_y = 610
    bw = 170
    gap = 35
    x0 = 45
    top_keys = ["in", "buf", "s0", "e0", "j0", "g1", "s1", "e1", "j1"]
    top_labels = {
        "in": ("画像\nstream", "#ffffff"),
        "buf": ("入力/ライン\nバッファ", COLORS["fifo"]),
        "s0": ("Stage 0\n浅いCNN", COLORS["stage"]),
        "e0": ("Exit Head 0\n小型分類", COLORS["head"]),
        "j0": ("Judge 0\n通過/棄却/継続", COLORS["judge"]),
        "g1": ("Gate 1\nvalid制御", COLORS["gate"]),
        "s1": ("Stage 1\n中間CNN", COLORS["stage"]),
        "e1": ("Exit Head 1\n小型分類", COLORS["head"]),
        "j1": ("Judge 1\n通過/棄却/継続", COLORS["judge"]),
    }
    boxes = {}
    for i, key in enumerate(top_keys):
        boxes[key] = (x0 + i * (bw + gap), top_y, x0 + i * (bw + gap) + bw, top_y + 92)

    bottom_keys = ["g2", "sf", "ef", "jf"]
    bottom_labels = {
        "g2": ("Gate 2\nvalid制御", COLORS["gate"]),
        "sf": ("Final Stage\n重いCNN", COLORS["stage"]),
        "ef": ("Final Head\n最終分類", COLORS["head"]),
        "jf": ("Final Judge\n通過/棄却", COLORS["judge"]),
    }
    for i, key in enumerate(bottom_keys):
        boxes[key] = (555 + i * 245, bot_y, 555 + i * 245 + 190, bot_y + 92)

    out_r = (1505, 590, 1815, 664)
    rej_r = (1505, 742, 1815, 816)

    for key in top_keys:
        label, fill = top_labels[key]
        box(draw, boxes[key], label, fill, size=19 if key.startswith("j") else 22, bold=key.startswith("j"))
    for key in bottom_keys:
        label, fill = bottom_labels[key]
        box(draw, boxes[key], label, fill, size=22, bold=key == "jf")
    box(draw, out_r, "出力FIFO: ラベル採用", COLORS["pass"], outline=COLORS["green"], size=24)
    box(draw, rej_r, "棄却FIFO: 排出/再検査", COLORS["reject"], outline=COLORS["red"], size=24)

    for a, b in zip(top_keys, top_keys[1:]):
        ar = boxes[a]
        br = boxes[b]
        arrow(draw, (ar[2], (ar[1] + ar[3]) // 2), (br[0], (br[1] + br[3]) // 2), width=4)
    poly_arrow(draw, [(boxes["j1"][0] + bw // 2, boxes["j1"][3]), (boxes["j1"][0] + bw // 2, 545), (650, 545), (650, boxes["g2"][1])], label="継続", label_pos=(1040, 523))
    for a, b in zip(bottom_keys, bottom_keys[1:]):
        ar = boxes[a]
        br = boxes[b]
        arrow(draw, (ar[2], (ar[1] + ar[3]) // 2), (br[0], (br[1] + br[3]) // 2), width=4)
    arrow(draw, (boxes["jf"][2], bot_y + 32), (out_r[0], out_r[1] + 37), color=COLORS["green"], label="通過", label_pos=(1458, 622))
    arrow(draw, (boxes["jf"][2], bot_y + 72), (rej_r[0], rej_r[1] + 37), color=COLORS["red"], label="棄却", label_pos=(1458, 756))

    j0cx = (boxes["j0"][0] + boxes["j0"][2]) // 2
    j1cx = (boxes["j1"][0] + boxes["j1"][2]) // 2
    out_mid = (out_r[0], (out_r[1] + out_r[3]) // 2)
    rej_mid = (rej_r[0], (rej_r[1] + rej_r[3]) // 2)
    poly_arrow(draw, [(j0cx, top_y), (j0cx, 250), (1435, 250), out_mid], color=COLORS["green"], label="早期通過", label_pos=(1220, 226))
    poly_arrow(draw, [(j0cx, top_y + 92), (j0cx, 850), (1435, 850), rej_mid], color=COLORS["red"], label="早期棄却", label_pos=(1220, 875))
    poly_arrow(draw, [(j1cx, top_y), (j1cx, 285), (1465, 285), out_mid], color=COLORS["green"], label="早期通過", label_pos=(1600, 302))
    poly_arrow(draw, [(j1cx, top_y + 92), (j1cx, 890), (1465, 890), rej_mid], color=COLORS["red"], label="早期棄却", label_pos=(1600, 912))

    ctrl = (610, 150, 1290, 245)
    box(draw, ctrl, "制御FSM / 閾値レジスタ / 後段改善見込み判定パラメータ", "#ffffff", outline=COLORS["accent"], size=24, bold=True)
    for tx, ty in [(boxes["j0"][0] + bw // 2, top_y), (boxes["g1"][0] + bw // 2, top_y), (boxes["j1"][0] + bw // 2, top_y), (boxes["g2"][0] + 95, bot_y), (boxes["jf"][0] + 95, bot_y)]:
        draw.line((950, 245, tx, ty), fill="#bfdbfe", width=1)

    legend_y = 940
    draw.text((70, legend_y), "FPGAで効く点", font=font(28, True), fill=COLORS["ink"])
    bullets = [
        "Judgeは小さな比較器・LUT・固定小数点積和として実装しやすい",
        "Gateで後段CNNへのvalidを落とし、パイプライン占有と動的電力を削る",
        "通過/棄却はFIFOで即時分流し、ストリーム処理を止めない",
    ]
    for i, item in enumerate(bullets):
        draw.ellipse((72, legend_y + 48 + i * 34, 84, legend_y + 60 + i * 34), fill=COLORS["accent"])
        draw.text((96, legend_y + 38 + i * 34), item, font=font(22), fill=COLORS["muted"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "proposal_fpga_architecture.png", quality=95)


def main() -> None:
    create_flow_diagram()
    create_fpga_architecture_diagram()
    print("wrote results/proposal_system_flow.png")
    print("wrote results/proposal_fpga_architecture.png")


if __name__ == "__main__":
    main()
