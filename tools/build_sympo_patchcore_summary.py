from __future__ import annotations

import json
import math
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from pypdf import PdfReader
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import BaseDocTemplate, Image, KeepTogether, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "sympo"
FIG_DIR = OUT_DIR / "figures"

DOCX_OUT = OUT_DIR / "patchcore_profile_switch_summary.docx"
PDF_OUT = OUT_DIR / "patchcore_profile_switch_summary.pdf"
PDF_EN_OUT = OUT_DIR / "patchcore_profile_switch_summary_en.pdf"
PDF_JA_OUT = OUT_DIR / "patchcore_profile_switch_summary_ja.pdf"
TEX_EN_OUT = OUT_DIR / "patchcore_profile_switch_summary_en.tex"
TEX_JA_OUT = OUT_DIR / "patchcore_profile_switch_summary_ja.tex"
FIG_SYSTEM = FIG_DIR / "system_architecture.png"
FIG_PATCHCORE = FIG_DIR / "patchcore_process.png"
FIG_RESULT = FIG_DIR / "profile_switch_result.png"
FIG_EQUATION = FIG_DIR / "relative_knn_cost_equation.png"

PROFILE_ROWS_EN = [
    ["Layer", "Grid", "D", "top-k", "Category"],
    ["layer3", "14", "512", "0.005", "bottle, carpet, hazelnut, leather"],
    ["layer3", "14", "512", "0.005", "metal_nut, tile, zipper"],
    ["layer3", "14", "512", "0.010", "screw, transistor"],
    ["layer3", "14", "512", "0.020", "capsule"],
    ["layer3", "14", "512", "0.050", "cable"],
    ["layer2+3", "14", "768", "0.005", "grid"],
    ["layer2+3", "14", "768", "0.010", "pill, wood"],
    ["layer2+3", "14", "768", "0.050", "toothbrush"],
]

PROFILE_ROWS_JA = [
    ["層", "grid", "D", "top-k", "カテゴリ"],
    ["layer3", "14", "512", "0.005", "bottle, carpet, hazelnut, leather"],
    ["layer3", "14", "512", "0.005", "metal_nut, tile, zipper"],
    ["layer3", "14", "512", "0.010", "screw, transistor"],
    ["layer3", "14", "512", "0.020", "capsule"],
    ["layer3", "14", "512", "0.050", "cable"],
    ["layer2+3", "14", "768", "0.005", "grid"],
    ["layer2+3", "14", "768", "0.010", "pill, wood"],
    ["layer2+3", "14", "768", "0.050", "toothbrush"],
]


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def get_font(size: int, bold: bool = False):
        candidates = [
            r"C:\Windows\Fonts\cambria.ttc",
            r"C:\Windows\Fonts\meiryob.ttc" if bold else r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()

    font = get_font(27)
    small_font = get_font(22)
    title_font = get_font(34, True)
    tiny_font = get_font(18)

    def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#111111", width: int = 4) -> None:
        draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for da in (0.45, -0.45):
            x2 = end[0] - 17 * math.cos(angle + da)
            y2 = end[1] - 17 * math.sin(angle + da)
            draw.line((end[0], end[1], x2, y2), fill=fill, width=width)

    def poly_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: str = "#111111", width: int = 4) -> None:
        for start, end in zip(points, points[1:]):
            draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
        start = points[-2]
        end = points[-1]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for da in (0.45, -0.45):
            x2 = end[0] - 17 * math.cos(angle + da)
            y2 = end[1] - 17 * math.sin(angle + da)
            draw.line((end[0], end[1], x2, y2), fill=fill, width=width)

    def box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: str = "#f3f3f3", outline: str = "#111111") -> None:
        draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=3)
        lines = text.split("\n")
        line_step = 24 if len(lines) >= 3 else 30
        max_h = max(draw.textbbox((0, 0), line, font=small_font)[3] for line in lines)
        y0 = y + h / 2 - ((len(lines) - 1) * line_step + max_h) / 2
        for j, line in enumerate(lines):
            tw = draw.textlength(line, font=small_font)
            draw.text((x + w / 2 - tw / 2, y0 + j * line_step), line, fill="#111111", font=small_font)

    img = PILImage.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 88), "Training", fill="#111111", font=font)
    draw.text((40, 258), "Inference", fill="#111111", font=font)
    training = [
        (185, 55, 165, 76, "Normal\nimage"),
        (415, 55, 165, 76, "CNN\nfeatures"),
        (645, 55, 170, 76, "Patch\nvectors"),
        (885, 55, 190, 76, "Memory\nbank"),
    ]
    inference = [
        (185, 225, 165, 76, "Test\nimage"),
        (415, 225, 165, 76, "CNN\nfeatures"),
        (645, 225, 170, 76, "Patch\nvectors"),
        (885, 225, 190, 76, "KNN\nsearch"),
    ]
    for item in training[:-1]:
        box(draw, *item)
    box(draw, *training[-1], fill="#f2f2f2", outline="#111111")
    for item in inference[:-1]:
        box(draw, *item)
    box(draw, *inference[-1], fill="#e0e0e0", outline="#111111")
    box(draw, 950, 330, 170, 60, "Score /\nJudge")
    for s, e in [
        ((350, 93), (415, 93)),
        ((580, 93), (645, 93)),
        ((815, 93), (885, 93)),
        ((350, 263), (415, 263)),
        ((580, 263), (645, 263)),
        ((815, 263), (885, 263)),
        ((980, 131), (980, 225)),
        ((980, 301), (1015, 330)),
    ]:
        arrow(draw, s, e)
    img.save(FIG_PATCHCORE)

    img = PILImage.new("RGB", (1360, 650), "white")
    draw = ImageDraw.Draw(img)
    draw.text((85, 25), "Maximum-effect examples", fill="#111111", font=title_font)

    cases = [
        ("AB\ncable+carpet", 52.59, 79.99, 1.000, 0.333),
        ("ABC\ncable+carpet+screw", 37.99, 66.33, 1.000, 0.222),
    ]

    def draw_panel(x0: int, y0: int, w: int, h: int, title: str, unit: str, values: list[tuple[str, float, float]], ymax: float, color: str) -> None:
        draw.text((x0 + 15, y0 - 45), title, fill="#111111", font=small_font)
        for i in range(6):
            y = y0 + i * h / 5
            draw.line((x0, y, x0 + w, y), fill="#dddddd", width=1)
        draw.line((x0, y0, x0, y0 + h), fill="#444444", width=2)
        draw.line((x0, y0 + h, x0 + w, y0 + h), fill="#444444", width=2)
        draw.text((x0 - 80, y0 + h / 2 - 15), unit, fill="#333333", font=small_font)
        group_gap = w / len(values)
        for i, (label, common, proposed) in enumerate(values):
            cx = x0 + group_gap * i + group_gap / 2
            for j, (name, value, fill) in enumerate([("common", common, "#b8b8b8"), ("proposed", proposed, color)]):
                bw = 54
                bx = cx - 65 + j * 70
                bh = max(1, value / ymax * h)
                draw.rectangle((bx, y0 + h - bh, bx + bw, y0 + h), fill=fill)
                text = f"{value:.1f}" if unit == "%" else f"{value:.3f}x"
                tw = draw.textlength(text, font=small_font)
                draw.text((bx + bw / 2 - tw / 2, y0 + h - bh - 30), text, fill="#111111", font=small_font)
            for k, line in enumerate(label.split("\n")):
                tw = draw.textlength(line, font=small_font)
                draw.text((cx - tw / 2, y0 + h + 18 + k * 27), line, fill="#111111", font=small_font)

    draw_panel(
        115,
        105,
        500,
        360,
        "Good-product pass rate",
        "%",
        [(label, common_good, proposed_good) for label, common_good, proposed_good, _, _ in cases],
        100.0,
        "#4a4a4a",
    )
    draw_panel(
        790,
        105,
        430,
        360,
        "Relative KNN search cost",
        "x",
        [(label, common_ops, proposed_ops) for label, _, _, common_ops, proposed_ops in cases],
        1.05,
        "#4a4a4a",
    )
    draw.rectangle((320, 570, 342, 592), fill="#b8b8b8")
    draw.text((355, 567), "Common profile + common bank", fill="#111111", font=small_font)
    draw.rectangle((745, 570, 767, 592), fill="#4a4a4a")
    draw.text((780, 567), "Proposed profile + bank switch", fill="#111111", font=small_font)
    img.save(FIG_RESULT)

    equation = PILImage.new("RGB", (1200, 210), "white")
    eq_draw = ImageDraw.Draw(equation)
    eq_font = get_font(45)
    eq_small = get_font(31)
    eq_draw.text((65, 35), "Cₘ(S) = Σ c∈S  p_c Pₘ,c Bₘ,c Dₘ,c", fill=(20, 20, 20), font=eq_font)
    eq_draw.text((65, 118), "Rₘ(S) = Cₘ(S) / C_common(S)", fill=(20, 20, 20), font=eq_font)
    eq_draw.text((91, 88), "category-weighted KNN operations", fill=(90, 90, 90), font=eq_small)
    equation.save(FIG_EQUATION)

    img = PILImage.new("RGB", (1200, 430), "white")
    draw = ImageDraw.Draw(img)
    box(draw, 45, 185, 145, 78, "Input\nimage")
    box(draw, 245, 185, 170, 78, "Shared\nCNN")
    box(draw, 480, 185, 190, 78, "Profiled\nfeatures")
    box(draw, 735, 185, 180, 78, "KNN\npipeline")
    box(draw, 970, 185, 175, 78, "Anomaly\njudge")
    box(draw, 350, 35, 260, 90, "Profile regs\nlayer/grid/D\ntop-k/thresh.", fill="#f0f0f0", outline="#111111")
    box(draw, 680, 35, 260, 90, "Bank range\nselector", fill="#f0f0f0", outline="#111111")
    draw.rectangle((650, 315, 1055, 392), fill="#f2f2f2", outline="#111111", width=3)
    for i, label in enumerate(["bank A", "bank B", "bank C"]):
        x0 = 680 + i * 120
        draw.rectangle((x0, 338, x0 + 95, 373), fill=["#eeeeee", "#dddddd", "#cccccc"][i], outline="#111111", width=2)
        tw = draw.textlength(label, font=tiny_font)
        draw.text((x0 + 47 - tw / 2, 344), label, fill="#111111", font=tiny_font)
    for s, e in [
        ((190, 224), (245, 224)),
        ((415, 224), (480, 224)),
        ((670, 224), (735, 224)),
        ((915, 224), (970, 224)),
    ]:
        arrow(draw, s, e)
    draw.text((90, 76), "Category ID", fill="#111111", font=small_font)
    poly_arrow(draw, [(230, 85), (300, 85), (300, 20), (480, 20), (480, 35)], fill="#111111", width=4)
    poly_arrow(draw, [(300, 20), (810, 20), (810, 35)], fill="#111111", width=4)
    poly_arrow(draw, [(480, 125), (480, 150), (575, 150), (575, 185)], fill="#111111", width=4)
    poly_arrow(draw, [(810, 125), (810, 150), (825, 150), (825, 185)], fill="#111111", width=4)
    poly_arrow(draw, [(855, 315), (855, 285), (825, 285), (825, 263)], fill="#111111", width=4)
    img.save(FIG_SYSTEM)


TITLE = "FPGA Implementation of a PatchCore-lite Anomaly Detection System with Category-wise Profile Switching"
TITLE_PDF = "FPGA Implementation of a PatchCore-lite Anomaly Detection System<br/>with Category-wise Profile Switching"
AUTHORS = "Shunya Yamada and Kazuyoshi Takagi"
AFFILIATION = "Computer Architecture Laboratory, Department of Information Engineering, Graduate School of Engineering, Mie University"
CONTACT = "(E-mail: shunya@arch.info.mie-u.ac.jp)"

ABSTRACT = (
    "PatchCore is a strong industrial anomaly detection method that stores normal patch features "
    "in a memory bank and detects anomalies by k-nearest neighbor (KNN) search. However, a multi-product "
    "inspection line forces one common configuration to cover different object categories, increasing "
    "KNN search cost and making FPGA implementation memory-bound. This paper studies a "
    "PatchCore-lite system that keeps one shared CNN backbone but switches category-specific profiles, "
    "including the extracted feature layer, grid resolution, top-k score setting, threshold, and memory-bank "
    "address. Experiments on MVTec AD indicate that the proposed switching reduces the theoretical KNN "
    "search cost while preserving or improving the good-product pass rate under a fixed defect false-pass constraint."
)


SECTIONS = [
    (
        "1. INTRODUCTION",
        [
            "Industrial visual inspection often needs high defect sensitivity and stable throughput. PatchCore [1] is attractive because it can be trained from normal samples only and has reported high image-level anomaly detection performance on MVTec AD [2]. Its main implementation cost is not back-propagation, but feature extraction and memory-bank KNN search.",
            "A simple FPGA implementation can prepare one common profile and one common memory bank for all products. This is easy to control, but it searches unnecessary bank entries and uses one feature setting even when each product category has different visual characteristics. This study investigates whether product-category information, which is already known before inspection in many lines, can be used to select a lighter profile without reconfiguring the FPGA bitstream.",
        ],
    ),
    (
        "2. PATCHCORE",
        [
            "PatchCore is an anomaly detection method based on normal feature memory. The training and inference flow is shown in Fig. 1. First, a backbone CNN extracts intermediate feature maps from an input image. These feature maps are converted into patch-level feature vectors. During training, feature vectors obtained only from normal images are stored as a memory bank. During inference, each input patch feature is compared with the memory bank by KNN search. If the input feature is far from the stored normal features, the image is judged anomalous.",
            "The main design parameters are the backbone CNN, the feature profile, and the memory bank. The backbone CNN, such as WideResNet50-2, extracts feature maps from the image. A profile is the set of lightweight choices after this CNN: layer2/layer3 indicate which intermediate CNN feature maps are used, grid 14 means that the feature map is sampled into 14 x 14 patch positions, D is the feature-vector dimension per patch, and top-k is the fraction of the largest patch anomaly scores used to form the image anomaly score.",
            "The memory bank stores representative normal patch features. A larger bank improves coverage of the normal distribution but increases memory traffic and KNN operations. k-center coreset selection is a standard technique that keeps representative bank entries while reducing the bank size [3]. To isolate the effect of category-wise profile switching, all compared systems use the same coreset-based bank budget in the main evaluation.",
        ],
    ),
    (
        "3. PROPOSED SYSTEM",
        [
            "The proposed system targets multi-category inspection on one FPGA, as illustrated in Fig. 2. It keeps one shared CNN datapath and switches only category-dependent profile registers and memory-bank ranges before inference. The switched parameters are feature layer, grid size, feature dimension, top-k aggregation, anomaly threshold, and the base address and search length of the category memory bank.",
            "The proposed contribution is category-wise execution control in addition to conventional memory-bank reduction. Existing PatchCore acceleration commonly uses coreset selection to reduce stored features. The proposed system further uses product-category information, which is available before inspection, to select the feature profile and bank range used for the current product. This reduces active KNN search compared with a common multi-category PatchCore, while reusing the CNN and KNN datapath instead of preparing independent engines for each category.",
        ],
    ),
    (
        "4. EXPERIMENTAL SETUP",
        [
            "Experiments used all 15 MVTec AD categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, and zipper. All two-category and three-category subsets were evaluated, giving 105 AB tasks and 455 ABC tasks, 560 tasks in total. Only normal training images were used to construct the memory bank, and both normal and defective test images were used for evaluation. The feature extractor was a PatchCore-like WideResNet50-2 backbone. The main comparison used k-center coreset selection with the same bank budget, bank/category = 3000, for all compared systems.",
            "The evaluated metric was the good-product pass rate under a defect false-pass constraint of 3% or lower. The defect false-pass rate is the fraction of defective samples judged normal. The good-product pass rate is the fraction of normal samples judged normal. The anomaly threshold was selected for each evaluated system so that the defect false-pass constraint was satisfied.",
            "The compared systems were: (i) default profile with a common bank, (ii) default profile with category-bank switching, (iii) one category-specific profile fixed and applied to all categories with category-bank switching, and (iv) category-specific profile and category-bank switching. For (iii), the fixed profile with the highest mean good-product pass rate was selected for each subset, giving a strong single-profile baseline. The default profile was layer2+3, grid 14, D=768, and top-k fraction 0.010. Category-specific profiles were obtained before the subset experiment by a single-category sweep at bank=12000. For each category, profiles whose good-product pass rate was within 2 percentage points of the default profile under the 3% defect false-pass constraint were kept, and the smallest profile by patch_count x D was selected. The selected profiles are listed in Table 1.",
        ],
    ),
    (
        "5. RESULTS",
        [
            "Tables 2 and 3 summarize the AB and ABC results. For the 105 two-category tasks, the proposed profile-and-bank switching achieved 85.53% mean good-product pass rate with 0.378x relative KNN search cost. For the 455 three-category tasks, it achieved 85.53% with 0.252x relative cost. Thus, the proposed system maintained good-product pass performance while reducing the active search work, especially when more categories shared one inspection system.",
            "Large improvements were observed in subsets where the common profile did not fit all categories well. In the cable+carpet task, the mean good-product pass rate increased from 52.59% to 79.99%, while the KNN search cost became 0.333x. In the cable+carpet+screw task, it increased from 37.99% to 66.33%, while the cost became 0.222x.",
        ],
    ),
    (
        "6. DISCUSSION",
        [
            "The results show an effect beyond memory-bank address switching. Category-bank switching reduced the search range, but the proposed system also improved the mean and minimum good-product pass rates by selecting the feature profile for each category. This indicates that useful feature layer, grid density, feature dimension, and top-k aggregation are category-dependent. For example, cable selected a larger top-k fraction, suggesting that categories with elongated structures or spatially distributed defects benefit from aggregating more high-score patches.",
            "The fixed-profile baseline clarifies why switching is needed. Even when a strong single profile is selected for each subset, one category's profile is still forced onto the other categories. The proposed system avoids this profile mismatch by assigning each category its own profile before inspection. Thus, the contribution is stable category-specialized execution on a shared multi-category system.",
            "The relative KNN search cost is modeled by Eq. (1). Since the main comparison fixes the bank budget per category, the additional reduction comes from profile parameters such as patch count and feature dimension. The lower relative cost in ABC tasks suggests that the benefit increases as the number of target categories increases.",
        ],
    ),
    (
        "7. FPGA COST ESTIMATE",
        [
            "On FPGA, the reduction maps to bank base address, search length, feature-dimension enable, and profile registers. The system can reuse one CNN and KNN datapath while changing the active memory range and loop bounds for the current category.",
            "Table 4 gives the FPGA-facing estimate. If 8-bit bank features and 64 distance lanes are assumed, KNN cycles and bank reads follow the relative KNN search cost. Under this model, the ABC result reduces KNN-side activity to about one quarter of the common-profile baseline.",
            "The remaining evaluation is to implement the profile-controlled KNN pipeline and measure latency, memory bandwidth, logic usage, on-chip memory usage, and dynamic power. The KNN engine mainly scales with the number of distance lanes and memory ports, whereas category switching adds only profile registers, address offsets, and small multiplexers. The implementation will verify whether this switching overhead is smaller than the saved KNN loops and memory traffic.",
        ],
    ),
    (
        "8. CONCLUSION",
        [
            "This study proposes category-wise profile switching for PatchCore-lite anomaly detection toward FPGA implementation. Experiments under a fixed defect false-pass constraint showed that profile switching can improve the good-product pass rate and reduce KNN search cost in multi-category inspection. The next step is FPGA implementation and measurement of the gap between the theoretical model and actual resource, latency, and power results.",
        ],
    ),
]

REFERENCES = [
    "[1] K. Roth, L. Pemula, J. Zepeda, B. Schoelkopf, T. Brox, and P. Gehler, \"Towards Total Recall in Industrial Anomaly Detection,\" CVPR, pp. 14318-14328, 2022.",
    "[2] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, \"MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,\" CVPR, 2019.",
    "[3] OpenVINO Anomalib Contributors, \"PatchCore model documentation,\" 2026.",
]

JA_TITLE = "カテゴリ別プロファイル切り替え機構を備えたPatchCore-lite異常検知システムのFPGA実装に向けた検討"
JA_AUTHORS = "山田 峻也・高木 一義"
JA_AFFILIATION = "三重大学大学院 工学研究科 情報工学専攻 コンピュータアーキテクチャ研究室"
JA_CONTACT = "（E-mail: shunya@arch.info.mie-u.ac.jp）"
JA_ABSTRACT = (
    "PatchCoreは，正常画像から得た特徴量をメモリバンクとして保持し，推論時に入力特徴量と正常特徴量の距離を近傍探索で求める異常検知手法である。"
    "高い検知性能を持つ一方で，複数カテゴリを一つの装置で扱う場合，共通設定では現在検査しているカテゴリに不要な特徴量やメモリバンクまで探索し，FPGA実装時のメモリアクセスと探索回数が大きくなる。"
    "本稿では，共通CNNを維持しつつ，検査カテゴリごとに特徴層，グリッド，top-k，しきい値，メモリバンク参照範囲を切り替えるPatchCore-lite異常検知システムを検討する。"
    "MVTec ADを用いた予備実験では，欠陥誤通過率を一定以下に制約した条件で，良品通過率を保ちながら近傍探索量を削減できることを確認した。"
)

JA_SECTIONS = [
    (
        "1. はじめに",
        [
            "工業検品では，欠陥品を良品として通してしまう誤通過を抑えつつ，良品をできるだけ多く通過させることが求められる。PatchCore[1]は正常例のみから学習でき，MVTec AD[2]などで高い異常検知性能が報告されているため，検品システムの土台として有力である。一方で，PatchCoreの推論ではCNNによる特徴抽出に加えて，正常特徴量を保存したメモリバンクとのk近傍探索（KNN探索）が必要になる。",
            "複数カテゴリを一つのFPGAで扱う場合，全カテゴリ共通の設定と共通メモリバンクを用いる構成が考えられる。しかしこの構成では，現在検査しているカテゴリに不要な正常特徴まで探索する。また，カテゴリごとに有効な特徴層やグリッド密度が異なる場合でも，同じ設定を用いる必要がある。本研究では，カテゴリ情報が検査前に分かっている状況を利用し，FPGA内部の動作モードをカテゴリごとに切り替える構成を検討する。",
        ],
    ),
    (
        "2. PatchCore",
        [
            "PatchCore[1]は，正常画像の特徴量を記憶しておき，入力画像がその正常特徴からどれだけ離れているかで異常を検知する手法であり，その処理流れを図1に示す。まず，入力画像からバックボーンCNNによって中間特徴マップを抽出する。次に，その特徴マップをパッチ単位の特徴ベクトルへ変換する。学習時には正常画像から得られた特徴ベクトルだけをメモリバンクに保存し，推論時には入力特徴とメモリバンクの距離を近傍探索で計算する。入力特徴が正常特徴から大きく離れていれば異常と判定する。",
            "PatchCoreで主に設定できる要素は，バックボーンCNN，profile，メモリバンクである。バックボーンCNNはWideResNet50-2などの特徴抽出器であり，画像から特徴マップを取り出す。profileは，CNN後段で用いる軽量な設定群を指す。layer2/layer3は利用するCNN中間特徴層，grid 14は特徴マップを14 x 14個のパッチ位置として扱う設定，Dは各パッチ特徴ベクトルの次元数，top-kは画像全体の異常スコアを作る際に上位何割のパッチ異常スコアを使うかを表す。",
            "メモリバンクは，正常画像から得た代表的なパッチ特徴を保存する領域である。バンクを大きくすると正常分布を広く覆えるが，推論時のメモリ読み出し量とKNN探索回数が増える。そこでPatchCoreでは，正常特徴を代表する点だけを残すk-center coresetが広く使われる[1][3]。本評価では，カテゴリ別profile切り替えの効果を分離するため，全ての比較構成で同じcoreset条件とbank数を用いる。",
        ],
    ),
    (
        "3. 提案手法",
        [
            "提案手法は，複数カテゴリを一つのFPGAで検査する状況を対象とし，構成を図2に示す。CNNの主要なデータパスは一つだけ用意し，検査カテゴリに応じてprofileレジスタとメモリバンク参照範囲を切り替える。切り替える値は，特徴層，グリッドサイズ，特徴次元，top-k集約，異常判定しきい値，カテゴリ別bankの開始アドレスと探索長である。これは画像ごとに別の推論モデルを動かす構成ではなく，検査対象が切り替わる直前に動作モードを更新する構成である。",
            "提案手法の特徴は，既存のメモリバンク削減に加えて，カテゴリごとに実行時のprofileを切り替える点にある。既存のPatchCore高速化では，k-center coresetにより保存する正常特徴量を削減する[1][3]。本研究ではさらに，検査ラインで事前に得られるカテゴリ情報を用いて，現在のカテゴリに対応した特徴層，集約方法，bank参照範囲を選択する。これにより，全カテゴリ共通PatchCoreより探索活動量を減らしつつ，カテゴリごとに完全に別のPatchCore回路を用意する場合より主要回路を共有できる。",
        ],
    ),
    (
        "4. 実験条件",
        [
            "実験にはMVTec AD[2]の全15カテゴリを用いた。対象カテゴリはbottle，cable，capsule，carpet，grid，hazelnut，leather，metal_nut，pill，screw，tile，toothbrush，transistor，wood，zipperである。全ての2カテゴリ組み合わせと3カテゴリ組み合わせを評価し，2カテゴリ検品105通り，3カテゴリ検品455通り，合計560通りを対象とした。メモリバンクの構築には正常な学習画像のみを用い，評価には正常画像と欠陥画像の両方を用いた。特徴抽出器にはWideResNet50-2を用いるPatchCore系の実装を用いた。主比較では，全ての比較構成でk-center coreset[1][3]を用い，bank/category=3000にそろえた。",
            "評価指標は，欠陥誤通過率を3%以下に制約したときの良品通過率である。欠陥誤通過率は欠陥品を正常として通してしまう割合，良品通過率は正常品を正常として通せる割合を意味する。異常判定しきい値は，各構成が欠陥誤通過率3%以下を満たす範囲で設定した。",
            "比較構成は，(1) デフォルトprofile+共通bank，(2) デフォルトprofile+カテゴリ別bank切り替え，(3) 一つのカテゴリ専用profileを全カテゴリに固定して使う構成+カテゴリ別bank切り替え，(4) カテゴリ別profile+カテゴリ別bank切り替えである。(3)では，各組み合わせで平均良品通過率が最大となる固定profileを採用し，強い単一profile基準として用いた。デフォルトprofileはlayer2+3，grid 14，D=768，top-k 0.010とした。カテゴリ専用profileは，組み合わせ実験の前に単一カテゴリ評価で決定した。bank=12000でprofile候補を探索し，欠陥誤通過率3%以下の条件で，デフォルトprofileからの良品通過率低下が2ポイント以内の候補を残した。その中でpatch_count x Dが最小のprofileを各カテゴリの専用profileとして選択した。選択したprofileを表1に示す。",
        ],
    ),
    (
        "5. 実験結果",
        [
            "表2，表3に2カテゴリ検品と3カテゴリ検品の結果を示す。2カテゴリ検品105通りでは，提案するprofile+bank切り替えの平均良品通過率は85.53%，相対KNN探索量は0.378倍であった。3カテゴリ検品455通りでは，平均良品通過率85.53%，相対KNN探索量0.252倍であった。提案構成は，良品通過性能を維持しながら，複数カテゴリを一つの検査システムで扱う際の探索活動量を削減した。",
            "共通profileが各カテゴリに十分適合しない組み合わせでは，より大きな改善が得られた。2カテゴリのcable+carpetでは，平均良品通過率が52.59%から79.99%へ向上し，KNN探索量は0.333倍となった。3カテゴリのcable+carpet+screwでは，平均良品通過率が37.99%から66.33%へ向上し，KNN探索量は0.222倍となった。",
        ],
    ),
    (
        "6. 考察",
        [
            "実験結果から，提案手法ではメモリバンク参照範囲の削減に加えて，profile選択による効果が得られた。カテゴリ別bank切り替えでは探索範囲を減らせるが，提案手法ではさらに，カテゴリごとに特徴層，grid，特徴次元，top-kを選ぶことで平均良品通過率と最低良品通過率が向上した。これは，有効な特徴の取り方がカテゴリに依存することを示している。例えばcableでは大きいtop-kが選択されており，細長い構造や空間的に広がる欠陥を持つカテゴリでは，より多くの高スコアパッチを集約する方が有効である可能性がある。",
            "単一固定profileとの比較は，profile切り替えの必要性を示す。単一固定profileは，各組み合わせに対して最も良い一つのprofileを選ぶ強い基準である。それでも，あるカテゴリ用のprofileを別カテゴリにも強制するため，組み合わせによって性能が不安定になる。提案手法は，カテゴリごとにprofileを割り当てることで，このprofile不一致を避けられる。",
            "近傍探索側の計算量は式(1)で表せる。主比較では各カテゴリのbank数をそろえているため，追加の削減効果はパッチ数や特徴次元などのprofile差から生じる。2カテゴリより3カテゴリで相対KNN探索量が小さくなったことから，対象カテゴリ数が増えるほどカテゴリ別切り替えの効果が表れやすいと考えられる。",
        ],
    ),
    (
        "7. FPGA実装に向けた見積もり",
        [
            "FPGA上では，提案手法の削減をbank開始アドレス，探索長，有効特徴次元，profileレジスタの制御に対応させられる。カテゴリごとに別々のPatchCore回路を用意するのではなく，共通のCNNとKNNデータパスを再利用し，メモリアクセス範囲とループ回数を切り替える構成になる。",
            "FPGA向け見積もりを表4に示す。期待できる効果は，共通CNN前段よりも，KNN探索とメモリアクセスに強く現れる。8bit特徴量を用い，距離計算器が1サイクルに64次元分を処理すると仮定すると，KNNサイクル数とbank読み出し回数は相対KNN探索量に比例する。この見積もりでは，3カテゴリ検品におけるKNN側の活動量は，共通profile構成の約4分の1になる。",
            "今後はprofile制御付きKNNパイプラインを実装し，レイテンシ，メモリ帯域，論理資源，オンチップメモリ，消費電力について，理論見積もりと実測値の差を評価する。KNN演算器の規模は主に距離計算レーン数とメモリポート数に依存し，提案手法で追加されるのはprofileレジスタ，bankアドレスオフセット，小規模な選択回路である。特に，この制御コストがKNNループ数とメモリアクセス削減による利益に対して十分小さいかを確認する。",
        ],
    ),
    (
        "8. まとめ",
        [
            "本稿では，PatchCore-lite異常検知をFPGAへ実装するためのカテゴリ別profile切り替え構成を検討した。提案の中心は，bank削減そのものではなく，カテゴリごとに有効なprofileを事前に求め，検査時に軽量に切り替える実行方式である。予備実験では，共通profileの基準構成より高い良品通過率と小さい相対KNN探索量を確認した。今後は，FPGA上でモード切り替え型KNNパイプラインを実装し，理論見積もりと実測の差を評価する。",
        ],
    ),
]

JA_REFERENCES = [
    "[1] K. Rothほか, \"Towards Total Recall in Industrial Anomaly Detection,\" CVPR, 2022.",
    "[2] P. Bergmannほか, \"MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,\" CVPR, 2019.",
    "[3] OpenVINO Anomalib Contributors, \"PatchCore model documentation,\" 2026.",
]


def add_columns(section, num_cols: int = 2, space_twips: int = 567) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols_el = cols[0] if cols else OxmlElement("w:cols")
    if not cols:
        sect_pr.append(cols_el)
    cols_el.set(qn("w:num"), str(num_cols))
    cols_el.set(qn("w:space"), str(space_twips))


def set_run_font(run, size: float = 10.0, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def add_para(doc: Document, text: str, size: float = 10.0, bold: bool = False, align=None, first_indent=True) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    if first_indent:
        p.paragraph_format.first_line_indent = Cm(0.4)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)


def build_docx() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)

    add_para(doc, TITLE, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False)
    add_para(doc, AUTHORS, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False)
    add_para(doc, AFFILIATION, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False)
    add_para(doc, CONTACT, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False)
    add_para(doc, "Abstract: " + ABSTRACT, size=10, first_indent=False)
    add_para(doc, "Keywords: industrial anomaly detection, PatchCore, FPGA, KNN search, profile switching", size=10, first_indent=False)

    body_sec = doc.add_section(WD_SECTION.CONTINUOUS)
    add_columns(body_sec, 2)
    for heading, paragraphs in SECTIONS:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(heading)
        set_run_font(r, 11, True)
        for para in paragraphs:
            add_para(doc, para)
        if heading == "2. PATCHCORE":
            doc.add_picture(str(FIG_PATCHCORE), width=Cm(7.8))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.add_run("Fig. 1 PatchCore training and inference flow.").font.size = Pt(8)
        if heading == "3. PROPOSED SYSTEM":
            doc.add_picture(str(FIG_SYSTEM), width=Cm(7.8))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.add_run("Fig. 2 Category-wise profile switching architecture.").font.size = Pt(8)
        if heading == "5. RESULTS":
            pass

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("REFERENCES")
    set_run_font(r, 11, True)
    for ref in REFERENCES:
        add_para(doc, ref, size=9, first_indent=False)

    doc.save(DOCX_OUT)


class TwoColumnDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=25 * mm,
            bottomMargin=25 * mm,
        )


def paragraph_style(name: str, size: float, leading: float, bold: bool = False, align: int = 0, first_indent: float = 0) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName="Times-Bold" if bold else "Times-Roman",
        fontSize=size,
        leading=leading,
        alignment=align,
        firstLineIndent=first_indent,
        spaceAfter=2.5,
    )


def result_table(rows: list[list[str]], font_name: str, bold_font_name: str, font_size: float = 7.1) -> Table:
    data = [["System", "Mean good", "Min good", "KNN cost"], *rows]
    table = Table(data, colWidths=[39 * mm, 13 * mm, 12 * mm, 14 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font_name, font_size),
                ("FONT", (0, 0), (-1, 0), bold_font_name, font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#7a7a7a")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    return table


def ab_result_table(font_name: str, bold_font_name: str, font_size: float = 7.1) -> Table:
    return result_table(
        [
            ["Common profile + common bank", "80.27%", "69.30%", "1.000x"],
            ["Category bank switch", "81.95%", "71.73%", "0.500x"],
            ["Single fixed profile (best)", "85.01%", "75.83%", "0.400x"],
            ["Proposed profile+bank switch", "85.53%", "76.48%", "0.378x"],
        ],
        font_name,
        bold_font_name,
        font_size,
    )


def abc_result_table(font_name: str, bold_font_name: str, font_size: float = 7.1) -> Table:
    return result_table(
        [
            ["Common profile + common bank", "79.27%", "60.43%", "1.000x"],
            ["Category bank switch", "81.95%", "65.44%", "0.333x"],
            ["Single fixed profile (best)", "84.49%", "68.77%", "0.271x"],
            ["Proposed profile+bank switch", "85.53%", "70.43%", "0.252x"],
        ],
        font_name,
        bold_font_name,
        font_size,
    )


def fpga_estimate_table(rows: list[list[str]], font_name: str, bold_font_name: str, font_size: float = 6.0) -> Table:
    table = Table(rows, colWidths=[31 * mm, 15 * mm, 12 * mm, 12 * mm, 9 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font_name, font_size),
                ("FONT", (0, 0), (-1, 0), bold_font_name, font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#7a7a7a")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.8),
                ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
            ]
        )
    )
    return table


def fpga_estimate_table_en(font_name: str, bold_font_name: str, font_size: float = 5.8) -> Table:
    return fpga_estimate_table(
        [
            ["System", "cycles/img", "ratio", "mem.", "BRAM"],
            ["Common profile+bank", "17.88M", "1.000x", "5.57MiB", "1267"],
            ["Bank switch", "7.06M", "0.395x", "5.57MiB", "1267"],
            ["Fixed profile+bank", "5.52M", "0.309x", "4.34MiB", "988"],
            ["Proposed switch", "5.23M", "0.293x", "4.11MiB", "936"],
        ],
        font_name,
        bold_font_name,
        font_size,
    )


def fpga_estimate_table_ja(font_name: str, bold_font_name: str, font_size: float = 5.8) -> Table:
    return fpga_estimate_table(
        [
            ["構成", "cycles/枚", "比", "容量", "BRAM"],
            ["共通profile+bank", "17.88M", "1.000x", "5.57MiB", "1267"],
            ["bank切替", "7.06M", "0.395x", "5.57MiB", "1267"],
            ["固定profile+bank", "5.52M", "0.309x", "4.34MiB", "988"],
            ["提案切替", "5.23M", "0.293x", "4.11MiB", "936"],
        ],
        font_name,
        bold_font_name,
        font_size,
    )


def equation_box(text: str, font_name: str, font_size: float = 6.5) -> Table:
    style = ParagraphStyle(
        "equation_box",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size + 1.3,
        alignment=1,
        spaceAfter=0,
    )
    table = Table([[Paragraph(text, style)]], colWidths=[78 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f7")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#666666")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def profile_table(rows: list[list[str]], font_name: str, bold_font_name: str, font_size: float = 5.1) -> Table:
    table = Table(rows, colWidths=[13 * mm, 8 * mm, 8 * mm, 12 * mm, 37 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font_name, font_size),
                ("FONT", (0, 0), (-1, 0), bold_font_name, font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#7a7a7a")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.2),
                ("TOPPADDING", (0, 0), (-1, -1), 1.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
            ]
        )
    )
    return table


def build_pdf() -> None:
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.platypus import Frame, FrameBreak, NextPageTemplate, PageTemplate

    doc = TwoColumnDoc(str(PDF_OUT))
    page_w, page_h = A4
    col_w = 80 * mm
    gap = 10 * mm
    left_x = 20 * mm
    right_x = left_x + col_w + gap
    bottom = 25 * mm
    top_margin = 25 * mm
    header_h = 62 * mm
    header_gap = 1 * mm
    body_top = page_h - top_margin - header_h - header_gap
    first_frames = [
        Frame(left_x, body_top + header_gap, 2 * col_w + gap, header_h, id="first_header"),
        Frame(left_x, bottom, col_w, body_top - bottom, id="first_left"),
        Frame(right_x, bottom, col_w, body_top - bottom, id="first_right"),
    ]
    body_h = page_h - top_margin - bottom
    two_col_frames = [
        Frame(left_x, bottom, col_w, body_h, id="left"),
        Frame(right_x, bottom, col_w, body_h, id="right"),
    ]
    doc.addPageTemplates([
        PageTemplate(id="FirstPage", frames=first_frames),
        PageTemplate(id="TwoCol", frames=two_col_frames),
    ])

    styles = {
        "title": paragraph_style("title", 12, 14, True, TA_CENTER),
        "author": paragraph_style("author", 11, 13, False, TA_CENTER),
        "body": paragraph_style("body", 8.25, 9.35, False, TA_LEFT, first_indent=4 * mm),
        "body_noindent": paragraph_style("body_noindent", 8.35, 9.4, False, TA_LEFT),
        "heading": paragraph_style("heading", 9.5, 10.6, True, TA_CENTER),
        "ref": paragraph_style("ref", 7.3, 8.2, False, TA_LEFT),
        "caption": paragraph_style("caption", 7.4, 8.4, False, TA_CENTER),
    }

    story = [
        Paragraph(TITLE_PDF, styles["title"]),
        Paragraph(AUTHORS, styles["author"]),
        Paragraph(AFFILIATION, styles["author"]),
        Paragraph(CONTACT, styles["author"]),
        Spacer(1, 1.5 * mm),
        Paragraph("<b>Abstract:</b> " + ABSTRACT, styles["body_noindent"]),
        Paragraph("<b>Keywords:</b> industrial anomaly detection, PatchCore, FPGA, KNN search, profile switching", styles["body_noindent"]),
        NextPageTemplate("TwoCol"),
        FrameBreak(),
    ]

    for heading, paragraphs in SECTIONS:
        if heading == "5. RESULTS":
            block = [Paragraph(heading, styles["heading"])]
            block.extend(Paragraph(para, styles["body"]) for para in paragraphs)
            block.append(ab_result_table("Times-Roman", "Times-Bold", 6.9))
            block.append(Paragraph("Table 2 Statistics for 105 AB tasks at <=3% defect false-pass rate.", styles["caption"]))
            block.append(abc_result_table("Times-Roman", "Times-Bold", 6.9))
            block.append(Paragraph("Table 3 Statistics for 455 ABC tasks at <=3% defect false-pass rate.", styles["caption"]))
            story.append(KeepTogether(block))
            continue
        if heading == "6. DISCUSSION":
            story.append(Paragraph(heading, styles["heading"]))
            for idx, para in enumerate(paragraphs):
                story.append(Paragraph(para, styles["body"]))
                if idx == 1:
                    story.append(KeepTogether([
                        Spacer(1, 0.6 * mm),
                        Image(str(FIG_EQUATION), width=72 * mm, height=12.6 * mm),
                        Paragraph(
                            "Eq. (1) Relative KNN search cost, where P is patch count, B is searched bank size, and D is feature dimension.",
                            styles["caption"],
                        ),
                    ]))
            continue
        if heading == "7. FPGA COST ESTIMATE":
            story.append(Paragraph(heading, styles["heading"]))
            for para in paragraphs:
                story.append(Paragraph(para, styles["body"]))
            story.append(KeepTogether([
                fpga_estimate_table_en("Times-Roman", "Times-Bold", 5.7),
                Paragraph("Table 4 FPGA-facing estimate at bank/category=3000, 8-bit features, and 64 distance lanes.", styles["caption"]),
            ]))
            continue
        story.append(Paragraph(heading, styles["heading"]))
        for para in paragraphs:
            story.append(Paragraph(para, styles["body"]))
        if heading == "2. PATCHCORE":
            story.append(KeepTogether([
                Image(str(FIG_PATCHCORE), width=64 * mm, height=24 * mm),
                Paragraph("Fig. 1 PatchCore training and inference flow.", styles["caption"]),
            ]))
        if heading == "3. PROPOSED SYSTEM":
            story.append(Image(str(FIG_SYSTEM), width=78 * mm, height=32 * mm))
            story.append(Paragraph("Fig. 2 Category-wise profile switching architecture.", styles["caption"]))
        if heading == "4. EXPERIMENTAL SETUP":
            story.append(KeepTogether([
                profile_table(PROFILE_ROWS_EN, "Times-Roman", "Times-Bold", 5.1),
                Paragraph("Table 1 Category-specific profiles selected by single-category sweep.", styles["caption"]),
            ]))
    story.append(Paragraph("REFERENCES", styles["heading"]))
    for ref in REFERENCES:
        story.append(Paragraph(ref, styles["ref"]))

    doc.build(story)


def safe_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_tex_sources() -> None:
    en_lines = [
        r"\documentclass[a4paper,twocolumn,10pt]{article}",
        r"\usepackage[margin=25mm,left=20mm,right=20mm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{times}",
        r"\pagestyle{empty}",
        r"\setlength{\columnsep}{10mm}",
        r"\title{\textbf{" + tex_escape(TITLE) + "}}",
        r"\author{" + tex_escape(AUTHORS) + r"\\ " + tex_escape(AFFILIATION) + r"\\ " + tex_escape(CONTACT) + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_escape(ABSTRACT),
        r"\end{abstract}",
        r"\noindent\textbf{Keywords:} industrial anomaly detection, PatchCore, FPGA, KNN search, profile switching",
    ]
    for heading, paragraphs in SECTIONS:
        en_lines.append(r"\section*{" + tex_escape(heading) + "}")
        for para in paragraphs:
            en_lines.append(tex_escape(para) + "\n")
        if heading == "2. PATCHCORE":
            en_lines.extend([
                r"\begin{figure}[h]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/patchcore_process.png}",
                r"\caption{PatchCore training and inference flow.}",
                r"\end{figure}",
            ])
        if heading == "3. PROPOSED SYSTEM":
            en_lines.extend([
                r"\begin{figure}[h]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/system_architecture.png}",
                r"\caption{Category-wise profile switching architecture.}",
                r"\end{figure}",
            ])
        if heading == "4. EXPERIMENTAL SETUP":
            en_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{Category-specific profiles selected by single-category sweep.}",
                r"\begin{tabular}{lcccl}\toprule Layer & Grid & D & top-k & Category\\\midrule",
                r"layer3 & 14 & 512 & 0.005 & bottle, carpet, hazelnut, leather\\",
                r"layer3 & 14 & 512 & 0.005 & metal\_nut, tile, zipper\\",
                r"layer3 & 14 & 512 & 0.010 & screw, transistor\\",
                r"layer3 & 14 & 512 & 0.020 & capsule\\",
                r"layer3 & 14 & 512 & 0.050 & cable\\",
                r"layer2+3 & 14 & 768 & 0.005 & grid\\",
                r"layer2+3 & 14 & 768 & 0.010 & pill, wood\\",
                r"layer2+3 & 14 & 768 & 0.050 & toothbrush\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
        if heading == "5. RESULTS":
            en_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{Statistics for 105 AB tasks at <=3\% defect false-pass rate.}",
                r"\begin{tabular}{lccc}\toprule System & Mean good & Min good & KNN cost\\\midrule",
                r"Common profile + common bank & 80.27\% & 69.30\% & 1.000x\\",
                r"Category bank switch & 81.95\% & 71.73\% & 0.500x\\",
                r"Single fixed profile (best) & 85.01\% & 75.83\% & 0.400x\\",
                r"Proposed profile+bank switch & 85.53\% & 76.48\% & 0.378x\\",
                r"\bottomrule\end{tabular}\end{table}",
                r"\begin{table}[h]\centering\small",
                r"\caption{Statistics for 455 ABC tasks at <=3\% defect false-pass rate.}",
                r"\begin{tabular}{lccc}\toprule System & Mean good & Min good & KNN cost\\\midrule",
                r"Common profile + common bank & 79.27\% & 60.43\% & 1.000x\\",
                r"Category bank switch & 81.95\% & 65.44\% & 0.333x\\",
                r"Single fixed profile (best) & 84.49\% & 68.77\% & 0.271x\\",
                r"Proposed profile+bank switch & 85.53\% & 70.43\% & 0.252x\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
        if heading == "6. DISCUSSION":
            en_lines.extend([
                r"\begin{equation}",
                r"C_m(S)=\sum_{c\in S}p_c P_{m,c}B_{m,c}D_{m,c},\quad R_m=\frac{C_m(S)}{C_{\rm common}(S)}",
                r"\end{equation}",
                r"\noindent where $P$, $B$, and $D$ denote patch count, searched bank size, and feature dimension.",
            ])
        if heading == "7. FPGA COST ESTIMATE":
            en_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{FPGA-facing estimate at bank/category=3000, 8-bit features, and 64 distance lanes.}",
                r"\begin{tabular}{lcccc}\toprule System & cycles/img & ratio & mem. & BRAM\\\midrule",
                r"Common profile+bank & 17.88M & 1.000x & 5.57MiB & 1267\\",
                r"Bank switch & 7.06M & 0.395x & 5.57MiB & 1267\\",
                r"Fixed profile+bank & 5.52M & 0.309x & 4.34MiB & 988\\",
                r"Proposed switch & 5.23M & 0.293x & 4.11MiB & 936\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
    en_lines.append(r"\section*{REFERENCES}")
    for ref in REFERENCES:
        en_lines.append(r"\noindent " + tex_escape(ref) + r"\\")
    en_lines.append(r"\end{document}")
    TEX_EN_OUT.write_text("\n".join(en_lines), encoding="utf-8")

    ja_lines = [
        r"\documentclass[a4paper,twocolumn,10pt]{ltjsarticle}",
        r"\usepackage[margin=25mm,left=20mm,right=20mm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\pagestyle{empty}",
        r"\setlength{\columnsep}{10mm}",
        r"\title{\textbf{" + tex_escape(JA_TITLE) + "}}",
        r"\author{" + tex_escape(JA_AUTHORS) + r"\\ " + tex_escape(JA_AFFILIATION) + r"\\ " + tex_escape(JA_CONTACT) + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_escape(JA_ABSTRACT),
        r"\end{abstract}",
        r"\noindent\textbf{キーワード:} 産業用異常検知，PatchCore，FPGA，近傍探索，プロファイル切り替え",
    ]
    for heading, paragraphs in JA_SECTIONS:
        ja_lines.append(r"\section*{" + tex_escape(heading) + "}")
        for para in paragraphs:
            ja_lines.append(tex_escape(para) + "\n")
        if heading == "2. PatchCore":
            ja_lines.extend([
                r"\begin{figure}[h]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/patchcore_process.png}",
                r"\caption{PatchCoreの学習時・推論時の処理流れ}",
                r"\end{figure}",
            ])
        if heading == "3. 提案手法":
            ja_lines.extend([
                r"\begin{figure}[h]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/system_architecture.png}",
                r"\caption{カテゴリ別profile切り替え構成}",
                r"\end{figure}",
            ])
        if heading == "4. 実験条件":
            ja_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{単一カテゴリ探索で選択したカテゴリ専用profile}",
                r"\begin{tabular}{lcccl}\toprule 層 & grid & D & top-k & カテゴリ\\\midrule",
                r"layer3 & 14 & 512 & 0.005 & bottle, carpet, hazelnut, leather\\",
                r"layer3 & 14 & 512 & 0.005 & metal\_nut, tile, zipper\\",
                r"layer3 & 14 & 512 & 0.010 & screw, transistor\\",
                r"layer3 & 14 & 512 & 0.020 & capsule\\",
                r"layer3 & 14 & 512 & 0.050 & cable\\",
                r"layer2+3 & 14 & 768 & 0.005 & grid\\",
                r"layer2+3 & 14 & 768 & 0.010 & pill, wood\\",
                r"layer2+3 & 14 & 768 & 0.050 & toothbrush\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
        if heading == "5. 実験結果":
            ja_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{欠陥誤通過率3\%以下での2カテゴリ検品105通りの統計}",
                r"\begin{tabular}{lccc}\toprule 構成 & 平均良品 & 最低良品 & NN探索量\\\midrule",
                r"共通profile + 共通bank & 80.27\% & 69.30\% & 1.000x\\",
                r"カテゴリ別bank切替 & 81.95\% & 71.73\% & 0.500x\\",
                r"単一固定profile(best) & 85.01\% & 75.83\% & 0.400x\\",
                r"提案: profile+bank切替 & 85.53\% & 76.48\% & 0.378x\\",
                r"\bottomrule\end{tabular}\end{table}",
                r"\begin{table}[h]\centering\small",
                r"\caption{欠陥誤通過率3\%以下での3カテゴリ検品455通りの統計}",
                r"\begin{tabular}{lccc}\toprule 構成 & 平均良品 & 最低良品 & NN探索量\\\midrule",
                r"共通profile + 共通bank & 79.27\% & 60.43\% & 1.000x\\",
                r"カテゴリ別bank切替 & 81.95\% & 65.44\% & 0.333x\\",
                r"単一固定profile(best) & 84.49\% & 68.77\% & 0.271x\\",
                r"提案: profile+bank切替 & 85.53\% & 70.43\% & 0.252x\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
        if heading == "6. 考察":
            ja_lines.extend([
                r"\begin{equation}",
                r"C_m(S)=\sum_{c\in S}p_c P_{m,c}B_{m,c}D_{m,c},\quad R_m=\frac{C_m(S)}{C_{\rm common}(S)}",
                r"\end{equation}",
                r"\noindent $P$はパッチ数，$B$は探索bank数，$D$は特徴次元を表す。",
            ])
        if heading == "7. FPGA実装に向けた見積もり":
            ja_lines.extend([
                r"\begin{table}[h]\centering\small",
                r"\caption{bank/category=3000，8bit特徴量，64並列距離計算レーンでのFPGA向け見積もり}",
                r"\begin{tabular}{lcccc}\toprule 構成 & cycles/枚 & 比 & 容量 & BRAM\\\midrule",
                r"共通profile+bank & 17.88M & 1.000x & 5.57MiB & 1267\\",
                r"bank切替 & 7.06M & 0.395x & 5.57MiB & 1267\\",
                r"固定profile+bank & 5.52M & 0.309x & 4.34MiB & 988\\",
                r"提案切替 & 5.23M & 0.293x & 4.11MiB & 936\\",
                r"\bottomrule\end{tabular}\end{table}",
            ])
    ja_lines.append(r"\section*{参考文献}")
    for ref in JA_REFERENCES:
        ja_lines.append(r"\noindent " + tex_escape(ref) + r"\\")
    ja_lines.append(r"\end{document}")
    TEX_JA_OUT.write_text("\n".join(ja_lines), encoding="utf-8")


def build_japanese_pdf() -> None:
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import Frame, FrameBreak, NextPageTemplate, PageTemplate

    pdfmetrics.registerFont(TTFont("Meiryo", r"C:\Windows\Fonts\meiryo.ttc"))
    pdfmetrics.registerFont(TTFont("Meiryo-Bold", r"C:\Windows\Fonts\meiryob.ttc"))

    doc = TwoColumnDoc(str(PDF_JA_OUT))
    page_w, page_h = A4
    col_w = 80 * mm
    gap = 10 * mm
    left_x = 20 * mm
    right_x = left_x + col_w + gap
    bottom = 25 * mm
    top_margin = 25 * mm
    header_h = 56 * mm
    header_gap = 5 * mm
    body_top = page_h - top_margin - header_h - header_gap
    first_frames = [
        Frame(left_x, body_top + header_gap, 2 * col_w + gap, header_h, id="first_header"),
        Frame(left_x, bottom, col_w, body_top - bottom, id="first_left"),
        Frame(right_x, bottom, col_w, body_top - bottom, id="first_right"),
    ]
    body_h = page_h - top_margin - bottom
    two_col_frames = [
        Frame(left_x, bottom, col_w, body_h, id="left"),
        Frame(right_x, bottom, col_w, body_h, id="right"),
    ]
    doc.addPageTemplates([
        PageTemplate(id="FirstPage", frames=first_frames),
        PageTemplate(id="TwoCol", frames=two_col_frames),
    ])

    def ja_style(name: str, size: float, leading: float, bold: bool = False, align: int = TA_LEFT, first_indent: float = 0) -> ParagraphStyle:
        return ParagraphStyle(
            name,
            fontName="Meiryo-Bold" if bold else "Meiryo",
            fontSize=size,
            leading=leading,
            alignment=align,
            firstLineIndent=first_indent,
            spaceAfter=2.7,
            wordWrap="CJK",
        )

    styles = {
        "title": ja_style("ja_title", 11.2, 13.2, True, TA_CENTER),
        "author": ja_style("ja_author", 9.6, 11.2, False, TA_CENTER),
        "body": ja_style("ja_body", 7.9, 9.4, False, TA_LEFT, first_indent=4 * mm),
        "body_noindent": ja_style("ja_body_noindent", 7.9, 9.4, False, TA_LEFT),
        "heading": ja_style("ja_heading", 9.2, 10.8, True, TA_CENTER),
        "caption": ja_style("ja_caption", 7.1, 8.2, False, TA_CENTER),
        "ref": ja_style("ja_ref", 7.0, 8.1, False, TA_LEFT),
    }

    story = [
        Paragraph(safe_xml(JA_TITLE), styles["title"]),
        Paragraph(safe_xml(JA_AUTHORS), styles["author"]),
        Paragraph(safe_xml(JA_AFFILIATION), styles["author"]),
        Paragraph(safe_xml(JA_CONTACT), styles["author"]),
        Spacer(1, 1.5 * mm),
        Paragraph("<b>概要：</b>" + safe_xml(JA_ABSTRACT), styles["body_noindent"]),
        Paragraph("<b>キーワード：</b>産業用異常検知，PatchCore，FPGA，近傍探索，プロファイル切り替え", styles["body_noindent"]),
        NextPageTemplate("TwoCol"),
        FrameBreak(),
    ]

    for heading, paragraphs in JA_SECTIONS:
        if heading == "5. 実験結果":
            table_ab = Table(
                [
                    ["構成", "平均良品", "最低良品", "NN探索量"],
                    ["共通profile + 共通bank", "80.27%", "69.30%", "1.000x"],
                    ["カテゴリ別bank切替", "81.95%", "71.73%", "0.500x"],
                    ["単一固定profile(best)", "85.01%", "75.83%", "0.400x"],
                    ["提案: profile+bank切替", "85.53%", "76.48%", "0.378x"],
                ],
                colWidths=[41 * mm, 13 * mm, 12 * mm, 12 * mm],
            )
            table_abc = Table(
                [
                    ["構成", "平均良品", "最低良品", "NN探索量"],
                    ["共通profile + 共通bank", "79.27%", "60.43%", "1.000x"],
                    ["カテゴリ別bank切替", "81.95%", "65.44%", "0.333x"],
                    ["単一固定profile(best)", "84.49%", "68.77%", "0.271x"],
                    ["提案: profile+bank切替", "85.53%", "70.43%", "0.252x"],
                ],
                colWidths=[41 * mm, 13 * mm, 12 * mm, 12 * mm],
            )
            for table in (table_ab, table_abc):
                table.setStyle(
                    TableStyle(
                        [
                            ("FONT", (0, 0), (-1, -1), "Meiryo", 6.0),
                            ("FONT", (0, 0), (-1, 0), "Meiryo-Bold", 6.0),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#7a7a7a")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                        ]
                    )
                )
            block = [Paragraph(safe_xml(heading), styles["heading"])]
            block.extend(Paragraph(safe_xml(para), styles["body"]) for para in paragraphs)
            block.append(table_ab)
            block.append(Paragraph("表2 欠陥誤通過率3%以下での2カテゴリ検品105通りの統計", styles["caption"]))
            block.append(table_abc)
            block.append(Paragraph("表3 欠陥誤通過率3%以下での3カテゴリ検品455通りの統計", styles["caption"]))
            story.append(KeepTogether(block))
            continue
        if heading == "6. 考察":
            story.append(Paragraph(safe_xml(heading), styles["heading"]))
            for idx, para in enumerate(paragraphs):
                story.append(Paragraph(safe_xml(para), styles["body"]))
                if idx == 1:
                    story.append(KeepTogether([
                        Spacer(1, 0.6 * mm),
                        Image(str(FIG_EQUATION), width=72 * mm, height=12.6 * mm),
                        Paragraph(
                            "式(1) 相対KNN探索量。Pはパッチ数，Bは探索bank数，Dは特徴次元を表す。",
                            styles["caption"],
                        ),
                    ]))
            continue
        if heading == "7. FPGA実装に向けた見積もり":
            block = [Paragraph(safe_xml(heading), styles["heading"])]
            block.extend(Paragraph(safe_xml(para), styles["body"]) for para in paragraphs)
            block.append(fpga_estimate_table_ja("Meiryo", "Meiryo-Bold", 5.6))
            block.append(Paragraph("表4 bank/category=3000，8bit特徴量，64並列距離計算レーンでのFPGA向け見積もり", styles["caption"]))
            story.append(KeepTogether(block))
            continue
        if heading == "8. まとめ":
            block = [Paragraph(safe_xml(heading), styles["heading"])]
            block.extend(Paragraph(safe_xml(para), styles["body"]) for para in paragraphs)
            story.append(KeepTogether(block))
            continue
        story.append(Paragraph(safe_xml(heading), styles["heading"]))
        for para in paragraphs:
            story.append(Paragraph(safe_xml(para), styles["body"]))
        if heading == "2. PatchCore":
            story.append(Image(str(FIG_PATCHCORE), width=78 * mm, height=30 * mm))
            story.append(Paragraph("図1 PatchCoreの学習時・推論時の処理流れ", styles["caption"]))
        if heading == "3. 提案手法":
            story.append(Image(str(FIG_SYSTEM), width=78 * mm, height=32 * mm))
            story.append(Paragraph("図2 カテゴリ別profile切り替え構成", styles["caption"]))
        if heading == "4. 実験条件":
            story.append(KeepTogether([
                profile_table(PROFILE_ROWS_JA, "Meiryo", "Meiryo-Bold", 4.9),
                Paragraph("表1 単一カテゴリ探索で選択したカテゴリ専用profile", styles["caption"]),
            ]))
    story.append(Paragraph("参考文献", styles["heading"]))
    for ref in JA_REFERENCES:
        story.append(Paragraph(safe_xml(ref), styles["ref"]))

    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_figures()
    build_docx()
    build_pdf()
    if PDF_EN_OUT != PDF_OUT:
        PDF_EN_OUT.write_bytes(PDF_OUT.read_bytes())
    build_japanese_pdf()
    write_tex_sources()
    pages_en = len(PdfReader(str(PDF_OUT)).pages)
    pages_ja = len(PdfReader(str(PDF_JA_OUT)).pages)
    if pages_en != 2:
        raise RuntimeError(f"Expected a 2-page English PDF, got {pages_en} pages: {PDF_OUT}")
    if pages_ja != 2:
        raise RuntimeError(f"Expected a 2-page Japanese PDF, got {pages_ja} pages: {PDF_JA_OUT}")
    print(json.dumps({
        "docx": str(DOCX_OUT),
        "pdf": str(PDF_OUT),
        "pdf_en": str(PDF_EN_OUT),
        "pdf_ja": str(PDF_JA_OUT),
        "tex_en": str(TEX_EN_OUT),
        "tex_ja": str(TEX_JA_OUT),
        "pages_en": pages_en,
        "pages_ja": pages_ja,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
