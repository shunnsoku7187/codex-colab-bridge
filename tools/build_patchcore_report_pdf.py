from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "patchcore_fpga_professor_report.pdf"
FIG = ROOT / "docs" / "figures"


def setup_fonts():
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ja-title",
            parent=base["Title"],
            fontName="HeiseiKakuGo-W5",
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=7 * mm,
        ),
        "meta": ParagraphStyle(
            "ja-meta",
            parent=base["Normal"],
            fontName="HeiseiMin-W3",
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),
        "h1": ParagraphStyle(
            "ja-h1",
            parent=base["Heading1"],
            fontName="HeiseiKakuGo-W5",
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#1f3b64"),
            spaceBefore=7 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "ja-h2",
            parent=base["Heading2"],
            fontName="HeiseiKakuGo-W5",
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#2d4f78"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "ja-body",
            parent=base["BodyText"],
            fontName="HeiseiMin-W3",
            fontSize=9.2,
            leading=14.0,
            alignment=TA_LEFT,
            spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "ja-small",
            parent=base["BodyText"],
            fontName="HeiseiMin-W3",
            fontSize=8,
            leading=11,
            spaceAfter=1.5 * mm,
        ),
        "caption": ParagraphStyle(
            "ja-caption",
            parent=base["BodyText"],
            fontName="HeiseiMin-W3",
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4 * mm,
        ),
        "quote": ParagraphStyle(
            "ja-quote",
            parent=base["BodyText"],
            fontName="HeiseiKakuGo-W5",
            fontSize=9,
            leading=14,
            leftIndent=8 * mm,
            rightIndent=8 * mm,
            borderColor=colors.HexColor("#b8c7d9"),
            borderWidth=0.8,
            borderPadding=5,
            backColor=colors.HexColor("#f5f8fb"),
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
    }


def p(text, sty):
    return Paragraph(text, sty)


def bullets(items, sty):
    return ListFlowable(
        [ListItem(Paragraph(item, sty), leftIndent=4 * mm) for item in items],
        bulletType="bullet",
        leftIndent=6 * mm,
        bulletFontName="HeiseiMin-W3",
        bulletFontSize=8,
    )


def number_list(items, sty):
    return ListFlowable(
        [ListItem(Paragraph(item, sty), leftIndent=4 * mm) for item in items],
        bulletType="1",
        leftIndent=7 * mm,
        bulletFontName="HeiseiMin-W3",
        bulletFontSize=8,
    )


def table(data, col_widths, font_size=7.1, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5df")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "HeiseiKakuGo-W5"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
        ]
    t.setStyle(TableStyle(style))
    return t


def fig(path, caption, width_mm=155):
    img = Image(str(FIG / path))
    img._restrictSize(width_mm * mm, 105 * mm)
    return KeepTogether([img, p(caption, S["caption"])])


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("HeiseiMin-W3", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(198 * mm, 10 * mm, f"{doc.page}")
    canvas.drawString(18 * mm, 10 * mm, "カテゴリ別プロファイル型PatchCore-lite FPGA説明資料")
    canvas.restoreState()


def build_story():
    story = []
    story += [
        p("カテゴリ別プロファイル型PatchCore-liteのFPGA実装に向けた説明資料", S["title"]),
        p("山田 俊介<br/>2026年8月24日", S["meta"]),
        p("概要", S["h1"]),
        p(
            "本資料では、検品向け異常検知手法であるPatchCoreを題材に、"
            "単にPatchCoreをFPGAに載せるのではなく、検品対象カテゴリごとに必要な"
            "バックボーンCNN、中間特徴層、パッチグリッド、メモリバンク、KNN探索量を"
            "事前に決定し、その構成をFPGA上で切り替える方式を提案する。",
            S["body"],
        ),
        p(
            "現在までの実験から、PatchCoreの重い部分であるKNN探索は平均で基準構成の"
            "0.0109倍、中央値で0.0035倍まで削減できる見込みが得られた。"
            "一方、15カテゴリすべてに対して厳しい検品品質を同時に満たすことは現時点では"
            "難しく、初回実装はhazelnutなど成立しやすいカテゴリに絞るのが妥当である。",
            S["body"],
        ),
        p("PatchCoreとは", S["h1"]),
        p("検品タスクにおける位置づけ", S["h2"]),
        p(
            "PatchCoreは、工業製品の外観検査で用いられる異常検知手法である。"
            "通常の画像分類モデルは良品画像と欠陥画像の両方を用いて学習する。"
            "一方、PatchCoreは正常画像のみを用いて「正常らしさ」を記憶し、"
            "検査画像が正常分布からどれだけ離れているかを測る。",
            S["body"],
        ),
        p(
            "MVTec ADは15種類の工業検品カテゴリを含む代表的な異常検知ベンチマークであり、"
            "正常学習画像と正常/異常のテスト画像を持つ。",
            S["body"],
        ),
        fig("patchcore_flow.png", "図1 PatchCoreの基本動作。重い部分はメモリバンク容量とKNN探索である。"),
        p("基本動作", S["h2"]),
        number_list(
            [
                "<b>準備段階:</b> 正常画像をCNN特徴抽出器に通し、画像内の局所特徴をパッチ特徴として取り出す。その正常パッチ特徴をメモリバンクに保存する。",
                "<b>推論段階:</b> 検査画像から同様にパッチ特徴を取り出し、各パッチ特徴に対してメモリバンク内の最近傍特徴をKNN探索で探す。近い正常特徴が見つからないほど異常スコアが高くなる。",
            ],
            S["body"],
        ),
        p(
            "<b>パッチ特徴</b>: 画像内の局所領域ごとに取り出した特徴。局所的な傷や汚れを検出しやすい。<br/>"
            "<b>メモリバンク</b>: 正常画像から得られたパッチ特徴の集合。<br/>"
            "<b>KNN探索</b>: 検査画像の各パッチ特徴について、メモリバンク内の近い特徴を探す処理。<br/>"
            "<b>異常スコア</b>: 検査画像のパッチ特徴と正常メモリバンクの距離から計算される値。",
            S["body"],
        ),
        p("強みと弱み", S["h2"]),
        p(
            "PatchCoreの強みは、欠陥画像を大量に用意しなくても検品タスクを構成しやすい点である。"
            "弱みは、メモリバンクを大きくするとメモリ容量とKNN探索量が増える点である。",
            S["body"],
        ),
        p("KNN探索の概算演算量: C_KNN ∝ N_patch × N_bank × D", S["quote"]),
    ]

    story += [
        PageBreak(),
        p("カテゴリを絞れば削減できる根拠", S["h1"]),
        p("実験の目的", S["h2"]),
        p(
            "MVTec ADの15カテゴリを対象に、PatchCoreの構成をどこまで削減できるかを調べた。"
            "単に平均AUROCを見るのではなく、欠陥誤通過、良品通過、KNN演算量、メモリバンク量を"
            "分けて評価した。",
            S["body"],
        ),
        bullets(
            [
                "<b>欠陥誤通過</b>: 欠陥品を良品として通してしまう割合。",
                "<b>良品通過</b>: 良品を良品として通せる割合。",
                "<b>KNN演算量</b>: メモリバンク探索に必要な相対演算量。",
                "<b>メモリバンク量</b>: FPGA上で保持すべき特徴量の相対サイズ。",
            ],
            S["body"],
        ),
        p("全カテゴリ共通設定との比較", S["h2"]),
        table(
            [
                ["設定", "良品通過", "KNN演算", "読み取り"],
                ["全カテゴリ共通: 基準構成", "59.34%", "1.0000倍", "最も重い参照点"],
                ["全カテゴリ共通: KNN 20%以下", "57.19%", "0.1667倍", "共通軽量化で品質を残せる限界の目安"],
                ["全カテゴリ共通: KNN 10%以下", "44.88%", "0.0834倍", "さらに軽くすると良品通過が大きく下がる"],
                ["カテゴリ別選択設定、holdout平均", "54.59%", "0.0206倍", "KNNは非常に小さいが、品質はカテゴリ別に扱う必要がある"],
            ],
            [42 * mm, 22 * mm, 23 * mm, 73 * mm],
        ),
        Spacer(1, 3 * mm),
        p(
            "カテゴリ別プロファイルにより、KNN探索コストは平均0.0109倍、中央値0.0035倍となった。"
            "CNN特徴抽出とKNN探索を合わせた総コスト近似でも、平均0.2687倍、中央値0.2035倍であった。",
            S["body"],
        ),
        fig("profiled_reduction_bars.png", "図2 カテゴリ別プロファイルによる削減効果。基準構成を1.0倍とした相対値。"),
        p(
            "全カテゴリ分のメモリバンクを単純に基準構成で保持すると131.660 MiBが必要である。"
            "一方、カテゴリ別に選択したバンクだけを保持する場合は3.288 MiBとなり、"
            "常駐メモリは約97.50%削減できる。",
            S["body"],
        ),
        p("品質面の限界", S["h2"]),
        p(
            "選択済みプロファイル設定が欠陥誤通過3%以下を満たしたのは5/15カテゴリ、"
            "5%以下では9/15カテゴリであった。したがって、現時点で主張すべきことは"
            "「全カテゴリに万能に安全な軽量PatchCoreを得た」ではない。",
            S["body"],
        ),
        p(
            "検品対象カテゴリが事前に分かる場合、そのカテゴリに必要な特徴層・バンク量・探索量だけを選ぶことで、FPGA実装に向いた小さい構成を作れる。",
            S["quote"],
        ),
        table(
            [
                ["カテゴリ", "選択設定", "良品通過", "欠陥誤通過", "総コスト近似", "KNN演算"],
                ["hazelnut", "res18_l23_g14_b500", "92.00%", "1.14%", "0.1008倍", "0.0104倍"],
                ["zipper", "wrn_l3_g14_b1500", "75.00%", "2.71%", "0.5220倍", "0.0833倍"],
                ["wood", "res18_l23_g7_b125", "71.11%", "2.00%", "0.0957倍", "0.0007倍"],
                ["cable", "res18_l23_g7_b1500", "33.79%", "2.61%", "0.0995倍", "0.0078倍"],
            ],
            [22 * mm, 44 * mm, 22 * mm, 24 * mm, 26 * mm, 22 * mm],
        ),
    ]

    story += [
        PageBreak(),
        p("PatchCoreのどこを削るのか", S["h1"]),
        p(
            "本研究で削減対象にしているのは、メモリバンクだけではない。PatchCoreの処理順に沿って見ると、"
            "削減対象はバックボーンCNN、中間特徴層、パッチグリッド、メモリバンク、異常スコアと閾値に整理できる。",
            S["body"],
        ),
        number_list(
            [
                "<b>バックボーンCNN</b>: 入力画像から特徴量を抽出する計算本体。WideResNet-50-2からResNet34、ResNet18、MobileNetV3、EfficientNet-B0への置換を調べた。",
                "<b>中間特徴層</b>: 浅い層はエッジや局所模様に近い情報を持ち、深い層は部品形状や構造に近い情報を持つ。複数層を使うと情報量は増えるが、特徴次元数とKNN探索量も増える。",
                "<b>パッチグリッド</b>: 画像をどれだけ細かい位置単位で見るか。14x14では196個、7x7なら49個になり、KNN探索量は約4分の1になる。",
                "<b>メモリバンク</b>: 正常画像から保存する正常パッチ特徴の集合。小さくするとメモリ量と探索量は減るが、正常分布を覆えなくなる危険がある。",
                "<b>異常スコアと閾値</b>: パッチ距離から画像全体の異常スコアを作り、欠陥誤通過率を満たす閾値を選ぶ。",
            ],
            S["body"],
        ),
        p(
            "最近傍探索コストは C_NN = P × B × D で近似できる。相対コストは "
            "C'_NN / C_NN = (P'/P) × (B'/B) × (D'/D) で表せる。"
            "この式から、メモリバンク削減だけでなく、パッチグリッド削減と特徴次元削減もKNN探索量に直接効く。",
            S["quote"],
        ),
        p("普通のPatchCoreとの比較", S["h2"]),
        table(
            [
                ["比較対象", "構成の考え方", "本研究との差分"],
                ["通常のPatchCore", "WideResNet-50-2、layer2+layer3、大きめバンクを基準にする", "高性能だが、FPGAではCNN計算、特徴次元、バンク量、KNN探索が重い"],
                ["バンク削減のみ", "バックボーンと特徴層は固定し、正常パッチ数だけを減らす", "メモリとKNNは減るが、特徴抽出本体や特徴次元は残る"],
                ["一律軽量PatchCore", "全カテゴリに同じ軽量バックボーンやgridを使う", "削れるカテゴリには過剰、削れないカテゴリには不足する"],
                ["提案方向", "カテゴリごとにバックボーン、特徴層、grid、bank、閾値を選ぶ", "削れるカテゴリだけ大胆に削り、必要なカテゴリには強い構成を残せる"],
            ],
            [33 * mm, 57 * mm, 70 * mm],
        ),
        p("カテゴリ別切り替えが必要な理由", S["h1"]),
        p(
            "MVTec AD全15カテゴリに対して、WideResNet-50-2、ResNet34、ResNet18、MobileNetV3、EfficientNet-B0を含む"
            "22種類の特徴抽出プロファイルを評価した。欠陥誤通過3%以下、基準構成からの良品通過低下2ポイント以内という条件で、"
            "選ばれた最小構成のバックボーンはカテゴリごとに分散した。",
            S["body"],
        ),
        table(
            [
                ["バックボーン", "採用カテゴリ数"],
                ["EfficientNet-B0", "5"],
                ["ResNet34", "4"],
                ["ResNet18", "2"],
                ["WideResNet-50-2", "2"],
                ["MobileNetV3 Large", "1"],
                ["MobileNetV3 Small", "1"],
            ],
            [80 * mm, 35 * mm],
        ),
        p(
            "この切り替えは、画像1枚ごとに異なる経路を選ぶ動的ルーティングではない。"
            "検品動作の前にカテゴリモードを選び、対応するバックボーン設定、特徴層、grid、メモリバンク、閾値を有効化する。",
            S["body"],
        ),
    ]

    story += [
        PageBreak(),
        p("兼用検品システムとしての比較", S["h1"]),
        p(
            "本研究の比較対象は単一カテゴリ専用PatchCoreではなく、複数カテゴリに対応する兼用検品システムである。"
            "カテゴリiの出現頻度をp_i、そのカテゴリ用プロファイルの計算量をC_iとすると、"
            "提案する切替型システムの期待計算量は E[C_switch] = Σ p_i C_i である。",
            S["body"],
        ),
        p(
            "全カテゴリで同じ共通構成を使う兼用機では、毎回同じ計算量C_commonを支払う。"
            "したがって、提案方式が有利になる条件は Σ p_i C_i < C_common で表せる。",
            S["quote"],
        ),
        table(
            [
                ["システム", "平均良品通過", "最低良品通過", "平均NN計算", "最大NN計算", "読み取り"],
                ["標準共通PatchCore", "74.11%", "31.71%", "1.000000倍", "1.000000倍", "全カテゴリで標準構成を使うため重い"],
                ["品質維持の共通軽量", "81.94%", "31.71%", "0.666667倍", "0.666667倍", "共通構成のままでは十分には軽くならない"],
                ["10%制約の共通軽量", "68.41%", "2.44%", "0.083446倍", "0.085034倍", "軽くできるが一部カテゴリで品質が崩れる"],
                ["bankだけ切替", "76.28%", "31.71%", "0.234198倍", "1.000000倍", "メモリだけでは重いカテゴリが残る"],
                ["提案: full profile切替", "76.34%", "43.48%", "0.031596倍", "0.187500倍", "bank、grid、特徴次元を同時に削れる"],
            ],
            [35 * mm, 23 * mm, 23 * mm, 24 * mm, 24 * mm, 37 * mm],
            font_size=6.4,
        ),
        Spacer(1, 3 * mm),
        fig("mvtec_patchcore_multicategory_system_compare_001.png", "図3 兼用検品システムとして見た場合の平均KNN計算量と平均良品通過率。"),
        p(
            "提案するfull profile切替は、標準共通PatchCoreに対して平均NN計算量を0.031596倍まで下げる。"
            "これは約96.84%の削減である。bankだけ切替と比べても、平均良品通過はほぼ同等である一方、"
            "平均NN計算量は0.234198倍から0.031596倍へ下がる。",
            S["body"],
        ),
        p(
            "本研究の提案点は「1カテゴリ専用にチューニングできる」ことではなく、"
            "カテゴリ専用チューニング済みの複数プロファイルを1つのFPGA上で切り替え、"
            "兼用機でありながら専用機に近い計算効率を得る点である。",
            S["quote"],
        ),
    ]

    story += [
        PageBreak(),
        p("削減可否を説明する仮説とデータ", S["h1"]),
        p("仮説", S["h2"]),
        p(
            "PatchCore-lite化が成立するカテゴリでは、削減後の特徴空間でも正常画像の異常スコア上位側と"
            "欠陥画像の異常スコア下位側に分離余裕が残る。一方、成立しないカテゴリでは、削減によって"
            "正常/異常スコア分布が重なり、閾値分離の余裕が失われる。",
            S["quote"],
        ),
        p(
            "欠陥誤通過率αを満たすしきい値を τ_α = Q_α(S_defect) と定義する。"
            "良品95%を通すために必要なスコア境界を Q_0.95(S_good) とし、分離余裕を "
            "M95 = τ_α - Q_0.95(S_good) と定義する。M95 > 0なら欠陥誤通過をα以下に抑えながら、"
            "良品95%を通す余裕が残る。",
            S["body"],
        ),
        table(
            [
                ["カテゴリ", "削減構成", "良品通過", "C_NN比", "M95", "判断"],
                ["bottle", "effb0_g10_b500", "95.0%", "0.001772", "0.0199", "削れる根拠あり"],
                ["hazelnut", "res34_g14_b500", "97.5%", "0.010417", "0.0036", "削れる根拠あり"],
                ["leather", "effb0_g14_b125", "96.9%", "0.000868", "0.0525", "削れる根拠あり"],
                ["tile", "effb0_g14_b250", "87.9%", "0.001736", "-0.0116", "境界例"],
                ["zipper", "wrn_l3_g14_b1500", "84.4%", "0.083333", "-0.0228", "境界例"],
                ["metal_nut", "wrn_l3_g14_b1500", "77.3%", "0.083333", "-0.0086", "強め構成が必要"],
                ["screw", "res34_g14_b9000", "17.1%", "0.187500", "-0.0395", "土台から弱い"],
            ],
            [23 * mm, 43 * mm, 21 * mm, 25 * mm, 20 * mm, 35 * mm],
            font_size=6.7,
        ),
        Spacer(1, 3 * mm),
        fig("mvtec_patchcore_reduction_limit_probe_002.png", "図4 削減構成の良品通過率、KNN計算量比、分離余裕M95。"),
        p(
            "bottle、hazelnut、leatherでは、C_NNを1.1%以下まで削ってもM95が正であり、"
            "軽量特徴空間でも正常/異常を分ける余裕が残っている。一方、cable、capsule、grid、pillなどでは"
            "M95が負となり、欠陥を止める閾値を選ぶと良品も大きく落ちる。",
            S["body"],
        ),
        p("現時点での科学的根拠", S["h2"]),
        bullets(
            [
                "正常画像の見た目が安定し、欠陥が正常分布から外れ値として現れるカテゴリでは、軽量バックボーン、小さい特徴次元、小さいバンクでもM95が残るため削れる。",
                "正常形状や局所構造のばらつきが大きいカテゴリ、または欠陥が正常変動に近いカテゴリでは、削減後にM95が負になり、しきい値分離が難しくなる。",
                "カテゴリ別プロファイルは単なる実験的チューニングではなく、C_NN = PBDを小さくしつつM95 > 0を保てるかを調べる設計問題として定式化できる。",
            ],
            S["body"],
        ),
        p(
            "min_profile C_NN(profile)  subject to  M95(profile) > 0,  FPR_defect <= α",
            S["quote"],
        ),
    ]

    story += [
        PageBreak(),
        p("目指す提案システム", S["h1"]),
        p(
            "提案システムは、リアルタイムに画像ごとに複雑なルーティングを行う方式ではない。"
            "検品ラインでは通常、その時間帯に検査する製品カテゴリは事前に分かっている。"
            "そこで、検品動作の前にカテゴリを選択し、そのカテゴリに対応するPatchCore-lite構成をFPGA上で有効化する。",
            S["body"],
        ),
        bullets(
            [
                "使用するCNN特徴層",
                "パッチグリッドサイズ",
                "メモリバンクサイズ",
                "異常判定閾値",
                "距離計算器に流す特徴次元とバンク配置",
            ],
            S["body"],
        ),
        fig("proposed_patchcore_fpga.png", "図5 提案するカテゴリ別プロファイル型PatchCore-lite FPGA構成。"),
        p(
            "hazelnutを検査する時間帯には、hazelnut用に選択した特徴抽出設定、メモリバンク、閾値を読み込む。"
            "別カテゴリに切り替える場合は、検品前に設定レジスタとメモリバンクを切り替える。",
            S["body"],
        ),
        p("既存手法に対する優位性", S["h1"]),
        p(
            "通常のPatchCoreは高精度である一方、メモリバンクとKNN探索が重い。"
            "本研究の提案は、PatchCoreの検出原理そのものを変えるのではなく、"
            "検品対象ごとに必要な特徴量とメモリバンクだけを残し、FPGA実装可能な形に縮約する点にある。",
            S["body"],
        ),
        p(
            "CNN分類器やU-Net系のセグメンテーション器は、欠陥画像と欠陥ラベルが十分に用意できる場合には強力である。"
            "しかし、実際の検品では未知欠陥や少数欠陥が問題になりやすい。PatchCore系は正常画像のみから正常分布を作るため、"
            "欠陥データ収集コストが高い状況に向いている。",
            S["body"],
        ),
        p(
            "カスケード方式、パラレル方式、BranchyNet型早期終了は、主に画像分類モデルの推論を途中で切り替える考え方である。"
            "本提案は画像ごとの分岐判断ではなく、検品対象カテゴリが事前に分かるという前提を使い、固定されたモードごとの資源量とレイテンシを評価する。",
            S["body"],
        ),
    ]

    story += [
        PageBreak(),
        p("FPGA化で得られることと見積もり性能", S["h1"]),
        p(
            "PatchCoreのKNN探索は、同じ形の距離計算を多数のメモリバンク特徴に対して繰り返す。"
            "この処理は、FPGA上で並列距離計算器として実装しやすい。",
            S["body"],
        ),
        bullets(
            [
                "メモリバンクのビット幅と配置",
                "距離計算器の並列数",
                "ストリーミング入力に対するパイプライン処理",
                "閾値比較による通過/排出判定",
                "カテゴリモード切替時のバンク選択",
            ],
            S["body"],
        ),
        table(
            [
                ["項目", "値", "意味"],
                ["KNN演算量平均", "0.0109倍", "距離探索の演算回数は約98.9%削減"],
                ["KNN演算量中央値", "0.0035倍", "多くのカテゴリではさらに小さい"],
                ["総コスト近似平均", "0.2687倍", "CNN特徴抽出とKNN探索を合わせても削減が残る"],
                ["全カテゴリ分バンク", "3.288 MiB", "基準構成131.660 MiBから約97.5%削減"],
                ["512並列KNN平均", "0.193 ms", "削減後バンクに対する距離探索レイテンシ見積もり"],
                ["512並列KNN最大", "1.470 ms", "選択カテゴリ中の最大見積もり"],
            ],
            [38 * mm, 30 * mm, 92 * mm],
        ),
        p(
            "ただし、これは実装前のモデルであり、実際の速度・電力・資源量はFPGA実装後に測定する必要がある。"
            "特に、固定小数点化やint8/int4化により異常スコアの分布が変わる可能性がある。",
            S["body"],
        ),
        p("今後の作業", S["h1"]),
        number_list(
            [
                "hazelnutを対象に、選択済みPatchCore-lite構成を固定する。",
                "ソフトウェア側で同じ構成の入出力を固定し、FPGA実装の検証データを作る。",
                "KNN距離計算部をまずFPGA上に実装する。",
                "その後、特徴抽出部、メモリバンク配置、閾値判定を含めた全体構成へ拡張する。",
                "実測値を用いて、コストモデルとの差、速度、電力、品質変化を評価する。",
                "単一カテゴリで成立した後、カテゴリモード切替型の複数カテゴリ対応へ拡張する。",
            ],
            S["body"],
        ),
        p(
            "研究テーマ: 検品品質制約を考慮したメモリバンク型異常検知のカテゴリ別プロファイルとFPGA実装評価",
            S["quote"],
        ),
        p("参考文献", S["h1"]),
        p(
            "[1] K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022.<br/>"
            "[2] MVTec Software GmbH, “MVTec AD: Industrial Anomaly Detection Dataset.”<br/>"
            "[3] S. Teerapittayanon et al., “BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks,” ICPR 2016.<br/>"
            "[4] W. Wu et al., “MAD-Flow: An Efficient Deployment Flow for Memory-Bank-Based Anomaly Detection on FPGA-SoCs,” IEEE IoT Journal, 2025.<br/>"
            "[5] “PatchCore によるシール溶接不良検出システムの検出速度の向上,” 産業応用工学会全国大会講演論文集, 2023.",
            S["small"],
        ),
    ]
    return story


if __name__ == "__main__":
    setup_fonts()
    S = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    doc.build(build_story(), onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)
