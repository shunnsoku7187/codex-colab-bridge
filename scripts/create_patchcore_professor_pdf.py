"""Generate the professor-facing PatchCore FPGA report as a PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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


OUT = Path("output/pdf/patchcore_fpga_professor_report.pdf")
FIG = Path("docs/figures")


def register_fonts() -> tuple[str, str]:
    regular = "C:/Windows/Fonts/NotoSansJP-VF.ttf"
    bold = "C:/Windows/Fonts/YuGothB.ttc"
    pdfmetrics.registerFont(TTFont("JP", regular))
    pdfmetrics.registerFont(TTFont("JP-Bold", bold))
    return "JP", "JP-Bold"


def styles() -> dict[str, ParagraphStyle]:
    regular, bold = register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=regular,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=15,
            leading=20,
            spaceBefore=10,
            spaceAfter=7,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=12.5,
            leading=17,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#1e293b"),
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.4,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            fontName=regular,
            fontSize=8,
            leading=11,
            wordWrap="CJK",
            textColor=colors.HexColor("#475569"),
        ),
        "table": ParagraphStyle(
            "table",
            fontName=regular,
            fontSize=7.8,
            leading=10,
            wordWrap="CJK",
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            fontName=bold,
            fontSize=7.8,
            leading=10,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName=regular,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            spaceBefore=3,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "quote",
            fontName=regular,
            fontSize=9.4,
            leading=15,
            leftIndent=9 * mm,
            rightIndent=6 * mm,
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=1,
            borderPadding=6,
            backColor=colors.HexColor("#f8fafc"),
            wordWrap="CJK",
        ),
    }


def p(text: str, st: dict[str, ParagraphStyle], key: str = "body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), st[key])


def bullets(items: list[str], st: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, st), bulletColor=colors.HexColor("#2563eb")) for item in items],
        bulletType="bullet",
        leftIndent=12,
        bulletFontName="JP",
    )


def table(data: list[list[str]], st: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    rows = []
    for i, row in enumerate(data):
        rows.append([p(str(cell), st, "table_bold" if i == 0 else "table") for cell in row])
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def fig(path: str, caption: str, st: dict[str, ParagraphStyle], width: float = 165 * mm) -> KeepTogether:
    img = Image(str(FIG / path))
    img._restrictSize(width, 100 * mm)
    return KeepTogether([img, p(caption, st, "caption")])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("JP", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(22 * mm, 12 * mm, "カテゴリ別プロファイル型PatchCore-lite FPGA")
    canvas.drawRightString(188 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    st = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="カテゴリ別プロファイル型PatchCore-liteのFPGA実装に向けた説明資料",
        author="山田 俊介",
    )

    story = []
    story.append(p("カテゴリ別プロファイル型PatchCore-liteの<br/>FPGA実装に向けた説明資料", st, "title"))
    story.append(p("山田 俊介 / 2026年8月23日", st, "subtitle"))
    story.append(p("概要", st, "h1"))
    story.append(
        p(
            "本資料では，検品向け異常検知手法であるPatchCoreを題材に，"
            "「単にPatchCoreをFPGAに載せる」のではなく，検品対象カテゴリごとに必要な"
            "メモリバンクとKNN探索量を事前に決定し，その構成をFPGA上で切り替える方式を説明する。",
            st,
        )
    )
    story.append(
        p(
            "現在までの実験から，PatchCoreの重い部分であるKNN探索は平均で基準構成の0.0109倍，"
            "中央値で0.0035倍まで削減できる見込みが得られた。全15カテゴリ分のメモリバンクも，"
            "131.660 MiBから3.288 MiBへ削減できる。一方，全カテゴリで厳しい検品品質を満たす主張は"
            "現時点では難しく，初回実装はhazelnutなど成立しやすいカテゴリに絞る。",
            st,
        )
    )

    story.append(p("1. PatchCoreとは", st, "h1"))
    story.append(p("検品タスクにおける位置づけ", st, "h2"))
    story.append(
        p(
            "PatchCoreは，工業製品の外観検査で用いられる異常検知手法である。"
            "通常の画像分類モデルは良品画像と欠陥画像の両方を使うが，実際の検品現場では"
            "欠陥画像を十分に集めにくい。PatchCoreは正常画像のみから正常らしさを記憶し，"
            "検査画像が正常分布からどれだけ離れているかを測る。",
            st,
        )
    )
    story.append(
        p(
            "元論文では，MVTec ADにおいて高い画像レベル異常検知性能を示しており，"
            "欠陥例が少ない cold-start な検品タスクに強い。MVTec ADは15種類の工業検品カテゴリを含む"
            "代表的な異常検知ベンチマークである。",
            st,
        )
    )
    story.append(fig("patchcore_flow.png", "図1 PatchCoreの基本動作。重い部分はメモリバンク容量とKNN探索である。", st))
    story.append(PageBreak())
    story.append(p("用語整理", st, "h2"))
    story.append(
        table(
            [
                ["用語", "意味"],
                ["パッチ特徴", "画像全体ではなく，画像内の局所領域ごとに取り出した特徴。傷や汚れの局所検出に向く。"],
                ["メモリバンク", "正常画像から得られたパッチ特徴の集合。推論時に正常らしさの基準になる。"],
                ["KNN探索", "検査画像の各パッチ特徴について，メモリバンク内の近い正常特徴を探す処理。精度を支える一方で重い。"],
                ["異常スコア", "検査画像のパッチ特徴と正常メモリバンクの距離から計算される値。高いほど異常。"],
            ],
            st,
            [32 * mm, 130 * mm],
        )
    )
    story.append(p("強みと弱み", st, "h2"))
    story.append(
        p(
            "PatchCoreの強みは，欠陥画像を大量に用意しなくても検品タスクを構成しやすい点である。"
            "一方で，メモリバンクを大きくすると正常特徴を細かく保持できるが，メモリ容量とKNN探索量が増える。"
            "概算では KNN探索量は「画像パッチ数 × バンク特徴数 × 特徴次元」に比例する。",
            st,
        )
    )

    story.append(p("2. カテゴリを絞れば削減できる根拠", st, "h1"))
    story.append(
        p(
            "今回の実験では，MVTec ADの15カテゴリを対象に，特徴層，パッチグリッド，バンクサイズ，閾値を変え，"
            "良品通過，欠陥誤通過，KNN演算量，メモリバンク量を分けて評価した。検品ではAUROCだけでなく，"
            "欠陥品を良品として通してしまう欠陥誤通過を分けて見る必要がある。",
            st,
        )
    )
    story.append(
        table(
            [
                ["設定", "良品通過", "KNN演算", "読み取り"],
                ["全カテゴリ共通: 基準構成", "59.34%", "1.0000倍", "最も重い参照点。"],
                ["全カテゴリ共通: KNN 20%以下", "57.19%", "0.1667倍", "共通軽量化で品質を残せる限界の目安。"],
                ["全カテゴリ共通: KNN 10%以下", "44.88%", "0.0834倍", "さらに軽くすると良品通過が大きく下がる。"],
                ["カテゴリ別選択設定，holdout平均", "54.59%", "0.0206倍", "KNNは非常に小さいが，安全性はカテゴリ別に扱う必要がある。"],
            ],
            st,
            [58 * mm, 24 * mm, 24 * mm, 56 * mm],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(fig("profiled_reduction_bars.png", "図2 カテゴリ別プロファイルによる削減効果。基準構成を1.0倍とした相対値。", st, 150 * mm))
    story.append(
        p(
            "全カテゴリ共通の軽量設定では，KNNを小さくすると良品通過が大きく下がる。"
            "一方，カテゴリ別に構成を選ぶと，KNN探索は平均0.0109倍，中央値0.0035倍となった。"
            "全カテゴリ分のメモリバンクも，基準構成の131.660 MiBから3.288 MiBへ削減できる。",
            st,
        )
    )
    story.append(p("品質面の限界", st, "h2"))
    story.append(
        p(
            "ただし，全15カテゴリで厳しい検品品質を満たせたわけではない。選択済みプロファイル設定が"
            "欠陥誤通過3%以下を満たしたのは5/15カテゴリ，5%以下では9/15カテゴリであった。"
            "したがって，現時点で主張すべきことは「万能な軽量PatchCore」ではなく，"
            "カテゴリが事前に分かる検品対象に対し，必要な構成だけを選んでFPGA資源へ落とす方式である。",
            st,
        )
    )
    story.append(
        table(
            [
                ["カテゴリ", "選択設定", "良品通過", "欠陥誤通過", "総コスト近似", "KNN演算"],
                ["hazelnut", "res18_l23_g14_b500", "92.00%", "1.14%", "0.1008倍", "0.0104倍"],
                ["zipper", "wrn_l3_g14_b1500", "75.00%", "2.71%", "0.5220倍", "0.0833倍"],
                ["wood", "res18_l23_g7_b125", "71.11%", "2.00%", "0.0957倍", "0.0007倍"],
                ["cable", "res18_l23_g7_b1500", "33.79%", "2.61%", "0.0995倍", "0.0078倍"],
            ],
            st,
            [25 * mm, 43 * mm, 22 * mm, 25 * mm, 26 * mm, 21 * mm],
        )
    )
    story.append(
        p(
            "初回実装としては，良品通過92.00%，欠陥誤通過1.14%，総コスト近似0.1008倍のhazelnutが最も扱いやすい。",
            st,
        )
    )

    story.append(p("3. 目指す提案システム", st, "h1"))
    story.append(
        p(
            "提案システムは，画像ごとに複雑なルーティングを行う方式ではない。検品ラインでは，通常，"
            "その時間帯に検査する製品カテゴリは事前に分かっている。そこで，検品動作の前にカテゴリを選択し，"
            "そのカテゴリに対応するPatchCore-lite構成をFPGA上で有効化する。",
            st,
        )
    )
    story.append(fig("proposed_patchcore_fpga.png", "図3 提案するカテゴリ別プロファイル型PatchCore-lite FPGA構成。", st))
    story.append(
        p(
            "カテゴリごとに，使用するCNN特徴層，パッチグリッドサイズ，メモリバンクサイズ，異常判定閾値，"
            "距離計算器に流す特徴次元とバンク配置を事前に決める。別カテゴリに切り替える場合は，"
            "検品前に設定レジスタとメモリバンクを切り替える。",
            st,
        )
    )
    story.append(
        p(
            "この方式は，画像ごとの分岐判断を主張の中心に置かない。"
            "「検品対象カテゴリが既知である」という工業検品の前提を利用する，"
            "カテゴリ別の実行モードを持つFPGAアクセラレータである。",
            st,
            "quote",
        )
    )

    story.append(PageBreak())
    story.append(p("4. 既存手法に対する優位性", st, "h1"))
    story.append(
        table(
            [
                ["比較対象", "既存側の強み/弱み", "本提案の立ち位置"],
                ["通常PatchCore", "高精度だがメモリバンクとKNN探索が重い。", "検品カテゴリごとに必要な構成だけを残し，FPGA向けに縮約する。"],
                ["分類器/U-Net", "欠陥ラベルが十分あれば強いが，未知欠陥や少数欠陥に弱い。", "正常画像のみで構成しやすいPatchCore系を対象にする。"],
                ["カスケード/早期終了", "画像ごとの分岐で平均計算量を下げる。", "分岐予測ではなく，カテゴリ既知性を使って固定モードの資源量を下げる。"],
                ["MAD-Flow等", "メモリバンク型異常検知のFPGA化を扱う近い先行研究。", "AUROCだけでなく欠陥誤通過/良品通過制約とカテゴリ別資源配分を主張点にする。"],
            ],
            st,
            [32 * mm, 62 * mm, 68 * mm],
        )
    )
    story.append(
        p(
            "このため，本研究を「PatchCoreをFPGAに載せた」とだけ主張するのは弱い。"
            "狙うべき差分は，検品品質制約を満たす範囲で，カテゴリごとに特徴層・グリッド・バンク量を選び，"
            "その選択結果をFPGAのメモリ量，距離計算並列度，レイテンシへ対応づけることである。",
            st,
        )
    )

    story.append(p("5. FPGA化で得られることと見積もり性能", st, "h1"))
    story.append(
        p(
            "PatchCoreのKNN探索は，同じ形の距離計算を多数のメモリバンク特徴に対して繰り返す。"
            "この処理は，FPGA上で並列距離計算器として実装しやすい。カテゴリ別プロファイルにより"
            "バンクサイズを小さくできると，必要なBRAM/URAM/外部メモリアクセス量が減り，"
            "距離計算器の並列化も現実的になる。",
            st,
        )
    )
    story.append(
        table(
            [
                ["項目", "値", "意味"],
                ["KNN演算量平均", "0.0109倍", "距離探索の演算回数は約98.9%削減。"],
                ["KNN演算量中央値", "0.0035倍", "多くのカテゴリではさらに小さい。"],
                ["総コスト近似平均", "0.2687倍", "CNN特徴抽出とKNN探索を合わせても削減が残る。"],
                ["全カテゴリ分バンク", "3.288 MiB", "基準構成131.660 MiBから約97.5%削減。"],
                ["512並列KNN平均", "0.193 ms", "削減後バンクに対する距離探索レイテンシ見積もり。"],
                ["512並列KNN最大", "1.470 ms", "選択カテゴリ中の最大見積もり。"],
            ],
            st,
            [40 * mm, 30 * mm, 92 * mm],
        )
    )
    story.append(
        p(
            "ただし，これは実装前のモデルである。実際の速度，電力，資源量はFPGA実装後に測定する必要がある。"
            "特に固定小数点化やint8/int4化では異常スコアの分布が変わる可能性があるため，"
            "欠陥誤通過と良品通過を再評価する。",
            st,
        )
    )
    story.append(p("今後の作業", st, "h1"))
    story.append(
        bullets(
            [
                "hazelnutを対象に，選択済みPatchCore-lite構成を固定する。",
                "ソフトウェア側で同じ構成の入出力を固定し，FPGA実装の検証データを作る。",
                "KNN距離計算部をまずFPGA上に実装する。",
                "特徴抽出部，メモリバンク配置，閾値判定を含めた全体構成へ拡張する。",
                "実測値を用いて，コストモデルとの差，速度，電力，品質変化を評価する。",
                "単一カテゴリで成立した後，カテゴリモード切替型の複数カテゴリ対応へ拡張する。",
            ],
            st,
        )
    )
    story.append(
        p(
            "研究テーマ案: 検品品質制約を考慮したメモリバンク型異常検知のカテゴリ別プロファイルとFPGA実装評価",
            st,
            "quote",
        )
    )

    story.append(PageBreak())
    story.append(p("参考文献", st, "h1"))
    refs = [
        "K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html",
        "MVTec Software GmbH, “MVTec AD: Industrial Anomaly Detection Dataset.” https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "S. Teerapittayanon, B. McDanel, and H. T. Kung, “BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks,” ICPR 2016. https://doi.org/10.1109/ICPR.2016.7900006",
        "W. Wu et al., “MAD-Flow: An Efficient Deployment Flow for Memory-Bank-Based Anomaly Detection on FPGA-SoCs,” IEEE Internet of Things Journal, 2025. https://doi.org/10.1109/JIOT.2025.3605880",
        "“PatchCore によるシール溶接不良検出システムの検出速度の向上,” 産業応用工学会全国大会講演論文集, 2023. https://doi.org/10.12792/iiae2023.026",
    ]
    story.append(ListFlowable([ListItem(p(r, st, "small")) for r in refs], bulletType="1", leftIndent=16))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
