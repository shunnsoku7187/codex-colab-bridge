"""Bilingual A0 research posters, with shared geometry and embedded fonts."""
from pathlib import Path
from math import atan2, cos, sin, pi

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A0
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib import colors
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'poster'
# Half-size design coordinates; output uses true A0 dimensions.
W, H = A0[0] / 2, A0[1] / 2
M, GAP = 42, 36
CW = (W - 2 * M - GAP) / 2
RIGHT = M + CW + GAP
AB = [(80.27, 1.000), (81.95, .500), (85.01, .400), (85.53, .378)]
ABC = [(79.27, 1.000), (81.95, .333), (84.49, .271), (85.53, .252)]


def fonts():
    for name, filename in [('EN', 'arial.ttf'), ('ENB', 'arialbd.ttf'),
                           ('JA', 'meiryo.ttc'), ('JAB', 'meiryob.ttc')]:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
    for name in ['EN', 'JA']:
        pdfmetrics.registerFontFamily(name, normal=name, bold=name + 'B', italic=name, boldItalic=name + 'B')


class Poster:
    def __init__(self, ja):
        self.ja = ja
        self.font = 'JA' if ja else 'EN'
        self.path = OUT / f'patchcore_profile_switch_poster_{"ja" if ja else "en"}.pdf'
        self.c = canvas.Canvas(str(self.path), pagesize=A0)
        self.c.scale(2, 2)
        self.c.setTitle('Category-wise Profile Switching for PatchCore-lite')
        self.c.setAuthor('Shunya Yamada; Kazuyoshi Takagi')
        self.text_boxes = []

    def t(self, en, ja):
        return ja if self.ja else en

    def rule(self, x, y, w, thick=1, gray=.15):
        self.c.setStrokeGray(gray)
        self.c.setLineWidth(thick)
        self.c.line(x, H-y, x+w, H-y)

    def rect(self, x, y, w, h, gray=1, border=True):
        self.c.setFillGray(gray)
        self.c.setStrokeGray(.2)
        self.c.setLineWidth(.8)
        self.c.rect(x, H-y-h, w, h, fill=1, stroke=int(border))

    def text(self, txt, x, y, w, size=16, bold=False, align=0, max_h=None):
        style = ParagraphStyle('p', fontName=self.font + ('B' if bold else ''),
                               fontSize=size, leading=size*1.35, alignment=align,
                               wordWrap='CJK' if self.ja else None,
                               textColor=colors.black, spaceAfter=0)
        p = Paragraph(txt, style)
        _, h = p.wrap(w, H)
        if max_h is not None and h > max_h + .1:
            raise ValueError(f'Text overflow ({h:.1f}>{max_h}): {txt}')
        assert x >= 0 and x + w <= W + .1 and y+h < H
        p.drawOn(self.c, x, H-y-h)
        self.text_boxes.append((x, y, w, h))
        return y+h

    def heading(self, num, title, x, y, w):
        self.text(f'{num}  {title}', x, y, w, 22, True, max_h=32)
        self.rule(x, y+36, w, 1.5)

    def node(self, txt, x, y, w, h=58, gray=.96):
        self.rect(x, y, w, h, gray)
        n = Paragraph(txt, ParagraphStyle('node', fontName=self.font+'B', fontSize=15,
            leading=19, alignment=1, wordWrap='CJK' if self.ja else None))
        _, th = n.wrap(w-14, h)
        assert th <= h-8, txt
        n.drawOn(self.c, x+7, H-y-(h+th)/2)

    def arrow(self, points, dashed=False):
        self.c.setStrokeGray(.15)
        self.c.setFillGray(.15)
        self.c.setLineWidth(1.5)
        self.c.setDash(4, 3) if dashed else self.c.setDash()
        for (x,y), (xx,yy) in zip(points, points[1:]):
            self.c.line(x,H-y,xx,H-yy)
        self.c.setDash()
        x,y=points[-1]; a=atan2(y-points[-2][1],x-points[-2][0])
        p=self.c.beginPath();p.moveTo(x,H-y)
        for d in [-.42,.42]:
            p.lineTo(x-8*cos(a+d), H-(y-8*sin(a+d)))
        p.close();self.c.drawPath(p, fill=1, stroke=0)

    def header(self):
        title = self.t('FPGA Implementation of a PatchCore-lite<br/>Anomaly Detection System<br/>with Category-wise Profile Switching',
                       'カテゴリ別プロファイル切り替え機構を備えた<br/>PatchCore-lite異常検知システムの<br/>FPGA実装に向けた検討')
        self.text(title,M,30,W-2*M,30,True,align=1,max_h=125)
        self.text(self.t('Shunya Yamada and Kazuyoshi Takagi','山田 峻也・高木 一義'),M,161,W-2*M,17,True,align=1)
        self.text(self.t('Computer Architecture Laboratory, Department of Information Engineering,<br/>Graduate School of Engineering, Mie University  |  shunya@arch.info.mie-u.ac.jp',
                        '三重大学大学院 工学研究科 情報工学専攻 コンピュータアーキテクチャ研究室<br/>shunya@arch.info.mie-u.ac.jp'),M,188,W-2*M,13,align=1)
        self.rule(M,237,W-2*M,2)
        self.text(self.t('Category-specific feature settings and memory banks on one shared inspection engine',
                        '検査回路を共有し、検査対象に合わせて特徴量の設定と正常特徴のメモリを切り替える。'),
                  M,249,W-2*M,18,True,align=1,max_h=50)

    def patchcore(self):
        x=M
        self.heading('1',self.t('PatchCore anomaly detection','PatchCoreの仕組み'),x,303,CW)
        self.text(self.t('PatchCore [1] detects defects by comparing image features with stored normal examples. A memory bank is a set of representative normal patch features.',
                        'PatchCore [1]は、画像の特徴を正常例と比較して異常を検知する。メモリバンクとは、正常画像の各部分（パッチ）から得た代表的な特徴ベクトルの集合である。'),x,352,CW,max_h=90)
        bw=(CW-48)/3; y=460
        self.node(self.t('Image + CNN','画像 + CNN'),x,y,bw)
        self.node(self.t('Patch feature<br/>vectors','パッチごとの<br/>特徴ベクトル'),x+bw+24,y,bw)
        self.node(self.t('Nearest-neighbor<br/>(NN) search','近傍探索（NN）<br/>による距離計算'),x+2*(bw+24),y,bw)
        for i in range(2):
            self.arrow([(x+(i+1)*bw+i*24,y+29),(x+(i+1)*(bw+24),y+29)])
        bx=x+2*(bw+24)
        self.node(self.t('Normal memory<br/>bank','正常特徴の<br/>メモリバンク'),bx,558,bw,52)
        self.arrow([(bx+bw/2,558),(bx+bw/2,518)])
        self.text(self.t('Normal training patches<br/>→ k-center coreset selection<br/>→ representative features',
                        '正常な学習画像のパッチ<br/>→ k-center coresetで代表点を選択<br/>→ メモリバンクへ保存'),x,548,2*bw+24,15,max_h=68)
        self.text(self.t('Large distance from the bank → high anomaly score.<br/>A threshold separates normal and defective images.',
                        '正常特徴から遠いほど異常スコアが高くなる。<br/>画像単位に集約したスコアをしきい値で判定する。'),x,630,CW,max_h=65)

    def proposal(self):
        x=RIGHT
        self.heading('2',self.t('Category-wise profile and bank switching','カテゴリ別profile・bank切り替え'),x,303,CW)
        self.text(self.t('Useful feature settings differ between categories. A known product category selects its profile and normal memory bank before inspection. The proposed FPGA design shares one CNN and NN engine.',
                        '有効な特徴量の設定はカテゴリによって異なる。検査前に対象カテゴリが分かる状況で、専用profileと正常特徴のbankを選択する。提案するFPGA構成ではCNNとNN回路をカテゴリ間で共有する。'),x,352,CW,max_h=90)
        self.node(self.t('Category A / B / C','検査カテゴリ A / B / C'),x,456,185,54)
        self.node(self.t('Profile + bank selection','profile + bankの選択'),x+214,456,CW-214,54)
        self.arrow([(x+185,483),(x+214,483)])
        bw=(CW-48)/3
        for i,label in enumerate([self.t('Shared CNN','共通CNN'),self.t('Feature selection','特徴量の選択'),self.t('Bank search','bank探索')]):
            self.node(label,x+i*(bw+24),565,bw,54)
        for i in range(2):
            self.arrow([(x+(i+1)*bw+i*24,592),(x+(i+1)*(bw+24),592)])
        self.arrow([(x+CW-70,510),(x+CW-70,535),(x+CW/2,535),(x+CW/2,565)],True)
        self.arrow([(x+CW-70,535),(x+CW-bw/2,535),(x+CW-bw/2,565)],True)
        self.text(self.t('<b>Profile:</b> feature layer and dimension, patch grid, score aggregation and threshold. Bank selection changes the address range searched.',
                        '<b>profile：</b>抽出する特徴層と次元数、パッチの配置、スコア集約方法、判定しきい値。bankは探索するアドレス範囲を切り替える。'),x,637,CW,max_h=76)

    def setup(self):
        self.heading('3',self.t('Experiment and comparison','実験条件と比較構成'),M,736,W-2*M)
        x=M
        self.text(self.t('<b>Dataset:</b> MVTec AD [2], all 15 categories.<br/><b>Tasks:</b> all 105 pairs and all 455 triples.<br/><b>Backbone:</b> shared WideResNet50-2.<br/><b>Bank:</b> k-center coreset, 3,000 vectors/category.',
                        '<b>データ：</b>MVTec AD [2]、全15カテゴリ。<br/><b>組み合わせ：</b>2カテゴリ105通り、3カテゴリ455通り。<br/><b>CNN：</b>WideResNet50-2を共用。<br/><b>bank：</b>k-center coreset、各カテゴリ3,000点。'),x,786,CW,max_h=108)
        self.text(self.t('<b>Profile selection:</b> a prior single-category sweep at bank=12,000 retained candidates within 2 percentage points of the default normal-pass rate, then minimized patch count × feature dimension.',
                        '<b>専用profileの決定：</b>bank=12,000の単一カテゴリ評価で、標準設定からの良品通過率低下が2ポイント以内の候補を残し、パッチ数×特徴次元数を最小化した。'),x,893,CW,15,max_h=104)
        labels=self.t([
            ('I','Default profile + pooled bank'),('II','Default profile + category-bank switching'),
            ('III','Best fixed profile + category-bank switching'),('IV','Category-profile + category-bank switching')],
            [('I','標準profile + 対象カテゴリの共通bank'),('II','標準profile + カテゴリ別bank切り替え'),
             ('III','最良固定profile + カテゴリ別bank切り替え'),('IV','専用profile + カテゴリ別bank切り替え')])
        for i,(label,desc) in enumerate(labels):
            yy=786+i*35
            if i==3:self.rect(RIGHT-6,yy-5,CW+12,33,.92,False)
            self.text(label,RIGHT,yy,40,16,True)
            self.text(desc,RIGHT+42,yy,CW-42,15,i==3,max_h=25)
        self.text(self.t('III uses the best-performing fixed profile for each pair/triple. I pools only the target categories. All systems retain the same total bank-entry budget.',
                        'IIIは各組み合わせで最も良品通過率が高い固定profile。Iの共通bankは対象カテゴリのみを含む。全方式で保存するbankの総点数をそろえた。'),RIGHT,935,CW,15,max_h=65)
        self.text(self.t('Default: layer2+3, 14 × 14 patches, 768 dimensions; image score = mean of the highest 1% of patch scores.',
                        '標準設定：layer2+3、14×14パッチ、768次元。画像スコアはパッチスコア上位1%の平均。'),
                  M,994,W-2*M,13,max_h=20)

    def table(self,x,y,title,rows):
        self.text(title,x,y,CW,18,True)
        yy=y+36
        self.rule(x,yy,CW,1.2)
        self.text(self.t('System','構成'),x+8,yy+10,135,15,True)
        self.text(self.t('Normal pass (%) ↑','良品通過率 (%) ↑'),x+140,yy+10,210,15,True,align=2)
        self.text(self.t('NN work (×) ↓','NN探索量 (倍) ↓'),x+355,yy+10,CW-365,15,True,align=2)
        for i,(good,cost) in enumerate(rows):
            top=yy+43+i*34
            if i==3:self.rect(x,top,CW,34,.92,False)
            self.text(['I','II','III',self.t('IV  Proposed','IV  提案')][i],x+8,top+5,135,16,i==3)
            self.text(f'{good:.2f}',x+140,top+5,210,16,i==3,align=2)
            self.text(f'{cost:.3f}',x+355,top+5,CW-365,16,i==3,align=2)
        self.rule(x,yy+179,CW,1.2)

    def results(self):
        self.heading('4',self.t('Detection performance and search cost','検知性能と探索コスト'),M,1020,W-2*M)
        self.text(self.t('Thresholds allow at most 3% of defects to pass. Normal pass is the fraction of normal images accepted. Tables show subset means. NN work is proportional to patch count × searched bank entries × feature dimension, normalized to I.',
                        '欠陥誤通過率3%以下となるしきい値で評価。良品通過率は正常画像のうち良品と判定した割合で、表は組み合わせ間の平均。NN探索量はパッチ数×探索bank点数×特徴次元数に比例し、Iを1とした比で示す。'),M,1070,W-2*M,16,max_h=68)
        self.table(M,1150,self.t('Two categories · 105 combinations','2カテゴリ・105通り'),AB)
        self.table(RIGHT,1150,self.t('Three categories · 455 combinations','3カテゴリ・455通り'),ABC)
        self.text(self.t('Compared with bank switching alone (II), the proposal raises mean normal pass by 3.58 percentage points and reduces NN work by about 24%. Compared with the best fixed profile (III), the mean gains are 0.52 points for pairs and 1.04 points for triples.',
                        'bankのみ切り替えるIIに対し、提案は平均良品通過率を3.58ポイント改善し、NN探索量を約24%削減した。最良固定profileのIIIに対しても、平均通過率は2カテゴリで0.52ポイント、3カテゴリで1.04ポイント高かった。'),M,1380,W-2*M,16,max_h=66)

    def footer(self):
        # The full-width concluding band ties measured quality to modeled hardware work.
        self.heading('5',self.t('FPGA implementation: control the active work','FPGA実装：必要な処理範囲を制御'),M,1461,W-2*M)
        self.text(self.t('<b>Nearest-neighbor (NN) search:</b> C = P × B × D.<br/>P: patches/image; B: bank entries searched;<br/>D: dimensions/feature. Category switching reduces B;<br/>profile switching can reduce P and D.',
                        '<b>近傍探索（NN）の計算量：</b>C = P × B × D。<br/>P：画像のパッチ数、B：探索するbank点数、<br/>D：特徴次元数。bank切り替えはBを、<br/>profile切り替えはPやDを削減する。'),M,1510,CW,15,max_h=92)
        self.text(self.t('Implement profile registers, dimension masks and bank address control on shared CNN / NN hardware. The 0.252× triple-category result predicts NN-side work, not total latency or power. Hardware measurement remains future work.',
                        '共通CNN・NN回路に設定レジスタ、特徴次元の有効制御、bankアドレス制御を実装する。3カテゴリでの0.252倍はNN側の処理量の比であり、装置全体の速度・電力比ではない。実機評価は今後行う。'),RIGHT,1510,CW,15,max_h=107)
        self.rule(M,1634,W-2*M,.8)
        self.text('[1] K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR, 2022.<br/>[2] P. Bergmann et al., “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR, 2019.',M,1643,W-2*M,10,max_h=32)

    def build(self):
        self.header();self.patchcore();self.proposal();self.setup();self.results();self.footer()
        self.c.save()
        doc=PdfReader(str(self.path))
        assert len(doc.pages)==1
        assert abs(float(doc.pages[0].mediabox.width)-A0[0])<1
        pdf=pdfium.PdfDocument(str(self.path))
        pdf[0].render(scale=.65).to_pil().save(OUT/f'preview_{"ja" if self.ja else "en"}.png')
        print(self.path)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    fonts()
    for ja in [False,True]:Poster(ja).build()
