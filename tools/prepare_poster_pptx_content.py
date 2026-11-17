"""Reuse the approved poster copy and geometry as editable presentation data."""
import json
import re
from html import unescape
from build_patchcore_profile_switch_poster import Poster, W, H, CW, ROOT, fonts
from reportlab.pdfbase import pdfmetrics

BUILD = ROOT / 'tmp' / 'poster_pptx'


class Capture(Poster):
    def __init__(self, ja):
        self.ja = ja
        self.font = 'JA' if ja else 'EN'
        self.items = []

    def rule(self, x, y, w, thick=1, gray=.15):
        self.items.append(dict(kind='rule', x=x, y=y, w=w, thick=thick))

    def rect(self, x, y, w, h, gray=1, border=True):
        self.items.append(dict(kind='rect', x=x, y=y, w=w, h=h, gray=gray, border=border))

    def text(self, txt, x, y, w, size=16, bold=False, align=0, max_h=None):
        plain = unescape(re.sub('<[^>]+>', '', txt.replace('<br/>', '\n')))
        # Explicit lines preserve the reviewed layout across PowerPoint renderers.
        lines = []
        for paragraph in plain.split('\n'):
            tokens = re.findall(r'[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*|.', paragraph) if self.ja else paragraph.split(' ')
            line = ''
            for token in tokens:
                proposal = line + ('' if self.ja or not line else ' ') + token
                if self.ja and token in '。、，．：；）］」』！？':
                    line = proposal
                    continue
                if line and pdfmetrics.stringWidth(proposal, self.font+('B' if bold else ''), size) > w-6:
                    lines.append(line); line=token
                else: line=proposal
            lines.append(line)
        h = len(lines) * size * 1.35 + 5
        self.items.append(dict(kind='text', text='\n'.join(lines), x=x, y=y, w=w,
                               h=h, size=size, bold=bold, align=align))
        return y+h

    def node(self, txt, x, y, w, h=58, gray=.96):
        self.items.append(dict(kind='node', text=txt.replace('<br/>','\n'),x=x,y=y,w=w,h=h))

    def arrow(self, points, dashed=False):
        self.items.append(dict(kind='arrow', points=points, dashed=dashed))

    def table(self, x, y, title, rows):
        self.text(title,x,y,CW,18,True)
        self.items.append(dict(kind='table', x=x,y=y+36,w=CW,h=179,values=[
            [self.t('System','構成'),self.t('Normal pass (%)','良品通過率 (%)'),self.t('NN work (I = 1)','NN探索量 (I = 1)')],
            *[[['I','II','III',self.t('IV  Proposed','IV  提案')][i],f'{a:.2f}',f'{b:.3f}'] for i,(a,b) in enumerate(rows)]
        ]))

    def header(self):
        super().header()
        self.items = [i for i in self.items if i.get('y') != 249]

    def results(self):
        from build_patchcore_profile_switch_poster import M, RIGHT, AB, ABC
        self.heading('4', self.t('Detection performance and search cost','検知性能と探索コスト'),M,1020,W-2*M)
        self.text(self.t('Operating point: at most 3% of defects pass.\nNormal pass: fraction of normal images accepted.',
                         '評価条件：欠陥誤通過率3%以下。\n良品通過率：正常画像を良品と判定した割合。'),M,1072,CW,16)
        self.text(self.t('NN work = patches × searched bank entries × dimensions.\nTables report mean ratios relative to I for each subset size.',
                         'NN探索量＝パッチ数×探索bank点数×特徴次元数。\n表は構成Iを1とした比と良品通過率の平均を示す。'),RIGHT,1072,CW,15)
        self.table(M,1150,self.t('Two categories: 105 combinations','2カテゴリ：105通り'),AB)
        self.table(RIGHT,1150,self.t('Three categories: 455 combinations','3カテゴリ：455通り'),ABC)
        self.text(self.t('Versus bank switching alone (II): +3.58 points in normal\npass and about 24% less NN work for both task sizes.',
                         'bankのみ切り替えるIIに対し、良品通過率は平均3.58ポイント向上。\nNN探索量はIIからさらに約24%削減。'),M,1380,CW,16,True)
        self.text(self.t('Versus the best fixed profile (III): mean normal-pass\ngains of 0.52 points for pairs and 1.04 points for triples.',
                         '最良固定profileのIIIに対する平均通過率の向上：\n2カテゴリで0.52ポイント、3カテゴリで1.04ポイント。'),RIGHT,1380,CW,15)

    def build(self):
        self.header();self.patchcore();self.proposal();self.setup();self.results()
        from build_patchcore_profile_switch_poster import RIGHT
        self.items=[i for i in self.items if not (
            i.get('x',0)>=RIGHT-6 and 781<=i.get('y',0)<=916
            and i['kind'] in ('text','rect'))]
        self.items.append(dict(kind='comparison_table',x=RIGHT,y=786,w=CW,h=146,values=[
            [self.t('System','構成'),'Profile',self.t('Memory bank','メモリバンク')],
            ['I',self.t('Default','標準設定'),self.t('Pooled','対象カテゴリを統合')],
            ['II',self.t('Default','標準設定'),self.t('Category switching','カテゴリ別に切り替え')],
            ['III',self.t('Best fixed profile','最良の専用設定に固定'),self.t('Category switching','カテゴリ別に切り替え')],
            [self.t('IV  Proposed','IV  提案'),self.t('Category switching','カテゴリ別に切り替え'),self.t('Category switching','カテゴリ別に切り替え')]
        ]))
        default = next(i for i in self.items if i.get('y') == 994)
        self.items.remove(default)
        self.text(self.t('Default: layer2+3, 14 × 14 patches, 768 dimensions.\nImage score: mean of the highest 1% of patch scores.',
                         '標準設定：layer2+3、14×14パッチ、768次元。\n画像スコア：パッチスコア上位1%の平均。'),default['x'],966,CW,13)
        # Recover vertical room for the hardware evidence without shrinking text.
        for item in self.items:
            if item['kind']=='arrow':
                item['points']=[(x,303+(y-303)*.9) for x,y in item['points']]
            elif item.get('y',0)>=303:
                item['y']=303+(item['y']-303)*.9
                if item['kind'] in ('table','node','rect','comparison_table'):
                    item['h']*=.9
        # Keep the top finding separate from the first section headings.
        for item in self.items:
            if item['kind']=='arrow':
                item['points']=[(x,y+10) for x,y in item['points']]
            elif 303<=item.get('y',0)<690:
                item['y']+=10
        self.hardware()
        # Remove the former takeaway band while keeping the footer in place.
        def close_header_gap(y):
            if 303 <= y < 690:
                return y - 50
            if 690 <= y < 940:
                return y - 35
            if 940 <= y < 1340:
                return y - 20
            return y
        for item in self.items:
            if item['kind'] == 'arrow':
                item['points'] = [(x, close_header_gap(y)) for x,y in item['points']]
            elif 'y' in item:
                item['y'] = close_header_gap(item['y'])
        self.n_layout()
        (BUILD / f'content_{"ja" if self.ja else "en"}.json').write_text(
            json.dumps(dict(width=W,height=H,items=self.items),ensure_ascii=False),encoding='utf-8')

    def n_layout(self):
        from build_patchcore_profile_switch_poster import M, RIGHT
        old = self.items
        assert len(old) == 57, 'Review section mapping after changing source content.'
        self.items = old[:4]
        # The two process diagrams keep their horizontal geometry and text sizes.
        for first,last,origin,target in [(4,16,M,263),(16,30,RIGHT,610)]:
            for source in old[first:last]:
                o = dict(source)
                if o['kind'] == 'arrow':
                    o['points'] = [(x-origin+M,target+(y-263)*.90) for x,y in o['points']]
                else:
                    o['x'] = o['x']-origin+M
                    o['y'] = target+(o['y']-263)*.90
                    if o['kind'] == 'node':
                        o['h'] *= .90
                        o['switchPart'] = first == 16 and 'CNN' not in o['text']
                self.items.append(o)

        def place(index,x,y,text=None):
            o = dict(old[index])
            o.update(x=x,y=y,w=CW)
            if o['kind'] == 'text':
                self.text(text if text is not None else o['text'],x,y,CW,o['size'],o['bold'],o['align'])
                if index in (43,49):
                    self.items[-1]['emphasis'] = True
            else:
                self.items.append(o)

        place(30,M,990)
        place(32,M,1040)
        place(33,M,1160)
        place(46,M,1250)
        place(45,M,1330)
        self.items[-1]['h'] = 170
        place(34,M,1530)

        place(35,RIGHT,263)
        place(37,RIGHT,313)
        place(38,RIGHT,375)
        place(39,RIGHT,445)
        self.result_charts(RIGHT,481,old[40]['values'])
        place(41,RIGHT,695)
        self.result_charts(RIGHT,731,old[42]['values'])
        place(43,RIGHT,940)
        place(44,RIGHT,1010)

        place(47,RIGHT,1120,self.t('5  FPGA estimates and next steps','5  FPGAの見積もりと実装計画'))
        place(49,RIGHT,1170,self.t('Estimated NN cycles: 70.7% fewer than I.\nCNN, memory stalls and switching overhead excluded.',
                                 'NN処理サイクルは構成I比70.7%減の見積もり。\nCNN処理・メモリ待ち・切り替え負荷を除く。'))
        place(50,RIGHT,1230)
        place(51,RIGHT,1300)
        place(52,RIGHT,1402,self.t('Bank-only storage, packed by feature dimension.\nFixed-width storage requires 5.57 MiB for both I and IV.',
                                'bank容量は有効次元だけを詰めて格納した値。固定幅格納ではI・IVとも5.57 MiB。'))
        place(53,RIGHT,1450)
        place(54,RIGHT,1540)
        self.items.extend(old[55:57])
        for o in self.items:
            if o['kind'] != 'text' or not (290 < o['y'] < 1620) or o['size'] >= 18:
                continue
            if abs(o['y']-461.45) < 1:
                continue
            content = o['text'].replace('\n', '' if self.ja else ' ')
            content = content.replace('Tables report', 'Charts report').replace('表は', 'グラフは')
            parts = re.split(r'(?<=。)', content) if self.ja else re.split(r'(?<=\.)\s+(?=[A-Z0-9])', content)
            o['bullets'] = [part.strip() for part in parts if part.strip()]
            if abs(o['y']-649.69) < 1:
                o['bullets'] = ([
                    '検品前に、既知の対象カテゴリに応じたprofileと正常メモリバンクを選択する。',
                    '提案するFPGA構成では、CNN・NN回路を共用する。'
                ] if self.ja else [
                    'Before inspection, the known category selects its feature profile and normal memory bank.',
                    'The proposed FPGA design shares one CNN and NN engine.'
                ])
            # Native paragraph wrapping provides a true hanging indent.
            o['h'] += 24
        if self.ja:
            for o in list(self.items):
                if not o.get('bullets'):
                    continue
                wrapped = []
                for part in o['bullets']:
                    self.text(part,0,0,o['w']-20,o['size'],o['bold'])
                    wrapped.append(self.items.pop()['text'])
                o['bullets'] = wrapped

    def result_charts(self,x,y,values):
        for column in (1,2):
            self.items.append(dict(kind='bar_chart',x=x+(column-1)*(CW+16)/2,
                y=y,w=(CW-16)/2,h=193,metric=column,
                values=[float(row[column])/(100 if column==1 else 1) for row in values[1:]],
                title=self.t('Normal pass (%)','良品通過率 (%)') if column==1 else self.t('NN work (I = 1)','NN探索量 (I = 1)')))

    def hardware(self):
        from build_patchcore_profile_switch_poster import M, RIGHT
        self.heading('5',self.t('FPGA cost model and implementation plan','FPGAのコスト見積もりと実装計画'),M,1340,W-2*M)
        self.text(self.t('Estimated NN cycles: 70.7% fewer than I, excluding CNN processing, memory stalls and switching overhead.',
                            'NN処理サイクルは構成I比70.7%減の見積もり（CNN処理・メモリ待ち・切り替え負荷を除く）。'),M,1390,W-2*M,16,True)
        self.text(self.t('Reference mean: 105 pairs and 120 triples.\n3,000 entries/category, 8-bit bank features.\nNN model: 64 dimensions/cycle, no memory stalls.',
                            '参照条件の平均：2カテゴリ105通り、3カテゴリ120通り。\n各カテゴリ3,000点、bank特徴値は8bit。\nNNモデル：64次元/サイクル、メモリ待ちなし。'),M,1450,CW,14)
        self.items.append(dict(kind='hardware_table',x=M,y=1508,w=CW,h=88,values=[
            [self.t('System','構成'),self.t('NN (Mcycles/image)','NN (百万cycle/枚)'),self.t('Bank (MiB)','bank容量 (MiB)')],
            [self.t('I  Common','I  共通'),'17.88','5.57'],
            [self.t('IV  Proposed','IV  提案'),'5.23','4.11']]))
        self.text(self.t('Bank-only storage, packed by feature dimension.\nFixed-width storage requires 5.57 MiB for both I and IV.',
                            '容量はbankのみ。有効次元だけを詰めて格納する。\n固定幅格納ではI・IVとも5.57 MiBとなる。'),M,1605,CW,13)
        self.text(self.t('One shared CNN and NN engine use profile registers and bank address control. Switching requires register updates rather than a new bitstream. Fewer bank reads and distance calculations may reduce latency and energy.',
                            '共通CNN・NN回路にprofile設定レジスタとbankアドレス制御を設ける。レジスタ更新で切り替え、回路再構成は行わない。bank読み出しと距離計算の削減による遅延・エネルギー低減を期待する。'),RIGHT,1450,CW,15)
        self.text(self.t('Implement the mode-controlled NN pipeline in RTL/HLS. Check 8-bit accuracy, switching overhead and memory stalls. Synthesis and board tests will quantify LUT/DSP/BRAM use, end-to-end latency and power.',
                            '今後はモード制御付きNNパイプラインをRTL/HLSで実装する。8bit化の精度、切り替え負荷、メモリ待ちを検証し、合成と実機評価でLUT・DSP・BRAM使用量、全体の遅延・消費電力を確認する。'),RIGHT,1540,CW,15)
        for item in self.items:
            if 1450 <= item.get('y',0) <= 1605:
                item['y'] -= 18
        self.rule(M,1643,W-2*M,.8)
        self.text('[1] K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR, 2022.<br/>[2] P. Bergmann et al., “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR, 2019.',M,1651,W-2*M,10)


if __name__=='__main__':
    BUILD.mkdir(parents=True,exist_ok=True)
    fonts()
    for ja in [False,True]:Capture(ja).build()
