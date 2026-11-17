"""Two-page bilingual question-and-answer print sheet."""
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,KeepTogether,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from build_patchcore_profile_switch_poster import fonts
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
DATA=[
('新しさ','What is the main contribution?',
'Coreset selection and feature tuning are existing approaches. We evaluate the added benefit of switching profiles, beyond bank switching alone.',
'既存のbank削減技術そのものではなく、profileも切り替える追加効果と共有FPGA構成が主題。世界初とは断言しない。'),
('両方を切り替える理由','Why switch both the profile and the bank?',
'Different categories need different feature settings. Bank switching selects the normal examples. Profile switching also changes how we extract and compare features.',
'bank＝比較する正常例。profile＝特徴の取り出し方・比較方法。'),
('profileの内容','What does a profile include?',
'It includes the feature layer, feature dimension, patch grid, score aggregation, and decision threshold.',
'特徴層・次元数・パッチ配置・スコア集約・しきい値。別のCNN全体に切り替えるわけではない。'),
('カテゴリの取得','How do you identify the product category?',
'We assume that the category is known before inspection. The production schedule can provide it. We do not classify the category automatically.',
'カテゴリは検品前に既知。画像からカテゴリを自動判定する仕組みではない。'),
('用途','What is a practical application?',
'A production line that handles different products in separate batches. The system changes its profile and bank when the product type changes.',
'製品をロット単位で切り替えるライン。ランダム混在より、切り替えが事前に分かる運用。'),
('coresetと精度','Does coreset selection preserve accuracy?',
'It selects representative normal features to make the bank smaller. It aims to preserve performance, but does not guarantee it. We use three thousand vectors per category.',
'正常特徴を広く代表する点を選ぶ既存技術。精度無劣化の保証ではない。今回は各カテゴリ3,000点に統一。'),
('公平性・bank点数','Is the improvement simply caused by a smaller bank?',
'All systems keep three thousand entries per category. System one searches the pooled bank. The others search only the matching category bank.',
'ABは全構成で計6,000点、ABCは9,000点。Iも対象カテゴリのみ。II→IVの削減は点数削減ではなく特徴次元・パッチ数などによる。'),
('24%の比較基準','What is the reference for the twenty-four percent reduction?',
'It is compared with system two, which switches only the bank. The charts use system one as the reference.',
'24%はII比。AB：1 − 0.378/0.500＝24.4%。ABC：約24.4%。グラフの倍率はI基準。時間・電力の実測ではない。'),
('3.58ポイントの意味','What does the improvement of three point five eight points mean?',
'Compared with system two, mean normal pass increases from eighty-one point nine five percent to eighty-five point five three percent.',
'81.95% → 85.53%＝平均良品通過率が3.58ポイント向上。分類正解率ではない。'),
('欠陥誤通過3%','How did you apply the three-percent limit?',
'We did not remove images. We chose thresholds with at most three percent defect pass in the evaluation. This is a comparison condition, not a safety guarantee.',
'条件を満たすしきい値の中から良品通過率が最大の点を選択。評価データ上の値で、未知データでの3%以下を保証しない。'),
('最良固定profileとの比較','Why not use the best fixed profile?',
'We included that as system three. Our method achieved higher mean normal pass and lower mean N N work, but not in every combination.',
'IIIは対象カテゴリの専用profile候補から最良の固定設定を選ぶ。IVの平均良品通過率はABで+0.52、ABCで+1.04ポイント。'),
('profileの選定','How did you select the profiles?',
'We tested each category separately. We kept profiles no more than two points below default normal pass, then minimized patch count times feature dimension.',
'試した候補内で低下2ポイント以内→パッチ数×次元数を最小化。選定時12,000点／比較時3,000点。最高精度だけで選んだのではない。'),
('過学習・汎化','Could profile selection overfit the dataset?',
'Yes, that is a risk. Testing all category combinations does not prove performance on unseen images. We need independent data to validate the settings and thresholds.',
'105ペア・455トリプルは組み合わせの網羅性。独立した560データセットではない。新しい現場への汎化は未証明。'),
('FPGA実装状況','Have you implemented it on an FPGA?',
'Not yet. We have software results and a theoretical N N cost estimate. Hardware implementation and measurement are the next steps.',
'現状はソフト評価と理論見積もり。実装・実測は今後。'),
('FPGAの必要性','Could a CPU also switch profiles?',
'Yes. Our FPGA design reuses one CNN and N N engine, with profile registers and bank address control. Speed and energy benefits still need hardware tests.',
'切り替え自体はCPUでも可能。FPGAでは共有パイプラインの構成を狙う。CPUより速い・省電力、回路面積が減った、とは未実測で言わない。'),
('70.7%と全体速度','Is the whole system faster by seventy point seven percent?',
'No. It is a reduction in estimated N N cycles only. It excludes CNN processing, memory stalls, and switching overhead.',
'70.7%はI比、17.88M→5.23Mサイクル。参照は105ペア＋120トリプル（検知評価は105＋455）。8bit・64次元/サイクルのモデル。'),
('bank容量の減少','Why is bank storage smaller with the same entry count?',
'Some profiles use fewer feature dimensions. Packed storage uses less space per entry. With fixed-width storage, both systems need the same bank capacity.',
'点数は同じでも1点あたりの次元数が違う。見積もりは5.57→4.11 MiB。固定幅保存では両者5.57 MiB。CNNなどは含まない。'),
('切り替え負荷','Could switching overhead cancel the benefit?',
'Yes, especially with frequent switching. We plan register updates and bank address control, but memory transfer costs must also be measured.',
'bankがメモリにあればアドレス変更。転送が必要なら別途負荷が発生。ロット内の処理で償却できるかを今後測る。新bitstreamは不要の設計。'),
]
fonts()
title=ParagraphStyle('title',fontName='JAB',fontSize=12,leading=16,spaceAfter=7)
label=ParagraphStyle('label',fontName='JAB',fontSize=9,leading=12,spaceAfter=2)
q=ParagraphStyle('q',fontName='ENB',fontSize=9,leading=11,spaceAfter=2)
a=ParagraphStyle('a',fontName='EN',fontSize=10,leading=12.5,spaceAfter=2)
ja=ParagraphStyle('ja',fontName='JA',fontSize=8,leading=10.5,wordWrap='CJK',textColor=colors.HexColor('#444444'))
story=[]
for i,(name,question,answer,note) in enumerate(DATA,1):
    if i in (1,10):
        if i==10:story.append(PageBreak())
        story.append(Paragraph('想定質問・回答カンペ  /  '+('研究内容・結果' if i==1 else '評価条件・FPGA'),title))
    story.append(KeepTogether([Paragraph(f'{i:02d}  {escape(name)}',label),
        Paragraph('Q. '+escape(question),q),Paragraph('A. '+escape(answer),a),
        Paragraph(escape(note),ja),Spacer(1,5)]))
    if i==9:
        story.append(Paragraph('Could you say that again more slowly, please?<br/>Do you mean the software results or the FPGA estimate?',q))
    if i==len(DATA):
        story.append(Paragraph('We have not evaluated that yet.<br/>I would need to check the experimental details to answer accurately.',q))
def footer(c,d):
    c.setFont('EN',8)
    c.drawRightString(A4[0]-32,18,str(d.page))
out=ROOT/'output/pdf/patchcore_poster_qa_cheatsheet.pdf'
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=32,rightMargin=32,topMargin=25,bottomMargin=28,
    title='PatchCore Poster Q&A',author='Shunya Yamada').build(story,onFirstPage=footer,onLaterPages=footer)
pages=len(PdfReader(out).pages)
assert pages==2,pages
print(out,'pages=',pages)
md=['# 想定質問と回答例', '', '英語の回答と日本語の補足。提出済みポスターと最終原稿に対応。', '']
for i,(name,question,answer,note) in enumerate(DATA,1):
    md.extend([f'## {i:02d} {name}', '', f'**Q. {question}**', '', f'A. {answer}', '', note, ''])
(ROOT/'output/poster/patchcore_poster_qa_cheatsheet.md').write_text('\n'.join(md),encoding='utf-8')
doc=pdfium.PdfDocument(str(out))
for i,page in enumerate(doc):
    page.render(scale=1.2).to_pil().save(ROOT/f'tmp/pdfs/talk/qa_{i+1}.png')
