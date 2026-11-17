"""Print editions of the final talk, preserving cue text and emphasis."""
from pathlib import Path
import re
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from build_patchcore_profile_switch_poster import fonts
import pypdfium2 as pdfium
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / 'tmp/pdfs/talk'
TMP.mkdir(parents=True, exist_ok=True)
fonts()

def markup(s):
    s = escape(s)
    s = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
    return re.sub(r'`([^`]+)`', r'\1', s)

def footer(c, doc):
    c.setFont('EN', 9)
    c.setFillColor(colors.HexColor('#555555'))
    c.drawString(42, 24, 'Shunya Yamada | Poster talk')
    c.drawRightString(A4[0]-42, 24, str(doc.page))

for lang, stem in [('en', 'patchcore_poster_talk_en_final'), ('ja', 'patchcore_poster_talk_ja_guide_final')]:
    lines = (ROOT/'output/poster'/f'{stem}.md').read_text(encoding='utf-8').splitlines()
    # Print only speech (or its translation), never rehearsal instructions.
    start = next(i for i,s in enumerate(lines) if s.startswith('## Opening' if lang=='en' else '## 冒頭'))
    lines = lines[start:]
    if lang == 'ja':
        stop = next(i for i,s in enumerate(lines) if s.startswith('## 数字'))
        lines = lines[:stop]
        lines = [s for s in lines if not s.startswith('**拾う言葉：**')]
        first_table = next(i for i,s in enumerate(lines) if s.startswith('|'))
        end_table = first_table
        while end_table < len(lines) and lines[end_table].startswith('|'):
            end_table += 1
        lines[first_table:end_table] = [
            '4種類のシステムを比較しました。',
            '構成Iは標準profileと、対象カテゴリを統合したbankを使います。',
            '構成IIは標準profileのまま、bankだけを切り替えます。',
            '構成IIIはbankを切り替えますが、各ペア・トリプルで最良のprofileに固定します。',
            '構成IVが提案手法です。カテゴリごとにprofileとbankの両方を切り替えます。',
        ]
    body = ParagraphStyle('body', fontName='EN' if lang=='en' else 'JA', fontSize=11 if lang=='en' else 10.5,
                          leading=14 if lang=='en' else 16, spaceAfter=2, wordWrap='CJK' if lang=='ja' else None)
    cue = ParagraphStyle('cue', fontName='JA', fontSize=9, leading=13, spaceAfter=7, textColor=colors.HexColor('#444444'),wordWrap='CJK')
    heading = ParagraphStyle('heading', parent=body, fontName='ENB' if lang=='en' else 'JAB',fontSize=14,leading=19,spaceBefore=8,spaceAfter=8,keepWithNext=True)
    title = ParagraphStyle('title', parent=heading,fontSize=18,leading=24)
    story=[]
    i=0
    while i<len(lines):
        line=lines[i].strip()
        i+=1
        if not line:
            story.append(Spacer(1,4));continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]);i+=1
            cells=[[Paragraph(markup(v.strip()),cue) for v in row.strip('|').split('|')] for row in rows if not re.match(r'^\|[- :|]+\|$',row)]
            table=Table(cells, colWidths=[(A4[0]-84)/len(cells[0])]*len(cells[0]),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.append(table);continue
        if line.startswith('# '):
            story.append(Paragraph(markup(line[2:]),title));continue
        if line.startswith('## '):
            if story:
                story.append(Spacer(1,12))
            continue
        if line.startswith('>'):
            continue
        if line.startswith('- '):
            story.append(Paragraph(markup(line[2:]),body,bulletText='・'));continue
        japanese=bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]',line))
        story.append(Paragraph(markup(line),cue if lang=='en' and japanese else body))
    path=OUT/f'{stem}.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42,
                      title='Poster Talk - Final Script' if lang=='en' else 'ポスター口頭発表 日本語ガイド',author='Shunya Yamada').build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(path)
    print(path, 'pages=',len(reader.pages),'text_chars=',sum(len(p.extract_text()) for p in reader.pages))
    doc=pdfium.PdfDocument(str(path))
    for n,page in enumerate(doc):
        page.render(scale=1.15).to_pil().save(TMP/f'{lang}_{n+1}.png')
