"""Preserve native bullet properties omitted by the current PPTX exporter."""
import json
import copy
import sys
import zipfile
from xml.etree import ElementTree as E

path, content = sys.argv[1:]
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
for prefix, uri in ns.items():
    E.register_namespace(prefix, uri)
items = json.load(open(content, encoding='utf-8'))['items']
lookup = {o['text'][:65]: o for o in items if o.get('bullets')}
# Short spoken anchors, not entire paragraphs, carry the visual emphasis.
anchors = [
    'PatchCore [1] detects defects', 'A memory bank',
    'Before inspection', 'The proposed FPGA design shares', 'Profile:',
    'Dataset:', 'Tasks:', 'Profile selection:',
    'Operating point:', 'Normal pass:', 'NN work =',
    'Versus bank switching alone (II):', 'Estimated NN cycles:',
    'Switching requires register updates', 'Check 8-bit accuracy',
    'PatchCore [1]は', 'メモリバンクとは', '検品前に', '提案するFPGA構成では',
    'profile：', 'データ：', '組み合わせ：', '専用profileの決定：',
    '評価条件：', '良品通過率：', 'NN探索量＝',
    'bankのみ切り替えるIIに対し', 'NN処理サイクルは',
    'レジスタ更新で切り替え', '8bit化の精度',
]
with zipfile.ZipFile(path) as z:
    files = {n: z.read(n) for n in z.namelist()}
root = E.fromstring(files['ppt/slides/slide1.xml'])
for shape in root.findall('.//p:sp', ns):
    prop = shape.find('p:nvSpPr/p:cNvPr', ns)
    if prop is None or prop.get('name') not in lookup:
        continue
    for para in shape.findall('p:txBody/a:p', ns):
        paragraph_text = ''.join(t.text or '' for t in para.findall('.//a:t', ns))
        anchor = next((a for a in anchors if paragraph_text.startswith(a)), None)
        pr = para.find('a:pPr', ns)
        if pr is None:
            pr = E.Element('{%s}pPr' % ns['a'])
            para.insert(0, pr)
        pr.set('marL', '316992')
        pr.set('indent', '-243840')
        for child in list(pr):
            if child.tag.rsplit('}',1)[-1] in ('buNone','buChar','buAutoNum','buSzPct','buFont','buFontTx'):
                pr.remove(child)
        # DrawingML requires bullet properties before tabLst/defRPr/extLst.
        insert_at = next((i for i,c in enumerate(pr) if c.tag.rsplit('}',1)[-1]
                          in ('tabLst','defRPr','extLst')), len(pr))
        bullet_properties = []
        if anchor:
            bullet_properties.append(E.Element('{%s}buSzPct' % ns['a'], {'val': '150000'}))
        bullet_properties.append(E.Element('{%s}buFont' % ns['a'], {'typeface': 'Arial'}))
        bullet_properties.append(E.Element('{%s}buChar' % ns['a'], {'char': '\u2022'}))
        for offset, child in enumerate(bullet_properties):
            pr.insert(insert_at + offset, child)
        remaining = len(anchor) if anchor else 0
        for run in list(para.findall('a:r', ns)):
            t = run.find('a:t', ns)
            if not remaining or t is None:
                continue
            value = t.text or ''
            take = min(remaining, len(value))
            if take < len(value):
                suffix = copy.deepcopy(run)
                suffix.find('a:t', ns).text = value[take:]
                para.insert(list(para).index(run)+1, suffix)
                t.text = value[:take]
            style = run.find('a:rPr', ns)
            if style is None:
                style = E.Element('{%s}rPr' % ns['a'])
                run.insert(0, style)
            style.set('b', '1')
            remaining -= take
        # Soft line breaks must be a:br, not newlines inside a:t (PowerPoint
        # otherwise repeats the bullet on continuation lines).
        for run in list(para.findall('a:r', ns)):
            t = run.find('a:t', ns)
            if t is None or '\n' not in (t.text or ''):
                continue
            pieces = t.text.split('\n')
            index = list(para).index(run)
            para.remove(run)
            for i, piece in enumerate(pieces):
                if i:
                    para.insert(index, E.Element('{%s}br' % ns['a']))
                    index += 1
                segment = copy.deepcopy(run)
                segment.find('a:t', ns).text = piece
                para.insert(index, segment)
                index += 1
files['ppt/slides/slide1.xml'] = E.tostring(root, encoding='utf-8', xml_declaration=True)
with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, payload in files.items():
        z.writestr(name, payload)
