from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "patchcore_fpga_professor_report.pdf"
OUT = ROOT / "tmp" / "pdf_render"


def main():
    print("size", PDF.stat().st_size)
    with pdfplumber.open(PDF) as pdf:
        print("pages", len(pdf.pages))
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    for key in ["PatchCore", "MVTec AD", "FPGA", "hazelnut", "0.031596", "96.84", "M95"]:
        print(key, key in text)
    print("replacement", "\ufffd" in text)

    OUT.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(PDF))
    thumbs = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=0.7)
        im = bitmap.to_pil().convert("RGB")
        path = OUT / f"patchcore_report_page_{i + 1:02d}.png"
        im.save(path)
        stat = ImageStat.Stat(im.convert("L"))
        print(path, im.size, "mean", round(stat.mean[0], 2))
        thumbs.append(im.resize((int(im.width * 0.35), int(im.height * 0.35))))

    w = max(t.width for t in thumbs)
    h = sum(t.height for t in thumbs)
    sheet = Image.new("RGB", (w, h), "white")
    y = 0
    for thumb in thumbs:
        sheet.paste(thumb, (0, y))
        y += thumb.height
    sheet_path = OUT / "patchcore_report_contact_sheet.png"
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main()
