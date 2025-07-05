from pdf2image import convert_from_path

import os
import json
import fitz

from utils.config import TMP_DIR

def pdf_to_png(pdf_path: str, dpi: int = 300) -> str:
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise RuntimeError(f"Keine Seiten in PDF: {pdf_path}")
    img = pages[0].convert("RGB")
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    png_path = os.path.join(TMP_DIR, f"{base}_page1.png")
    img.save(png_path, "PNG")
    return png_path

# turn a pdf with multiple pages into a list of pngs
def pdf_to_png_multiple(pdf_path: str, dpi: int = 300) -> list[str]:
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise RuntimeError(f"Keine Seiten in PDF: {pdf_path}")

    png_paths = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, img in enumerate(pages):
        img = img.convert("RGB")
        png_path = os.path.join(TMP_DIR, f"{base}_page{i + 1}.png")
        img.save(png_path, "PNG")
        png_paths.append(png_path)

    return png_paths

def extract_text_if_searchable(pdf_path: str) -> str:
    """
    Open the PDF at `pdf_path`, extract all text, and return it as a JSON string.
    If no text is found (i.e. likely a scanned/image PDF), returns "".
    Zwischenspeicherung analog zu ocr_png_to_text().
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    full_text = []
    for page in doc:
        text = page.get_text()
        full_text.append(text)

    text_combined = "\n".join(full_text).strip()

    # zwischenspeichern, falls Text vorhanden
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")
    if text_combined:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_combined)
        return json.dumps(text_combined, ensure_ascii=False)

    # kein Text-Layer
    return ""