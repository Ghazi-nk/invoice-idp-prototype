from pdf2image import convert_from_path

import os
import json
import fitz

from utils.config import TMP_DIR

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


def pdf_to_png_with_pymupdf(pdf_path: str, zoom: float = 3.0) -> list[str]:
    """
    Konvertiert PDF-Seiten zu PNG mit hoher Qualität (durch PyMuPDF).
    zoom=3.0 entspricht etwa 300 DPI.
    """
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise RuntimeError(f"Keine Seiten in PDF: {pdf_path}")

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    png_paths = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(zoom, zoom)  # zoom 2.0~200dpi, 3.0~300dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path = os.path.join(TMP_DIR, f"{base}_page{i + 1}.png")
        pix.save(png_path)
        png_paths.append(png_path)

    return png_paths
