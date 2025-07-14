# FILE: utils/pdf_utils.py

import base64
import tempfile
from contextlib import contextmanager
from pdf2image import convert_from_path
import os
import json
import fitz

from config import TMP_DIR

@contextmanager
def save_base64_to_temp_pdf(base64_string: str):
    """
    Decodes a base64 string and saves it to a temporary PDF file.
    """
    temp_file_path = None
    try:
        pdf_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_file_path = temp_file.name
        yield temp_file_path
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def encode_image_to_base64(image_path: str) -> str:
    """
    Reads an image file and encodes its content into a base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# ... (The rest of your existing functions remain unchanged) ...

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
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")
    if text_combined:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_combined)
        return json.dumps(text_combined, ensure_ascii=False)
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
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path = os.path.join(TMP_DIR, f"{base}_page{i + 1}.png")
        pix.save(png_path)
        png_paths.append(png_path)
    return png_paths