from pdf2image import convert_from_path
from PIL import Image
import os

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