import os
import json
import fitz  # PyMuPDF

from config import TMP_DIR

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
