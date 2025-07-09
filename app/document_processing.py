"""pipeline.py

Rewritten orchestration layer that re‑uses **all existing helpers** and exposes
five clean, reusable functions:

1. `pdf_to_images`                  – PDF ➜ list[PNG] (multi‑page, uses TMP_DIR)
2. `ocr_pdf`                        – run one selected OCR engine on the PDF
3. `clean_ocr_text`                 – normalise & regex‑clean the OCR output
4. `extract_invoice_fields_from_text` – ask Ollama & parse JSON
5. `extract_invoice_fields_from_pdf`  – one‑call end‑to‑end helper

The old wrappers (`easyocr_process_pdf`, `tesseract_process_pdf`, …) are kept
unchanged for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List

from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.ocr import (
    easyocr_png_to_text,
    tesseract_png_to_text,
    paddleocr_png_to_text,
)

from pre_processing import (
    preprocess_doctr_output,
    preprocess_layoutlm_output,
    preprocess_plain_text_output,
)
from utils.config import SAMPLE_PDF_PATH
from utils.llm_utils import ollama_extract_invoice_fields
from utils.pdf_utils import pdf_to_png_with_pymupdf



# =============================================================================
# --- Document digitalization -------------------------------------------------
# =============================================================================

def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable[[str], str]) -> str | None:
    # ... (Diese Funktion bleibt unverändert)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__
    print(f"\n[Info] Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'…")
    full_text: str = ""
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path)
        for i, png_page in enumerate(png_pages):
            page_num = i + 1
            # WICHTIG: Die "--- Seite X ---" Header werden von process_pdf_with_ocr
            # hinzugefügt und von preprocess_plain_text_output wieder entfernt.
            # Das ist konsistent.
            full_text += f"\n--- Seite {page_num} ---\n"
            extracted_text = ocr_function(png_page)
            full_text += extracted_text
            print(f"[Info] Seite {page_num} von '{base_name}.pdf' erfolgreich verarbeitet.")
        return full_text
    except Exception as e:
        print(
            f"[Error] Ein Fehler ist bei der Verarbeitung von '{base_name}.pdf' mit '{ocr_engine_name}' aufgetreten: {e}")
        return None


# --- Original per‑engine convenience wrappers --------------------------------
def easyocr_process_pdf(pdf_path: str) -> str | None:
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text)


def tesseract_process_pdf(pdf_path: str) -> str | None:
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text)


def paddleocr_process_pdf(pdf_path: str) -> str | None:
    return process_pdf_with_ocr(pdf_path, paddleocr_png_to_text)


def layoutlm_process_pdf(pdf_path: str) -> str | None:
    return process_pdf_with_ocr(pdf_path, layoutlm_image_to_text)


# =============================================================================
# --- Five reusable pipeline building blocks ---------------------------------
# =============================================================================

# 1) PDF -> images
def pdf_to_images(pdf_path: str) -> List[str]:
    return pdf_to_png_with_pymupdf(pdf_path)


# 2) OCR step
_OCR_ENGINE_MAP: Dict[str, Callable[[str], str]] = {
    "easyocr": easyocr_png_to_text,
    "tesseract": tesseract_png_to_text,
    "paddleocr": paddleocr_png_to_text,
    "layoutlm": layoutlm_image_to_text,
}


def ocr_pdf(pdf_path: str, *, engine: str = "tesseract") -> str:
    engine = engine.lower()
    if engine == "doctr":
        return doctr_pdf_to_text(pdf_path)
    if engine not in _OCR_ENGINE_MAP:
        raise ValueError(
            f"Unsupported OCR engine '{engine}'. Valid options are: {', '.join(_OCR_ENGINE_MAP)} or 'doctr'.")

    if engine == "layoutlm":
        return layoutlm_process_pdf(pdf_path) or ""
    return process_pdf_with_ocr(pdf_path, _OCR_ENGINE_MAP[engine]) or ""


# 3) Cleaning / normalisation --- NEU ---
def clean_ocr_text(raw_text: str, *, engine: str) -> str:
    """Wählt den passenden Pre-Processing-Flow basierend auf der Engine."""
    engine = engine.lower()
    if engine == "doctr":
        return preprocess_doctr_output(raw_text)
    if engine == "layoutlm":
        return raw_text

    # Default für Tesseract, EasyOCR, PaddleOCR etc.
    return preprocess_plain_text_output(raw_text)


# 4) LLM post-processing
def extract_invoice_fields_from_text(clean_text: str) -> Dict:
    return ollama_extract_invoice_fields(clean_text)


# 5) End-to-end helper --- AKTUALISIERT ---
def extract_invoice_fields_from_pdf(pdf_path: str, *, engine: str = "easyocr", clean: bool = True) -> Dict:
    """Full pipeline: PDF ➜ OCR ➜ clean ➜ Ollama ➜ dict."""
    raw_text = ocr_pdf(pdf_path, engine=engine)
    print(f"[Info] OCR für '{os.path.basename(pdf_path)}' mit '{engine}' abgeschlossen.")
    final_text = raw_text
    # Die Engine wird jetzt an die clean-Funktion weitergereicht
    if clean:
        final_text = clean_ocr_text(raw_text, engine=engine)
        print(f"[Info] Bereinigter Text: '{final_text[:200]}...'")  # Log-Ausgabe gekürzt
    return extract_invoice_fields_from_text(final_text)


# ... (Der Rest der Datei, inklusive __main__ Block, kann gleich bleiben)
# Der __main__ Block ruft bereits extract_invoice_fields_from_pdf mit der Engine auf,
# daher sind dort keine Änderungen nötig.

if __name__ == "__main__":
    print("\n=== Pipeline demo (Tesseract) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="tesseract")

    print("\n=== Pipeline demo (LayoutLM) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="layoutlm")

    print("\n=== Pipeline demo (Doctr) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="doctr")

    print("\n=== Pipeline demo (EasyOCR) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="easyocr")

    print("\n=== Pipeline demo (PaddleOCR) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="paddleocr")
    print("\n=== Pipeline demo (all engines) completed ===")