"""pipeline.py

Rewritten orchestration layer that re‑uses **all existing helpers** and exposes
five clean, reusable functions:

1. `pdf_to_images`                  – PDF ➜ list[PNG] (multi‑page, uses TMP_DIR)
2. `ocr_pdf`                        – run one selected OCR engine on the PDF, returns text per page
3. `clean_ocr_text`                 – normalise & regex‑clean a single string of OCR output
4. `extract_invoice_fields_from_text` – ask Ollama & parse JSON from a list of page texts
5. `extract_invoice_fields_from_pdf`  – one‑call end‑to‑end helper (handles page-by-page processing)

The old wrappers (`easyocr_process_pdf`, `tesseract_process_pdf`, …) are kept
unchanged for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List

# Correctly import the newly refactored doctr function
from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.ocr import (
    easyocr_png_to_text,
    tesseract_png_to_text,
    paddleocr_png_to_text,
)

from utils.pre_processing import (
    preprocess_plain_text_output,
)
from utils.config import SAMPLE_PDF_PATH
from utils.llm_utils import ollama_extract_invoice_fields
from utils.pdf_utils import pdf_to_png_with_pymupdf


# =============================================================================
# --- Document digitalization -------------------------------------------------
# =============================================================================

def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable[[str], str]) -> List[str] | None:
    """
    Processes each page of a PDF with a given OCR function.
    (This function is for engines that work on a page-image-by-page-image basis).
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__
    print(f"\n[Info] Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'…")

    pages_text: List[str] = []
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path)
        for i, png_page in enumerate(png_pages):
            page_num = i + 1
            extracted_text = ocr_function(png_page)
            pages_text.append(extracted_text)
            print(f"[Info] Seite {page_num} von '{base_name}.pdf' erfolgreich verarbeitet.")
        return pages_text
    except Exception as e:
        print(
            f"[Error] Ein Fehler ist bei der Verarbeitung von '{base_name}.pdf' mit '{ocr_engine_name}' aufgetreten: {e}")
        return None


# --- Original per‑engine convenience wrappers are now less critical but kept for compatibility ---
def easyocr_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text)


def tesseract_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text)


def paddleocr_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, paddleocr_png_to_text)


def layoutlm_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, layoutlm_image_to_text)


# =============================================================================
# --- Five reusable pipeline building blocks ---------------------------------
# =============================================================================

# 1) PDF -> images
def pdf_to_images(pdf_path: str) -> List[str]:
    return pdf_to_png_with_pymupdf(pdf_path)


# 2) OCR step --- CLEANED UP ---
# Doctr is now a standard PDF-to-List[str] function.
# Other engines are image-to-str functions wrapped by process_pdf_with_ocr.
_OCR_ENGINE_PDF_MAP: Dict[str, Callable[[str], List[str] | None]] = {
    "doctr": doctr_pdf_to_text,
    "easyocr": easyocr_process_pdf,
    "tesseract": tesseract_process_pdf,
    "paddleocr": paddleocr_process_pdf,
    "layoutlm": layoutlm_process_pdf,
}


def ocr_pdf(pdf_path: str, *, engine: str = "easyocr") -> List[str]:
    """
    Runs a selected OCR engine on a PDF and returns a list of text per page.
    """
    engine = engine.lower()
    if engine not in _OCR_ENGINE_PDF_MAP:
        valid_options = ", ".join(_OCR_ENGINE_PDF_MAP.keys())
        raise ValueError(f"Unsupported OCR engine '{engine}'. Valid options are: {valid_options}.")

    # All engines now conform to the same interface: pdf_path -> List[str]
    ocr_function = _OCR_ENGINE_PDF_MAP[engine]
    result = ocr_function(pdf_path)
    return result or []


# 3) Cleaning / normalisation
def clean_ocr_text(raw_text: str, *, engine: str) -> str:
    """
    Wählt den passenden Pre-Processing-Flow basierend auf der Engine.
    NOTE: The 'doctr' text now contains bbox data. The preprocessor might
    need adjustment in the future if this data is to be handled specially.
    """
    engine = engine.lower()
    if engine == "doctr":
        # The preprocess_doctr_output might need to be reviewed to handle
        # the new text + bbox format. For now, it's passed through.
        #return preprocess_doctr_output(raw_text) #todo: fix and use doctr preprocessor
        return raw_text
    if engine == "layoutlm":
        return preprocess_plain_text_output(raw_text)

    # Default für Tesseract, EasyOCR, PaddleOCR etc.
    return preprocess_plain_text_output(raw_text)


# 4) LLM post-processing
def extract_invoice_fields_from_text(cleaned_pages: List[str]) -> Dict:
    """Asks the LLM to extract fields from a list of page texts."""
    return ollama_extract_invoice_fields(cleaned_pages)


# 5) End-to-end helper
def extract_invoice_fields_from_pdf(pdf_path: str, *, engine: str = "easyocr", clean: bool = True) -> Dict:
    """Full pipeline: PDF ➜ OCR (per page) ➜ clean (per page) ➜ Ollama ➜ dict."""
    # 1. Get raw text for each page
    pages_raw_text = ocr_pdf(pdf_path, engine=engine)
    print(f"[Info] OCR für '{os.path.basename(pdf_path)}' mit '{engine}' abgeschlossen. {len(pages_raw_text)} Seiten gefunden.")

    final_text_parts: List[str] = []
    # 2. Clean text for each page individually if requested
    if clean:
        for i, page_text in enumerate(pages_raw_text):
            cleaned_page_text = clean_ocr_text(page_text, engine=engine)
            final_text_parts.append(cleaned_page_text)
            print(f"[Info] Seite {i+1} bereinigt.")
        # Quick log of the cleaned content
        if final_text_parts:
             print(f"[Info] Gekürzte Vorschau des bereinigten Textes: '{(' '.join(final_text_parts))}...'")
    else:
        final_text_parts = pages_raw_text

    # 3. Extract fields using the LLM with the list of cleaned page texts
    return extract_invoice_fields_from_text(final_text_parts)

def get_available_engines() -> List[str]:
    """Returns a list of all supported OCR engine names."""
    return list(_OCR_ENGINE_PDF_MAP.keys())


# ... (The __main__ block remains unchanged)

if __name__ == "__main__":

    print("\n=== Pipeline demo (Doctr) ===")
    fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="doctr")

    # print("\n=== Pipeline demo (Tesseract) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="tesseract")

    # print("\n=== Pipeline demo (LayoutLM) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="layoutlm")
    #
    # print("\n=== Pipeline demo (EasyOCR) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="easyocr")
    #
    # print("\n=== Pipeline demo (PaddleOCR) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="paddleocr")
    # print("\n=== Pipeline demo (all engines) completed ===")