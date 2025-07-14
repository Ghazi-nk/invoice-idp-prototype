from __future__ import annotations

import os
from typing import Callable, Dict, List

from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.ocr import (
    easyocr_png_to_text,
    tesseract_png_to_text,
    paddleocr_pdf_to_text,
)

from document_digitalization.pre_processing import preprocess_plain_text_output

from post_processing import finalize_extracted_fields, verify_and_correct_fields
from config import SAMPLE_PDF_PATH
from semantic_extraction import ollama_extract_invoice_fields
from document_digitalization.pdf_utils import pdf_to_png_with_pymupdf


def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable[[str], str]) -> List[str] | None:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__
    print(f"\n[Info] Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'…")
    pages_text: List[str] = []
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path) # switch pdf to png converter here!
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


def easyocr_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text)

def tesseract_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text)

def layoutlm_process_pdf(pdf_path: str) -> List[str] | None:
    return process_pdf_with_ocr(pdf_path, layoutlm_image_to_text)


_OCR_ENGINE_PDF_MAP: Dict[str, Callable[[str], List[str] | None]] = {
    "doctr": doctr_pdf_to_text,
    "easyocr": easyocr_process_pdf,
    "tesseract": tesseract_process_pdf,
    "paddleocr": paddleocr_pdf_to_text,
    "layoutlm": layoutlm_process_pdf,
}


def ocr_pdf(pdf_path: str, *, engine: str = "paddleocr") -> List[str]:
    engine = engine.lower()
    if engine not in _OCR_ENGINE_PDF_MAP:
        valid_options = ", ".join(_OCR_ENGINE_PDF_MAP.keys())
        raise ValueError(f"Unsupported OCR engine '{engine}'. Valid options are: {valid_options}.")
    ocr_function = _OCR_ENGINE_PDF_MAP[engine]
    result = ocr_function(pdf_path)
    return result or []


def clean_ocr_text(raw_text: str, *, engine: str) -> str:
    return preprocess_plain_text_output(raw_text)


def extract_invoice_fields_from_pdf(pdf_path: str, *, engine: str = "paddleocr", clean: bool = True) -> Dict:
    """Full pipeline: PDF ➜ OCR ➜ clean ➜ LLM ➜ verify & correct ➜ post-process ➜ final dict."""
    pages_raw_text = ocr_pdf(pdf_path, engine=engine)
    print(
        f"[Info] OCR für '{os.path.basename(pdf_path)}' mit '{engine}' abgeschlossen. {len(pages_raw_text)} Seiten gefunden.")

    final_text_parts: List[str] = []
    if clean:
        for i, page_text in enumerate(pages_raw_text):
            cleaned_page_text = clean_ocr_text(page_text, engine=engine)
            final_text_parts.append(cleaned_page_text)
            print(f"[Info] Seite {i + 1} bereinigt.")
        if final_text_parts:
            print(f"[Info] Gekürzte Vorschau des bereinigten Textes: '{(' '.join(final_text_parts))[:200]}...'")
    else:
        final_text_parts = pages_raw_text

    llm_output = ollama_extract_invoice_fields(final_text_parts)

    full_text_for_verification = "\n".join(final_text_parts)
    corrected_dict = verify_and_correct_fields(llm_output, full_text_for_verification)

    final_dict = finalize_extracted_fields(corrected_dict)

    return final_dict


def get_available_engines() -> List[str]:
    return list(_OCR_ENGINE_PDF_MAP.keys())

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