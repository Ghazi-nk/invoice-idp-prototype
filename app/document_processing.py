import os
from typing import Callable

from utils.pdf_utils import pdf_to_png_multiple, extract_text_if_searchable, pdf_to_png_with_pymupdf
from document_digitalization.ocr import easyocr_png_to_text, tesseract_png_to_text, paddleocr_png_to_text
from utils.llm_utils import ask_ollama_for_invoice_fields
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text

from utils.config import SAMPLE_PDF_PATH, SAMPLE_M_PDF_PATH


def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable[[str], str]) -> str | None:

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__

    print(f"\n[Info] Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'...")

    full_text = ""
    try:
        # 1. PDF in eine Liste von PNGs umwandeln
        png_pages = pdf_to_png_with_pymupdf(pdf_path)

        # 2. Jede Seite mit der übergebenen OCR-Funktion verarbeiten
        for i, png_page in enumerate(png_pages):
            page_num = i + 1
            full_text += f"\n--- Seite {page_num} ---\n"

            # Hier wird die als Argument übergebene Funktion aufgerufen
            extracted_text = ocr_function(png_page)
            full_text += extracted_text

            print(f"[Info] Seite {page_num} von '{base_name}.pdf' erfolgreich verarbeitet.")

        return full_text

    except Exception as e:
        print(
            f"[Error] Ein Fehler ist bei der Verarbeitung von '{base_name}.pdf' mit '{ocr_engine_name}' aufgetreten: {e}")
        return None


# --- Wrapper-Funktionen (optional, aber für Klarheit nützlich) ---
# Diese Funktionen machen den Aufruf für eine bestimmte Engine einfacher.

def easyocr_process_pdf(pdf_path: str) -> str | None:
    """Verarbeitet ein PDF spezifisch mit EasyOCR."""
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text)


def tesseract_process_pdf(pdf_path: str) -> str | None:
    """Verarbeitet ein PDF spezifisch mit Tesseract."""
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text)


def paddleocr_process_pdf(pdf_path: str) -> str | None:
    """Verarbeitet ein PDF spezifisch mit PaddleOCR."""
    return process_pdf_with_ocr(pdf_path, paddleocr_png_to_text)

def layoutlm_process_pdf(pdf_path: str) -> str | None:
    """Verarbeitet ein PDF spezifisch mit PaddleOCR."""
    return process_pdf_with_ocr(pdf_path, layoutlm_image_to_text)







if __name__ == "__main__":
    # Example usage
    #print(paddleocr_process_pdf(SAMPLE_M_PDF_PATH))
    print(layoutlm_process_pdf(SAMPLE_M_PDF_PATH))
    #print(tesseract_process_pdf(SAMPLE_M_PDF_PATH))
    #print(easyocr_process_pdf(SAMPLE_M_PDF_PATH))
    #print(doctr_pdf_to_text(SAMPLE_M_PDF_PATH))