import os
from utils.pdf_utils import pdf_to_png
from tesseract_ocr import ocr_png_to_text
from utils.llm_utils import ask_ollama_for_invoice_fields
from utils.searchable_pdf import extract_text_if_searchable
from layoutlmv3_png2txt import layoutlm_image_to_text
from doctr_pdf2txt import doctr_pdf_to_text


def process_single_pdf_ocr(pdf_path: str, process_searchable: bool = False) -> dict | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        if process_searchable:
            text = extract_text_if_searchable(pdf_path)
            if text== "":
                png = pdf_to_png(pdf_path)
                text = ocr_png_to_text(png)
        else:
            png = pdf_to_png(pdf_path)
            text = ocr_png_to_text(png)
        data = ask_ollama_for_invoice_fields(text)
        return data
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None


def process_single_pdf_layoutlm(pdf_path: str, process_searchable: bool = False) -> dict | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        if process_searchable:
            text = extract_text_if_searchable(pdf_path)
            if text== "":
                png = pdf_to_png(pdf_path)
                text = layoutlm_image_to_text(png)
        else:
            png = pdf_to_png(pdf_path)
            text = layoutlm_image_to_text(png)
        data = ask_ollama_for_invoice_fields(text)
        return data
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None


def process_single_pdf_doctr(pdf_path: str, process_searchable: bool = False) -> dict | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        if process_searchable:
            text = extract_text_if_searchable(pdf_path)
            if text== "":
                text = doctr_pdf_to_text(pdf_path)
        else:
            text = doctr_pdf_to_text(pdf_path)
        data = ask_ollama_for_invoice_fields(text)
        return data
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None