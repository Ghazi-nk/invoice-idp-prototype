import os
from pdf_utils import pdf_to_png
from ocr_utils import ocr_png_to_text
from llm_utils import ask_ollama_for_invoice_fields
from searchable_pdf import extract_text_if_searchable


def process_single_pdf(pdf_path: str, process_searchable: bool) -> dict | None:
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