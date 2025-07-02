import os
from utils.pdf_utils import pdf_to_png
from ocr import ocr_png_to_text
from utils.llm_utils import ask_ollama_for_invoice_fields
from utils.searchable_pdf import extract_text_if_searchable
from layoutlmv3_png2txt import layoutlm_image_to_text
from doctr_pdf2txt import doctr_pdf_to_text

from utils.config import SAMPLE_FOLDER


def process_single_pdf_ocr(pdf_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        png = pdf_to_png(pdf_path)
        text = ocr_png_to_text(png)

        return text
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None


def process_single_pdf_layoutlm(pdf_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        png = pdf_to_png(pdf_path)
        text = layoutlm_image_to_text(png)

        return text
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None


def process_single_pdf_doctr(pdf_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    try:
        text = doctr_pdf_to_text(pdf_path)
        return text
    except Exception as e:
        print(f"[Error] {base}: {e}")
        return None


def process_multiple_pdfs(folder_path: str, pipeline_type: str, process_searchable: bool = False) -> list[dict | None]:
    results = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, fname)

        if not process_searchable or extract_text_if_searchable(pdf_path) == "":

            if pipeline_type == "ocr":
                text = process_single_pdf_ocr(pdf_path)
            elif pipeline_type == "layoutlm":
                text = process_single_pdf_layoutlm(pdf_path)
            elif pipeline_type == "doctr":
                text = process_single_pdf_doctr(pdf_path)
            else:
                print(f"[Error] Unbekannter Pipeline-Typ: {pipeline_type}")
                continue

        else:
            text = extract_text_if_searchable(pdf_path)

        result = ask_ollama_for_invoice_fields(text)

        results.append(result)

    return results



if __name__ == "__main__":
    # ── STATIC EXAMPLE ARGS ───────────────────────────────────────────────────
    pipeline_type     = "doctr"                      # one of: "ocr", "layoutlm", "doctr"
    process_searchable = False                       # True = use embedded text when available
    # ────────────────────────────────────────────────────────────────────────

    results = process_multiple_pdfs(
        folder_path=SAMPLE_FOLDER,
        pipeline_type=pipeline_type,
        process_searchable=process_searchable
    )

    for idx, invoice_fields in enumerate(results, start=1):
        print(f"\n--- Result {idx} ---")
        print(invoice_fields)
