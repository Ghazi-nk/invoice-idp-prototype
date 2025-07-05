import os
from utils.pdf_utils import pdf_to_png, pdf_to_png_multiple, extract_text_if_searchable, pdf_to_png_with_pymupdf
from document_digitalization.ocr import easyocr_png_to_text
from utils.llm_utils import ask_ollama_for_invoice_fields
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text

from utils.config import SAMPLE_PDF_PATH, SAMPLE_M_PDF_PATH

def easyocr_process_pdf_invoice(pdf_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Verarbeite {base}.pdf...")
    text = ""
    try:
        pngs = pdf_to_png_with_pymupdf(pdf_path)
        for png in pngs:
            # Append page number to the text
            text += "\npage number " + str(pngs.index(png) + 1) + "\n"
            text += easyocr_png_to_text(png)
            print(f"[Info] {base}.pdf Seite {pngs.index(png) + 1} verarbeitet.") #todo: remove
            print(f"[Info] {base}.pdf Seite {pngs.index(png) + 1} Text: {text}...")  #todo: remove
        print(f"text: {text}...")  #todo: remove
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


# def process_multiple_pdfs(folder_path: str, pipeline_type: str, process_searchable: bool = False) -> list[dict | None]:
#     results = []
#     for fname in os.listdir(folder_path):
#         if not fname.lower().endswith(".pdf"):
#             continue
#         pdf_path = os.path.join(folder_path, fname)
#
#         if not process_searchable or extract_text_if_searchable(pdf_path) == "":
#
#             if pipeline_type == "ocr":
#                 text = process_single_pdf_easyocr(pdf_path)
#             elif pipeline_type == "layoutlm":
#                 text = process_single_pdf_layoutlm(pdf_path)
#             elif pipeline_type == "doctr":
#                 text = process_single_pdf_doctr(pdf_path)
#             else:
#                 print(f"[Error] Unbekannter Pipeline-Typ: {pipeline_type}")
#                 continue
#
#         else:
#             text = extract_text_if_searchable(pdf_path)
#
#         result = ask_ollama_for_invoice_fields(text)
#
#         results.append(result)
#
#     return results



if __name__ == "__main__":
    # Example usage
    results = easyocr_process_pdf_invoice(SAMPLE_M_PDF_PATH)