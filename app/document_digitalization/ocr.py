import subprocess
import os
import json


import easyocr


from typing import List

from paddleocr import PaddleOCR

from app.config import TMP_DIR, TESSERACT_CMD
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr")

def tesseract_png_to_text(png_path: str) -> str:
    """Extracts text from a PNG image using Tesseract OCR. Returns the result as a JSON string."""
    base = os.path.splitext(os.path.basename(png_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")
    cmd = [TESSERACT_CMD, png_path, os.path.splitext(txt_path)[0], "-l", "deu"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(txt_path):
        logger.error(f"OCR-Text nicht gefunden: {txt_path}")
        raise RuntimeError(f"OCR-Text nicht gefunden: {txt_path}")
    results = json.dumps((open(txt_path, encoding="utf-8").read()), ensure_ascii=False)
    return results

def easyocr_png_to_text(png_path: str, languages: List[str] = ['de']) -> str:
    """Extracts text from a PNG image using EasyOCR. Returns the result as a JSON string."""
    if easyocr is None:
        logger.error("EasyOCR is not installed. Please install with `pip install easyocr torch`.")
        raise RuntimeError("EasyOCR is not installed. Please install with `pip install easyocr torch`.")
    reader = easyocr.Reader(languages, gpu=False)
    results = reader.readtext(png_path)
    texts = [res[1] for res in results]
    full_text = "\n".join(texts)
    return json.dumps(full_text, ensure_ascii=False)

def paddleocr_pdf_to_text(pdf_path: str, lang: str = 'german') -> list[str]:
    """Extracts text from each page of a PDF using PaddleOCR. Returns a list of strings (one per page)."""
    if PaddleOCR is None:
        logger.error("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
        raise RuntimeError("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    results = ocr.predict(pdf_path) or []
    page_texts = []
    for page_result in results:
        if isinstance(page_result, dict) and 'rec_texts' in page_result and isinstance(page_result['rec_texts'], list):
            rec_texts = [(t if isinstance(t, str) else "") for t in page_result['rec_texts'] if t is not None]
            page_text = "\n".join(rec_texts)
        else:
            page_text = str(page_result)
        page_texts.append(page_text)
    return page_texts


if __name__ == "__main__":
    # Example usage
    png_file = "../results/input/BRE-03.pdf"  # Replace with your image file path
    try:
        # text = tesseract_png_to_text(png_file)
        # logger.info(f"Tesseract OCR Output: {text}")

        # easyocr_text = easyocr_png_to_text(png_file)
        # logger.info(f"EasyOCR Output: {easyocr_text}")

        paddleocr_text = paddleocr_pdf_to_text(png_file)
        logger.info(f"PaddleOCR Output: {paddleocr_text}")

    except RuntimeError as e:
        logger.error(f"Error: {e}")