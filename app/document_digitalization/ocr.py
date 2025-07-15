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

def paddleocr_pdf_to_text(png_path: str, lang: str = 'german') -> str:
    """Extracts text from a PNG image using PaddleOCR. Returns the result as a JSON string."""
    if PaddleOCR is None:
        logger.error("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
        raise RuntimeError("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    result = ocr.predict(png_path)
    texts = []
    if result and result[0]:
        image_result = result[0]
        if 'rec_texts' in image_result:
            texts = image_result['rec_texts']
    full_text = "\n".join(texts)
    return json.dumps(full_text, ensure_ascii=False)


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