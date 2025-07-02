import subprocess
import os
import json

import easyocr

#from paddleocr import PaddleOCR

from typing import List
from utils.config import TMP_DIR, TESSERACT_CMD

def tesseract_png_to_text(png_path: str) -> str:
    base = os.path.splitext(os.path.basename(png_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")
    cmd = [TESSERACT_CMD, png_path, os.path.splitext(txt_path)[0], "-l", "deu"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"OCR-Text nicht gefunden: {txt_path}")
    results = json.dumps((open(txt_path, encoding="utf-8").read()), ensure_ascii=False)
    return results
#
# def easyocr_png_to_text(png_path: str, languages: List[str] = ['deu']) -> str:
#
#     if easyocr is None:
#         raise RuntimeError("EasyOCR is not installed. Please install with `pip install easyocr torch`.")
#
#     # Initialize reader (disable GPU by default)
#     reader = easyocr.Reader(languages, gpu=False)
#     # Perform OCR
#     results = reader.readtext(png_path)
#     # Extract only text portions
#     texts = [res[1] for res in results]
#     full_text = "\n".join(texts)
#     return json.dumps(full_text, ensure_ascii=False)


# def paddleocr_png_to_text_paddleocr(png_path: str, lang: str = 'de') -> str:
#
#     if PaddleOCR is None:
#         raise RuntimeError("PaddleOCR is not installed. Please install with `pip install paddlepaddle paddleocr`.")
#
#     # Initialize OCR with angle detection
#     ocr = PaddleOCR(use_angle_cls=True, lang=lang)
#     # Perform OCR
#     result = ocr.ocr(png_path, cls=True)
#     # Extract only text portions
#     texts = [line[1][0] for line in result]
#     full_text = "\n".join(texts)
#     return json.dumps(full_text, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    png_file = "results/tmp/BRE-02_page1.png"  # Replace with your image file path
    try:
        text = tesseract_png_to_text(png_file)
        print("Tesseract OCR Output:", text)

       # easyocr_text = easyocr_png_to_text(png_file)
        #print("EasyOCR Output:", easyocr_text)

        # paddleocr_text = paddleocr_png_to_text_paddleocr(png_file)
        # print("PaddleOCR Output:", paddleocr_text)

    except RuntimeError as e:
        print(f"Error: {e}")