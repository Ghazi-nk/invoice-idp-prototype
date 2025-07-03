import subprocess
import os
import json

import easyocr

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

def easyocr_png_to_text(png_path: str, languages: List[str] = ['de']) -> str:

    if easyocr is None:
        raise RuntimeError("EasyOCR is not installed. Please install with `pip install easyocr torch`.")

    # Initialize reader (disable GPU by default)
    reader = easyocr.Reader(languages, gpu=False)
    # Perform OCR
    results = reader.readtext(png_path)
    # Extract only text portions
    texts = [res[1] for res in results]
    full_text = "\n".join(texts)
    return json.dumps(full_text, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    png_file = "results/tmp/BRE-03_page1.png"  # Replace with your image file path
    try:
        text = tesseract_png_to_text(png_file)
        print("Tesseract OCR Output:", text)

        #easyocr_text = easyocr_png_to_text(png_file)
        #print("EasyOCR Output:", easyocr_text)

    except RuntimeError as e:
        print(f"Error: {e}")