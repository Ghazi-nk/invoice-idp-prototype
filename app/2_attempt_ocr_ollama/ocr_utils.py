import subprocess
import os
from config import TMP_DIR, TESSERACT_CMD

def ocr_png_to_text(png_path: str) -> str:
    base = os.path.splitext(os.path.basename(png_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")
    cmd = [TESSERACT_CMD, png_path, os.path.splitext(txt_path)[0], "-l", "deu"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"OCR-Text nicht gefunden: {txt_path}")
    return open(txt_path, encoding="utf-8").read()
