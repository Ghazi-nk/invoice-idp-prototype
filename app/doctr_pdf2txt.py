import json
import os
from pathlib import Path
from typing import List, Dict, Any

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Adjust this to wherever your temp files should live
from utils.config import TMP_DIR


# -------------------------
# Core OCR pipeline
# -------------------------

def run_ocr(pdf_path: str):
    """Run Doctr OCR on ``pdf_path`` and return the raw result object."""
    predictor = ocr_predictor(pretrained=True)
    document = DocumentFile.from_pdf(pdf_path)
    return predictor(document)


# -------------------------
# Post‑processing helpers
# -------------------------

def extract_word_boxes(result) -> List[Dict[str, Any]]:
    """Flatten a Doctr ``result`` into a list of word‑level dictionaries.

    Each item has the form ``{"text": str, "bbox": [x0, y0, x1, y1], "page": int}``.
    The bounding boxes are **absolute pixel coordinates** derived from the page dimensions,
    mirroring the format you asked for.
    """
    data = result.export()
    words: List[Dict[str, Any]] = []

    for page_index, page in enumerate(data.get("pages", []), start=1):
        # Doctr stores dimensions as (height, width)
        height, width = page.get("dimensions", (1, 1))

        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    # Geometry is [[x0, y0], [x1, y1]] in *relative* coords (0‒1)
                    (x0, y0), (x1, y1) = word.get("geometry", ((0, 0), (0, 0)))
                    abs_bbox = [
                        int(x0 * width),
                        int(y0 * height),
                        int(x1 * width),
                        int(y1 * height),
                    ]
                    words.append({
                        "text": word.get("value", ""),
                        "bbox": abs_bbox,
                        "page": page_index,
                    })
    return words


def save_json(obj: Any, pdf_path: str) -> str:
    """Persist *obj* as JSON next to *pdf_path* and return the file path."""
    base_name = Path(pdf_path).stem
    out_path = Path(TMP_DIR) / f"{base_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf‑8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return str(out_path)


# -------------------------
# Public convenience wrapper
# -------------------------

def doctr_pdf_to_text(pdf_path: str, *, save: bool = True) -> List[Dict[str, Any]]:
    """Run OCR and return a word‑box list (and optionally save to disk)."""
    result = run_ocr(pdf_path)
    word_boxes = extract_word_boxes(result)

    if save:
        path = save_json(word_boxes, pdf_path)
        print(f"Saved OCR word boxes ➜ {path}")

    return word_boxes


# -------------------------
# CLI entry point for quick manual testing
# -------------------------

if __name__ == "__main__":
        boxes = doctr_pdf_to_text("results/input/BRE-02.pdf", save=True)
        print(json.dumps(boxes, ensure_ascii=False))
