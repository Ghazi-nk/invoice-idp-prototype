import json

from pathlib import Path
from typing import List, Dict, Any

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


from utils.config import TMP_DIR



# Core OCR pipeline
def _run_ocr(pdf_path: str):
    """Run Doctr OCR on ``pdf_path`` and return the raw result object."""
    predictor = ocr_predictor(pretrained=True)
    document = DocumentFile.from_pdf(pdf_path)
    return predictor(document)


# -------------------------
# Post-processing helpers
# -------------------------

def _extract_line_boxes(result) -> List[Dict[str, Any]]:
    """Flatten a Doctr ``result`` into a list of line-level dictionaries.

    Each item has the form {"text": str, "bbox": [x0, y0, x1, y1], "page": int}.
    The bounding boxes are **absolute pixel coordinates** derived from the page dimensions.
    """
    data = result.export()
    lines: List[Dict[str, Any]] = []

    for page_index, page in enumerate(data.get("pages", []), start=1):
        # Doctr stores dimensions as (width, height)
        width, height = page.get("dimensions", (1, 1))

        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                (x0n, y0n), (x1n, y1n) = line.get("geometry", ((0, 0), (0, 0)))
                abs_bbox = [
                    int(x0n * width),
                    int(y0n * height),
                    int(x1n * width),
                    int(y1n * height),
                ]
                text = " ".join(w.get("value", "") for w in line.get("words", []))
                lines.append({
                    "text": text,
                    "bbox": abs_bbox,
                    "page": page_index,
                })
    return lines


def _save_lines_as_txt(lines: List[Dict[str, Any]], pdf_path: str) -> str:
    """Save each line dict as a JSON line in a .txt file and return its path."""
    base = Path(pdf_path).stem
    out_path = Path(TMP_DIR) / f"{base}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return str(out_path)


# -------------------------
# Public convenience wrapper
# -------------------------

def doctr_pdf_to_text(pdf_path: str) -> str:
    """
    Run OCR on a PDF, extract line-level text boxes,
    save each as JSON-lines in TMP_DIR/<basename>.txt,
    and return the full JSON array as a string.
    """
    result = _run_ocr(pdf_path)
    lines = _extract_line_boxes(result)
    # save as JSON-lines
    _save_lines_as_txt(lines, pdf_path)
    # return JSON array string
    return json.dumps(lines, ensure_ascii=False)


# -------------------------
# CLI entry point for quick manual testing
# -------------------------

if __name__ == "__main__":
    output_str = doctr_pdf_to_text("../results/input/BRE-02.pdf")
    print(output_str)
