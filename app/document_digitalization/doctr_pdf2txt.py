# FILE: document_digitalization/doctr_pdf2txt.py
# VERSION 2: ADVANCED (with line merging)

import json
from typing import List, Dict, Any

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def _run_ocr(pdf_path: str) -> Dict[str, Any]:
    """Run Doctr OCR on ``pdf_path`` and return the raw result object."""
    predictor = ocr_predictor(pretrained=True)
    document = DocumentFile.from_pdf(pdf_path)
    return predictor(document)


def _extract_and_group_lines_by_page(result: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Processes a Doctr result to group all line objects by their page number.
    Each line object will have its bounding box converted to absolute integer coordinates.
    """
    data = result.export()
    pages_to_lines: Dict[int, List[Dict[str, Any]]] = {}

    for page_index, page in enumerate(data.get("pages", []), start=1):
        width, height = page.get("dimensions", (1, 1))
        pages_to_lines.setdefault(page_index, [])

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
                pages_to_lines[page_index].append({
                    "text": text,
                    "bbox": abs_bbox,
                })
    return pages_to_lines


def _merge_lines_on_page(lines: List[Dict[str, Any]], y_tolerance: int = 10) -> List[Dict[str, Any]]:
    """
    Merges horizontally adjacent text lines that are on the same vertical level.
    """
    if not lines:
        return []

    # Sort lines primarily by their vertical position, then horizontal
    lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))

    merged_lines = []
    current_line = lines[0]

    for next_line in lines[1:]:
        # Check for vertical alignment
        y_center_current = (current_line['bbox'][1] + current_line['bbox'][3]) / 2
        y_center_next = (next_line['bbox'][1] + next_line['bbox'][3]) / 2

        if abs(y_center_current - y_center_next) < y_tolerance:
            # Vertically aligned, merge them
            current_line['text'] += " " + next_line['text']
            # Update bbox to be the union of the two
            current_line['bbox'][0] = min(current_line['bbox'][0], next_line['bbox'][0])
            current_line['bbox'][1] = min(current_line['bbox'][1], next_line['bbox'][1])
            current_line['bbox'][2] = max(current_line['bbox'][2], next_line['bbox'][2])
            current_line['bbox'][3] = max(current_line['bbox'][3], next_line['bbox'][3])
        else:
            # Not aligned, finalize the current line and start a new one
            merged_lines.append(current_line)
            current_line = next_line

    merged_lines.append(current_line)  # Add the last processed line
    return merged_lines


def doctr_pdf_to_text(pdf_path: str) -> List[str]:
    """
    Runs Doctr OCR, merges fragmented lines on each page, and then converts
    the result into a compact JSON string for the LLM.
    """
    result = _run_ocr(pdf_path)
    pages_with_lines = _extract_and_group_lines_by_page(result)
    final_page_strings: List[str] = []

    sorted_page_numbers = sorted(pages_with_lines.keys())
    for page_num in sorted_page_numbers:
        raw_lines = pages_with_lines[page_num]

        # Apply the merging logic before creating the JSON
        merged_page_lines = _merge_lines_on_page(raw_lines)

        json_string = json.dumps(merged_page_lines, ensure_ascii=False, separators=(',', ':'))
        final_page_strings.append(json_string)

    return final_page_strings