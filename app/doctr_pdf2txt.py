import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from utils.config import TMP_DIR


def run_ocr(pdf_path: str):
    """
    Run OCR on a PDF and return the raw result object.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        OCR result object.
    """
    predictor = ocr_predictor(pretrained=True)
    document = DocumentFile.from_pdf(pdf_path)
    return predictor(document)


def reconstruct_text(result) -> str:
    data = result.export()
    lines_out = []

    for page in data.get('pages', []):
        for block in page.get('blocks', []):
            lines = block.get('lines', [])
            counts = [len(line['words']) for line in lines]

            if _is_uniform_row_count(counts):
                lines_out.extend(_render_table(lines))
            else:
                lines_out.extend(_render_paragraph(lines))

            lines_out.append('')  # paragraph/table spacer

    return "\n".join(lines_out)


def _is_uniform_row_count(counts: list[int]) -> bool:
    """Detects if every row has the same number of words (and at least two rows)."""
    return len(counts) >= 2 and len(set(counts)) == 1


def _render_table(lines: list[dict]) -> list[str]:
    """
    Renders a table block:
    - first line is header
    - second line is a dash-separator
    - remaining lines are normal rows
    """
    header = _make_line(lines[0]['words'])
    separator = ''.join('-' if ch != ' ' else ' ' for ch in header)

    out = [header, separator]
    for row in lines[1:]:
        out.append(_make_line(row['words']))
    return out


def _render_paragraph(lines: list[dict]) -> list[str]:
    """Renders a normal text block (one output line per input line)."""
    return [_make_line(line['words']) for line in lines]


def _make_line(words):
    char_width = 150
    buf = [' '] * char_width
    for w in sorted(words, key=lambda w: w.get('geometry', w.get('bbox'))[0][0]):
        geom = w.get('geometry', w.get('bbox'))
        x0 = geom[0][0] if isinstance(geom[0], (list, tuple)) else geom[0]
        col = int(x0 * char_width)
        for i, ch in enumerate(w['value']):
            if 0 <= col + i < char_width:
                buf[col + i] = ch
    return ''.join(buf).rstrip()

def doctr_pdf_to_text(pdf_path: str, reconstructed: bool = True) -> str:

    result = run_ocr(pdf_path)
    if reconstructed:
        results = reconstruct_text(result)
    else:
        results = result.render()

    _save_and_serialize(results, pdf_path)
    return results


def _save_and_serialize(text: str, pdf_path: str) -> str:
    """
    Save the OCR text to a file and return the path.
    """
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(TMP_DIR, f"{base}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return out_path

if __name__ == '__main__':
    text = doctr_pdf_to_text("results/input/BRE-02.pdf")
    print(text)
