from doctr.io import DocumentFile
from doctr.models import ocr_predictor

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


def reconstruct_text(result, char_width: int = 150) -> str:
    """
    Reconstruct text from an OCR result using fixed-width alignment.

    Args:
        result: OCR result object.
        char_width: Width of the fixed character buffer per line.

    Returns:
        A multi-line string of reconstructed text.
    """
    data = result.export()
    lines_out = []

    def _make_line(words):
        buf = [' '] * char_width
        for w in sorted(words, key=lambda w: w.get('geometry', w.get('bbox'))[0][0]):
            geom = w.get('geometry', w.get('bbox'))
            x0 = geom[0][0] if isinstance(geom[0], (list, tuple)) else geom[0]
            col = int(x0 * char_width)
            for i, ch in enumerate(w['value']):
                if 0 <= col + i < char_width:
                    buf[col + i] = ch
        return ''.join(buf).rstrip()

    for page in data.get('pages', []):
        for block in page.get('blocks', []):
            lines = block.get('lines', [])
            counts = [len(l['words']) for l in lines]
            is_table = len(counts) >= 2 and len({*counts}) == 1
            if is_table:
                header = _make_line(lines[0]['words'])
                sep = ''.join('-' if ch != ' ' else ' ' for ch in header)
                lines_out.extend([header, sep])
                for row in lines[1:]:
                    lines_out.append(_make_line(row['words']))
                lines_out.append('')
            else:
                for line in lines:
                    lines_out.append(_make_line(line['words']))
                lines_out.append('')

    return "\n".join(lines_out)


def extract_text_from_pdf(pdf_path: str, char_width: int = 150, reconstructed: bool = True) -> str:
    """
    Extract text from a PDF via OCR, optionally reconstructing layout.

    Args:
        pdf_path: Path to the input PDF file.
        char_width: Width of the fixed character buffer per line.
        reconstructed: If False, return raw OCR text; if True, return reconstructed layout.

    Returns:
        OCR text as a string.
    """
    result = run_ocr(pdf_path)
    if not reconstructed:
        return result.render()
    return reconstruct_text(result, char_width)


if __name__ == '__main__':
    text = extract_text_from_pdf("results/input/BRE-02.pdf")
    print(text)
