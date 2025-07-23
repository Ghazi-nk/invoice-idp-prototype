"""
DocTr OCR-Modul für PDF-Dokumente.
"""
import logging
from typing import List, Dict, Any, Union

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doctr_pdf2txt")


def doctr_pdf_to_text(pdf_path: str) -> List[str]:
    """
    Run DocTr OCR auf einer PDF-Datei und gibt reinen Text zurück.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
    
    Returns:
        Liste von Strings (einer pro Seite) mit reinem Text
    """
    try:
        # 1. OCR durchführen
        predictor = ocr_predictor(pretrained=True)
        document = DocumentFile.from_pdf(pdf_path)
        result = predictor(document)
        
        # 2. Daten extrahieren und nach Seiten gruppieren
        data = result.export()
        pages_to_lines = {}
        
        # Text extrahieren
        for page_index, page in enumerate(data.get("pages", []), start=1):
            width, height = page.get("dimensions", (1, 1))
            pages_to_lines.setdefault(page_index, [])
            
            # Linien aus Blöcken extrahieren
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
        
        # Reiner Text ohne Koordinaten nach Seiten
        text_pages = []
        for page_num in sorted(pages_to_lines.keys()):
            lines = pages_to_lines[page_num]
            lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
            plain_text = "\n".join(line["text"] for line in lines)
            text_pages.append(plain_text)
        
        return text_pages
            
    except Exception as e:
        logger.exception(f"doctr_pdf_to_text failed for '{pdf_path}': {e}")
        raise
