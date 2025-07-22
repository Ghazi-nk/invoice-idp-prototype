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


def doctr_pdf_to_text(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[List[Dict[str, Any]]]]:
    """
    Run DocTr OCR auf einer PDF-Datei und gibt entweder reinen Text oder strukturierte Daten zurück.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        include_bbox: Wenn True, werden strukturierte Daten mit Bounding-Box-Koordinaten zurückgegeben
                     Wenn False, wird reiner Text ohne Koordinaten zurückgegeben
    
    Returns:
        Bei include_bbox=False: Liste von Strings (einer pro Seite) mit reinem Text
        Bei include_bbox=True: Liste von Listen von Dictionaries mit 'text' und 'bbox' Keys
    """
    try:
        # 1. OCR durchführen
        predictor = ocr_predictor(pretrained=True)
        document = DocumentFile.from_pdf(pdf_path)
        result = predictor(document)
        
        # 2. Daten extrahieren und nach Seiten gruppieren
        data = result.export()
        pages_to_lines = {}
        
        # Text und Bounding-Boxen extrahieren
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
        
        # 3. Ausgabeformat generieren
        output = []
        for page_num in sorted(pages_to_lines.keys()):
            lines = pages_to_lines[page_num]
            
            # Linien nach vertikaler Position sortieren
            lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
            
            # Horizontal benachbarte Linien auf gleicher Höhe zusammenführen
            merged_lines = []
            if lines:
                current_line = lines[0]
                for next_line in lines[1:]:
                    y_center_current = (current_line['bbox'][1] + current_line['bbox'][3]) / 2
                    y_center_next = (next_line['bbox'][1] + next_line['bbox'][3]) / 2
                    
                    # Wenn auf ähnlicher Höhe, zusammenführen
                    if abs(y_center_current - y_center_next) < 10:
                        current_line['text'] += " " + next_line['text']
                        # Bounding-Box erweitern
                        current_line['bbox'][0] = min(current_line['bbox'][0], next_line['bbox'][0])
                        current_line['bbox'][1] = min(current_line['bbox'][1], next_line['bbox'][1])
                        current_line['bbox'][2] = max(current_line['bbox'][2], next_line['bbox'][2])
                        current_line['bbox'][3] = max(current_line['bbox'][3], next_line['bbox'][3])
                    else:
                        # Neue Zeile beginnen
                        merged_lines.append(current_line)
                        current_line = next_line
                
                merged_lines.append(current_line)  # Letzte Zeile hinzufügen
            
            # Je nach Parameter zurückgeben
            if include_bbox:
                # Strukturierte Daten mit Text und Bounding-Boxen
                output.append([{'text': line['text'], 'bbox': line['bbox']} for line in merged_lines])
            else:
                # Reiner Text ohne Koordinaten - einfach die Textzeilen zusammenfügen
                plain_text = "\n".join(line["text"] for line in merged_lines)
                output.append(plain_text)
        
        return output
            
    except Exception as e:
        logger.exception(f"doctr_pdf_to_text failed for '{pdf_path}': {e}")
        raise
