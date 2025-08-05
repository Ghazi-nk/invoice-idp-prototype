"""
DocTR OCR-Engine für professionelle Dokumentenverarbeitung.

Dieses Modul implementiert die Integration von Mindee DocTR (Document Text Recognition),
einer modernen OCR-Engine die speziell für Geschäftsdokumente entwickelt wurde.
DocTR bietet direkte PDF-Verarbeitung mit layout-bewusster Texterkennung.

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""
import logging
from typing import List, Dict, Any, Union

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from app.logging_config import ocr_logger


def doctr_pdf_to_text(pdf_path: str) -> List[str]:
    """
    Führt DocTR OCR auf einer PDF-Datei durch und extrahiert strukturierten Text.
    
    Diese Funktion verwendet Mindee DocTR für direkte PDF-Verarbeitung
    mit layout-bewusster Texterkennung. Sie extrahiert Text aus Blöcken
    und Linien und organisiert ihn nach Seiten.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
    
    Returns:
        List[str]: Liste von Textinhalten pro Seite
        
    Raises:
        Exception: Bei PDF-Verarbeitungsfehlern oder DocTR-Fehlern

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
        ocr_logger.exception(f"DocTR OCR fehlgeschlagen für '{pdf_path}': {e}")
        raise
