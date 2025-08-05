"""
OCR-Engine-Manager für Multi-Engine-Texterkennung.

Dieses Modul implementiert einen einheitlichen Manager für verschiedene OCR-Engines
zur Texterkennung aus PDF-Dokumenten. Es bietet eine abstrahierte Schnittstelle
für die Verwendung von Tesseract, PaddleOCR, EasyOCR, DocTR und LayoutLMv3.

Unterstützte OCR-Engines:
- Tesseract: Klassische OCR-Engine von Google
- PaddleOCR: Hochperformante OCR von PaddlePaddle
- EasyOCR: Benutzerfreundliche OCR mit Deep Learning
- DocTR: Document Text Recognition von Mindee
- LayoutLMv3: Layout-bewusste OCR von Microsoft

Architektur:
PDF → PNG-Konvertierung → Engine-spezifische OCR → Einheitliches Text-Format

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List



from app.ocr.doctr_pdf2txt import doctr_pdf_to_text
from app.ocr.layoutlmv3_png2txt import layoutlm_image_to_text
from app.ocr.tesseract_ocr import tesseract_png_to_text
from app.ocr.easyocr_engine import easyocr_png_to_text
from app.ocr.paddle_ocr import paddleocr_pdf_to_text
from app.ocr.pdf_utils import pdf_to_png_with_pymupdf
from app.logging_config import ocr_logger


def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable) -> List[str] | None:
    """
    Verarbeitet eine PDF-Datei mit der angegebenen OCR-Funktion.
    
    Diese generische Funktion konvertiert PDF-Seiten zu PNG-Bildern und
    wendet die spezifizierte OCR-Engine auf jede Seite an. Sie dient als
    einheitliche Schnittstelle für verschiedene OCR-Engines.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        ocr_function (Callable): OCR-Funktion zur Texterkennung
    
    Returns:
        List[str] | None: Liste von erkanntem Text pro Seite oder None bei Fehlern  
    
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__
    ocr_logger.info(f"Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'…")
    pages_content = []
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path)
        for i, png_page in enumerate(png_pages):
            page_num = i + 1
            extracted_content = ocr_function(png_page)
            pages_content.append(extracted_content)
            ocr_logger.info(f"Seite {page_num} von '{base_name}.pdf' erfolgreich verarbeitet.")
        return pages_content
    except Exception as e:
        ocr_logger.exception(f"Ein Fehler ist bei der Verarbeitung von '{base_name}.pdf' mit '{ocr_engine_name}' aufgetreten: {e}")
        return None


def easyocr_process_pdf(pdf_path: str) -> List[str] | None:
    """
    Verarbeitet PDF mit EasyOCR-Engine.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        List[str] | None: Erkannter Text pro Seite oder None bei Fehlern
    """
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text)

def tesseract_process_pdf(pdf_path: str) -> List[str] | None:
    """
    Verarbeitet PDF mit Tesseract-OCR-Engine.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        List[str] | None: Erkannter Text pro Seite oder None bei Fehlern
    """
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text)

def layoutlm_process_pdf(pdf_path: str) -> List[str] | None:
    """
    Verarbeitet PDF mit LayoutLMv3-Engine.
    
    LayoutLMv3 ist eine layout-bewusste OCR-Engine, die sowohl Text als auch
    die räumliche Anordnung von Dokumentelementen berücksichtigt.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        List[str] | None: Erkannter Text pro Seite oder None bei Fehlern

    """
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path)
        results = []
        for png_page in png_pages:
            result = layoutlm_image_to_text(png_page)
            results.append(result)
                
        return results
    except Exception as e:
        ocr_logger.exception(f"Fehler bei LayoutLM für '{pdf_path}': {e}")
        return None

def doctr_process_pdf(pdf_path: str) -> List[str] | None:
    """
    Verarbeitet PDF mit DocTR-Engine.
    
    DocTR (Document Text Recognition) ist eine moderne OCR-Engine von Mindee,
    die speziell für Dokumentenverarbeitung optimiert ist.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        List[str] | None: Erkannter Text pro Seite oder None bei Fehlern
        
    Note:
        DocTR verarbeitet PDFs direkt ohne PNG-Konvertierung.
    """
    try:
        return doctr_pdf_to_text(pdf_path)
    except Exception as e:
        ocr_logger.exception(f"Fehler bei Doctr OCR für '{pdf_path}': {e}")
        return None

def paddleocr_process_pdf(pdf_path: str) -> List[str] | None:
    """
    Verarbeitet PDF mit PaddleOCR-Engine.
    
    PaddleOCR ist eine hochperformante OCR-Engine von Baidu PaddlePaddle,
    die für verschiedene Sprachen und Layouts optimiert ist.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        List[str] | None: Erkannter Text pro Seite oder None bei Fehlern

    """
    try:
        return paddleocr_pdf_to_text(pdf_path)
    except Exception as e:
        ocr_logger.exception(f"Fehler bei PaddleOCR für '{pdf_path}': {e}")
        return None

# Engine-Mapping: Verbindet Engine-Namen mit ihren Verarbeitungsfunktionen
_OCR_ENGINE_PDF_MAP: Dict[str, Callable] = {
    "doctr": doctr_process_pdf,
    "easyocr": easyocr_process_pdf,
    "tesseract": tesseract_process_pdf,
    "paddleocr": paddleocr_process_pdf,
    "layoutlm": layoutlm_process_pdf,
}


def ocr_pdf(pdf_path: str, *, engine: str = "paddleocr") -> List[str]:
    """
    Verarbeitet eine PDF-Datei mit der angegebenen OCR-Engine.
    
    Diese Hauptfunktion dient als einheitliche Schnittstelle für alle
    verfügbaren OCR-Engines. Sie validiert die Engine-Auswahl und
    delegiert die Verarbeitung an die entsprechende Engine-Funktion.
    
    Args:
        pdf_path (str): Pfad zur zu verarbeitenden PDF-Datei
        engine (str, optional): Name der OCR-Engine. Verfügbare Optionen:
            - "tesseract": Google Tesseract OCR
            - "paddleocr": Baidu PaddleOCR (Standard)
            - "easyocr": EasyOCR mit Deep Learning
            - "doctr": Mindee DocTR
            - "layoutlm": Microsoft LayoutLMv3
        
    Returns:
        List[str]: Liste von erkanntem Text pro Seite
        
    Raises:
        ValueError: Bei ungültiger Engine-Auswahl

    """
    engine = engine.lower()
    if engine not in _OCR_ENGINE_PDF_MAP:
        valid_options = ", ".join(_OCR_ENGINE_PDF_MAP.keys())
        raise ValueError(f"Unsupported OCR engine '{engine}'. Valid options are: {valid_options}.")
    
    ocr_function = _OCR_ENGINE_PDF_MAP[engine]
    result = ocr_function(pdf_path)
    
    if result is None:
        ocr_logger.error(f"OCR failed for '{pdf_path}' with engine '{engine}'. Returning empty list.")
        return []
    
    return result


def get_available_engines() -> List[str]:
    """
    Gibt eine Liste aller verfügbaren OCR-Engines zurück.
    
    Diese Funktion wird für API-Endpunkte und Validierungen verwendet,
    um dynamisch die verfügbaren OCR-Optionen abzufragen.
    
    Returns:
        List[str]: Namen aller verfügbaren OCR-Engines
        
    """
    return list(_OCR_ENGINE_PDF_MAP.keys()) 