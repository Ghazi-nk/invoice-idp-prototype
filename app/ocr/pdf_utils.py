"""
PDF-Utilities für Dokumentenverarbeitung und -konvertierung.

Dieses Modul stellt grundlegende Funktionen für die Verarbeitung von PDF-Dokumenten
bereit, einschließlich Base64-Dekodierung, PNG-Konvertierung und Textextraktion.
Es dient als Grundlage für die OCR-Pipeline und API-Endpunkte.

Hauptfunktionen:
- Base64-PDF-Dekodierung mit automatischem Cleanup
- PDF-zu-PNG-Konvertierung für OCR-Verarbeitung
- Extrahierung von durchsuchbarem Text aus PDFs

Verwendete Bibliotheken:
- PyMuPDF (fitz): Hochperformante PDF-Verarbeitung
- tempfile: Sichere temporäre Dateiverwaltung

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import base64
import logging
import os
import tempfile
from contextlib import contextmanager

import fitz

from app.config import TMP_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_utils")

@contextmanager
def save_base64_to_temp_pdf(base64_string: str):
    """
    Dekodiert einen Base64-String und speichert ihn als temporäre PDF-Datei.
    
    Diese Context-Manager-Funktion dekodiert Base64-kodierte PDF-Daten,
    speichert sie in einer temporären Datei und gewährleistet automatisches
    Cleanup nach der Verwendung. Ideal für API-Endpunkte, die PDF-Uploads
    als Base64-Strings empfangen.
    
    Args:
        base64_string (str): Base64-kodierte PDF-Daten
        
    Yields:
        str: Pfad zur temporären PDF-Datei
        
    Raises:
        Exception: Bei Base64-Dekodierungsfehlern oder Dateisystemfehlern
        
    Example:
        >>> with save_base64_to_temp_pdf(pdf_base64) as temp_path:
        ...     result = process_pdf(temp_path)
        # Automatisches Cleanup der temporären Datei
        
    Note:
        Die temporäre Datei wird automatisch nach Verlassen des Context gelöscht,
        auch bei Exceptions. Fehler beim Cleanup werden als Warnungen geloggt.
    """
    temp_file_path = None
    try:
        pdf_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_file_path = temp_file.name
        yield temp_file_path
    except Exception as e:
        logger.exception("Failed to save base64 PDF to temp file:")
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")

def extract_text_if_searchable(pdf_path: str) -> list[str]:
    """
    Extrahiert durchsuchbaren Text aus einer PDF-Datei.
    
    Diese Funktion öffnet eine PDF-Datei und extrahiert vorhandenen Text
    ohne OCR-Verarbeitung. Sie ist ideal für PDFs mit eingebettetem Text
    und kann als schnelle Alternative zur OCR-Verarbeitung dienen.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        list[str]: Liste von Textinhalten pro Seite. Leere Strings für
                  Seiten ohne durchsuchbaren Text.
                  
    Raises:
        RuntimeError: Bei PDF-Öffnungsfehlern oder beschädigten Dateien
        
    Note:
        Gibt leere Liste zurück für gescannte PDFs ohne eingebetteten Text.     
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.exception(f"Could not open PDF: {pdf_path}")
        raise RuntimeError(f"Could not open PDF: {e}")
    page_texts = []
    for page in doc:
        text = page.get_text()
        if isinstance(text, str):
            page_texts.append(text.strip())
        else:
            page_texts.append("")
    return page_texts

def pdf_to_png_with_pymupdf(pdf_path: str, zoom: float = 3.0) -> list[str]:
    """
    Konvertiert PDF-Seiten zu hochauflösenden PNG-Bildern.
    
    Diese Funktion verwendet PyMuPDF für die Konvertierung von PDF-Seiten
    zu PNG-Bildern mit konfigurierbarer Auflösung. Die resultierenden
    Bilder werden für OCR-Verarbeitung optimiert.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        zoom (float, optional): Zoom-Faktor für die Auflösung.
                               3.0 entspricht etwa 300 DPI. Defaults to 3.0.
                               
    Returns:
        list[str]: Liste von Pfaden zu den generierten PNG-Dateien,
                  ein Pfad pro PDF-Seite
                  
    Raises:
        RuntimeError: Bei PDF-Öffnungsfehlern oder leeren PDFs
        Exception: Bei Konvertierungsfehlern oder Dateisystemfehlern
        
     """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            logger.error(f"Keine Seiten in PDF: {pdf_path}")
            raise RuntimeError(f"Keine Seiten in PDF: {pdf_path}")
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        png_paths = []
        for i, page in enumerate(doc):
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_path = os.path.join(TMP_DIR, f"{base}_page{i + 1}.png")
            pix.save(png_path)
            png_paths.append(png_path)
        return png_paths
    except Exception as e:
        logger.exception(f"Failed to convert PDF to PNGs with PyMuPDF: {pdf_path}")
        raise