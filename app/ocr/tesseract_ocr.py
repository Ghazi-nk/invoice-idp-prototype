"""
Tesseract OCR-Engine für Texterkennung aus Bildern.

Dieses Modul implementiert die Integration von Tesseract OCR
für die Texterkennung aus PNG-Bildern. Tesseract ist eine der
ältesten und bewährtesten Open-Source-OCR-Engines.

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import logging

from pytesseract import image_to_string


from app.logging_config import ocr_logger

def tesseract_png_to_text(png_path: str) -> str:
    """
    Extrahiert Text aus einem PNG-Bild mittels Tesseract OCR.

    Args:
        png_path (str): Pfad zur PNG-Bilddatei
    
    Returns:
        str: Erkannter Text als String
        
    Raises:
        Exception: Bei Tesseract-Fehlern oder ungültigen Bildpfaden
        
    Note:
        - Verwendet deutsche Spracherkennung ('deu')
        - PSM 3: Vollautomatische Seitensegmentierung ohne OSD
    """
    # Use pytesseract to directly get the text, without relying on file system
    ocr_logger.info(f"Processing image: {png_path}")
    
    try:
        # Get text directly from pytesseract
        raw_text = image_to_string(png_path, lang='deu', config='--psm 3')
        ocr_logger.debug(f"Extracted {len(raw_text)} characters with Tesseract")
        return raw_text
            
    except Exception as e:
        ocr_logger.exception(f"Error processing {png_path} with Tesseract: {e}")
        raise


if __name__ == "__main__":
    from app.config import SAMPLE_PNG_PATH
    """Testmodus: Verarbeitet das Beispiel-PNG mit Tesseract."""
    if SAMPLE_PNG_PATH:
        try:
            result = tesseract_png_to_text(SAMPLE_PNG_PATH)
            ocr_logger.info(f"Tesseract OCR Ergebnis: {len(result)} Zeichen extrahiert")
            ocr_logger.debug(f"Tesseract Textvorschau: {result[:200] + '...' if len(result) > 200 else result}")
        except Exception as e:
            ocr_logger.error(f"Tesseract-Test fehlgeschlagen: {e}")
    else:
        ocr_logger.error("SAMPLE_PNG_PATH nicht konfiguriert")