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


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tesseract_ocr")

def tesseract_png_to_text(png_path: str) -> str:
    """
    Extrahiert Text aus einem PNG-Bild mittels Tesseract OCR.
    
    #todo: tesseract engine nutzung und note dass dies aufm local machine installiert werden muss!
    #todo: dies muss auch in die readme erwähnt werden!
    
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
    logger.info(f"Processing image: {png_path}")
    
    try:
        # Get text directly from pytesseract
        raw_text = image_to_string(png_path, lang='deu', config='--psm 3')
        logger.debug(f"Extracted {len(raw_text)} characters with Tesseract")
        return raw_text
            
    except Exception as e:
        logger.exception(f"Error processing {png_path} with Tesseract: {e}")
        raise


if __name__ == "__main__":
    from app.config import SAMPLE_PNG_PATH
    """Testmodus: Verarbeitet das Beispiel-PNG mit Tesseract."""
    if SAMPLE_PNG_PATH:
        try:
            result = tesseract_png_to_text(SAMPLE_PNG_PATH)
            print(f"Tesseract OCR Ergebnis ({len(result)} Zeichen):")
            print(result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            logger.error(f"Tesseract-Test fehlgeschlagen: {e}")
    else:
        logger.error("SAMPLE_PNG_PATH nicht konfiguriert")