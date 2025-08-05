"""
PaddleOCR-Engine für hochperformante Texterkennung.

Dieses Modul implementiert die Integration von PaddleOCR,
einer modernen Deep-Learning-basierten OCR-Engine mit Unterstützung
für verschiedene Sprachen und komplexe Dokumentenlayouts.

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import sys
from typing import List

from paddleocr import PaddleOCR
from app.logging_config import ocr_logger


def paddleocr_pdf_to_text(pdf_path: str, lang: str = 'german') -> List[str]:
    """
    Extrahiert Text aus jeder Seite einer PDF-Datei mittels PaddleOCR.
    
    Diese Funktion nutzt PaddleOCR für die direkte PDF-Verarbeitung
    ohne vorherige PNG-Konvertierung. Sie behandelt OCRResult-Objekte
    und extrahiert strukturierten Text pro Seite.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        lang (str, optional): Sprache für PaddleOCR. Defaults to 'german'.
        
    Returns:
        List[str]: Liste von erkanntem Text pro Seite
        
    Raises:
        ImportError: Wenn PaddleOCR nicht installiert ist
        Exception: Bei PDF-Verarbeitungsfehlern

    """
    if PaddleOCR is None:
        ocr_logger.error("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
        raise ImportError("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")

    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=False, lang=lang)
    results = ocr.predict(pdf_path) or []
    ocr_logger.info(f"PaddleOCR detected content across {len(results)} pages")

    # Extract text
    page_texts = []

    # Handle OCRResult objects
    for page_idx, page_result in enumerate(results):
        # Handle paddlex.inference.pipelines.ocr.result.OCRResult objects
        ocr_result_class = str(type(page_result))
        if 'OCRResult' in ocr_result_class:
            # Try to access text and boxes
            try:
                # Check if it has a dictionary interface with rec_texts and rec_boxes
                if hasattr(page_result, 'get') or isinstance(page_result, dict):
                    # Extract with dictionary interface
                    data_dict = page_result
                    if isinstance(page_result, object) and hasattr(page_result, 'get'):
                        data_dict = {k: page_result.get(k) for k in ['rec_texts', 'rec_boxes']}

                    texts = data_dict.get('rec_texts', [])
                    boxes = data_dict.get('rec_boxes', [])

                # If we found texts
                if 'texts' in locals() and len(texts) > 0:
                    # Join texts for this page
                    page_text = "\n".join(str(t) for t in texts if t is not None)
                    page_texts.append(page_text)

            except Exception as e:
                ocr_logger.exception(f"Error processing OCRResult for page {page_idx+1}: {e}")

    return page_texts


if __name__ == "__main__":
    from app.config import SAMPLE_PDF_PATH
    import os
    
    if not SAMPLE_PDF_PATH:
        ocr_logger.error("SAMPLE_PDF_PATH not set")
        sys.exit(1)
        
    if not os.path.exists(SAMPLE_PDF_PATH):
        ocr_logger.error(f"Sample PDF file not found: {SAMPLE_PDF_PATH}")
        sys.exit(1)
        
    ocr_logger.info(f"Testing with sample PDF: {SAMPLE_PDF_PATH}")
    
    try:
        # Test real PaddleOCR
        ocr_logger.info("Testing PaddleOCR extraction")
        texts = paddleocr_pdf_to_text(SAMPLE_PDF_PATH)
        ocr_logger.info(f"Extracted text from {len(texts)} pages with PaddleOCR")
        
        # Print first few pages
        if texts:
            preview = texts[0][:200] if len(texts[0]) > 200 else texts[0]
            ocr_logger.info(f"First page content preview: {preview}")
        else:
            ocr_logger.warning("No text found in the PDF. Check if PaddleOCR is installed and working correctly.")
            
    except Exception as e:
        ocr_logger.exception(f"Error testing PaddleOCR: {e}")
        

