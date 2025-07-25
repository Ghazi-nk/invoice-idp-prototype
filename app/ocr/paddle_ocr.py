import sys
import logging
from typing import List, Dict, Any, Union

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddle_ocr")


def paddleocr_pdf_to_text(pdf_path: str, lang: str = 'german') -> List[str]:
    """
    Extracts text from each page of a PDF using PaddleOCR.

    Args:
        pdf_path: Path to the PDF file
        lang: Language for PaddleOCR

    Returns:
        List of text strings (one per page)
    """
    if PaddleOCR is None:
        logger.error("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
        raise ImportError("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")

    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=False, lang=lang)
    results = ocr.predict(pdf_path) or []
    logger.info(f"PaddleOCR detected content across {len(results)} pages")

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
                logger.exception(f"Error processing OCRResult for page {page_idx+1}: {e}")

    return page_texts


if __name__ == "__main__":
    from app.config import SAMPLE_PDF_PATH
    import os
    
    if not SAMPLE_PDF_PATH:
        logger.error("SAMPLE_PDF_PATH not set")
        sys.exit(1)
        
    if not os.path.exists(SAMPLE_PDF_PATH):
        logger.error(f"Sample PDF file not found: {SAMPLE_PDF_PATH}")
        sys.exit(1)
        
    print(f"Testing with sample PDF: {SAMPLE_PDF_PATH}")
    
    try:
        # Test real PaddleOCR
        print("\n=== Testing PaddleOCR extraction ===")
        texts = paddleocr_pdf_to_text(SAMPLE_PDF_PATH)
        print(f"Extracted text from {len(texts)} pages with PaddleOCR")
        
        # Print first few pages
        if texts:
            print("\nFirst page content preview:")
            preview = texts[0][:200] if len(texts[0]) > 200 else texts[0]
            print(preview)
        else:
            print("No text found in the PDF. Check if PaddleOCR is installed and working correctly.")
            
    except Exception as e:
        logger.exception(f"Error testing PaddleOCR: {e}")
        

