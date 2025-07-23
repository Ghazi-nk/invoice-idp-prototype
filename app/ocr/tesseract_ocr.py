import logging
from typing import List, Dict, Any, Union

from pytesseract import image_to_string, image_to_data

from app.config import SAMPLE_PNG_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tesseract_ocr")

def tesseract_png_to_text(png_path: str) -> str:
    """
    Extracts text from a PNG image using Tesseract OCR.
    
    Args:
        png_path: Path to the PNG file
    
    Returns:
        OCR text as a string
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
    print(tesseract_png_to_text(SAMPLE_PNG_PATH))