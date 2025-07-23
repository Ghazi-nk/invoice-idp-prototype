import sys
import logging
from typing import List, Dict, Any, Union, cast

try:
    import easyocr
except ImportError:
    easyocr = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("easyocr_engine")

def easyocr_png_to_text(png_path: str, languages: List[str] = ['de']) -> str:
    """
    Extracts text from a PNG image using EasyOCR.
    
    Args:
        png_path: Path to the PNG file
        languages: List of language codes for EasyOCR
    
    Returns:
        OCR text as a string
    """
    try:
        import easyocr
    except ImportError:
        logger.error("EasyOCR is not installed. Please install with `pip install easyocr torch`.")
        raise RuntimeError("EasyOCR is not installed. Please install with `pip install easyocr torch`.")
    
    reader = easyocr.Reader(languages, gpu=False)
    # EasyOCR returns: [[bbox, text, confidence], ...]
    results = reader.readtext(png_path)
    logger.info(f"EasyOCR found {len(results)} text regions")
    
    # Return just the text, joined by newlines
    texts = []
    for detection in results:
        if len(detection) >= 2:
            # Safe indexing with type check
            detection_list = cast(List, detection)
            text = detection_list[1] if len(detection_list) > 1 else ""
            if text:
                texts.append(text)
    
    return "\n".join(texts)


if __name__ == "__main__":
    from app.config import SAMPLE_PNG_PATH
    
    # Example usage
    if not SAMPLE_PNG_PATH:
        logger.error("SAMPLE_PNG_PATH not set in environment")
        sys.exit(1)
        
    # Convert None to default values if needed
    png_file = SAMPLE_PNG_PATH or ""
    
    if not png_file:
        logger.error("Sample PNG path cannot be empty")
        sys.exit(1)
        
    try:
        # Show text output
        text_output = easyocr_png_to_text(png_file)
        preview = text_output[:200] if len(text_output) > 200 else text_output
        logger.info(f"EasyOCR Text Output:\n{preview}...")
    
    except RuntimeError as e:
        logger.error(f"Error: {e}") 