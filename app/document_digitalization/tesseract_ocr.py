import os
import sys
import logging
from typing import List, Dict, Any, Union
import tempfile

from pytesseract import image_to_string

from app.config import TMP_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tesseract_ocr")

def tesseract_png_to_text(png_path: str, return_bbox: bool = False) -> Union[str, List[Dict[str, Any]]]:
    """
    Extracts text from a PNG image using Tesseract OCR.
    
    Args:
        png_path: Path to the PNG file
        return_bbox: If True, returns structured data with text and bbox info
                    If False, returns just the text as a string
    
    Returns:
        When return_bbox=False: OCR text as a string
        When return_bbox=True: List of dicts with 'text' and 'bbox' for standardize_ocr_output
    """
    # Use pytesseract to directly get the text, without relying on file system
    logger.info(f"Processing image: {png_path}")
    
    try:
        # Get text directly from pytesseract
        raw_text = image_to_string(png_path, lang='deu', config='--psm 6')
        logger.debug(f"Extracted {len(raw_text)} characters with Tesseract")
        
        if return_bbox:
            # For Tesseract, we don't have real bbox information without additional processing
            # So we'll create synthetic bboxes based on line positions
            lines = raw_text.strip().split('\n')
            items: List[Dict[str, Any]] = []
            
            # Create simple bounding boxes based on line number
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    # Create a synthetic bbox: [x0, y0, x1, y1]
                    # With x spanning the whole width and y based on line number
                    y_position = i * 30 + 15  # Simple estimate: 30 pixels per line, centered at 15
                    items.append({
                        'text': line,
                        'bbox': [0, y_position - 10, 1000, y_position + 10]  # Default 20px height
                    })
                    
            return items
        else:
            # Return just the raw text
            return raw_text
            
    except Exception as e:
        logger.exception(f"Error processing {png_path} with Tesseract: {e}")
        raise


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
        # Show regular text output
        text_output = tesseract_png_to_text(png_file, return_bbox=False)
        if isinstance(text_output, str):
            preview = text_output[:200] if len(text_output) > 200 else text_output
            logger.info(f"Tesseract OCR Text Output:\n{preview}...")
        
        # Show bbox output
        bbox_output = tesseract_png_to_text(png_file, return_bbox=True)
        logger.info(f"Tesseract OCR Bbox Output (first 3 items):")
        if isinstance(bbox_output, list):
            sample_items = bbox_output[:3] if len(bbox_output) > 3 else bbox_output
            for i, item in enumerate(sample_items):
                text_preview = item.get('text', '')[:30]  # Get first 30 chars of text
                text_preview += '...' if len(item.get('text', '')) > 30 else ''
                logger.info(f"  Item {i+1}: text='{text_preview}', bbox={item.get('bbox')}")
    
    except RuntimeError as e:
        logger.error(f"Error: {e}") 