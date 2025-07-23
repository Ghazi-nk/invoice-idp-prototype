import os
import sys
import logging
from typing import List, Dict, Any, Union
import tempfile

from pytesseract import image_to_string, image_to_boxes, image_to_data

from app.config import SAMPLE_PNG_PATH

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
            # Get structured data with bounding box info
            data_dict = image_to_data(png_path, lang='deu', output_type='dict', config='--psm 6')
            
            # Convert to list of dicts with 'text' and 'bbox' keys
            result = []
            
            # Extract useful information for each word/line
            n_boxes = len(data_dict['text'])
            for i in range(n_boxes):
                # Skip empty text entries
                if not data_dict['text'][i].strip():
                    continue
                    
                # Skip entries with very low confidence
                if int(data_dict['conf'][i]) < 0:
                    continue
                
                # Extract bounding box coordinates
                left = int(data_dict['left'][i])
                top = int(data_dict['top'][i])
                width = int(data_dict['width'][i])
                height = int(data_dict['height'][i])
                
                # Format as [x0, y0, x1, y1] for standardize_ocr_output
                bbox = [left, top, left + width, top + height]
                
                result.append({
                    'text': data_dict['text'][i],
                    'bbox': bbox
                })
                
            return result
        else:
            # Return just the raw text
            return raw_text
            
    except Exception as e:
        logger.exception(f"Error processing {png_path} with Tesseract: {e}")
        raise


if __name__ == "__main__":
    print(tesseract_png_to_text(SAMPLE_PNG_PATH, return_bbox=True))