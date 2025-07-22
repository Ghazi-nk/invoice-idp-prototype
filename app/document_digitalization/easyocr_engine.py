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

def easyocr_png_to_text(png_path: str, languages: List[str] = ['de'], return_bbox: bool = False) -> Union[str, List[Dict[str, Any]]]:
    """
    Extracts text from a PNG image using EasyOCR.
    
    Args:
        png_path: Path to the PNG file
        languages: List of language codes for EasyOCR
        return_bbox: If True, returns structured data with text and bbox info
                    If False, returns just the text as a string
    
    Returns:
        When return_bbox=False: OCR text as a string
        When return_bbox=True: List of dicts with 'text' and 'bbox' for standardize_ocr_output
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
    
    if return_bbox:
        # EasyOCR already provides bbox information - convert to our standard format
        items: List[Dict[str, Any]] = []
        for detection in results:
            if len(detection) >= 2:
                # Safe casting for type checking
                detection_list = cast(List, detection)
                
                # EasyOCR provides bbox as 4 points, we need [x0, y0, x1, y1]
                bbox_points = detection_list[0] if len(detection_list) > 0 else None
                text = detection_list[1] if len(detection_list) > 1 else ""
                
                # Handle different bbox formats
                if isinstance(bbox_points, list) and len(bbox_points) >= 4:
                    # Convert polygon points to bounding box [x0, y0, x1, y1]
                    x_coords = [p[0] for p in bbox_points if isinstance(p, (list, tuple)) and len(p) >= 2]
                    y_coords = [p[1] for p in bbox_points if isinstance(p, (list, tuple)) and len(p) >= 2]
                    
                    if x_coords and y_coords:
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        
                        items.append({
                            'text': text,
                            'bbox': bbox
                        })
                
        return items
    else:
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
        # Show regular text output
        text_output = easyocr_png_to_text(png_file, return_bbox=False)
        if isinstance(text_output, str):
            preview = text_output[:200] if len(text_output) > 200 else text_output
            logger.info(f"EasyOCR Text Output:\n{preview}...")
        
        # Show bbox output
        bbox_output = easyocr_png_to_text(png_file, return_bbox=True)
        logger.info(f"EasyOCR Bbox Output (first 3 items):")
        if isinstance(bbox_output, list):
            sample_items = bbox_output[:3] if len(bbox_output) > 3 else bbox_output
            for i, item in enumerate(sample_items):
                text_preview = item.get('text', '')[:30]  # Get first 30 chars of text
                text_preview += '...' if len(item.get('text', '')) > 30 else ''
                logger.info(f"  Item {i+1}: text='{text_preview}', bbox={item.get('bbox')}")
    
    except RuntimeError as e:
        logger.error(f"Error: {e}") 