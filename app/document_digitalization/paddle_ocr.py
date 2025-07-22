import sys
import logging
from typing import List, Dict, Any, Union, cast

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddle_ocr")

def paddleocr_pdf_to_text(pdf_path: str, lang: str = 'german', return_bbox: bool = False) -> Union[List[str], List[List[Dict[str, Any]]]]:
    """
    Extracts text from each page of a PDF using PaddleOCR.
    
    Args:
        pdf_path: Path to the PDF file
        lang: Language for PaddleOCR
        return_bbox: If True, returns structured data with text and bbox info
                    If False, returns just the text strings
    
    Returns:
        When return_bbox=False: List of text strings (one per page)
        When return_bbox=True: List of lists of dicts with 'text' and 'bbox' (one list per page)
    """
    if PaddleOCR is None:
        logger.error("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
        raise RuntimeError("PaddleOCR is not installed. Please install with `pip install paddleocr paddlepaddle`.")
    
    ocr = PaddleOCR(use_angle_cls=False, lang=lang)
    results = ocr.predict(pdf_path) or []
    logger.info(f"PaddleOCR detected content across {len(results)} pages")
    
    if return_bbox:
        # Process results to get structured data with bbox info
        all_pages_items: List[List[Dict[str, Any]]] = []
        
        for page_idx, page_result in enumerate(results):
            page_items: List[Dict[str, Any]] = []
            
            # Handle different PaddleOCR output formats
            if isinstance(page_result, dict) and 'rec_texts' in page_result and 'rec_boxes' in page_result:
                # Modern PaddleOCR format with separate text and bbox lists
                texts = page_result.get('rec_texts', [])
                boxes = page_result.get('rec_boxes', [])
                
                # Combine texts and boxes
                for i, (text, box) in enumerate(zip(texts, boxes)):
                    if text is None:
                        continue
                        
                    # Ensure we have a valid bbox
                    if isinstance(box, list) and len(box) >= 4:
                        # PaddleOCR boxes may be in different format, convert to [x0,y0,x1,y1]
                        # Typically they provide 4 points [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
                        x_coords = [p[0] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                        y_coords = [p[1] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                        
                        if x_coords and y_coords:
                            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        else:
                            bbox = [0, page_idx * 1000 + i * 30, 1000, page_idx * 1000 + (i + 1) * 30]
                    else:
                        bbox = [0, page_idx * 1000 + i * 30, 1000, page_idx * 1000 + (i + 1) * 30]
                    
                    page_items.append({
                        'text': str(text),
                        'bbox': bbox,
                        'page': page_idx + 1
                    })
            elif isinstance(page_result, list):
                # Handle when page_result is a list of detection results
                for i, item in enumerate(page_result):
                    # Different PaddleOCR versions might structure results differently
                    # Try to extract text and bbox
                    text = ""
                    bbox = [0, page_idx * 1000 + i * 30, 1000, page_idx * 1000 + (i + 1) * 30]
                    
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        item_list = cast(List, item)
                        text = item_list[1] if len(item_list) > 1 and item_list[1] is not None else ""
                        if len(item_list) > 0 and isinstance(item_list[0], (list, tuple)):
                            # Try to construct bbox from points
                            try:
                                bbox_data = cast(List, item_list[0])
                                x_coords = [p[0] for p in bbox_data if isinstance(p, (list, tuple)) and len(p) >= 2]
                                y_coords = [p[1] for p in bbox_data if isinstance(p, (list, tuple)) and len(p) >= 2]
                                if x_coords and y_coords:
                                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            except (IndexError, TypeError):
                                pass
                    elif isinstance(item, dict):
                        text = item.get('text', '')
                        box = item.get('bbox', [])
                        if box:
                            try:
                                bbox = [box[0], box[1], box[2], box[3]]
                            except (IndexError, TypeError):
                                pass
                    
                    page_items.append({
                        'text': str(text),
                        'bbox': bbox,
                        'page': page_idx + 1
                    })
            
            all_pages_items.append(page_items)
            
        return all_pages_items
    else:
        # Return just the text content for each page
        page_texts = []
        for page_result in results:
            if isinstance(page_result, dict) and 'rec_texts' in page_result and isinstance(page_result['rec_texts'], list):
                rec_texts = [(t if isinstance(t, str) else "") for t in page_result['rec_texts'] if t is not None]
                page_text = "\n".join(rec_texts)
            else:
                # Try to extract text from other formats
                texts = []
                if isinstance(page_result, list):
                    for item in page_result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            item_list = cast(List, item)
                            text = item_list[1] if len(item_list) > 1 and item_list[1] is not None else ""
                            if text:
                                texts.append(str(text))
                        elif isinstance(item, dict) and 'text' in item:
                            texts.append(str(item['text']))
                        
                page_text = "\n".join(texts) if texts else str(page_result)
                
            page_texts.append(page_text)
            
        return page_texts


if __name__ == "__main__":
    from app.config import SAMPLE_PDF_PATH
    
    # Example usage
    if not SAMPLE_PDF_PATH:
        logger.error("SAMPLE_PDF_PATH not set in environment")
        sys.exit(1)
        
    # Convert None to default values if needed
    pdf_file = SAMPLE_PDF_PATH or ""
    
    if not pdf_file:
        logger.error("Sample PDF path cannot be empty")
        sys.exit(1)
        
    try:
        # Show regular text output
        text_output = paddleocr_pdf_to_text(pdf_file, return_bbox=False)
        if isinstance(text_output, list) and text_output:
            first_page = text_output[0]
            preview = first_page[:200] if first_page and len(first_page) > 200 else first_page
            logger.info(f"PaddleOCR Text Output (first page):\n{preview}...")
        
        # Show bbox output
        bbox_output = paddleocr_pdf_to_text(pdf_file, return_bbox=True)
        logger.info(f"PaddleOCR Bbox Output:")
        if isinstance(bbox_output, list) and bbox_output:
            for page_idx, page_items in enumerate(bbox_output):
                logger.info(f"  Page {page_idx + 1}: {len(page_items)} items")
                if page_items:
                    sample_items = page_items[:2] if len(page_items) > 2 else page_items
                    for i, item in enumerate(sample_items):
                        # Safe item access
                        if isinstance(item, dict):
                            text = item.get('text', '')
                            bbox = item.get('bbox', [])
                            text_preview = text[:30] if len(text) > 30 else text
                            text_preview += '...' if len(text) > 30 else ''
                            logger.info(f"    Item {i+1}: text='{text_preview}', bbox={bbox}")
    
    except RuntimeError as e:
        logger.error(f"Error: {e}") 