import re
import logging
from typing import List, Dict, Any, Union, Tuple, Sequence, cast

# Setup logging (for consistency, though rarely needed here)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pre_processing")


def standardize_ocr_output(text_with_bbox: Union[List[Dict[str, Any]], Sequence[Any], str], 
                          format_type: str = "formatted_string") -> Union[str, List[Dict[str, Any]]]:
    """
    Converts various OCR output formats into a standardized format compatible with doctr and layoutlm.
    
    Args:
        text_with_bbox: Input in various formats:
            - List of dicts with 'text' and 'bbox' keys
            - List of lists (like from EasyOCR): [[bbox, text, confidence], ...]
            - Raw string with some structure
        format_type: Output format type
            - "formatted_string": Returns a string with format "[y=123] "text content""
            - "objects": Returns a list of objects with text and bbox keys
    
    Returns:
        Either a formatted string or a list of standardized objects based on format_type
    """
    try:
        # Initialize the standardized output
        standardized_items: List[Dict[str, Any]] = []
        
        # Case 1: Already a list of dicts with 'text' and 'bbox'
        if isinstance(text_with_bbox, list) and all(isinstance(item, dict) and 'text' in item and 'bbox' in item 
                                                  for item in text_with_bbox):
            standardized_items = list(text_with_bbox)  # Create a copy to avoid type issues
        
        # Case 2: List of lists (like from EasyOCR)
        elif isinstance(text_with_bbox, list) and all(isinstance(item, list) and len(item) >= 2 for item in text_with_bbox):
            # EasyOCR format: [[bbox, text, confidence], ...]
            for item in text_with_bbox:
                if len(item) >= 2:
                    # Safe indexing with type handling
                    try:
                        # Cast item to list to help with type checking
                        item_list = cast(List, item)
                        bbox_data = item_list[0] if len(item_list) > 0 else None
                        text_content = item_list[1] if len(item_list) > 1 else ""
                    except (IndexError, TypeError):
                        bbox_data = None
                        text_content = ""
                    
                    # Ensure bbox has the correct format [x0, y0, x1, y1]
                    x0, y0, x1, y1 = 0, 0, 100, 100  # Default values
                    
                    if isinstance(bbox_data, (list, tuple)):
                        try:
                            if len(bbox_data) == 4:  # [x0, y0, x1, y1]
                                # Handle list of coordinates
                                bbox_list = cast(List, bbox_data)
                                x0 = int(bbox_list[0])
                                y0 = int(bbox_list[1])
                                x1 = int(bbox_list[2])
                                y1 = int(bbox_list[3])
                            elif len(bbox_data) == 2:
                                # Try to handle [(x0,y0), (x1,y1)] format
                                points = cast(List, bbox_data)
                                if (isinstance(points[0], (list, tuple)) and 
                                   isinstance(points[1], (list, tuple)) and
                                   len(points[0]) >= 2 and len(points[1]) >= 2):
                                    
                                    point1 = cast(List, points[0])
                                    point2 = cast(List, points[1])
                                    x0 = int(point1[0])
                                    y0 = int(point1[1])
                                    x1 = int(point2[0])
                                    y1 = int(point2[1])
                        except (IndexError, ValueError, TypeError):
                            pass  # Keep default values
                                
                    text = str(text_content)  # Convert to string regardless of type
                    standardized_items.append({
                        'text': text,
                        'bbox': [x0, y0, x1, y1]
                    })
        
        # Case 3: Raw string (with some expected structure)
        elif isinstance(text_with_bbox, str):
            # Try to parse strings that might contain bounding box information
            lines = text_with_bbox.strip().split('\n')
            for i, line in enumerate(lines):
                # Use simple y-coordinate heuristic based on line number
                y_value = i * 30 + 30  # Simple estimate: 30 pixels per line
                
                # Check if there's a bounding box pattern like [x,y,x,y] in the text
                bbox_match = re.search(r'\[(\d+),(\d+),(\d+),(\d+)\]', line)
                if bbox_match:
                    x0, y0, x1, y1 = map(int, bbox_match.groups())
                    # Remove the bbox part from the text
                    text = re.sub(r'\[(\d+),(\d+),(\d+),(\d+)\]', '', line).strip()
                else:
                    # Default bounding box spans the whole width
                    x0, y0, x1, y1 = 0, y_value - 15, 1000, y_value + 15
                    text = line.strip()
                
                standardized_items.append({
                    'text': text,
                    'bbox': [x0, y0, x1, y1]
                })
                
        # If we still have no items, create a basic fallback
        if not standardized_items and isinstance(text_with_bbox, str):
            lines = text_with_bbox.strip().split('\n')
            for i, line in enumerate(lines):
                y_value = i * 30 + 30
                standardized_items.append({
                    'text': line.strip(),
                    'bbox': [0, y_value - 15, 1000, y_value + 15]
                })
        
        # Format the output according to format_type
        if format_type == "formatted_string":
            formatted_lines = []
            for item in standardized_items:
                if item['text'].strip():  # Skip empty lines
                    x0, y0, x1, y1 = item['bbox']
                    y_center = int((y0 + y1) / 2)
                    x_center = int((x0 + x1) / 2)
                    formatted_lines.append(f'[x={x_center} ,y={y_center}] "{item["text"]}"')
            
            return "\n".join(formatted_lines)
        else:
            return standardized_items
            
    except Exception as e:
        logger.exception("Error during OCR output standardization:")
        return text_with_bbox if isinstance(text_with_bbox, str) else ""