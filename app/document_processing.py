from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Union, Any, cast, Tuple

import logging

from app.document_digitalization.doctr_pdf2txt import doctr_pdf_to_text
from app.document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from app.document_digitalization.tesseract_ocr import tesseract_png_to_text
from app.document_digitalization.easyocr_engine import easyocr_png_to_text
from app.document_digitalization.paddle_ocr import paddleocr_pdf_to_text
from app.document_digitalization.pre_processing import standardize_ocr_output

from app.post_processing import finalize_extracted_fields, verify_and_correct_fields
from app.config import SAMPLE_PDF_PATH
from app.semantic_extraction import ollama_extract_invoice_fields
from app.document_digitalization.pdf_utils import pdf_to_png_with_pymupdf


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_processing")


def process_pdf_with_ocr(pdf_path: str, ocr_function: Callable, include_bbox: bool = False) -> Union[List[str], List[Dict[str, Any]]] | None:
    """
    Process a PDF file with the given OCR function.
    
    Args:
        pdf_path: Path to the PDF file
        ocr_function: OCR function to use
        include_bbox: Whether to return bbox information
    
    Returns:
        List of processed text per page, or None on error
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_engine_name = ocr_function.__name__
    logger.info(f"Verarbeite '{base_name}.pdf' mit der Engine '{ocr_engine_name}'…")
    pages_content = []
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path) # switch pdf to png converter here!
        for i, png_page in enumerate(png_pages):
            page_num = i + 1
            # Pass the include_bbox parameter to the OCR function
            extracted_content = ocr_function(png_page, return_bbox=include_bbox)
            pages_content.append(extracted_content)
            logger.info(f"Seite {page_num} von '{base_name}.pdf' erfolgreich verarbeitet.")
        return pages_content
    except Exception as e:
        logger.exception(f"Ein Fehler ist bei der Verarbeitung von '{base_name}.pdf' mit '{ocr_engine_name}' aufgetreten: {e}")
        return None


def easyocr_process_pdf(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[Dict[str, Any]]] | None:
    return process_pdf_with_ocr(pdf_path, easyocr_png_to_text, include_bbox)

def tesseract_process_pdf(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[Dict[str, Any]]] | None:
    return process_pdf_with_ocr(pdf_path, tesseract_png_to_text, include_bbox)

def layoutlm_process_pdf(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[List[Dict[str, Any]]]] | None:
    """Adapter for layoutlm_image_to_text to match expected signature."""
    try:
        png_pages: List[str] = pdf_to_png_with_pymupdf(pdf_path)
        results = []
        for png_page in png_pages:
            # Pass the include_bbox parameter directly to layoutlm_image_to_text
            # The function now handles the formatting internally
            result = layoutlm_image_to_text(png_page, include_bbox=include_bbox)
            results.append(result)
                
        return results
    except Exception as e:
        logger.exception(f"Fehler bei LayoutLM für '{pdf_path}': {e}")
        return None

# Adapter for doctr_pdf_to_text to match expected signature
def doctr_process_pdf(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[List[Dict[str, Any]]]] | None:
    """Adapter for doctr_pdf_to_text to match expected signature."""
    try:
        # Pass the include_bbox parameter directly to doctr_pdf_to_text
        # The function now handles the formatting internally
        return doctr_pdf_to_text(pdf_path, include_bbox=include_bbox)
    except Exception as e:
        logger.exception(f"Fehler bei Doctr OCR für '{pdf_path}': {e}")
        return None

# Adapter for paddleocr_pdf_to_text to match expected signature
def paddleocr_process_pdf(pdf_path: str, include_bbox: bool = False) -> Union[List[str], List[List[Dict[str, Any]]]] | None:
    try:
        return paddleocr_pdf_to_text(pdf_path, return_bbox=include_bbox)
    except Exception as e:
        logger.exception(f"Fehler bei PaddleOCR für '{pdf_path}': {e}")
        return None

_OCR_ENGINE_PDF_MAP: Dict[str, Callable] = {
    "doctr": doctr_process_pdf,
    "easyocr": easyocr_process_pdf,
    "tesseract": tesseract_process_pdf,
    "paddleocr": paddleocr_process_pdf,
    "layoutlm": layoutlm_process_pdf,
}


def ocr_pdf(pdf_path: str, *, engine: str = "paddleocr", include_bbox: bool = False) -> List[Union[str, List[Dict[str, Any]]]]:
    """
    Process a PDF file with the specified OCR engine.
    
    Args:
        pdf_path: Path to the PDF file
        engine: OCR engine to use
        include_bbox: Whether to include bounding box information
        
    Returns:
        When include_bbox=False: List of text strings (one per page)
        When include_bbox=True: List of lists of dicts with text and bbox info
    """
    engine = engine.lower()
    if engine not in _OCR_ENGINE_PDF_MAP:
        valid_options = ", ".join(_OCR_ENGINE_PDF_MAP.keys())
        raise ValueError(f"Unsupported OCR engine '{engine}'. Valid options are: {valid_options}.")
    
    ocr_function = _OCR_ENGINE_PDF_MAP[engine]
    result = ocr_function(pdf_path, include_bbox)
    
    if result is None:
        logger.error(f"OCR failed for '{pdf_path}' with engine '{engine}'. Returning empty list.")
        return []
    
    return result



def extract_invoice_fields_from_pdf(pdf_path: str, *, engine: str = "paddleocr", clean: bool = True, include_bbox: bool = False) -> Tuple[Dict, float, float]:
    """
    Full pipeline: PDF ➜ OCR ➜ clean ➜ LLM ➜ verify & correct ➜ post-process ➜ final dict.
    
    If include_bbox=True, use standardize_ocr_output to convert bbox data to formatted text.
    
    Returns:
        A tuple containing (extracted_fields, ollama_duration, processing_duration)
    """
    start_time = time.perf_counter()
    
    pages_raw_content = ocr_pdf(pdf_path, engine=engine, include_bbox=include_bbox)
    logger.info(f"OCR für '{os.path.basename(pdf_path)}' mit '{engine}' abgeschlossen. {len(pages_raw_content)} Seiten gefunden.")
    print(f"raw ocr text of {engine}: {pages_raw_content}")  # todo: remove debugprint

    final_text_parts: List[str] = []
    if include_bbox:
        # Convert bbox data to standardized format
        for page_content in pages_raw_content:
            if page_content:  # Skip empty pages
                try:
                    # Handle different bbox formats from different engines
                    formatted_text = ""
                    
                    # Handle single list of dictionaries (e.g., from layoutlm, tesseract)
                    if isinstance(page_content, list) and all(isinstance(item, dict) for item in page_content if item):
                        formatted_text = standardize_ocr_output(page_content, format_type="formatted_string")
                    
                    # Handle nested list of dictionaries (e.g., from doctr, paddleocr)
                    elif isinstance(page_content, list) and all(isinstance(item, list) for item in page_content if item):
                        formatted_text = standardize_ocr_output(page_content, format_type="formatted_string")
                    print(f"standardized text of engine {engine}: {formatted_text}") #todo: remove debugprint
                    if formatted_text and isinstance(formatted_text, str):
                        # if clean:
                        #     formatted_text = preprocess_plain_text_output(formatted_text)
                        final_text_parts.append(formatted_text)
                except Exception as e:
                    logger.exception(f"Error processing page with bbox data: {e}")
        print(f"final text of engine {engine}: {final_text_parts}")  # todo: remove debugprint
    else:
        # Handle plain text content
        for page_text in pages_raw_content:
            if page_text and isinstance(page_text, str):
                text_content = page_text
                # if clean:
                #     text_content = preprocess_plain_text_output(text_content)
                final_text_parts.append(text_content)
    
    if not final_text_parts:
        logger.error(f"No text content extracted from {pdf_path} with engine {engine}")
        raise ValueError("Input ocr_pages list cannot be empty.")
        
    logger.info(f"Extracted {len(final_text_parts)} pages of text content")
    logger.info(f"Gekürzte Vorschau des Textes: '{(' '.join(final_text_parts))[:200]}...'")

    # Now ollama_extract_invoice_fields returns a tuple (fields, ollama_duration)
    llm_output, ollama_duration = ollama_extract_invoice_fields(final_text_parts, include_bbox=include_bbox)

    full_text_for_verification = "\n".join(final_text_parts)
    corrected_dict = verify_and_correct_fields(llm_output, full_text_for_verification)

    final_dict = finalize_extracted_fields(corrected_dict)
    
    processing_duration = time.perf_counter() - start_time - ollama_duration

    return final_dict, ollama_duration, processing_duration


def get_available_engines() -> List[str]:
    return list(_OCR_ENGINE_PDF_MAP.keys())

if __name__ == "__main__":
    if not isinstance(SAMPLE_PDF_PATH, str) or not SAMPLE_PDF_PATH:
        logger.error("SAMPLE_PDF_PATH is not set or is not a string. Please check your configuration.")
    else:
        logger.info("\n=== Pipeline demo (Doctr) ===")
        fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="doctr")

    # print("\n=== Pipeline demo (Tesseract) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="tesseract")

    # print("\n=== Pipeline demo (LayoutLM) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="layoutlm")
    #
    # print("\n=== Pipeline demo (EasyOCR) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="easyocr")
    #
    # print("\n=== Pipeline demo (PaddleOCR) ===")
    # fields = extract_invoice_fields_from_pdf(SAMPLE_PDF_PATH, engine="paddleocr")
    # print("\n=== Pipeline demo (all engines) completed ===")