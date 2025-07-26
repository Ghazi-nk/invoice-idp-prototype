from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import logging

from app.ocr.ocr_manager import ocr_pdf
from app.post_processing import finalize_extracted_fields, verify_and_correct_fields
from app.config import SAMPLE_PDF_PATH
from app.semantic_extraction_strategies.semantic_extraction import ollama_extract_invoice_fields


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")


def process_invoice(pdf_path: str, *, engine: str = "paddleocr") -> Tuple[Dict, float, float]:
    """
    Full pipeline: PDF ➜ OCR ➜ LLM ➜ verify & correct ➜ post-process ➜ final dict.
    
    Args:
        pdf_path: Path to the PDF file to process
        engine: OCR engine to use (default: paddleocr)
    
    Returns:
        A tuple containing (extracted_fields, ollama_duration, processing_duration)
    """
    start_time = time.perf_counter()
    
    pages_raw_content = ocr_pdf(pdf_path, engine=engine)
    logger.info(f"OCR für '{os.path.basename(pdf_path)}' mit '{engine}' abgeschlossen. {len(pages_raw_content)} Seiten gefunden.")
    print(f"raw ocr text of {engine}: {pages_raw_content}")  # todo: remove debugprint

    # Handle plain text content directly
    final_text_parts = []
    for page_text in pages_raw_content:
        if page_text and isinstance(page_text, str):
            final_text_parts.append(page_text)
    
    if not final_text_parts:
        logger.error(f"No text content extracted from {pdf_path} with engine {engine}")
        raise ValueError("No text content was extracted from the PDF.")
        
    logger.info(f"Extracted {len(final_text_parts)} pages of text content")
    logger.info(f"Gekürzte Vorschau des Textes: '{(' '.join(final_text_parts))[:200]}...'")

    # Extract fields using LLM without bbox
    llm_output, ollama_duration = ollama_extract_invoice_fields(final_text_parts)

    full_text_for_verification = "\n".join(final_text_parts)
    corrected_dict = verify_and_correct_fields(llm_output, full_text_for_verification)

    final_dict = finalize_extracted_fields(corrected_dict)
    
    processing_duration = time.perf_counter() - start_time - ollama_duration

    return final_dict, ollama_duration, processing_duration


# Legacy alias for backward compatibility
def extract_invoice_fields_from_pdf(pdf_path: str, *, engine: str = "paddleocr") -> Tuple[Dict, float, float]:
    """
    Legacy alias for process_invoice function.
    
    Args:
        pdf_path: Path to the PDF file to process
        engine: OCR engine to use (default: paddleocr)
    
    Returns:
        A tuple containing (extracted_fields, ollama_duration, processing_duration)
    """
    return process_invoice(pdf_path, engine=engine)


if __name__ == "__main__":
    if not isinstance(SAMPLE_PDF_PATH, str) or not SAMPLE_PDF_PATH:
        logger.error("SAMPLE_PDF_PATH is not set or is not a string. Please check your configuration.")
    else:
        logger.info("\n=== Pipeline demo (Doctr) ===")
        fields = process_invoice(SAMPLE_PDF_PATH, engine="doctr") 