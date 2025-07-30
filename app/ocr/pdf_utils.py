# FILE: utils/pdf_utils.py

import base64
import logging
import os
import tempfile
from contextlib import contextmanager

import fitz

from app.config import TMP_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_utils")

@contextmanager
def save_base64_to_temp_pdf(base64_string: str):
    """
    Decodes a base64 string and saves it to a temporary PDF file.
    Yields the path to the temporary file, and ensures cleanup.
    Logs errors if decoding or file operations fail.
    """
    temp_file_path = None
    try:
        pdf_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_file_path = temp_file.name
        yield temp_file_path
    except Exception as e:
        logger.exception("Failed to save base64 PDF to temp file:")
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")

def encode_image_to_base64(image_path: str) -> str:
    """
    Reads an image file and encodes its content into a base64 string.
    Logs errors if file reading fails.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.exception(f"Failed to encode image to base64: {image_path}")
        raise


def extract_text_if_searchable(pdf_path: str) -> list[str]:
    """
    Open the PDF at `pdf_path`, extract all text, and return a list of strings (one per page).
    If no text is found (i.e. likely a scanned/image PDF), returns an empty list.
    Logs errors if PDF cannot be opened or read.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.exception(f"Could not open PDF: {pdf_path}")
        raise RuntimeError(f"Could not open PDF: {e}")
    page_texts = []
    for page in doc:
        text = page.get_text()
        if isinstance(text, str):
            page_texts.append(text.strip())
        else:
            page_texts.append("")
    return page_texts

def pdf_to_png_with_pymupdf(pdf_path: str, zoom: float = 3.0) -> list[str]:
    """
    Converts PDF pages to PNG images with high quality using PyMuPDF.
    zoom=3.0 corresponds to about 300 DPI.
    Logs errors if conversion fails.
    """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            logger.error(f"Keine Seiten in PDF: {pdf_path}")
            raise RuntimeError(f"Keine Seiten in PDF: {pdf_path}")
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        png_paths = []
        for i, page in enumerate(doc):
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_path = os.path.join(TMP_DIR, f"{base}_page{i + 1}.png")
            pix.save(png_path)
            png_paths.append(png_path)
        return png_paths
    except Exception as e:
        logger.exception(f"Failed to convert PDF to PNGs with PyMuPDF: {pdf_path}")
        raise