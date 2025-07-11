# FILE: app/api_server.py

import os
import tempfile
import base64
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# --- Main pipeline functions ---
from document_processing import (
    extract_invoice_fields_from_pdf,
    get_available_engines,
    ocr_pdf
)
from utils.pdf_utils import pdf_to_png_with_pymupdf,save_base64_to_temp_pdf, encode_image_to_base64

# --- API Setup ---
app = FastAPI(
    title="Invoice Extraction API",
    description="An API to extract structured data and perform OCR utilities on PDF invoices.",
    version="1.0.0"
)

DEFAULT_ENGINE = "tesseract"


# =============================================================================
# --- Request Models ---
# =============================================================================

class InvoiceRequest(BaseModel):
    """Request body for extraction and OCR endpoints."""
    invoice_base64: str = Field(..., description="The PDF invoice file encoded as a base64 string.")
    engine: Optional[str] = Field(DEFAULT_ENGINE, description=f"The OCR engine to use. Defaults to '{DEFAULT_ENGINE}'.")


class PDFRequest(BaseModel):
    """Request body for PDF utility endpoints."""
    invoice_base64: str = Field(..., description="The PDF file encoded as a base64 string.")


# =============================================================================
# --- API Endpoints ---
# =============================================================================

@app.post("/api/v1/extract", summary="Extract Invoice Data")
def extract_data(request: InvoiceRequest):
    """
    Receives a base64-encoded PDF invoice and returns the extracted structured data as JSON.
    """
    engine_to_use = request.engine.lower()
    if engine_to_use not in get_available_engines():
        print(f"Warning: Invalid engine '{engine_to_use}' requested. Falling back to default.")
        engine_to_use = DEFAULT_ENGINE

    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            extracted_data = extract_invoice_fields_from_pdf(
                pdf_path=temp_pdf_path,
                engine=engine_to_use,
                clean=True
            )
            return extracted_data
    except Exception as e:
        handle_error(e)


@app.post("/api/v1/ocr", summary="Get Raw OCR Text")
def get_ocr_text(request: InvoiceRequest) -> List[str]:
    """
    Receives a base64-encoded PDF invoice and returns the raw OCR text per page.
    """
    engine_to_use = request.engine.lower()
    if engine_to_use not in get_available_engines():
        print(f"Warning: Invalid engine '{engine_to_use}' requested. Falling back to default.")
        engine_to_use = DEFAULT_ENGINE

    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            # Call the OCR step directly
            ocr_text_per_page = ocr_pdf(pdf_path=temp_pdf_path, engine=engine_to_use)
            return ocr_text_per_page
    except Exception as e:
        handle_error(e)


@app.post("/api/v1/to-images", summary="Convert PDF to Images (Base64)")
def pdf_to_images_base64(request: PDFRequest):
    """
    Receives a base64-encoded PDF and returns a list of base64-encoded PNG images for each page.
    """
    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            # Generate image paths
            image_paths = pdf_to_png_with_pymupdf(temp_pdf_path)

            # Encode images to base64
            encoded_images = []
            for img_path in image_paths:
                encoded_images.append(encode_image_to_base64(img_path))
                # Clean up the generated image file immediately
                if os.path.exists(img_path):
                    os.remove(img_path)

            return {"images": encoded_images}
    except Exception as e:
        handle_error(e)


@app.post("/api/v1/to-pdf", summary="Convert Base64 to PDF File")
def base64_to_pdf_file(request: PDFRequest):
    """
    Receives a base64-encoded string and returns it as a downloadable PDF file.
    """
    try:
        pdf_data = base64.b64decode(request.invoice_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

    return Response(content=pdf_data, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=invoice.pdf"})


def handle_error(e: Exception):
    """Helper to raise appropriate HTTP exceptions."""
    if isinstance(e, HTTPException):
        raise e
    elif isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# To run this server:
# 1. Install fastapi and uvicorn: pip install fastapi "uvicorn[standard]"
# 2. From your project root directory, run: uvicorn app.api_server:app --reload