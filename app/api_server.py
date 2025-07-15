# FILE: app/api_server.py

import os
import base64
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

# --- Main pipeline functions ---
from app.document_processing import (
    extract_invoice_fields_from_pdf,
    get_available_engines,
    ocr_pdf
)
from app.document_digitalization.pdf_utils import pdf_to_png_with_pymupdf, save_base64_to_temp_pdf, encode_image_to_base64

import logging

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
# --- Response Models ---
# =============================================================================

class ExtractedInvoiceResponse(BaseModel):
    """Response model for extracted invoice data."""
    data: dict

class OCRTextResponse(BaseModel):
    """Response model for OCR text extraction."""
    ocr_text: List[str]

class ImagesResponse(BaseModel):
    """Response model for PDF-to-images conversion."""
    images: List[str]


# =============================================================================
# --- API Endpoints ---
# =============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice_api")


def select_engine(engine: Optional[str]) -> str:
    """Helper to select and validate the OCR engine, falling back to default if needed."""
    if not engine:
        return DEFAULT_ENGINE
    engine_to_use = engine.lower()
    if engine_to_use not in get_available_engines():
        logger.warning(f"Invalid engine '{engine_to_use}' requested. Falling back to default.")
        return DEFAULT_ENGINE
    return engine_to_use


@app.post("/api/v1/extract", summary="Extract Invoice Data", response_model=ExtractedInvoiceResponse)
def extract_data(request: InvoiceRequest) -> ExtractedInvoiceResponse:
    """
    Receives a base64-encoded PDF invoice and returns the extracted structured data as JSON.
    
    Request Example:
    {
        "invoice_base64": "...",
        "engine": "tesseract"
    }
    
    Response Example:
    {
        "data": {"field1": "value1", ...}
    }
    """
    engine_to_use = select_engine(request.engine)
    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            extracted_data = extract_invoice_fields_from_pdf(
                pdf_path=temp_pdf_path,
                engine=engine_to_use,
                clean=True
            )
            return ExtractedInvoiceResponse(data=extracted_data)
    except Exception as e:
        handle_error(e)
        return ExtractedInvoiceResponse(data={})


@app.post("/api/v1/ocr", summary="Get Raw OCR Text", response_model=OCRTextResponse)
def get_ocr_text(request: InvoiceRequest) -> OCRTextResponse:
    """
    Receives a base64-encoded PDF invoice and returns the raw OCR text per page.
    
    Request Example:
    {
        "invoice_base64": "...",
        "engine": "tesseract"
    }
    
    Response Example:
    {
        "ocr_text": ["page 1 text", "page 2 text", ...]
    }
    """
    engine_to_use = select_engine(request.engine)
    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            ocr_text_per_page = ocr_pdf(pdf_path=temp_pdf_path, engine=engine_to_use)
            return OCRTextResponse(ocr_text=ocr_text_per_page)
    except Exception as e:
        handle_error(e)
        return OCRTextResponse(ocr_text=[])


@app.post("/api/v1/to-images", summary="Convert PDF to Images (Base64)", response_model=ImagesResponse)
def pdf_to_images_base64(request: PDFRequest) -> ImagesResponse:
    """
    Receives a base64-encoded PDF and returns a list of base64-encoded PNG images for each page.
    
    Request Example:
    {
        "invoice_base64": "..."
    }
    
    Response Example:
    {
        "images": ["base64img1", "base64img2", ...]
    }
    """
    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            image_paths = pdf_to_png_with_pymupdf(temp_pdf_path)
            encoded_images = []
            for img_path in image_paths:
                encoded_images.append(encode_image_to_base64(img_path))
                if os.path.exists(img_path):
                    os.remove(img_path)

            return ImagesResponse(images=encoded_images)
    except Exception as e:
        handle_error(e)
        return ImagesResponse(images=[])


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
    """Helper to raise appropriate HTTP exceptions and log errors."""
    if isinstance(e, HTTPException):
        logger.error(f"HTTPException: {e.detail}")
        raise e
    elif isinstance(e, FileNotFoundError):
        logger.error(f"FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    else:
        logger.exception("An unexpected server error occurred:")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# To run this server:
# 1. Install fastapi and uvicorn: pip install fastapi "uvicorn[standard]"
# 2. From your project root directory, run: uvicorn app.api_server:app --reload