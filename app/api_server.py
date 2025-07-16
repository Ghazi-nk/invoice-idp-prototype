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
from app.document_digitalization.pdf_utils import pdf_to_png_with_pymupdf, save_base64_to_temp_pdf, encode_image_to_base64, extract_text_if_searchable
from app.semantic_extraction import ollama_extract_invoice_fields

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

class OCRTextResponse(BaseModel):
    """Response model for OCR text extraction."""
    ocr_text: List[str]

class ImagesResponse(BaseModel):
    """Response model for PDF-to-images conversion."""
    images: List[str]


class InvoiceExtractionResponse(BaseModel):
    invoice_date: Optional[str] = Field(default=None, example="01.01.2024")
    vendor_name: Optional[str] = Field(default=None, example="Max Mustermann GmbH")
    invoice_number: Optional[str] = Field(default=None, example="RE-2024-0001")
    recipient_name: Optional[str] = Field(default=None, example="Erika Musterfrau AG")
    total_amount: Optional[float] = Field(default=None, example=1234.56)
    currency: Optional[str] = Field(default=None, example="EUR")
    purchase_order_number: Optional[str] = Field(default=None, example=None)
    ust_id: Optional[str] = Field(default=None, example=None, alias="ust-id")
    iban: Optional[str] = Field(default=None, example="DE00123456781234567890")
    tax_rate: Optional[float] = Field(default=None, example=19.0)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "invoice_date": "01.01.2024",
                "vendor_name": "Max Mustermann GmbH",
                "invoice_number": "RE-2024-0001",
                "recipient_name": "Erika Musterfrau AG",
                "total_amount": 1234.56,
                "currency": "EUR",
                "purchase_order_number": None,
                "ust-id": None,
                "iban": "DE00123456781234567890",
                "tax_rate": 19.0
            }
        }


class SearchableTextRequest(BaseModel):
    invoice_base64: str = Field(..., description="The PDF file encoded as a base64 string.")

class SearchableTextResponse(BaseModel):
    text: str = Field(..., description="All extracted searchable text from the PDF.")

class LLMExtractRequest(BaseModel):
    ocr_pages: List[str] = Field(..., description="List of OCR text per page.")

class LLMExtractResponse(BaseModel):
    fields: dict = Field(..., description="Extracted invoice fields from LLM.")


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


@app.post("/api/v1/extract", summary="Extract Invoice Data", response_model=InvoiceExtractionResponse)
def extract_data(request: InvoiceRequest) -> InvoiceExtractionResponse:
    """
    Receives a base64-encoded PDF invoice and returns the extracted structured data as JSON.
    
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
            # Map 'ust-id' to 'ust_id' for the model
            if 'ust-id' in extracted_data:
                extracted_data['ust_id'] = extracted_data.pop('ust-id')
            return InvoiceExtractionResponse(**extracted_data)
    except Exception as e:
        handle_error(e)
        return InvoiceExtractionResponse(
            invoice_date="",
            vendor_name="",
            invoice_number="",
            recipient_name="",
            total_amount=0.0,
            currency="",
            purchase_order_number=None,
            ust_id=None,
            iban="",
            tax_rate=0.0
        )


@app.post("/api/v1/ocr", summary="Get Raw OCR Text", response_model=OCRTextResponse)
def get_ocr_text(request: InvoiceRequest) -> OCRTextResponse:
    """
    Receives a base64-encoded PDF invoice and returns the raw OCR text per page.

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


@app.get("/api/v1/ocr-engines", summary="List Available OCR Engines")
def get_ocr_engines() -> List[str]:
    """
    Returns a list of available OCR engines.
    """
    return get_available_engines()

@app.post("/api/v1/extract-searchable-text", summary="Extract Searchable Text from PDF", response_model=SearchableTextResponse)
def extract_searchable_text(request: SearchableTextRequest) -> SearchableTextResponse:
    """
    Extracts all searchable text from a PDF (no OCR). Returns as a single string.
    """
    try:
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")
            text_json = extract_text_if_searchable(temp_pdf_path)
            # text_json is a JSON string, so parse it
            import json
            text = json.loads(text_json) if text_json else ""
            return SearchableTextResponse(text=text)
    except Exception as e:
        handle_error(e)
        return SearchableTextResponse(text="")

@app.post("/api/v1/llm-extract", summary="LLM-based Field Extraction", response_model=LLMExtractResponse)
def llm_extract(request: LLMExtractRequest) -> LLMExtractResponse:
    """
    Runs LLM-based field extraction on provided OCR text pages. Returns extracted invoice fields.
    """
    try:
        fields = ollama_extract_invoice_fields(request.ocr_pages)
        return LLMExtractResponse(fields=fields)
    except Exception as e:
        handle_error(e)
        return LLMExtractResponse(fields={})


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