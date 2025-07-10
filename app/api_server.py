import os
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Main pipeline functions ---
from document_processing import extract_invoice_fields_from_pdf, get_available_engines
from utils.pdf_utils import save_base64_to_temp_pdf

# --- API Setup ---
app = FastAPI(
    title="Invoice Extraction API",
    description="An API to extract structured data from PDF invoices using various OCR engines.",
    version="1.0.0"
)

# --- Default Engine Configuration ---
# Tesseract has shown good, consistent performance
DEFAULT_ENGINE = "tesseract"


class InvoiceRequest(BaseModel):
    """Defines the structure of the API request body."""
    invoice_base64: str = Field(...,
                                description="The PDF invoice file encoded as a base64 string.")
    engine: Optional[str] = Field(DEFAULT_ENGINE,
                                  description=f"The OCR engine to use. Defaults to '{DEFAULT_ENGINE}'.")


@app.post("/api/v1/extract", summary="Extract Invoice Data")
def extract_data(request: InvoiceRequest):
    """
    Receives a base64-encoded PDF invoice and an engine name, processes it,
    and returns the extracted structured data as JSON.
    """
    # 1. Validate the chosen engine
    engine_to_use = request.engine.lower()
    if engine_to_use not in get_available_engines():
        print(f"Warning: Invalid engine '{engine_to_use}' requested. Falling back to default.")
        engine_to_use = DEFAULT_ENGINE

    # 2. Decode the base64 string and save to a temporary file
    try:
        # The 'with' statement ensures the temp file is automatically deleted
        with save_base64_to_temp_pdf(request.invoice_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            # 3. Process the file using the main pipeline
            try:
                extracted_data = extract_invoice_fields_from_pdf(
                    pdf_path=temp_pdf_path,
                    engine=engine_to_use,
                    clean=True
                )
                return extracted_data
            except Exception as e:
                # Handle errors during the main pipeline processing
                raise HTTPException(status_code=500,
                                    detail=f"An error occurred during invoice processing: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during file handling
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# To run this server:
# 1. Make sure you have fastapi and uvicorn installed: pip install fastapi "uvicorn[standard]"
# 2. From your project root directory, run: uvicorn app.api_server:app --reload