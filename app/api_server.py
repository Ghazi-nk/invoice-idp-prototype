import logging
from typing import Optional, List


from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

# --- Main pipeline functions ---
from app.pipeline import extract_invoice_fields_from_pdf

from app.ocr.ocr_manager import ocr_pdf, get_available_engines
from app.ocr.pdf_utils import save_base64_to_temp_pdf, extract_text_if_searchable
from semantic_extraction import ollama_extract_invoice_fields, ollama_process_with_custom_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

# --- API Setup ---
app = FastAPI(
    title="Invoice Extraction API",
    description="An API to extract structured data and perform OCR utilities on PDF invoices.",
    version="1.0.0"
)

DEFAULT_ENGINE = "tesseract"

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "API is running"}

# =============================================================================
# --- Request Models ---
# =============================================================================

class BaseRequest(BaseModel):
    """Base request model for endpoints that require a PDF file."""
    pdf_base64: str = Field(..., description="The PDF file encoded as a base64 string.")
    engine: Optional[str] = Field(DEFAULT_ENGINE, description=f"The OCR engine to use. Defaults to '{DEFAULT_ENGINE}'.")

class PDFQueryRequest(BaseRequest):
    """Request body for PDF query endpoint with custom prompt."""
    prompt: str = Field(..., description="The prompt to send to Ollama.")

class LLMExtractRequest(BaseModel):
    """Request for LLM-based invoice field extraction from pre-processed OCR text."""
    ocr_pages: List[str] = Field(..., description="List of OCR text per page.")

# =============================================================================
# --- Response Models ---
# =============================================================================

class TextResponse(BaseModel):
    """Base response model for text data."""
    result: str = Field(..., description="Extracted text data.")

class OCRTextResponse(BaseModel):
    """Response model for OCR text extraction."""
    ocr_text: List[str] = Field(..., description="List of OCR text extracted from each page.")

class InvoiceExtractionResponse(BaseModel):
    """Response model for invoice data extraction."""
    invoice_date: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    recipient_name: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    purchase_order_number: Optional[str] = None
    ust_id: Optional[str] = None
    iban: Optional[str] = None
    tax_rate: Optional[float] = None

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


@app.post("/api/v1/invoice-extract", summary="Extract Invoice Data", response_model=InvoiceExtractionResponse)
def extract_data(request: BaseRequest) -> InvoiceExtractionResponse:
    """
    Receives a base64-encoded PDF invoice and returns the extracted structured data as JSON.
    
    """
    engine_to_use = select_engine(request.engine)
    try:
        with save_base64_to_temp_pdf(request.pdf_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            extracted_data_tuple = extract_invoice_fields_from_pdf(
                pdf_path=temp_pdf_path,
                engine=engine_to_use
            )
            # Unpack only the first element (dictionary) from the tuple
            extracted_data = extracted_data_tuple[0]
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
def get_ocr_text(request: BaseRequest) -> OCRTextResponse:
    """
    Receives a base64-encoded PDF invoice and returns the raw OCR text per page.

    """
    engine_to_use = select_engine(request.engine)
    try:
        with save_base64_to_temp_pdf(request.pdf_base64) as temp_pdf_path:
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

@app.post("/api/v1/extract-searchable-text", summary="Extract Searchable Text from PDF", response_model=OCRTextResponse)
def extract_searchable_text(request: BaseRequest) -> OCRTextResponse:
    """
    Extracts all searchable text from a PDF (no OCR). Returns as a single string.
    """
    try:
        with save_base64_to_temp_pdf(request.pdf_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")
            text_json = extract_text_if_searchable(temp_pdf_path)
            if not text_json:
                text_json = ["pdf is not searchable or no text found."]
                
            return OCRTextResponse(ocr_text=text_json)
    except Exception as e:
        handle_error(e)
        return OCRTextResponse(ocr_text=[""])

@app.post("/api/v1/llm-invoice-extract", summary="LLM-based Field Extraction", response_model=InvoiceExtractionResponse)
def llm_extract(request: LLMExtractRequest) -> InvoiceExtractionResponse:
    """
    Runs LLM-based field extraction on provided OCR text pages. Returns extracted invoice fields.
    """
    try:
        extracted_data = ollama_extract_invoice_fields(request.ocr_pages)
        # Map 'ust-id' to 'ust_id' for the model if needed
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


@app.post("/api/v1/pdf-query", summary="Query PDF with Custom Prompt", response_model=TextResponse)
def pdf_query(request: PDFQueryRequest) -> TextResponse:
    """
    Receives a base64-encoded PDF and a custom prompt, performs OCR, and returns the language model's response.
    """
    engine_to_use = select_engine(request.engine)
    try:
        with save_base64_to_temp_pdf(request.pdf_base64) as temp_pdf_path:
            if not temp_pdf_path:
                raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

            # Get OCR text from the PDF
            ocr_text_per_page = ocr_pdf(pdf_path=temp_pdf_path, engine=engine_to_use)
            
            # Process with custom prompt and get raw response
            response_content = ollama_process_with_custom_prompt(ocr_text_per_page, request.prompt)
            
            return TextResponse(result=response_content)
    except Exception as e:
        handle_error(e)
        return TextResponse(result=f"Error: {str(e)}")


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)