"""
FastAPI-basierte REST-API für Intelligent Document Processing.

Dieses Modul implementiert eine RESTful API zur Verarbeitung von PDF-Rechnungen
mit verschiedenen OCR-Engines und LLM-basierter semantischer Extraktion.
Die API bietet sowohl vollständige Pipeline-Verarbeitung als auch modulare
Endpunkte für einzelne Verarbeitungsschritte.

Verfügbare Endpunkte:
- POST /api/v1/invoice-extract: Vollständige Pipeline-Verarbeitung (PDF → strukturierte Daten)
- POST /api/v1/ocr: Reine OCR-Texterkennung 
- POST /api/v1/extract-searchable-text: Durchsuchbarer PDF-Text
- POST /api/v1/llm-invoice-extract: LLM-basierte Extraktion aus OCR-Text
- POST /api/v1/pdf-query: Freie Abfragen mit benutzerdefinierten Prompts
- GET /api/v1/ocr-engines: Liste der verfügbaren OCR-Engines
- GET /docs: openapi documentation

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import logging
from typing import Optional, List


from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

# --- Main pipeline functions ---
from app.pipeline import extract_invoice_fields_from_pdf

from app.ocr.ocr_manager import ocr_pdf, get_available_engines
from app.ocr.pdf_utils import save_base64_to_temp_pdf, extract_text_if_searchable
from app.semantic_extraction import ollama_extract_invoice_fields, ollama_process_with_custom_prompt

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
    """
    Basis-Request-Modell für Endpunkte, die eine PDF-Datei benötigen.
    
    Dieses Modell stellt die gemeinsamen Felder für alle PDF-verarbeitenden
    Endpunkte zur Verfügung und gewährleistet konsistente API-Schemas.
    """
    pdf_base64: str = Field(..., description="PDF-Datei als Base64-kodierter String")
    engine: Optional[str] = Field(DEFAULT_ENGINE, description=f"OCR-Engine zur Texterkennung. Standard: '{DEFAULT_ENGINE}'. Verfügbar: tesseract, paddleocr, easyocr, doctr, layoutlmv3")

class PDFQueryRequest(BaseRequest):
    """
    Request-Modell für PDF-Abfragen mit benutzerdefiniertem Prompt.
    
    Erweitert das BaseRequest um ein Prompt-Feld für freie Abfragen
    an das Dokument mittels LLM.
    """
    prompt: str = Field(..., description="Benutzerdefinierter Prompt für die Dokumentenabfrage")

class LLMExtractRequest(BaseModel):
    """
    Request-Modell für LLM-basierte Feldextraktion aus vorverarbeitetem OCR-Text.
    
    Dieser Request-Typ ermöglicht die direkte LLM-Verarbeitung von bereits
    extrahiertem OCR-Text ohne erneute PDF-Verarbeitung.
    """
    ocr_pages: List[str] = Field(..., description="Liste von OCR-Texten pro Seite")

# =============================================================================
# --- Response Models ---
# =============================================================================

class TextResponse(BaseModel):
    """
    Basis-Response-Modell für Textdaten.
    
    Wird für einfache Text-Antworten verwendet, z.B. für durchsuchbaren
    PDF-Text oder LLM-Antworten.
    """
    result: str = Field(..., description="Extrahierte Textdaten")

class OCRTextResponse(BaseModel):
    """
    Response-Modell für OCR-Texterkennung.
    
    Enthält den erkannten Text pro Seite als strukturierte Liste.
    """
    ocr_text: List[str] = Field(..., description="Liste von OCR-Text pro Seite")

class InvoiceExtractionResponse(BaseModel):
    """
    Response-Modell für strukturierte Rechnungsdaten-Extraktion.
    
    Definiert das Schema für alle extrahierbaren Rechnungsfelder gemäß
    dem IncomingInvoiceSchema. Alle Felder sind optional, da nicht jede
    Rechnung alle Informationen enthält.
    """
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
    """
    Wählt und validiert die OCR-Engine, mit Fallback auf Standard-Engine.
    
    Args:
        engine (Optional[str]): Gewünschte OCR-Engine oder None
        
    Returns:
        str: Validierte OCR-Engine als String
        
    Note:
        Fällt automatisch auf DEFAULT_ENGINE zurück bei ungültigen Eingaben.
    """
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
    Vollständige Pipeline-Verarbeitung: Extrahiert strukturierte Daten aus PDF-Rechnungen.
    
    Dieser Endpunkt führt die komplette IDP-Pipeline durch:
    1. Base64-PDF dekodieren und temporär speichern
    2. OCR-basierte Texterkennung mit gewählter Engine
    3. LLM-basierte semantische Extraktion der Rechnungsfelder
    4. Verifikation und Post-Processing der Daten
    
    Args:
        request (BaseRequest): Request mit Base64-kodierter PDF und OCR-Engine
        
    Returns:
        InvoiceExtractionResponse: Strukturierte Rechnungsdaten
        
    Raises:
        HTTPException: Bei ungültigen PDF-Daten oder Verarbeitungsfehlern

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
    Extrahiert reinen OCR-Text aus einer PDF-Rechnung.
    
    Dieser Endpunkt führt nur die OCR-Verarbeitung durch und gibt
    den erkannten Text zurück, ohne LLM-basierte Feldextraktion.
    
    Args:
        request (BaseRequest): Request mit Base64-PDF und OCR-Engine
        
    Returns:
        OCRTextResponse: Roher OCR-Text pro Seite
        
    Raises:
        HTTPException: Bei ungültigen PDF-Daten oder OCR-Fehlern
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
    Gibt eine Liste aller verfügbaren OCR-Engines zurück.
    
    Returns:
        List[str]: Namen aller konfigurierten OCR-Engines
        
    
    """
    return get_available_engines()

@app.post("/api/v1/extract-searchable-text", summary="Extract Searchable Text from PDF", response_model=OCRTextResponse)
def extract_searchable_text(request: BaseRequest) -> OCRTextResponse:
    """
    Extrahiert durchsuchbaren Text aus einer PDF ohne OCR-Verarbeitung.
    
    Dieser Endpunkt nutzt eingebetteten Text in PDFs und ist deutlich
    schneller als OCR-Verarbeitung. Für gescannte PDFs ohne Text
    sollte der /ocr Endpunkt verwendet werden.
    
    Args:
        request (BaseRequest): Request mit Base64-PDF
        
    Returns:
        OCRTextResponse: Extrahierter Text pro Seite oder Fehlermeldung
        
    Note:
        Gibt Fehlermeldung zurück wenn PDF keinen durchsuchbaren Text enthält.
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
    Führt LLM-basierte Feldextraktion auf vorhandenem OCR-Text durch.
    
    Dieser Endpunkt nimmt bereits extrahierten OCR-Text entgegen
    und führt nur die semantische Extraktion und Post-Processing durch.
    Ideal für modulare Verarbeitung oder wenn OCR bereits durchgeführt wurde.
    
    Args:
        request (LLMExtractRequest): Request mit OCR-Text-Seiten
        
    Returns:
        InvoiceExtractionResponse: Strukturierte Rechnungsdaten
    
    """
    try:
        extracted_data_tuple = ollama_extract_invoice_fields(request.ocr_pages)
        logger.info(f"Extracted data tuple type: {type(extracted_data_tuple)}")
        logger.info(f"Extracted data tuple length: {len(extracted_data_tuple) if hasattr(extracted_data_tuple, '__len__') else 'N/A'}")
        
        # Unpack the tuple to get just the dictionary (ignore the duration)
        extracted_data, duration = extracted_data_tuple
        logger.info(f"Extracted data type: {type(extracted_data)}")
        logger.info(f"Extracted data content: {extracted_data}")
        
        # Ensure extracted_data is a dictionary
        if not isinstance(extracted_data, dict):
            logger.error(f"Expected dict, got {type(extracted_data)}: {extracted_data}")
            raise ValueError(f"Expected dictionary from LLM extraction, got {type(extracted_data)}")
        
        # Map 'ust-id' to 'ust_id' for the model if needed
        if 'ust-id' in extracted_data:
            extracted_data['ust_id'] = extracted_data.pop('ust-id')
        
        return InvoiceExtractionResponse(**extracted_data)
    except Exception as e:
        logger.exception("Error in llm_extract endpoint:")
        # Return a default response instead of raising an exception
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
    Führt benutzerdefinierte Abfragen gegen PDF-Dokumente durch.
    
    Dieser Endpunkt ermöglicht freie Textabfragen an PDF-Dokumente
    mittels LLM. Im Gegensatz zur strukturierten Feldextraktion
    können hier beliebige Fragen gestellt werden.
    
    Args:
        request (PDFQueryRequest): Request mit Base64-PDF, Engine und Custom Prompt
        
    Returns:
        TextResponse: Freie LLM-Antwort auf die Benutzerabfrage
        
    Examples:
        - "Fasse den Inhalt dieser Rechnung zusammen"
        - "Welche Produkte wurden gekauft?"
        - "Gibt es Rabatte oder Sonderkonditionen?"
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
    """
    Zentrale Fehlerbehandlung für die API.
    
    Diese Funktion behandelt alle unerwarteten Exceptions in den
    API-Endpunkten und wirft entsprechende HTTPExceptions mit
    benutzerfreundlichen Fehlermeldungen.
    
    Args:
        e (Exception): Aufgetretene Exception
        
    Raises:
        HTTPException: HTTP 500 mit Fehlerbeschreibung
        
    Note:
        Loggt alle Fehler für Debugging und Monitoring.
    """
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