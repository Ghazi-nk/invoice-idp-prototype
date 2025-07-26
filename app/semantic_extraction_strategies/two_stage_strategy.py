"""
Two Stage Strategy: 
1. Extract interesting lines from each page that might contain field data
2. Extract specific fields from the collected lines
"""
import warnings
import time
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
import requests

from app.ocr.doctr_pdf2txt import doctr_pdf_to_text

from app.config import (
    CHAT_ENDPOINT,
    OLLAMA_MODEL, SAMPLE_PDF_PATH,
stage_1_prompt_system,
stage_1_prompt_user,
stage_2_prompt_system,
stage_2_prompt_user)



def load_prompt(filename: str) -> str:
    """Load prompt from file."""
    prompt_path = Path("resources/prompts/semantic_extraction/tow_stage") / filename
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found at {prompt_path}")
        return ""


def stage1_extract_interesting_lines(ocr_pages: List[str]) -> str:
    """
    Extracts lines of interest from each page and returns them as a single string.

    Args:
        ocr_pages: List of strings containing OCR text from each page

    Returns:
        A string containing all collected lines of interest from the pages.
    """
    try:
        if stage_1_prompt_system is None or stage_1_prompt_user is None:
            raise FileNotFoundError("Required prompt files are not set in environment variables.")

        system_prompt = Path(stage_1_prompt_system).read_text(encoding="utf-8")
        user_prompt = Path(stage_1_prompt_user).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find a required prompt file: {e}")

    # Build the initial message list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    # Add each page's text as a separate user message
    if not ocr_pages:
        raise ValueError("Input ocr_pages list cannot be empty.")

    num_pages = len(ocr_pages)
    for i, page_text in enumerate(ocr_pages):
        message_content = f"Here is the text for Page {i + 1} of {num_pages}:\n\n---\n{page_text}\n---"
        messages.append({"role": "user", "content": message_content})

    # Add the final user prompt to trigger the JSON generation
    messages.append({"role": "user", "content": user_prompt.strip()})

    # Send the complete conversation to the chat endpoint
    body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}

    # Suppress warnings
    warnings.filterwarnings("ignore")

    print(f"Sending {len(messages)} messages ({num_pages} pages) to the chat model...")
    ollama_start_time = time.perf_counter()
    resp = requests.post(CHAT_ENDPOINT, json=body, verify=False, timeout=600)
    ollama_duration = time.perf_counter() - ollama_start_time

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API Error: {resp.status_code} – {resp.text}")

    # Extract the JSON from the final assistant message
    try:
        raw_content = resp.json().get("message", {}).get("content", "")
    except (AttributeError, KeyError):
        raise ValueError("Ollama chat response is not in the expected format.")
    # Parse the JSON response
    print(f"Ollama response: {raw_content}")
    return raw_content, ollama_duration





def _extract_field_from_lines(collected_lines: str, field_name: str) -> str:
    """Extract a specific field from the collected lines using chat endpoint."""
    try:
        # Load prompts from files
        system_prompt_path = Path(stage_2_prompt_system)
        user_prompt_path = Path(stage_2_prompt_user)
        
        try:
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
            user_prompt_template = user_prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise RuntimeError(f"Could not find stage 2 prompt file: {e}")
        
        # Define detailed field prompts
        field_prompts = {
            "invoice_number": "Extrahiere die Rechnungsnummer. Suche nach Schlüsselwörtern wie 'Rechnungsnummer', 'Rechnung', 'Rechn.-Nr.', 'Rg.-Nr.', 'Belegnummer', 'Invoice No.' und dem darauffolgenden alphanumerischen Wert. Gib diesen als String zurück.",

            "invoice_date": "Extrahiere das Rechnungsdatum. Suche nach Schlüsselwörtern wie 'Datum', 'Rechnungsdatum', 'Date', 'Ausgestellt am'. Das Datum muss als String im Format 'DD.MM.YYYY' zurückgegeben werden. Formate wie 'YYYY-MM-DD' oder 'YYYY.MM.DD' müssen entsprechend umgewandelt werden.",

            "vendor_name": "Extrahiere den Namen des Rechnungsstellers (Verkäufer, Absender). Dieser steht typischerweise im Kopfbereich der Rechnung, oft als erstes oder größtes Logo/Firmenname.",

            "recipient_name": "Extrahiere den Namen des Rechnungsempfängers (Kunde). Suche gezielt nach Adressblöcken, die mit 'An:', 'Firma:', 'Kunde:', 'Rechnung an:' oder 'Recipient:' eingeleitet werden.",

            "total_amount": "Extrahiere den Brutto-Gesamtbetrag. Suche nach dem höchsten Betrag auf der Rechnung, oft in Verbindung mit Begriffen wie 'Gesamt', 'Total', 'Endbetrag', 'Summe', 'Zu zahlen'. Der Wert muss eine Zahl sein, mit einem Punkt als Dezimaltrennzeichen und exakt zwei Nachkommastellen. Entferne jegliche Tausendertrennzeichen. Beispiel: '1.234,56 €' wird zu 1234.56.",

            "currency": "Extrahiere die Währung. Gib den 3-stelligen ISO-Code zurück (z.B. 'EUR', 'CHF', 'USD'). Wenn das Symbol '€' oder der Text 'EUR' gefunden wird, gib 'EUR' zurück.",

            "purchase_order_number": "Extrahiere die Bestellnummer. Suche nach Schlüsselwörtern wie 'Bestellnummer', 'Bestellung', 'Auftragsnummer', 'PO Number', 'Order No.'. Wenn keine gefunden wird, gib explizit null zurück, keinen leeren String.",

            "ust_id": "Extrahiere die Umsatzsteuer-Identifikationsnummer (USt-IdNr. oder VAT ID). Der Wert muss mit zwei Großbuchstaben beginnen und einem der exakten EU-Regex-Muster entsprechen (z.B. DE gefolgt von 9 Ziffern). Ignoriere normale Steuernummern, Handelsregisternummern (HRB) oder Finanzamt-Nummern. Wenn kein gültiger Wert gefunden wird, gib explizit null zurück.",

            "iban": "Extrahiere die IBAN. Suche zuerst nach dem Schlüsselwort 'IBAN' oder 'Konto' und extrahiere dann die erste darauf folgende gültige IBAN, die dem Muster [A-Z]{2}[0-9A-Z]{13,32} entspricht. Wenn keine gefunden wird, gib explizit null zurück.",

            "tax_rate": "Extrahiere den primären Umsatzsteuersatz. Suche nach Prozentangaben (%) in der Nähe von Begriffen wie 'MwSt.', 'USt.', 'Mehrwertsteuer', 'Tax'. Der Wert muss eine Zahl mit einem Punkt als Dezimaltrennzeichen und zwei Nachkommastellen sein. Beispiel: '19%' wird zu 19.00."
        }
        
        field_prompt = field_prompts.get(field_name, f"Extrahiere das Feld: {field_name}")
        
        # Format user prompt
        user_prompt = user_prompt_template.format(
            text=collected_lines,
            field_prompt=field_prompt
        )
        
        # Build messages for chat endpoint
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
        
        # Send request to chat endpoint
        body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
        
        warnings.filterwarnings("ignore")
        resp = requests.post(CHAT_ENDPOINT, json=body, verify=False, timeout=120)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API Error: {resp.status_code} – {resp.text}")
        
        # Extract response
        try:
            raw_content = resp.json().get("message", {}).get("content", "")
        except (AttributeError, KeyError):
            raise ValueError("Ollama chat response is not in the expected format.")
        
        # Clean response
        if raw_content:
            response = raw_content.strip()
            # Remove common non-answers
            if response.lower() in ['null', 'none', 'nicht vorhanden', 'nicht gefunden', 'keine', 'kein', '', 'nicht_gefunden']:
                return None
            # Remove "ja" prefixes
            response = re.sub(r'^(ja,?\s*)', '', response, flags=re.IGNORECASE)
            return response
        
        return None
    except Exception as e:
        print(f"Error extracting field {field_name}: {e}")
        return None


def _normalize_date(date_str: str) -> str:
    """Normalize date to YYYY-MM-DD format."""
    if not date_str:
        return None
    
    import datetime
    
    # Try different date formats
    formats = [
        '%d.%m.%Y', '%d/%m/%Y', '%d.%m.%y', '%d/%m/%y',
        '%Y-%m-%d', '%Y/%m/%d'
    ]
    
    # Extract date pattern from string
    date_patterns = [
        r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b',
        r'\b(\d{4}-\d{1,2}-\d{1,2})\b'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            date_part = match.group(1)
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(date_part, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    
    return date_str


def _normalize_amount(amount_str: str) -> str:
    """Extract and normalize amount."""
    if not amount_str:
        return None
    
    # Find numbers
    number_match = re.search(r'\b(\d+[.,]?\d*)\b', amount_str)
    if number_match:
        return number_match.group(1).replace(',', '.')
    
    return None


def ollama_extract_invoice_fields(ocr_pages: List[str]) -> Tuple[Dict, float]:
    """
    Two-stage extraction:
    1. Collect interesting lines from each page
    2. Extract fields from collected lines
    """
    start_time = time.perf_counter()
    
    # Stage 1: Extract interesting lines from all pages
    print("Stage 1: Collecting interesting lines from all pages...")
    all_collected, stage1_duration = stage1_extract_interesting_lines(ocr_pages)
    print(f"Stage 1 complete. Collected {len(all_collected)} characters.")
    
    # Stage 2: Extract each field from collected lines
    print("Stage 2: Extracting fields from collected lines...")
    extracted_fields = {}
    
    field_names = [
        "invoice_number", "invoice_date", "vendor_name", "recipient_name",
        "total_amount", "currency", "purchase_order_number", "ust_id", 
        "iban", "tax_rate"
    ]
    
    for field_name in field_names:
        print(f"  Extracting {field_name}...")
        value = _extract_field_from_lines(all_collected, field_name)
        print(f"  feld {field_name}: {value}")
        
        # Apply field-specific normalization
        if field_name == "invoice_date" and value:
            value = _normalize_date(value)
        elif field_name == "total_amount" and value:
            value = _normalize_amount(value)
        elif field_name == "currency" and value:
            # Normalize currency
            currency_match = re.search(r'\b(EUR|USD|GBP|CHF|€|$|£)\b', value, re.IGNORECASE)
            if currency_match:
                currency = currency_match.group(1).upper()
                value = "EUR" if currency == "€" else currency
        
        extracted_fields[field_name] = value
    
    total_duration = time.perf_counter() - start_time
    print(f"Two-stage extraction complete in {total_duration:.2f}s")
    
    return extracted_fields, total_duration


# test run for extract lines
if __name__ == "__main__":
    ocr_pages = doctr_pdf_to_text(SAMPLE_PDF_PATH)
    print(f"Extracted ocr text: {ocr_pages}")

    print(f"=======================================")
    extracted_fields, duration = stage1_extract_interesting_lines(ocr_pages)
    print(f"Extracted Fields: {extracted_fields} ")

    print(f"========================================")
    field_keys = [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "recipient_name",
        "total_amount",
        "currency",
        "purchase_order_number",
        "ust_id",
        "iban",
        "tax_rate"
    ]
    for field in field_keys:
        print(f"feld {field}: {_extract_field_from_lines(extracted_fields, field)}")


