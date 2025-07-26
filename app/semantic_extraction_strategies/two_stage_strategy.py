"""
Two Stage Strategy: 
1. Extract interesting lines from each page that might contain field data
2. Extract specific fields from the collected lines
"""

import time
import json
import re
import requests
from typing import List, Dict, Tuple
from pathlib import Path
from app.config import CHAT_ENDPOINT, OLLAMA_MODEL


def load_prompt(filename: str) -> str:
    """Load prompt from file."""
    prompt_path = Path("resources/prompts/semantic_extraction/two_stage") / filename
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found at {prompt_path}")
        return ""


def _send_chat_request(messages: List[Dict[str, str]]) -> str:
    """Send a chat request to Ollama."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error in chat request: {e}")
        return ""


def _extract_interesting_lines(page_text: str) -> str:
    """Extract interesting lines from a single page that might contain invoice data."""
    try:
        messages = [
            {"role": "system", "content": "Extrahiere nur Zeilen die folgende Informationen enthalten könnten: Rechnungsnummer, Datum, Namen, Adressen, Beträge, Währung, IBAN, USt-ID, Bestellnummer. Gib nur diese relevanten Zeilen zurück."},
            {"role": "user", "content": f"Extrahiere relevante Zeilen aus diesem Text:\n\n{page_text}"}
        ]
        
        response = _send_chat_request(messages)
        return response if response else ""
    except Exception as e:
        print(f"Error extracting interesting lines: {e}")
        return ""


def _extract_field_from_lines(collected_lines: str, field_name: str) -> str:
    """Extract a specific field from the collected lines."""
    try:
        # Load prompts
        system_prompt = load_prompt("system_prompt.txt")
        user_prompt = load_prompt("user_prompt.txt")
        
        # Specific field extraction prompts
        field_prompts = {
            "invoice_number": "Finde die Rechnungsnummer. Format: Kann Buchstaben, Zahlen und Bindestriche enthalten (z.B. RE-2024-001, 10005238-8).",
            "invoice_date": "Finde das Rechnungsdatum. Konvertiere in Format DD.MM.YYYY.",
            "vendor_name": "Finde den Namen des Verkäufers/Anbieters/Absenders. Dies ist die Firma oder Person die die Rechnung ausstellt.",
            "recipient_name": "Finde den Namen des Empfängers/Kunden. Dies ist die Firma oder Person die die Rechnung erhält.",
            "total_amount": "Finde den Gesamtbetrag/Endbetrag der Rechnung. Nur die Zahl ohne Währung.",
            "currency": "Finde die Währung. Gib nur den Code zurück (EUR, USD, GBP, etc.).",
            "purchase_order_number": "Finde die Bestellnummer/Auftragsnummer falls vorhanden.",
            "ust_id": "Finde die Umsatzsteuer-ID (USt-ID) oder VAT-Nummer. Format beginnt meist mit Ländercode (z.B. DE123456789).",
            "iban": "Finde die IBAN für die Bankverbindung. Format: 2 Buchstaben gefolgt von Zahlen.",
            "tax_rate": "Finde den Steuersatz/MwSt-Satz. Nur die Prozentzahl ohne % Zeichen."
        }
        
        specific_prompt = field_prompts.get(field_name, f"Extrahiere: {field_name}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{specific_prompt}\n\nText:\n{collected_lines}\n\n{user_prompt}"}
        ]
        
        response = _send_chat_request(messages)
        
        # Clean response
        if response:
            response = response.strip()
            # Remove common non-answers
            if response.lower() in ['null', 'none', 'nicht vorhanden', 'nicht gefunden', 'keine', 'kein', '']:
                return None
            # Remove "ja" prefixes
            response = re.sub(r'^(ja,?\s*)', '', response, flags=re.IGNORECASE)
            return response
        
        return None
    except Exception as e:
        print(f"Error extracting field {field_name}: {e}")
        return None