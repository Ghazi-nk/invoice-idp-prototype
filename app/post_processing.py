"""
Post-Processing und Verifikation extrahierter Rechnungsdaten.

Dieses Modul implementiert regelbasierte Verifikation und Korrektur von
LLM-extrahierten Rechnungsdaten. Es nutzt Regex-Patterns und Heuristiken
zur Qualitätssicherung und Standardisierung der Ausgabedaten.

Funktionen:
- Verifikation von IBAN und USt-ID-Formaten
- Korrektur und Normalisierung von Datumsformaten  
- Datenvalidierung und -bereinigung
- Finalisierung der Ausgabestruktur

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import re
from datetime import datetime
from typing import Dict, Any, List



# =============================================================================
# --- Rule-Based Verification and Correction ---
# =============================================================================

UST_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(DE[0-9]{9})\b', re.IGNORECASE),
    re.compile(r'\b(ATU[0-9]{8})\b', re.IGNORECASE),
    re.compile(r'\b(DE[0-9]{8})\b', re.IGNORECASE),
]

# KORREKTUR: More precise IBAN pattern to avoid false matches
IBAN_PATTERN: re.Pattern = re.compile(
    r'\b([A-Z]{2}[O0]?\d{2}(?:\s*[A-Z0-9O]){11,28})\b',
    re.IGNORECASE
)

def verify_and_correct_fields(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Verifiziert und korrigiert extrahierte Felder mittels Regex-Patterns.
    
    Diese Funktion dient als "Sicherheitsnetz" für die LLM-Extraktion
    und korrigiert häufige OCR-Fehler sowie fehlende Werte durch
    regelbasierte Pattern-Erkennung im Volltext.
    
    Args:
        data (Dict[str, Any]): Von LLM extrahierte Rechnungsdaten
        full_text (str): Vollständiger OCR-Text für Pattern-Suche
        
    Returns:
        Dict[str, Any]: Korrigierte und verifizierte Daten
        
    Korrekturen:
        - IBAN: O→0 Korrektur, Leerzeichen-Entfernung, Prefix-Bereinigung
        - USt-ID: Pattern-Erkennung für DE/ATU-Formate
        - Automatische Nachsuche bei fehlenden kritischen Feldern
        
    Note:
        Überschreibt LLM-Werte nur bei eindeutigen Pattern-Matches.
        Loggt alle Korrekturen für Transparenz und Debugging.
    """
    if not isinstance(data, dict):
        return data

    # get iban and ust-id from data
    iban = (data.get('iban') or '').strip()
    ust_id = (data.get('ust-id') or '').strip()

    # --- Verify and Correct IBAN ---
    # Clean IBAN from common prefixes
    if iban:
        # Remove "IBAN:" prefix if present
        if iban.upper().startswith('IBAN:'):
            iban = iban[5:].strip()
            data['iban'] = iban
        
        # Fix O->0 and normalize spaces in LLM output
        if 'O' in iban or 'o' in iban or ' ' in iban:
            corrected_iban = iban.replace('O', '0').replace('o', '0').replace(' ', '')
            data['iban'] = corrected_iban
            print(f"[Info] Post-processing: Fixed IBAN OCR error from '{iban}' to '{corrected_iban}'.")
    
    if not iban:
        # Only search for IBAN if LLM found nothing
        all_iban_matches = IBAN_PATTERN.findall(full_text)
        valid_ibans = []
        for match in all_iban_matches:
            clean_iban = re.sub(r'\s+', '', match.upper()).replace('O', '0')
            # Validate typical IBAN length and format
            if 15 <= len(clean_iban) <= 32 and clean_iban[:2].isalpha():
                valid_ibans.append(clean_iban)
        
        if valid_ibans:
            data['iban'] = valid_ibans[-1]  # Take last/most relevant
            print(f"[Info] Post-processing: Found IBAN '{valid_ibans[-1]}'.")
        else:
            print(f"[Info] Post-processing: No valid IBAN found in text.")
    else:
        print(f"[Info] Post-processing: IBAN '{iban}' kept as extracted by LLM.")

    # --- Verify and Correct USt-Id ---
    # Find all USt-Id matches in the full text
    all_ust_ids = []
    for pattern in UST_ID_PATTERNS:
        matches = pattern.findall(full_text)
        all_ust_ids.extend([match.upper() for match in matches])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ust_ids = []
    for ust in all_ust_ids:
        if ust not in seen:
            seen.add(ust)
            unique_ust_ids.append(ust)
    
    if unique_ust_ids:
        # If current USt-Id exists and is in the list of found USt-Ids, keep it
        if ust_id and ust_id in unique_ust_ids:
            print(f"[Info] Post-processing: USt-Id '{ust_id}' is valid and found in text.")
        else:
            # Use the last found USt-Id as the most relevant one
            corrected_ust_id = unique_ust_ids[-1]
            if data.get('ust-id') != corrected_ust_id:
                print(f"[Info] Post-processing: Corrected USt-Id from '{ust_id}' to '{corrected_ust_id}' (last found in text).")
            data['ust-id'] = corrected_ust_id
    else:
        print(f"[Info] Post-processing: No valid USt-Id found in text. Current USt-Id: '{ust_id}'")

    return data

def canon_number(x: str | float | None) -> float | None:
    """
    Kanonisiert einen Zahlenwert zu einem Float.
    
    Diese Funktion normalisiert verschiedene Zahlenformate (String, Float, Int)
    zu einem einheitlichen Float-Wert mit 2 Dezimalstellen. Sie behandelt
    häufige Formatierungen wie Währungssymbole, Tausendertrennzeichen und
    Komma-als-Dezimaltrennzeichen.
    
    Args:
        x (str | float | None): Zu normalisierender Zahlenwert
        
    Returns:
        float | None: Normalisierter Float-Wert oder None bei ungültigen Eingaben
        
    Note:
        - Entfernt Währungssymbole (€), Leerzeichen und Apostrophe
        - Konvertiert Komma zu Punkt als Dezimaltrennzeichen
        - Rundet auf 2 Dezimalstellen für Geldbeträge
    """
    if x in (None, "", "null"): return None
    if isinstance(x, (int, float)): return round(float(x), 2)
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None


def canon_date(date_str: str) -> str | None:
    """
    Kanonisiert einen Datumsstring zum Format 'DD.MM.YYYY'.
    
    Diese Funktion parst verschiedene Datumsformate und konvertiert
    sie zu einem einheitlichen deutschen Datumsformat. Sie unterstützt
    sowohl ISO-Formate als auch deutsche und amerikanische Varianten.
    
    Args:
        date_str (str): Zu parsender Datumsstring
        
    Returns:
        str | None: Datum im Format 'DD.MM.YYYY' oder None bei ungültigen Formaten
        
    Unterstützte Formate:
        - DD.MM.YYYY (deutsch)
        - YYYY-MM-DD (ISO)
        - DD/MM/YYYY, YYYY/MM/DD
        - YYYY.MM.DD
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    # Try to parse common date formats including YYYY.MM.DD
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None  # If no format matched, return None


def finalize_extracted_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalisiert extrahierte Felder durch Datentyp-Konvertierung und Bereinigung.
    
    Diese Funktion nimmt das rohe Dictionary vom LLM und führt umfassende
    Post-Processing-Schritte durch: Zahlenkonvertierung, Datumsformatierung,
    Namensbereinigung und Entfernung von Adressteilen.
    
    Args:
        data (Dict[str, Any]): Rohe extrahierte Daten vom LLM
        
    Returns:
        Dict[str, Any]: Bereinigte und finalisierte Daten
        
    Verarbeitungsschritte:
        - total_amount, tax_rate → Float-Konvertierung
        - invoice_date → Datumsformatierung
        - purchase_order_number → Bereinigung von Beschreibungstexten
        - recipient_name, vendor_name → Adressentfernung und Längenkürzung
        
    Note:
        Loggt alle Bereinigungsschritte für Transparenz und Debugging.
    """
    if not isinstance(data, dict):
        return data

    if 'total_amount' in data:
        data['total_amount'] = canon_number(data['total_amount'])

    if 'tax_rate' in data:
        data['tax_rate'] = canon_number(data['tax_rate'])
        # Ensure tax_rate has exactly 2 decimal places
        if data['tax_rate'] is not None:
            data['tax_rate'] = round(float(data['tax_rate']), 2)

    if 'invoice_date' in data:
        data['invoice_date'] = canon_date(data['invoice_date'])
    
    # Clean up purchase_order_number - only remove if it's clearly descriptive text
    if 'purchase_order_number' in data and data['purchase_order_number']:
        po = data['purchase_order_number'].strip()
        # Only remove if contains descriptive words (not just dates/numbers)
        if any(word in po.lower() for word in ['erteilt', 'am:', 'datum']) and len(po) > 15:
            print(f"[Info] Post-processing: Cleaned invalid purchase order '{po}' → null")
            data['purchase_order_number'] = None
    
    # Clean up recipient_name - remove addresses and limit length
    if 'recipient_name' in data and data['recipient_name']:
        name = data['recipient_name'].strip()
        # Remove everything after dash if it contains street/number patterns
        if ' - ' in name:
            parts = name.split(' - ')
            # Check if second part looks like address (contains numbers or "str")
            if len(parts) > 1 and (any(char.isdigit() for char in parts[1]) or 'str' in parts[1].lower()):
                name = parts[0].strip()
                print(f"[Info] Post-processing: Removed address after dash from recipient → '{name}'")
        # Remove everything after comma if it looks like an address
        elif ',' in name and any(char.isdigit() for char in name.split(',')[-1]):
            name = name.split(',')[0].strip()
            print(f"[Info] Post-processing: Removed address from recipient name → '{name}'")
        # Remove c/o and handelnd für phrases
        elif 'handelnd für' in name.lower() or 'c/o' in name.lower():
            # Extract only the first part before these phrases
            parts = name.split('handelnd für')[0].split('c/o')[0]
            name = parts.strip()
            print(f"[Info] Post-processing: Shortened recipient name to '{name}'")
        # Limit to 80 characters
        if len(name) > 80:
            name = name[:80].strip()
            print(f"[Info] Post-processing: Truncated recipient name to 80 chars")
        data['recipient_name'] = name
    
    # Clean up vendor_name - remove addresses and "vertr. d." phrases
    if 'vendor_name' in data and data['vendor_name']:
        name = data['vendor_name'].strip()
        # Remove "vertr. d." or "vertreten durch" phrases
        if 'vertr. d.' in name or 'vertreten durch' in name:
            # Extract only the first company name
            name = name.split('vertr. d.')[0].split('vertreten durch')[0].strip()
            print(f"[Info] Post-processing: Cleaned vendor name to '{name}'")
        # Remove everything after comma if it looks like an address
        elif ',' in name and any(char.isdigit() for char in name.split(',')[-1]):
            name = name.split(',')[0].strip()
            print(f"[Info] Post-processing: Removed address from vendor name → '{name}'")
        data['vendor_name'] = name

    return data