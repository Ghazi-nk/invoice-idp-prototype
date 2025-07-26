# FILE: utils/post_processing.py
import re
from datetime import datetime
from typing import Dict, Any, List


# =============================================================================
# --- IBAN Validation without external dependencies ---
# =============================================================================

def _char_to_num(char: str) -> str:
    """Convert A-Z to 10-35 for IBAN validation."""
    if char.isdigit():
        return char
    return str(ord(char.upper()) - ord('A') + 10)

def _validate_iban(iban: str) -> bool:
    """Validate IBAN using MOD-97 algorithm."""
    if not iban or len(iban) < 15 or len(iban) > 34:
        return False
    
    # Remove spaces and convert to uppercase
    iban = re.sub(r'\s+', '', iban.upper())
    
    # Check if format is correct (2 letters + 2 digits + alphanumeric)
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban):
        return False
    
    # Move first 4 characters to end
    rearranged = iban[4:] + iban[:4]
    
    # Convert letters to numbers
    numeric_string = ''.join(_char_to_num(char) for char in rearranged)
    
    # Check MOD 97
    try:
        return int(numeric_string) % 97 == 1
    except ValueError:
        return False

# IBAN country lengths (for additional validation)
IBAN_LENGTHS = {
    'AD': 24, 'AE': 23, 'AL': 28, 'AT': 20, 'AZ': 28, 'BA': 20, 'BE': 16,
    'BG': 22, 'BH': 22, 'BR': 29, 'BY': 28, 'CH': 21, 'CR': 22, 'CY': 28,
    'CZ': 24, 'DE': 22, 'DK': 18, 'DO': 28, 'EE': 20, 'ES': 24, 'FI': 18,
    'FO': 18, 'FR': 27, 'GB': 22, 'GE': 22, 'GI': 23, 'GL': 18, 'GR': 27,
    'GT': 28, 'HR': 21, 'HU': 28, 'IE': 22, 'IL': 23, 'IS': 26, 'IT': 27,
    'JO': 30, 'KW': 30, 'KZ': 20, 'LB': 28, 'LC': 32, 'LI': 21, 'LT': 20,
    'LU': 20, 'LV': 21, 'MC': 27, 'MD': 24, 'ME': 22, 'MK': 19, 'MR': 27,
    'MT': 31, 'MU': 30, 'NL': 18, 'NO': 15, 'PK': 24, 'PL': 28, 'PS': 29,
    'PT': 25, 'QA': 29, 'RO': 24, 'RS': 22, 'SA': 24, 'SE': 24, 'SI': 19,
    'SK': 24, 'SM': 27, 'TN': 24, 'TR': 26, 'UA': 29, 'VG': 24, 'XK': 20
}


# =============================================================================
# --- Rule-Based Verification and Correction ---
# =============================================================================

UST_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(DE[0-9]{9})\b', re.IGNORECASE),
    re.compile(r'\b(ATU[0-9]{8})\b', re.IGNORECASE),
]

# KORREKTUR: Genaue IBAN-Patterns für europäische Länder
IBAN_PATTERNS: List[re.Pattern] = [
    # Deutschland: DE + 2 Ziffern + 16 Zeichen (alphanumerisch)
    re.compile(r'\b(DE[0-9]{2}[0-9A-Z]{16})\b', re.IGNORECASE),
    # Österreich: AT + 2 Ziffern + 16 Ziffern
    re.compile(r'\b(AT[0-9]{18})\b', re.IGNORECASE),
    # Schweiz: CH + 2 Ziffern + 5 Ziffern + 12 alphanumerisch
    re.compile(r'\b(CH[0-9]{2}[0-9]{5}[0-9A-Z]{12})\b', re.IGNORECASE),
    # Niederlande: NL + 2 Ziffern + 4 Buchstaben + 10 Ziffern
    re.compile(r'\b(NL[0-9]{2}[A-Z]{4}[0-9]{10})\b', re.IGNORECASE),
    # Frankreich: FR + 2 Ziffern + 10 Ziffern + 11 alphanumerisch + 2 Ziffern
    re.compile(r'\b(FR[0-9]{2}[0-9]{10}[0-9A-Z]{11}[0-9]{2})\b', re.IGNORECASE),
    # Belgien: BE + 2 Ziffern + 12 Ziffern
    re.compile(r'\b(BE[0-9]{14})\b', re.IGNORECASE),
    # Italien: IT + 2 Ziffern + 1 Buchstabe + 10 Ziffern + 12 alphanumerisch
    re.compile(r'\b(IT[0-9]{2}[A-Z][0-9]{10}[0-9A-Z]{12})\b', re.IGNORECASE),
    # Spanien: ES + 2 Ziffern + 20 Ziffern
    re.compile(r'\b(ES[0-9]{22})\b', re.IGNORECASE),
    # UK: GB + 2 Ziffern + 4 Buchstaben + 14 Ziffern
    re.compile(r'\b(GB[0-9]{2}[A-Z]{4}[0-9]{14})\b', re.IGNORECASE),
    # Weitere wichtige EU-Länder
    re.compile(r'\b(PL[0-9]{26})\b', re.IGNORECASE),  # Polen
    re.compile(r'\b(SE[0-9]{22})\b', re.IGNORECASE),  # Schweden
    re.compile(r'\b(DK[0-9]{16})\b', re.IGNORECASE),  # Dänemark
    re.compile(r'\b(NO[0-9]{13})\b', re.IGNORECASE),  # Norwegen
    re.compile(r'\b(FI[0-9]{16})\b', re.IGNORECASE),  # Finnland
]

# Fallback für andere europäische IBANs (allgemeineres Pattern)
IBAN_FALLBACK_PATTERN: re.Pattern = re.compile(r'\b([A-Z]{2}[0-9]{2}[\sA-Z0-9]{11,30})\b', re.IGNORECASE)


def verify_and_correct_fields(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Acts as a "safety net" to verify and correct fields with strong patterns,
    overwriting existing values if a valid pattern is found in the text.
    """
    if not isinstance(data, dict):
        return data

    # --- Verify and Correct IBAN ---
    iban_match = None
    candidate_iban = None

    # Zuerst versuchen wir die spezifischen europäischen IBAN-Patterns
    for pattern in IBAN_PATTERNS:
        match = pattern.search(full_text)
        if match:
            candidate_iban = re.sub(r'\s+', '', match.group(1).upper())
            
            # Additional length validation
            country_code = candidate_iban[:2]
            expected_length = IBAN_LENGTHS.get(country_code)
            if expected_length and len(candidate_iban) == expected_length:
                # Validate using MOD-97 algorithm
                if _validate_iban(candidate_iban):
                    iban_match = match
                    break

    # Falls kein spezifisches Pattern gefunden wurde, Fallback verwenden
    if not iban_match:
        fallback_match = IBAN_FALLBACK_PATTERN.search(full_text)
        if fallback_match:
            candidate_iban = re.sub(r'\s+', '', fallback_match.group(1).upper())
            
            # Additional length validation
            country_code = candidate_iban[:2]
            expected_length = IBAN_LENGTHS.get(country_code)
            if expected_length and len(candidate_iban) == expected_length:
                # Validate using MOD-97 algorithm
                if _validate_iban(candidate_iban):
                    iban_match = fallback_match

    if iban_match and candidate_iban:
        # Get the standardized electronic format (remove spaces, uppercase)
        corrected_iban = candidate_iban

        # Update the data only if the IBAN is valid and has changed
        if data.get('iban') != corrected_iban:
            print(f"[Info] Post-processing: Corrected IBAN to '{corrected_iban}'.")
        data['iban'] = corrected_iban
    else:
        # No valid IBAN found
        if candidate_iban:
            print(f"[Warning] Post-processing: Found invalid IBAN-like string '{candidate_iban}'. Discarding.")

    # --- Verify and Correct USt-Id ---
    for pattern in UST_ID_PATTERNS:
        ust_id_match = pattern.search(full_text)
        if ust_id_match:
            corrected_ust_id = ust_id_match.group(1).upper()
            if data.get('ust-id') != corrected_ust_id:
                 print(f"[Info] Post-processing: Corrected USt-Id to '{corrected_ust_id}'.")
            data['ust-id'] = corrected_ust_id
            break

    return data

def canon_money(x: str | float | None) -> float | None:
    """Canonicalize a money value to a float."""
    if x in (None, "", "null"): return None
    if isinstance(x, (int, float)): return round(float(x), 2)
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None

def canon_date(date_str: str) -> str | None:
    """Canonicalize a date string to 'DD.MM.YYYY' format."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # Try to parse common date formats
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None  # If no format matched, return None


def finalize_extracted_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the raw dictionary from the LLM and finalizes it by converting
    numeric fields from string to number types.
    """
    if not isinstance(data, dict):
        return data

    if 'total_amount' in data:
        data['total_amount'] = canon_money(data['total_amount'])

    if 'tax_rate' in data:
        data['tax_rate'] = canon_money(data['tax_rate'])

    if 'invoice_date' in data:
        data['invoice_date'] = canon_date(data['invoice_date'])

    return data