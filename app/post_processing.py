# FILE: utils/post_processing.py
import re
from datetime import datetime
from typing import Dict, Any, List



# =============================================================================
# --- Rule-Based Verification and Correction ---
# =============================================================================

UST_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(DE[0-9]{9})\b', re.IGNORECASE),
    re.compile(r'\b(ATU[0-9]{8})\b', re.IGNORECASE),
]

# KORREKTUR: The pattern now includes `\s` to allow for spaces within the IBAN.
IBAN_PATTERN: re.Pattern = re.compile(r'\b([A-Z]{2}[0-9]{2}[\sA-Z0-9]{11,30})\b', re.IGNORECASE)


def verify_and_correct_fields(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Acts as a "safety net" to verify and correct fields with strong patterns,
    overwriting existing values if a valid pattern is found in the text.
    """
    if not isinstance(data, dict):
        return data

    # --- Verify and Correct IBAN ---
    iban_match = IBAN_PATTERN.search(full_text)
    if iban_match:
        # This sub removes the spaces after they have been successfully matched.
        corrected_iban = re.sub(r'\s+', '', iban_match.group(1).upper())
        if data.get('iban') != corrected_iban:
            print(f"[Info] Post-processing: Corrected IBAN to '{corrected_iban}'.")
        data['iban'] = corrected_iban

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

def canon_number(x: str | float | None) -> float | None:
    """Canonicalize a number value to a float."""
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
        data['total_amount'] = canon_number(data['total_amount'])

    if 'tax_rate' in data:
        data['tax_rate'] = canon_number(data['tax_rate'])

    if 'invoice_date' in data:
        data['invoice_date'] = canon_date(data['invoice_date'])

    return data