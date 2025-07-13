# FILE: utils/post_processing.py
import re
from typing import Dict, Any, List

# This is a dummy setup for the function to be available.
# In your project, you'd use your actual import.
try:
    from utils.evaluation_utils import canon_money
except ImportError:
    def canon_money(value):
        if isinstance(value, (int, float)): return float(value)
        try:
            found = re.search(r'[-+]?\d*[\.,]?\d+', str(value).replace(',', '.'))
            return float(found.group(0)) if found else 0.0
        except (ValueError, AttributeError):
            return 0.0

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

# =============================================================================
# --- Final Data Type Conversion ---
# =============================================================================

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

    return data