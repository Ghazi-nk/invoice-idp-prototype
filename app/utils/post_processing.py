# FILE: utils/post_processing.py

import re
from typing import Dict, Any, List

# --- Import canonicalization function from the central evaluation utility ---
from utils.evaluation_utils import canon_money

# =============================================================================
# --- Rule-Based Verification and Correction ---
# =============================================================================

# Regex to find common European VAT ID formats.
# This can be expanded with more formats as needed.
UST_ID_PATTERNS = [
    re.compile(r'\b(DE[0-9]{9})\b', re.IGNORECASE),
    re.compile(r'\b(ATU[0-9]{8})\b', re.IGNORECASE),
    # Add other common EU VAT patterns here
]

# Regex for finding a valid IBAN
IBAN_PATTERN = re.compile(r'\b([A-Z]{2}[0-9]{2}[A-Z0-9]{11,30})\b', re.IGNORECASE)


def verify_and_correct_fields(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Acts as a "safety net" to verify and correct fields with strong patterns
    if the LLM failed to extract them.

    Args:
        data: The dictionary extracted by the LLM.
        full_text: The complete, raw OCR text of the document.

    Returns:
        The corrected dictionary.
    """
    if not isinstance(data, dict):
        return data

    # --- Verify and Correct IBAN ---
    # If iban is missing, search the text for a valid pattern.
    if not data.get('iban'):
        iban_match = IBAN_PATTERN.search(full_text)
        if iban_match:
            # Clean up the found IBAN (remove spaces) and update the dict
            data['iban'] = re.sub(r'\s+', '', iban_match.group(1).upper())
            print(f"[Info] Post-processing: Found and corrected IBAN to '{data['iban']}'.")

    # --- Verify and Correct USt-Id ---
    # If ust-id is missing, search for common European VAT ID patterns.
    if not data.get('ust-id'):
        for pattern in UST_ID_PATTERNS:
            ust_id_match = pattern.search(full_text)
            if ust_id_match:
                data['ust-id'] = ust_id_match.group(1).upper()
                print(f"[Info] Post-processing: Found and corrected USt-Id to '{data['ust-id']}'.")
                break # Stop after the first valid pattern is found

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

    # Convert total_amount from string to number
    if 'total_amount' in data:
        data['total_amount'] = canon_money(data['total_amount'])

    # Convert tax_rate from string to number
    if 'tax_rate' in data:
        data['tax_rate'] = canon_money(data['tax_rate'])

    return data