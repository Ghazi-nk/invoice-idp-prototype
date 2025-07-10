# utils/post_processing.py

from typing import Dict, Any


def canon_money(x: str | float | None) -> float | None:
    """
    Canonicalize a money value to a float.
    Handles German ("1.234,56") and standard ("1,234.56") formats.
    """
    if x in (None, "", "null"):
        return None
    if isinstance(x, (int, float)):
        return round(float(x), 2)

    x_str = str(x).strip().replace("'", "").replace("€", "").replace("$", "")

    # This logic handles both "1.234,56" -> "1234.56" and "1,234.56" -> "1234.56"
    if ',' in x_str and '.' in x_str:
        if x_str.rfind('.') > x_str.rfind(','):  # Format like 1,234.56
            x_str = x_str.replace(',', '')
        else:  # Format like 1.234,56
            x_str = x_str.replace('.', '').replace(',', '.')
    elif ',' in x_str:  # Format like 1234,56
        x_str = x_str.replace(',', '.')

    try:
        return round(float(x_str), 2)
    except (ValueError, TypeError):
        return None


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