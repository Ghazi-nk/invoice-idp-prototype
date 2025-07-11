from typing import Dict, Any

from utils.evaluation_utils import canon_money

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