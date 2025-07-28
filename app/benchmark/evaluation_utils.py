import re
import unicodedata
from typing import Dict

from app.post_processing import canon_number


def canon_text(s: str | None) -> str:
    """
    Canonicalize a string for comparison by normalizing, removing punctuation,
    and standardizing common business terms.
    """
    if not s:
        return ""
    # Normalize unicode characters (e.g., umlauts) and case
    s = unicodedata.normalize("NFKD", str(s).lower())

    # Standardize conjunctions and common abbreviations
    s = re.sub(r'\s*&\s*|\s*und\s*', ' and ', s)
    s = re.sub(r'\bgmbh\.?|\bgesellschaft mit beschränkter haftung', '', s)
    s = re.sub(r'\bag\.?|\baktiengesellschaft', '', s)

    # Remove all non-alphanumeric characters except for spaces
    s = ''.join(c for c in s if c.isalnum() or c.isspace())

    # Collapse multiple spaces into one and remove leading/trailing whitespace
    return ' '.join(s.split()).strip()


def is_name_match(expected: str | None, predicted: str | None) -> bool:
    """
    A more forgiving comparison for names. Returns True if the core words
    of the predicted name are found within the expected name.
    """
    if not expected or not predicted:
        return not expected and not predicted

    canon_expected = set(canon_text(expected).split())
    canon_predicted = set(canon_text(predicted).split())

    # Check if the set of predicted words is a subset of the expected words
    return canon_predicted.issubset(canon_expected)

def _canon_id(s: str | None) -> str:
    """Canonicalize an ID string."""
    if not s: return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def is_match(field: str, true_val, pred_val) -> bool:
    """Compares a true and predicted value based on the field type."""
    ID_KEYS = {"ust-id", "iban"}
    MONEY_KEYS = {"total_amount", "tax_rate"}
    NAME_KEYS = {"vendor_name", "recipient_name"}

    if true_val in (None, "", "null"): return pred_val in (None, "", "null")

    # Use the more forgiving comparison for name fields
    if field in NAME_KEYS:
        return is_name_match(true_val, pred_val)

    if field in MONEY_KEYS: return canon_number(true_val) == canon_number(pred_val)
    if field in ID_KEYS: return _canon_id(true_val) == _canon_id(pred_val)
    return canon_text(true_val) == canon_text(pred_val)


def check_acceptance(gt: Dict, pred: Dict) -> bool:
    """
    Checks if the extracted data meets the business rules for automated processing.
    """
    # Rule 1: With Purchase Order Number
    if is_match("purchase_order_number", gt.get("purchase_order_number"), pred.get("purchase_order_number")):
        return all([
            is_match("recipient_name", gt.get("recipient_name"), pred.get("recipient_name")),
            is_match("invoice_date", gt.get("invoice_date"), pred.get("invoice_date"))
        ])

    # Rule 2: Without Purchase Order Number
    else:
        # Check if at least one of iban or ust-id is a match
        id_match = any([
            is_match("iban", gt.get("iban"), pred.get("iban")),
            is_match("ust-id", gt.get("ust-id"), pred.get("ust-id"))
        ])

        return all([
            is_match("recipient_name", gt.get("recipient_name"), pred.get("recipient_name")),
            is_match("invoice_number", gt.get("invoice_number"), pred.get("invoice_number")),
            is_match("invoice_date", gt.get("invoice_date"), pred.get("invoice_date")),
            is_match("total_amount", gt.get("total_amount"), pred.get("total_amount")),
            is_match("currency", gt.get("currency"), pred.get("currency")),
            id_match
        ])