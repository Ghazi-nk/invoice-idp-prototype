# FILE: utils/pre_processing.py
# SIMPLIFIED

import re

# --- PRIVATE HELPER ---

def _preprocess_text_content(txt: str) -> str:
    """Cleans a single text block of typical OCR errors."""
    if not txt:
        return ""
    # Corrects common IBAN errors (A7 -> AT)
    txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
    # Removes superfluous apostrophes
    txt = txt.replace("'", "")
    # Corrects currency symbols
    txt = txt.replace("Â€", "€")
    # Standardizes decimal separators in amounts
    txt = re.sub(r"(\d),(\d{2})(\s*€?)", r"\1.\2\3", txt)
    # Joins words separated by a newline and hyphen
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    # Removes page number indicators from the start of lines
    txt = re.sub(r"^\s*page \d+:\s*", "", txt, flags=re.MULTILINE | re.IGNORECASE)
    return txt.strip()


# --- PUBLIC PRE-PROCESSING FLOW ---

def preprocess_plain_text_output(raw_text: str) -> str:
    """Cleans plain OCR text from engines like Tesseract, Doctr, etc."""
    # Removes our custom page headers like "--- Seite 1 ---"
    processed_text = re.sub(r"\n?---\s*Seite\s*\d+\s*---\n?", "\n", raw_text, flags=re.IGNORECASE)
    # Removes blank lines that can result from removing headers
    processed_text = "\n".join([line for line in processed_text.splitlines() if line.strip()])
    return _preprocess_text_content(processed_text)