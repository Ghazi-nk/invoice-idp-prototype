import re
import logging

# Setup logging (for consistency, though rarely needed here)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pre_processing")

def _preprocess_text_content(txt: str) -> str:
    """Cleans a single text block of typical OCR errors."""
    if not txt:
        return ""
    try:
        # Corrects common IBAN errors (A7 -> AT)
        txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
        # Removes superfluous apostrophes
        txt = txt.replace("'", "")
        # Corrects currency symbols
        txt = txt.replace("\u00c2\u20ac", "\u20ac")
        # Standardizes decimal separators in amounts
        txt = re.sub(r"(\d),(\d{2})(\s*\u20ac?)", r"\1.\2\3", txt)
        # Joins words separated by a newline and hyphen
        txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
        # Removes page number indicators from the start of lines
        txt = re.sub(r"^\s*page \d+:\s*", "", txt, flags=re.MULTILINE | re.IGNORECASE)
        return txt.strip()
    except Exception as e:
        logger.exception("Error during text preprocessing:")
        raise

def preprocess_plain_text_output(raw_text: str) -> str:
    """Cleans plain OCR text from engines like Tesseract, Doctr, etc."""
    try:
        # Removes our custom page headers like "--- Seite 1 ---"
        processed_text = re.sub(r"\n?---\s*Seite\s*\d+\s*---\n?", "\n", raw_text, flags=re.IGNORECASE)
        # Removes blank lines that can result from removing headers
        processed_text = "\n".join([line for line in processed_text.splitlines() if line.strip()])
        return _preprocess_text_content(processed_text)
    except Exception as e:
        logger.exception("Error during plain text output preprocessing:")
        raise