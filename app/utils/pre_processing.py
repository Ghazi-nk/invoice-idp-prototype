# pre_processing.py

import json
import re
from typing import List, Dict, Any


# --- PRIVATE HELPER -----------------------------------------------------------

def _preprocess_text_content(txt: str) -> str:
    """Bereinigt einen einzelnen Textblock von typischen OCR-Fehlern."""
    if not txt:
        return ""
    # Korrigiert häufige IBAN-Fehler (A7 -> AT)
    txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
    # Entfernt überflüssige Anführungszeichen
    txt = txt.replace("'", "")
    # Korrigiert Währungssymbole
    txt = txt.replace("Â€", "€")
    # Vereinheitlicht Dezimaltrennzeichen in Geldbeträgen
    txt = re.sub(r"(\d),(\d{2})(\s*€?)", r"\1.\2\3", txt)
    # Verbindet durch Zeilenumbruch getrennte Wörter
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    # Entfernt Seitenzahlanzeiger am Zeilenanfang
    txt = re.sub(r"^\s*page \d+:\s*", "", txt, flags=re.MULTILINE | re.IGNORECASE)
    return txt.strip()


def _format_doctr_output(elements: List[Dict[str, Any]]) -> str:
    """Formatiert Doctr-Output: Jede Zeile enthält Text, Bbox und Seite."""
    elements.sort(key=lambda item: (item.get('page', 1), item['bbox'][1], item['bbox'][0]))
    lines_with_all_info = [f"{item['text']} bbox={item['bbox']}" for item in elements]
    return "\n".join(lines_with_all_info)


def _format_layoutlm_output(elements: List[Dict[str, Any]]) -> str:
    """Formatiert LayoutLM-Output: Jede Zeile enthält Text und Bbox."""
    elements.sort(key=lambda item: (item.get('page', 1), item['bbox'][1], item['bbox'][0]))
    lines_with_bbox = [f"{item['text']} bbox={item['bbox']}" for item in elements]
    return "\n".join(lines_with_bbox)


# --- ÖFFENTLICHE PRE-PROCESSING FLOWS ------------------------------------------

def preprocess_plain_text_output(raw_text: str) -> str:
    """Bereinigt reinen OCR-Text von Engines wie Tesseract, EasyOCR etc."""
    # Entfernt unsere eigenen Seiten-Header wie "--- Seite 1 ---"
    processed_text = re.sub(r"\n?---\s*Seite\s*\d+\s*---\n?", "\n", raw_text, flags=re.IGNORECASE)
    # Entfernt leere Zeilen, die durch das Entfernen der Header entstehen können
    processed_text = "\n".join([line for line in processed_text.splitlines() if line.strip()])
    return _preprocess_text_content(processed_text)


