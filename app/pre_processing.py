import json
import re
from typing import List, Dict, Any


def preprocess_ocr_text(txt: str) -> str:
    """Bereinigt einen einzelnen Textblock von typischen OCR-Fehlern."""
    if not txt:
        return ""
    txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
    txt = txt.replace("'", "")
    txt = txt.replace("Â€", "€")
    txt = re.sub(r"(\d),(\d{2})(\s*€?)", r"\1.\2\3", txt)
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"^\s*page \d+:\s*", "", txt, flags=re.MULTILINE | re.IGNORECASE)
    return txt.strip()


def _format_doctr_output(elements: List[Dict[str, Any]]) -> str:
    """Formatiert Doctr-Output: Jede Zeile enthält Text, Bbox und Seite."""
    elements.sort(key=lambda item: (item.get('page', 1), item['bbox'][1], item['bbox'][0]))
    lines_with_all_info = [f"{item['text']} bbox={item['bbox']} page={item['page']}" for item in elements]
    return "\n".join(lines_with_all_info)


def _format_layoutlm_output(elements: List[Dict[str, Any]]) -> str:
    """Formatiert LayoutLM-Output: Jede Zeile enthält Text und Bbox (ohne Seite)."""
    elements.sort(key=lambda item: (item.get('page', 1), item['bbox'][1], item['bbox'][0]))
    lines_with_bbox = [f"{item['text']} bbox={item['bbox']}" for item in elements]
    return "\n".join(lines_with_bbox)


def _handle_doctr(raw_text: str) -> str:
    """Verarbeitet und formatiert das JSON-Array-Format von Doctr."""
    try:
        elements = json.loads(raw_text)
        return _format_doctr_output(elements)
    except json.JSONDecodeError:
        return raw_text


def _handle_layoutlm(raw_text: str) -> str:
    """
    Verarbeitet und formatiert das zeilenweise JSON-Format von LayoutLM.
    """
    elements = []
    current_page = 1
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        if line.lower().startswith('page '):
            try:
                current_page = int(re.findall(r'\d+', line)[0])
            except (IndexError, ValueError):
                pass
        elif line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                data['page'] = current_page
                elements.append(data)
            except json.JSONDecodeError:
                pass

    return _format_layoutlm_output(elements)


def normalize_ocr_output(ocr_output: str) -> str:
    """Erkennt automatisch das Format und normalisiert es."""
    stripped_output = ocr_output.strip()
    if stripped_output.startswith('[') and stripped_output.endswith(']'):
        return _handle_doctr(stripped_output)
    elif '"bbox":' in stripped_output:
        return _handle_layoutlm(stripped_output)
    else:
        lines = [line.strip() for line in ocr_output.strip().splitlines()]
        return "\n".join(lines)


def run_full_preprocessing_pipeline(raw_ocr_output: str) -> str:
    """Die zentrale Pipeline-Funktion."""
    normalized_text = normalize_ocr_output(raw_ocr_output)
    cleaned_text = preprocess_ocr_text(normalized_text)
    return cleaned_text


# --- TESTBLOCK (mit allen drei Formaten) ---

if __name__ == "__main__":
    # Testdaten definieren
    layoutlm_data = """
    page 2:
    {"text": "Alex Mustermann", "bbox": [104, 214, 206, 222]}
    page 1:
    {"text": "BETRAG NETTO 2247,00", "bbox": [671, 115, 879, 124]}
    """

    doctr_data = """
    [
        {"text": "Alex Mustermann", "bbox": [171, 251, 357, 268], "page": 2},
        {"text": "UMSATZSTEUER 426,93", "bbox": [666, 141, 879, 150], "page": 1}
    ]
    """

    ocr_data = """
        page 1:
        Autoreparatur 2,50 40,00 100,00
        UMSATZSTEUER 426,93
    """

    # Test 1: LayoutLM
    print("--- 1. Test: LayoutLM Format (NUR mit Bbox) ---")
    processed_layoutlm = run_full_preprocessing_pipeline(layoutlm_data)
    print(processed_layoutlm)
    expected_layoutlm = "BETRAG NETTO 2247.00 bbox=[671, 115, 879, 124]\nAlex Mustermann bbox=[104, 214, 206, 222]"
    assert processed_layoutlm == expected_layoutlm
    print("\n✅ LayoutLM-Test erfolgreich!")

    print("\n" + "=" * 50 + "\n")

    # Test 2: Doctr
    print("--- 2. Test: Doctr Format (mit Bbox UND Seite) ---")
    processed_doctr = run_full_preprocessing_pipeline(doctr_data)
    print(processed_doctr)
    expected_doctr = "UMSATZSTEUER 426.93 bbox=[666, 141, 879, 150] page=1\nAlex Mustermann bbox=[171, 251, 357, 268] page=2"
    assert processed_doctr == expected_doctr
    print("\n✅ Doctr-Test erfolgreich!")

    print("\n" + "=" * 50 + "\n")

    # Test 3: Reiner OCR-Text
    print("--- 3. Test: Reiner Text (OCR) Format ---")
    processed_ocr = run_full_preprocessing_pipeline(ocr_data)
    print(processed_ocr)
    expected_ocr = "Autoreparatur 2.50 40.00 100.00\nUMSATZSTEUER 426.93"
    assert processed_ocr == expected_ocr
    print("\n✅ Reiner-Text-Test erfolgreich!")

    print("\n\n🎉 Alle Tests erfolgreich!")
