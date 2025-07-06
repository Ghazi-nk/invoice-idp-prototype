import pytest
from pre_processing import preprocess_ocr_text, normalize_ocr_output

# #############################################################
# ## Tests für die Text-Reinigung (preprocess_ocr_text)      ##
# #############################################################

# Hier testen wir mit @pytest.mark.parametrize systematisch alle "komischen Zeichen"
# und Sonderfälle, die deine Funktion behandeln soll.
# Jedes Tupel in der Liste ist ein eigener kleiner Testfall.
@pytest.mark.parametrize("test_input, expected_output", [
    # Testfall 1: Leerer String
    ("", ""),
    # Testfall 2: None als Input
    (None, ""),
    # Testfall 3: Typischer OCR-Fehler bei IBANs
    ("Meine IBAN ist A740 1234 5678", "Meine IBAN ist AT40 1234 5678"),
    # Testfall 4: Apostrophe entfernen
    ("It's Ghazi's test", "Its Ghazis test"),
    # Testfall 5: Falsch kodiertes Euro-Zeichen
    ("Der Betrag: 500 Â€.", "Der Betrag: 500 €."),
    # Testfall 6: Komma als Dezimaltrennzeichen
    ("Summe: 1.234,56", "Summe: 1.234.56"),
    # Testfall 7: Komma als Dezimaltrennzeichen mit Euro-Symbol
    ("Rechnungsbetrag 99,95€", "Rechnungsbetrag 99.95€"),
    # Testfall 8: Wort mit Trennstrich über Zeilenumbruch
    ("Das ist ein wunder-\nschöner Testfall", "Das ist ein wunderschöner Testfall"),
    # Testfall 9: Seiten-Markierung am Zeilenanfang entfernen
    ("page 1:\nDas ist der Inhalt.", "Das ist der Inhalt."),
    # Testfall 10: Führende und nachfolgende Leerzeichen entfernen
    ("   \n   ein test   \n ", "ein test"),
    # Testfall 11: Alles kombiniert in einem "Chaos-Test"
    (" page 9: \nDas ist Ghazi's Test für 199,99 Â€ und eine wunder-\nschöne IBAN: A712345678",
     "Das ist Ghazis Test für 199.99 € und eine wunderschöne IBAN: AT12345678"),
])
def test_preprocess_ocr_text(test_input, expected_output):
    """
    Testet die Funktion 'preprocess_ocr_text' mit verschiedenen Eingaben.
    """
    assert preprocess_ocr_text(test_input) == expected_output

# #############################################################
# ## Tests für die Format-Normalisierung (normalize_ocr_output) ##
# #############################################################

def test_normalize_plain_ocr_text():
    """
    Testet die Verarbeitung von reinem Text (Tesseract, etc.).
    Sollte Leerzeichen am Zeilenanfang/-ende entfernen.
    """
    ocr_data = """
        page 1:
          Autoreparatur 2,50 40,00 100,00
        UMSATZSTEUER 426,93
    """
    expected = "page 1:\nAutoreparatur 2,50 40,00 100,00\nUMSATZSTEUER 426,93"
    assert normalize_ocr_output(ocr_data) == expected

def test_normalize_doctr_format():
    """
    Testet die Verarbeitung des Doctr-Formats.
    Achtet auf korrekte Sortierung und Formatierung (mit Seite).
    """
    doctr_data = """
    [
        {"text": "Alex Mustermann", "bbox": [171, 251, 357, 268], "page": 2},
        {"text": "SUMME 2673,93", "bbox": [724, 155, 878, 164], "page": 1},
        {"text": "UMSATZSTEUER 426,93", "bbox": [666, 141, 879, 150], "page": 1}
    ]
    """
    expected = ("UMSATZSTEUER 426,93 bbox=[666, 141, 879, 150] page=1\n"
                "SUMME 2673,93 bbox=[724, 155, 878, 164] page=1\n"
                "Alex Mustermann bbox=[171, 251, 357, 268] page=2")
    assert normalize_ocr_output(doctr_data) == expected

def test_normalize_layoutlm_format():
    """
    Testet die Verarbeitung des LayoutLM-Formats.
    Achtet auf korrekte Sortierung und Formatierung (ohne Seite).
    """
    layoutlm_data = """
    page 2:
    {"text": "Alex Mustermann", "bbox": [104, 214, 206, 222]}
    page 1:
    {"text": "BETRAG NETTO 2247,00", "bbox": [671, 115, 879, 124]}
    """
    expected = ("BETRAG NETTO 2247,00 bbox=[671, 115, 879, 124]\n"
                "Alex Mustermann bbox=[104, 214, 206, 222]")
    assert normalize_ocr_output(layoutlm_data) == expected