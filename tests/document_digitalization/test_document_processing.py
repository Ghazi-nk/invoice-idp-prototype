
import unittest
from unittest.mock import patch, MagicMock, call

# Der zu testende Code
import document_processing


class TestDocumentProcessing(unittest.TestCase):

    # 1. Test für die Haupt-Pipeline-Funktion
    @patch('document_processing.finalize_extracted_fields')
    @patch('document_processing.verify_and_correct_fields')
    @patch('document_processing.ollama_extract_invoice_fields')
    @patch('document_processing.clean_ocr_text')
    @patch('document_processing.ocr_pdf')
    def test_extract_invoice_fields_from_pdf_full_pipeline(
            self,
            mock_ocr_pdf,
            mock_clean_ocr_text,
            mock_ollama_extract,
            mock_verify_fields,
            mock_finalize_fields
    ):
        """
        Testet den kompletten, erfolgreichen Durchlauf der Pipeline mit Bereinigung.
        Alle externen Abhängigkeiten werden gemockt.
        """
        # --- Arrange: Lege das Verhalten der Mocks fest ---
        pdf_path = "/fake/path/to/invoice.pdf"
        engine = "easyocr"

        # 1. ocr_pdf gibt rohen Text für zwei Seiten zurück
        mock_ocr_pdf.return_value = ["Raw text page 1.", "Raw text page 2."]

        # 2. clean_ocr_text wird für jede Seite aufgerufen und gibt bereinigten Text zurück
        mock_clean_ocr_text.side_effect = ["Clean text page 1.", "Clean text page 2."]

        # 3. LLM-Extraktion gibt ein Dictionary zurück
        mock_ollama_extract.return_value = {"invoice_id": "INV-123", "total_amount": "150.00"}

        # 4. Verifizierung gibt ein (potenziell korrigiertes) Dictionary zurück
        mock_verify_fields.return_value = {"invoice_id": "INV-123", "total_amount": "150.00", "corrected": True}

        # 5. Finalisierung gibt das endgültige Dictionary zurück (z.B. mit konvertierten Zahlentypen)
        mock_finalize_fields.return_value = {"invoice_id": "INV-123", "total_amount": 150.00, "corrected": True}

        # --- Act: Rufe die zu testende Funktion auf ---
        result = document_processing.extract_invoice_fields_from_pdf(pdf_path, engine=engine)

        # --- Assert: Überprüfe, ob alles wie erwartet abgelaufen ist ---
        # Wurde die OCR-Funktion korrekt aufgerufen?
        mock_ocr_pdf.assert_called_once_with(pdf_path, engine=engine)

        # Wurde der Text für beide Seiten bereinigt?
        self.assertEqual(mock_clean_ocr_text.call_count, 2)
        mock_clean_ocr_text.assert_has_calls([
            call("Raw text page 1.", engine=engine),
            call("Raw text page 2.", engine=engine)
        ])

        # Wurde der LLM mit dem zusammengefügten, bereinigten Text aufgerufen?
        expected_llm_input = ["Clean text page 1.", "Clean text page 2."]
        mock_ollama_extract.assert_called_once_with(expected_llm_input)

        # Wurde die Verifizierung mit dem LLM-Output und dem Volltext aufgerufen?
        expected_full_text = "Clean text page 1.\nClean text page 2."
        mock_verify_fields.assert_called_once_with(
            {"invoice_id": "INV-123", "total_amount": "150.00"},
            expected_full_text
        )

        # Wurde die Finalisierung mit dem verifizierten Dictionary aufgerufen?
        mock_finalize_fields.assert_called_once_with(
            {"invoice_id": "INV-123", "total_amount": "150.00", "corrected": True}
        )

        # Ist das Endergebnis das, was die Finalisierungsfunktion zurückgegeben hat?
        self.assertEqual(result, {"invoice_id": "INV-123", "total_amount": 150.00, "corrected": True})

    # 2. Test für die Pipeline OHNE Bereinigung
    @patch('document_processing.finalize_extracted_fields')
    @patch('document_processing.verify_and_correct_fields')
    @patch('document_processing.ollama_extract_invoice_fields')
    @patch('document_processing.clean_ocr_text')
    @patch('document_processing.ocr_pdf')
    def test_extract_invoice_fields_from_pdf_without_cleaning(
            self,
            mock_ocr_pdf,
            mock_clean_ocr_text,
            mock_ollama_extract,
            mock_verify_fields,
            mock_finalize_fields
    ):
        """Testet die Pipeline, wenn `clean=False` gesetzt ist."""
        # --- Arrange ---
        pdf_path = "/fake/path/to/invoice.pdf"
        mock_ocr_pdf.return_value = ["Raw text page 1.", "Raw text page 2."]

        # --- Act ---
        document_processing.extract_invoice_fields_from_pdf(pdf_path)

        # --- Assert ---
        # Stelle sicher, dass die Bereinigungsfunktion NICHT aufgerufen wurde
        mock_clean_ocr_text.assert_not_called()

        # Stelle sicher, dass der LLM stattdessen den rohen Text erhalten hat
        expected_llm_input = ["Raw text page 1.", "Raw text page 2."]
        mock_ollama_extract.assert_called_once_with(expected_llm_input)

    # 3. Test für die OCR-Dispatcher-Funktion
    def test_ocr_pdf_raises_error_for_invalid_engine(self):
        """Testet, ob ocr_pdf bei einer unbekannten Engine einen ValueError auslöst."""
        with self.assertRaises(ValueError) as context:
            document_processing.ocr_pdf("dummy.pdf", engine="invalid_engine")

        self.assertIn("Unsupported OCR engine 'invalid_engine'", str(context.exception))

    # 4. Test der allgemeinen OCR-Verarbeitungsfunktion
    @patch('document_processing.pdf_to_png_with_pymupdf')
    def test_process_pdf_with_ocr(self, mock_pdf_to_png):
        """Testet die Logik von process_pdf_with_ocr in Isolation."""
        # --- Arrange ---
        # Mock für die PDF-zu-PNG-Konvertierung
        mock_pdf_to_png.return_value = ["/tmp/page1.png", "/tmp/page2.png"]

        # Ein Mock für eine beliebige OCR-Funktion
        mock_ocr_function = MagicMock(name="mock_ocr_engine")
        mock_ocr_function.side_effect = ["Text from page 1", "Text from page 2"]
        mock_ocr_function.__name__ = "mock_ocr_engine"  # Wichtig für die Log-Ausgabe

        # --- Act ---
        result = document_processing.process_pdf_with_ocr("dummy.pdf", mock_ocr_function)

        # --- Assert ---
        # Wurde die PNG-Konvertierung aufgerufen?
        mock_pdf_to_png.assert_called_once_with("dummy.pdf")

        # Wurde die OCR-Funktion für jede Seite aufgerufen?
        self.assertEqual(mock_ocr_function.call_count, 2)
        mock_ocr_function.assert_has_calls([call("/tmp/page1.png"), call("/tmp/page2.png")])

        # Ist das Ergebnis korrekt zusammengefügt?
        self.assertEqual(result, ["Text from page 1", "Text from page 2"])

    # 5. Test für den Fehlerfall in der OCR-Verarbeitung
    @patch('document_processing.pdf_to_png_with_pymupdf')
    def test_process_pdf_with_ocr_handles_exception(self, mock_pdf_to_png):
        """Testet, ob Fehler bei der Verarbeitung korrekt abgefangen werden."""
        # --- Arrange ---
        # Simuliere einen Fehler bei der PDF-Konvertierung
        mock_pdf_to_png.side_effect = Exception("Failed to open PDF")
        mock_ocr_function = MagicMock(name="mock_ocr_engine")
        mock_ocr_function.__name__ = "mock_ocr_engine"

        # --- Act