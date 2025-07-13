
import unittest
from unittest.mock import patch, call

# Der zu testende Code
from utils import post_processing


class TestVerifyAndCorrectFields(unittest.TestCase):
    """
    Testet die Funktion verify_and_correct_fields, die fehlende Felder
    mithilfe von Regex aus dem Volltext ergänzt.
    """

    def test_corrects_missing_iban(self):
        """Sollte eine fehlende IBAN aus dem Text extrahieren und bereinigen."""
        data = {"invoice_id": "123", "iban": "DE89370400440532013666"}
        full_text = "Bitte überweisen Sie an: DE89 3704 0044 0532 0130 00. Danke."

        # KORREKTUR: Das 'expected'-Dictionary muss natürlich auch die gefundene IBAN enthalten.
        expected = {
            "iban": "DE89370400440532013000",
            "invoice_id": "123s"
        }

        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(expected, result)

    def test_does_not_overwrite_existing_iban(self):
        """Sollte eine bereits vorhandene IBAN nicht überschreiben."""
        data = {"invoice_id": "123", "iban": "EXISTING_IBAN"}
        full_text = "Hier steht eine andere IBAN: DE89370400440532013000"
        # Das Original-Dictionary wird erwartet, keine Änderung
        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, data)

    def test_corrects_missing_ust_id(self):
        """Sollte eine fehlende USt-Id (VAT-ID) aus dem Text extrahieren."""
        data = {"invoice_id": "456"}
        full_text = "Unsere USt-Id lautet de123456789. Rechnung für Sie."
        expected = {
            "invoice_id": "456",
            "ust-id": "DE123456789"  # Großgeschrieben
        }
        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, expected)

    def test_does_not_overwrite_existing_ust_id(self):
        """Sollte eine bereits vorhandene USt-Id nicht überschreiben."""
        data = {"invoice_id": "456", "ust-id": "EXISTING_ID"}
        full_text = "Unsere USt-Id lautet DE123456789."
        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, data)

    def test_returns_unmodified_if_no_patterns_found(self):
        """Sollte das Dictionary nicht ändern, wenn keine Muster gefunden werden."""
        data = {"invoice_id": "789"}
        full_text = "Dies ist ein einfacher Text ohne spezielle Nummern."
        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, data)

    def test_handles_non_dict_input(self):
        """Sollte Eingaben, die kein Dictionary sind, unverändert zurückgeben."""
        self.assertEqual(post_processing.verify_and_correct_fields(None, "text"), None)
        self.assertEqual(post_processing.verify_and_correct_fields([], "text"), [])


class TestFinalizeExtractedFields(unittest.TestCase):
    """
    Testet die Funktion finalize_extracted_fields, die Datentypen konvertiert.
    Die Abhängigkeit `canon_money` wird gemockt.
    """

    @patch('utils.post_processing.canon_money')
    def test_finalizes_total_amount(self, mock_canon_money):
        """Sollte `canon_money` für total_amount aufrufen."""
        # Arrange: Mock gibt für jeden Input eine "kanonisierte" Zahl zurück
        mock_canon_money.return_value = 123.45
        data = {"total_amount": "123,45 EUR"}

        # Act
        result = post_processing.finalize_extracted_fields(data)

        # Assert
        mock_canon_money.assert_called_once_with("123,45 EUR")
        self.assertEqual(result["total_amount"], 123.45)

    @patch('utils.post_processing.canon_money')
    def test_finalizes_tax_rate(self, mock_canon_money):
        """Sollte `canon_money` für tax_rate aufrufen."""
        mock_canon_money.return_value = 19.0
        data = {"tax_rate": "19%"}

        result = post_processing.finalize_extracted_fields(data)

        mock_canon_money.assert_called_once_with("19%")
        self.assertEqual(result["tax_rate"], 19.0)

    @patch('utils.post_processing.canon_money')
    def test_finalizes_both_fields(self, mock_canon_money):
        """Sollte `canon_money` für beide Felder aufrufen, wenn vorhanden."""
        # side_effect erlaubt es, für verschiedene Aufrufe unterschiedliche Werte zurückzugeben
        mock_canon_money.side_effect = [150.00, 7.0]
        data = {"total_amount": "150.00", "tax_rate": "7"}

        result = post_processing.finalize_extracted_fields(data)

        # Überprüfe die Aufrufe
        self.assertEqual(mock_canon_money.call_count, 2)
        mock_canon_money.assert_has_calls([
            call("150.00"),
            call("7")
        ])

        # Überprüfe die Ergebnisse
        self.assertEqual(result["total_amount"], 150.00)
        self.assertEqual(result["tax_rate"], 7.0)

    @patch('utils.post_processing.canon_money')
    def test_handles_missing_keys_gracefully(self, mock_canon_money):
        """Sollte keine Fehler werfen oder etwas tun, wenn die Keys fehlen."""
        data = {"invoice_id": "some_id"}
        result = post_processing.finalize_extracted_fields(data)

        # `canon_money` sollte gar nicht erst aufgerufen werden
        mock_canon_money.assert_not_called()
        self.assertEqual(result, data)  # Das Dictionary bleibt ansonsten unverändert

    def test_handles_non_dict_input(self):
        """Sollte Eingaben, die kein Dictionary sind, unverändert zurückgeben."""
        self.assertEqual(post_processing.finalize_extracted_fields(None), None)
        self.assertEqual(post_processing.finalize_extracted_fields("a string"), "a string")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)