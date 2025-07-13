# FILE: test_post_processing.py
import unittest
from unittest.mock import patch, call

# Der zu testende Code
from utils import post_processing


class TestVerifyAndCorrectFields(unittest.TestCase):
    """
    Testet die Funktion post_processing.verify_and_correct_fields.
    """

    def test_overwrites_incorrect_values(self):
        """Sollte eine falsche IBAN und USt-Id mit Werten aus dem Text überschreiben."""
        data = {
            "invoice_date": "06.11.2012",
            "ust-id": "DE123456789",
            "iban": "DE89370400440532013666",
        }
        full_text = "Bitte überweisen Sie an: DE89 3704 0044 0532 0130 00. Danke. ust-id: DE123456789"

        expected = data.copy()
        expected['iban'] = 'DE89370400440532013000'
        expected['ust-id'] = 'DE123456789'

        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, expected)

    def test_fills_missing_values(self):
        """Sollte fehlende IBAN und USt-Id aus dem Text ergänzen."""
        data = {"invoice_date": "06.11.2012"}
        full_text = "Unsere Bankverbindung: ATU12345678 und IBAN: AT611904400234573201"

        expected = data.copy()
        expected['iban'] = 'AT611904400234573201'
        expected['ust-id'] = 'ATU12345678'

        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, expected)

    def test_does_nothing_if_no_patterns_found(self):
        """Sollte das Dictionary nicht ändern, wenn keine Muster im Text gefunden werden."""
        data = {"iban": "EXISTING-IBAN", "ust-id": "EXISTING-ID"}
        full_text = "Ein Text ohne relevante Informationen."

        expected = data.copy()
        result = post_processing.verify_and_correct_fields(data, full_text)
        self.assertEqual(result, expected)

    def test_handles_non_dict_input(self):
        """Sollte Eingaben, die kein Dictionary sind, unverändert zurückgeben."""
        self.assertEqual(post_processing.verify_and_correct_fields(None, "text"), None)
        self.assertEqual(post_processing.verify_and_correct_fields([], "text"), [])


class TestFinalizeExtractedFields(unittest.TestCase):
    """
    Testet die Funktion finalize_extracted_fields, die Datentypen konvertiert.
    """

    # Assuming canon_money is in the post_processing file for this example
    @patch('utils.post_processing.canon_money')
    def test_finalizes_total_amount(self, mock_canon_money):
        """Sollte `canon_money` für total_amount aufrufen."""
        mock_canon_money.return_value = 123.45
        data = {"total_amount": "123,45 EUR"}
        result = post_processing.finalize_extracted_fields(data)
        mock_canon_money.assert_called_once_with("123,45 EUR")
        self.assertEqual(result["total_amount"], 123.45)

    @patch('utils.post_processing.canon_money')
    def test_finalizes_both_fields(self, mock_canon_money):
        """Sollte `canon_money` für beide Felder aufrufen, wenn vorhanden."""
        mock_canon_money.side_effect = [150.00, 7.0]
        data = {"total_amount": "150.00", "tax_rate": "7"}
        result = post_processing.finalize_extracted_fields(data)
        mock_canon_money.assert_has_calls([call("150.00"), call("7")])
        self.assertEqual(result["total_amount"], 150.00)
        self.assertEqual(result["tax_rate"], 7.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)