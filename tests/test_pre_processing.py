import unittest

# The code to be tested
from utils import pre_processing


class TestPreProcessing(unittest.TestCase):
    """
    Tests the text pre-processing functions.
    """

    # --- Tests for the private helper: _preprocess_text_content ---

    def test_iban_correction(self):
        """Should correct the common 'A7' to 'AT' OCR error in IBANs."""
        text_in = "Our IBAN is A761 1904 4002 3457 3201."
        text_out = "Our IBAN is AT61 1904 4002 3457 3201."
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_apostrophe_removal(self):
        """Should remove all apostrophes from the text."""
        text_in = "It's the company's 'official' document."
        text_out = "Its the companys official document."
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_currency_symbol_correction(self):
        """Should correct the garbled 'Â€' to '€'."""
        text_in = "The total amount is 50.00 Â€."
        text_out = "The total amount is 50.00 €."
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_decimal_separator_standardization(self):
        """Should convert comma decimal separators to dots."""
        text_in = "Price: 1.234,56 and another price: 99,95€"
        text_out = "Price: 1,234.56 and another price: 99.95€"
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_hyphenated_word_joining(self):
        """Should join words split by a hyphen and a newline."""
        text_in = "This is a long docu-\nment that continues on the next line."
        text_out = "This is a long document that continues on the next line."
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_page_number_removal(self):
        """Should remove 'page X:' indicators from the start of lines."""
        text_in = "Some text\npage 2: continued text\n Page 3: more text"
        text_out = "Some text\ncontinued text\nmore text"
        self.assertEqual(pre_processing._preprocess_text_content(text_in), text_out)

    def test_handles_empty_input(self):
        """The helper function should handle empty or None input gracefully."""
        self.assertEqual(pre_processing._preprocess_text_content(""), "")
        self.assertEqual(pre_processing._preprocess_text_content(None), "")

    # --- Tests for the public function: preprocess_plain_text_output ---

    def test_removes_page_headers(self):
        """Should remove custom '--- Seite X ---' headers."""
        text_in = "Start of document.\n--- Seite 1 ---\nThis is page one."
        text_out = "Start of document.\nThis is page one."
        self.assertEqual(pre_processing.preprocess_plain_text_output(text_in), text_out)

    def test_removes_blank_lines_after_header_removal(self):
        """Should remove blank lines that can result from removing headers."""
        text_in = "Line 1\n--- Seite 5 ---\n\n\nLine 2"
        text_out = "Line 1\nLine 2"
        self.assertEqual(pre_processing.preprocess_plain_text_output(text_in), text_out)

    def test_full_preprocessing_flow(self):
        """Tests the entire preprocessing flow from raw text to cleaned text."""
        raw_text = (
            "--- Seite 1 ---\n"
            "An 'invoice' for Mustermann's company.\n"
            "Total: 1.234,56 Â€.\n"
            "\n"
            "--- Seite 2 ---\n"
            "page 2: The IBAN is A7123456789 and the docu-\n"
            "ment number is 42."
        )

        # This is the final, expected output after all rules are applied
        expected_clean_text = (
            "An invoice for Mustermanns company.\n"
            "Total: 1.234.56 €.\n"
            "The IBAN is AT123456789 and the document number is 42."
        )

        self.assertEqual(pre_processing.preprocess_plain_text_output(raw_text), expected_clean_text)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)