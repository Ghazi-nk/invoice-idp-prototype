"""Tests for the doctr_pdf2txt.py module."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ocr.doctr_pdf2txt import doctr_pdf_to_text


class TestDoctrOCR(unittest.TestCase):
    """Tests for the Doctr OCR implementation."""

    def setUp(self):
        """Set up test environment."""
        self.sample_pdf = str(Path(project_root) / "resources" / "samples" / "BRE-03.pdf")
        self.assertTrue(os.path.exists(self.sample_pdf), f"Test file not found: {self.sample_pdf}")
        
        # Create a mock Doctr result object
        self.mock_doctr_result = self._create_mock_doctr_result()
    
    def _create_mock_doctr_result(self):
        """Create a mock Doctr OCR result."""
        mock_result = MagicMock()
        
        # Set up export method to return a dict with expected structure
        mock_result.export.return_value = {
            "pages": [
                {
                    "dimensions": (1000, 800),
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "geometry": ((0.1, 0.1), (0.5, 0.15)),
                                    "words": [{"value": "Invoice"}, {"value": "Number:"}]
                                },
                                {
                                    "geometry": ((0.6, 0.1), (0.8, 0.15)),
                                    "words": [{"value": "12345"}]
                                }
                            ]
                        },
                        {
                            "lines": [
                                {
                                    "geometry": ((0.1, 0.3), (0.7, 0.35)),
                                    "words": [{"value": "Customer:"}, {"value": "ACME"}, {"value": "Corp"}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        return mock_result
    
    @patch('app.ocr.doctr_pdf2txt.ocr_predictor')
    @patch('app.ocr.doctr_pdf2txt.DocumentFile')
    def test_doctr_pdf_to_text(self, mock_doc_file, mock_predictor):
        """Test doctr_pdf_to_text with plain text output."""
        # Set up the mocks
        mock_doc_file.from_pdf.return_value = MagicMock()
        mock_predictor.return_value.return_value = self.mock_doctr_result
        
        # Call the function
        result = doctr_pdf_to_text(self.sample_pdf)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # One page
        self.assertIsInstance(result[0], str)
        
        # Check if the result contains expected text
        page_text = result[0]
        self.assertIn('Invoice Number:', page_text)
        self.assertIn('12345', page_text)
        self.assertIn('Customer: ACME Corp', page_text)
        
        # Verify that the mocks were called correctly
        mock_doc_file.from_pdf.assert_called_once_with(self.sample_pdf)
        mock_predictor.assert_called_once_with(pretrained=True)
    
    def test_real_doctr_execution(self):
        """Test real execution of doctr_pdf_to_text with a real PDF file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_DOCTR_TESTS'):
            self.skipTest("Skipping real Doctr test. Set RUN_REAL_DOCTR_TESTS=1 to enable.")
        
        # Test with plain text output
        text_result = doctr_pdf_to_text(self.sample_pdf)
        self.assertIsInstance(text_result, list)
        self.assertGreater(len(text_result), 0)
        for page in text_result:
            self.assertIsInstance(page, str)
            self.assertGreater(len(page), 0)


if __name__ == "__main__":
    unittest.main() 