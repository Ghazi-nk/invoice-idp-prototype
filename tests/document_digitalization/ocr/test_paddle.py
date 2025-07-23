"""Tests for the paddle_ocr.py module."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ocr.paddle_ocr import paddleocr_pdf_to_text


class TestPaddleOCR(unittest.TestCase):
    """Tests for the PaddleOCR implementation."""

    def setUp(self):
        """Set up test environment."""
        self.sample_pdf = str(Path(project_root) / "resources" / "samples" / "BRE-03.pdf")
        self.assertTrue(os.path.exists(self.sample_pdf), f"Test file not found: {self.sample_pdf}")
        
        # Create mock PaddleOCR result
        # PaddleOCR format: [
        #   [ [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)], ...
        # ]
        self.mock_paddle_result = [
            [
                # Page 1
                [
                    [[[100, 50], [300, 50], [300, 70], [100, 70]], ("Invoice", 0.95)],
                    [[[320, 50], [400, 50], [400, 70], [320, 70]], ("12345", 0.90)],
                    [[[100, 100], [350, 100], [350, 120], [100, 120]], ("Customer: ACME Corp", 0.85)]
                ]
            ],
            [
                # Page 2
                [
                    [[[100, 50], [400, 50], [400, 70], [100, 70]], ("Total Amount: $1,234.56", 0.95)],
                    [[[100, 100], [350, 100], [350, 120], [100, 120]], ("Due Date: 2023-01-15", 0.90)]
                ]
            ]
        ]
    
    @patch('app.document_digitalization.paddle_ocr.PaddleOCR')
    def test_paddleocr_pdf_to_text(self, mock_paddleocr_class):
        """Test paddleocr_pdf_to_text with plain text output."""
        # Set up mocks
        mock_paddleocr_instance = MagicMock()
        mock_paddleocr_instance.predict.return_value = self.mock_paddle_result
        mock_paddleocr_class.return_value = mock_paddleocr_instance
        
        # Call the function
        result = paddleocr_pdf_to_text(self.sample_pdf, lang='german')
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Two pages
        
        # Check if the results contain the expected text
        self.assertIn("Invoice", result[0])
        self.assertIn("12345", result[0])
        self.assertIn("Customer: ACME Corp", result[0])
        
        self.assertIn("Total Amount: $1,234.56", result[1])
        self.assertIn("Due Date: 2023-01-15", result[1])
        
        # Verify that PaddleOCR was instantiated correctly
        mock_paddleocr_class.assert_called_once()
        self.assertEqual(mock_paddleocr_class.call_args[1]['lang'], 'german')
        
        # Verify that predict method was called with the PDF path
        mock_paddleocr_instance.predict.assert_called_once_with(self.sample_pdf)
    
    def test_real_paddleocr_execution(self):
        """Test real execution of paddleocr_pdf_to_text with a real PDF file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_PADDLE_TESTS'):
            self.skipTest("Skipping real PaddleOCR test. Set RUN_REAL_PADDLE_TESTS=1 to enable.")
        
        # Test with plain text output
        text_result = paddleocr_pdf_to_text(self.sample_pdf, lang='german')
        self.assertIsInstance(text_result, list)
        self.assertGreater(len(text_result), 0)
        for page in text_result:
            self.assertIsInstance(page, str)


if __name__ == "__main__":
    unittest.main() 