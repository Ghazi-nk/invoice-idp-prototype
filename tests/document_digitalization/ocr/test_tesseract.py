"""Tests for the tesseract_ocr.py module."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_digitalization.tesseract_ocr import tesseract_png_to_text


class TestTesseractOCR(unittest.TestCase):
    """Tests for the Tesseract OCR implementation."""

    def setUp(self):
        """Set up test environment."""
        self.sample_png = str(Path(project_root) / "resources" / "samples" / "BRE-03.png")
        self.assertTrue(os.path.exists(self.sample_png), f"Test file not found: {self.sample_png}")
    
    @patch('app.document_digitalization.tesseract_ocr.image_to_string')
    def test_tesseract_png_to_text(self, mock_image_to_string):
        """Test tesseract_png_to_text functionality."""
        # Mock pytesseract's image_to_string to return predictable output
        mock_image_to_string.return_value = "Test line 1\nTest line 2"
        
        # Call the function
        result = tesseract_png_to_text(self.sample_png)
        
        # Verify the result
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test line 1\nTest line 2")
        
        # Verify that image_to_string was called once with the correct parameters
        mock_image_to_string.assert_called_once_with(self.sample_png, lang='deu', config='--psm 3')
    
    def test_real_tesseract_execution(self):
        """Test real execution of tesseract_png_to_text with a real PNG file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_TESSERACT_TESTS'):
            self.skipTest("Skipping real tesseract test. Set RUN_REAL_TESSERACT_TESTS=1 to enable.")
        
        # Test with plain text output
        text_result = tesseract_png_to_text(self.sample_png)
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)


if __name__ == "__main__":
    unittest.main() 