"""Tests for the easyocr_engine.py module."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_digitalization.easyocr_engine import easyocr_png_to_text


class TestEasyOCR(unittest.TestCase):
    """Tests for the EasyOCR implementation."""

    def setUp(self):
        """Set up test environment."""
        self.sample_png = str(Path(project_root) / "resources" / "samples" / "BRE-03.png")
        self.assertTrue(os.path.exists(self.sample_png), f"Test file not found: {self.sample_png}")
        
        # Mock EasyOCR reader output - Format: [[bbox, text, confidence], ...]
        # bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        self.mock_reader_output = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], "Text line 1", 0.95],
            [[[10, 60], [120, 60], [120, 80], [10, 80]], "Text line 2", 0.90]
        ]
    
    @patch('app.document_digitalization.easyocr_engine.easyocr')
    def test_easyocr_png_to_text(self, mock_easyocr):
        """Test easyocr_png_to_text basic functionality."""
        # Set up the mock Reader instance
        mock_reader_instance = MagicMock()
        mock_reader_instance.readtext.return_value = self.mock_reader_output
        mock_easyocr.Reader.return_value = mock_reader_instance
        
        # Call the function
        result = easyocr_png_to_text(self.sample_png, ['de'])
        
        # Verify the result
        self.assertIsInstance(result, str)
        self.assertIn("Text line 1", result)
        self.assertIn("Text line 2", result)
        
        # Verify that Reader class was initialized with correct parameters
        mock_easyocr.Reader.assert_called_once_with(['de'], gpu=False)
        
        # Verify that readtext was called with the image path
        mock_reader_instance.readtext.assert_called_once()
        args = mock_reader_instance.readtext.call_args[0]
        self.assertEqual(args[0], self.sample_png)
    
    def test_real_easyocr_execution(self):
        """Test real execution of easyocr_png_to_text with a real PNG file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_EASYOCR_TESTS'):
            self.skipTest("Skipping real EasyOCR test. Set RUN_REAL_EASYOCR_TESTS=1 to enable.")
        
        # Test with plain text output
        text_result = easyocr_png_to_text(self.sample_png, ['de'])
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)


if __name__ == "__main__":
    unittest.main() 