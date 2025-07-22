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
    
    @patch('app.document_digitalization.tesseract_ocr.pytesseract.image_to_string')
    def test_tesseract_png_to_text_without_bbox(self, mock_image_to_string):
        """Test tesseract_png_to_text without bounding boxes."""
        # Mock pytesseract's image_to_string to return predictable output
        mock_image_to_string.return_value = "Test line 1\nTest line 2"
        
        # Call the function
        result = tesseract_png_to_text(self.sample_png, return_bbox=False)
        
        # Verify the result
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test line 1\nTest line 2")
        
        # Verify that image_to_string was called once
        mock_image_to_string.assert_called_once()
    
    @patch('app.document_digitalization.tesseract_ocr.pytesseract.image_to_string')
    def test_tesseract_png_to_text_with_bbox(self, mock_image_to_string):
        """Test tesseract_png_to_text with bounding boxes."""
        # Mock pytesseract's image_to_string to return predictable output
        mock_image_to_string.return_value = "Test line 1\nTest line 2"
        
        # Call the function
        result = tesseract_png_to_text(self.sample_png, return_bbox=True)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertEqual(len(result), 2)  # Two lines of text
        
        # Verify the structure of the items
        for item in result:
            self.assertIn('text', item)
            self.assertIn('bbox', item)
            self.assertIsInstance(item['text'], str)
            self.assertIsInstance(item['bbox'], list)
            self.assertEqual(len(item['bbox']), 4)  # bbox should be [x0, y0, x1, y1]
        
        # Verify the content
        self.assertEqual(result[0]['text'], "Test line 1")
        self.assertEqual(result[1]['text'], "Test line 2")
        
        # Verify that image_to_string was called once
        mock_image_to_string.assert_called_once()
    
    def test_real_tesseract_execution(self):
        """Test real execution of tesseract_png_to_text with a real PNG file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_TESSERACT_TESTS'):
            self.skipTest("Skipping real tesseract test. Set RUN_REAL_TESSERACT_TESTS=1 to enable.")
        
        # Test without bbox
        text_result = tesseract_png_to_text(self.sample_png, return_bbox=False)
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)
        
        # Test with bbox
        bbox_result = tesseract_png_to_text(self.sample_png, return_bbox=True)
        self.assertIsInstance(bbox_result, list)
        self.assertTrue(all(isinstance(item, dict) for item in bbox_result))
        for item in bbox_result:
            self.assertIn('text', item)
            self.assertIn('bbox', item)


if __name__ == "__main__":
    unittest.main() 