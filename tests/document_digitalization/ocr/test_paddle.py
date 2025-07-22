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

from app.document_digitalization.paddle_ocr import paddleocr_pdf_to_text


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
    
    @patch('app.document_digitalization.paddle_ocr.os.path')
    @patch('app.document_digitalization.paddle_ocr.PaddleOCR')
    def test_paddleocr_pdf_to_text_without_bbox(self, mock_paddleocr_class, mock_os_path):
        """Test paddleocr_pdf_to_text without bounding boxes."""
        # Set up mocks
        mock_paddleocr_instance = MagicMock()
        mock_paddleocr_instance.ocr.return_value = self.mock_paddle_result
        mock_paddleocr_class.return_value = mock_paddleocr_instance
        
        # Mock os.path.exists to return True
        mock_os_path.exists.return_value = True
        
        # Call the function
        result = paddleocr_pdf_to_text(self.sample_pdf, lang='german', return_bbox=False)
        
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
        
        # Verify that ocr method was called with the PDF path
        mock_paddleocr_instance.ocr.assert_called_once_with(self.sample_pdf)
    
    @patch('app.document_digitalization.paddle_ocr.os.path')
    @patch('app.document_digitalization.paddle_ocr.PaddleOCR')
    def test_paddleocr_pdf_to_text_with_bbox(self, mock_paddleocr_class, mock_os_path):
        """Test paddleocr_pdf_to_text with bounding boxes."""
        # Set up mocks
        mock_paddleocr_instance = MagicMock()
        mock_paddleocr_instance.ocr.return_value = self.mock_paddle_result
        mock_paddleocr_class.return_value = mock_paddleocr_instance
        
        # Mock os.path.exists to return True
        mock_os_path.exists.return_value = True
        
        # Call the function
        result = paddleocr_pdf_to_text(self.sample_pdf, lang='german', return_bbox=True)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Two pages
        
        # Check the structure of each page
        for page in result:
            self.assertIsInstance(page, list)
            self.assertTrue(all(isinstance(item, dict) for item in page))
            
            # Verify the structure of each text item
            for item in page:
                self.assertIn('text', item)
                self.assertIn('bbox', item)
                self.assertIsInstance(item['text'], str)
                self.assertIsInstance(item['bbox'], list)
                self.assertEqual(len(item['bbox']), 4)  # bbox should be [x0, y0, x1, y1]
        
        # Check specific content
        self.assertEqual(len(result[0]), 3)  # Page 1: 3 text items
        self.assertEqual(len(result[1]), 2)  # Page 2: 2 text items
        
        self.assertEqual(result[0][0]['text'], "Invoice")
        self.assertEqual(result[0][1]['text'], "12345")
        self.assertEqual(result[0][2]['text'], "Customer: ACME Corp")
        
        self.assertEqual(result[1][0]['text'], "Total Amount: $1,234.56")
        self.assertEqual(result[1][1]['text'], "Due Date: 2023-01-15")
        
        # Verify that PaddleOCR was instantiated correctly
        mock_paddleocr_class.assert_called_once()
        self.assertEqual(mock_paddleocr_class.call_args[1]['lang'], 'german')
        
        # Verify that ocr method was called with the PDF path
        mock_paddleocr_instance.ocr.assert_called_once_with(self.sample_pdf)
    
    def test_real_paddleocr_execution(self):
        """Test real execution of paddleocr_pdf_to_text with a real PDF file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_PADDLE_TESTS'):
            self.skipTest("Skipping real PaddleOCR test. Set RUN_REAL_PADDLE_TESTS=1 to enable.")
        
        # Test without bbox
        text_result = paddleocr_pdf_to_text(self.sample_pdf, lang='german', return_bbox=False)
        self.assertIsInstance(text_result, list)
        self.assertGreater(len(text_result), 0)
        for page in text_result:
            self.assertIsInstance(page, str)
        
        # Test with bbox
        bbox_result = paddleocr_pdf_to_text(self.sample_pdf, lang='german', return_bbox=True)
        self.assertIsInstance(bbox_result, list)
        self.assertGreater(len(bbox_result), 0)
        for page in bbox_result:
            self.assertIsInstance(page, list)
            for item in page:
                self.assertIn('text', item)
                self.assertIn('bbox', item)


if __name__ == "__main__":
    unittest.main() 