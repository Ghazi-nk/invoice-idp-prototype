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

from app.document_digitalization.doctr_pdf2txt import doctr_pdf_to_text


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
    
    @patch('app.document_digitalization.doctr_pdf2txt._run_ocr')
    def test_doctr_pdf_to_text_without_bbox(self, mock_run_ocr):
        """Test doctr_pdf_to_text without bounding boxes."""
        # Set up the mock
        mock_run_ocr.return_value = self.mock_doctr_result
        
        # Call the function
        result = doctr_pdf_to_text(self.sample_pdf, include_bbox=False)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # One page
        self.assertIsInstance(result[0], str)
        
        # Check if the result contains expected formatted text with y-coordinates
        page_text = result[0]
        self.assertIn('[y=', page_text)  # Contains y-coordinates
        self.assertIn('"Invoice Number:"', page_text)  # Contains merged text
        self.assertIn('"12345"', page_text)
        self.assertIn('"Customer: ACME Corp"', page_text)
        
        # Verify that _run_ocr was called once with the PDF path
        mock_run_ocr.assert_called_once_with(self.sample_pdf)
    
    @patch('app.document_digitalization.doctr_pdf2txt._run_ocr')
    def test_doctr_pdf_to_text_with_bbox(self, mock_run_ocr):
        """Test doctr_pdf_to_text with bounding boxes."""
        # Set up the mock
        mock_run_ocr.return_value = self.mock_doctr_result
        
        # Call the function
        result = doctr_pdf_to_text(self.sample_pdf, include_bbox=True)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # One page
        
        # Check the structure of the first page
        page_items = result[0]
        self.assertIsInstance(page_items, list)
        self.assertTrue(all(isinstance(item, dict) for item in page_items))
        
        # Verify the structure of the items
        for item in page_items:
            self.assertIn('text', item)
            self.assertIn('bbox', item)
            self.assertIsInstance(item['text'], str)
            self.assertIsInstance(item['bbox'], list)
            self.assertEqual(len(item['bbox']), 4)  # bbox should be [x0, y0, x1, y1]
        
        # Check if contents are as expected
        self.assertEqual(len(page_items), 3)  # Three text lines
        self.assertEqual(page_items[0]['text'], "Invoice Number:")
        self.assertEqual(page_items[1]['text'], "12345")
        self.assertEqual(page_items[2]['text'], "Customer: ACME Corp")
        
        # Verify bounding box coordinates are reasonable
        for item in page_items:
            bbox = item['bbox']
            # Bounding boxes should be within the page dimensions
            self.assertTrue(0 <= bbox[0] <= 1000)
            self.assertTrue(0 <= bbox[1] <= 800)
            self.assertTrue(0 <= bbox[2] <= 1000)
            self.assertTrue(0 <= bbox[3] <= 800)
            # Second coordinate should be greater than first
            self.assertTrue(bbox[2] > bbox[0])
            self.assertTrue(bbox[3] > bbox[1])
        
        # Verify that _run_ocr was called once with the PDF path
        mock_run_ocr.assert_called_once_with(self.sample_pdf)
    
    def test_real_doctr_execution(self):
        """Test real execution of doctr_pdf_to_text with a real PDF file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_DOCTR_TESTS'):
            self.skipTest("Skipping real Doctr test. Set RUN_REAL_DOCTR_TESTS=1 to enable.")
        
        # Test without bbox
        text_result = doctr_pdf_to_text(self.sample_pdf, include_bbox=False)
        self.assertIsInstance(text_result, list)
        self.assertGreater(len(text_result), 0)
        for page in text_result:
            self.assertIsInstance(page, str)
            self.assertGreater(len(page), 0)
        
        # Test with bbox
        bbox_result = doctr_pdf_to_text(self.sample_pdf, include_bbox=True)
        self.assertIsInstance(bbox_result, list)
        self.assertGreater(len(bbox_result), 0)
        for page in bbox_result:
            self.assertIsInstance(page, list)
            self.assertGreater(len(page), 0)
            for item in page:
                self.assertIn('text', item)
                self.assertIn('bbox', item)


if __name__ == "__main__":
    unittest.main() 