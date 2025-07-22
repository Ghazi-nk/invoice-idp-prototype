"""Tests for the layoutlmv3_png2txt.py module."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text


class TestLayoutLMOCR(unittest.TestCase):
    """Tests for the LayoutLM OCR implementation."""

    def setUp(self):
        """Set up test environment."""
        self.sample_png = str(Path(project_root) / "resources" / "samples" / "BRE-03.png")
        self.assertTrue(os.path.exists(self.sample_png), f"Test file not found: {self.sample_png}")
        
        # Create mock objects and results
        self._setup_mocks()
    
    def _setup_mocks(self):
        """Set up mock objects for LayoutLM testing."""
        # Mock token data
        self.mock_token_data = [
            {"text": "Invoice", "bbox": [100, 50, 200, 70]},
            {"text": "Number", "bbox": [210, 50, 300, 70]},
            {"text": "12345", "bbox": [310, 50, 400, 70]},
            {"text": "Customer", "bbox": [100, 100, 200, 120]},
            {"text": "ACME", "bbox": [210, 100, 280, 120]},
            {"text": "Corp", "bbox": [290, 100, 350, 120]}
        ]
        
        # Mock chunk objects
        self.mock_chunk_objects = [
            {"text": "Invoice Number", "bbox": [100, 50, 300, 70]},
            {"text": "12345", "bbox": [310, 50, 400, 70]},
            {"text": "Customer ACME Corp", "bbox": [100, 100, 350, 120]}
        ]
    
    @patch('app.document_digitalization.layoutlmv3_png2txt._load_image')
    @patch('app.document_digitalization.layoutlmv3_png2txt._flatten_tokens')
    @patch('app.document_digitalization.layoutlmv3_png2txt._bucket_tokens_by_line')
    @patch('app.document_digitalization.layoutlmv3_png2txt._compute_dynamic_threshold')
    @patch('app.document_digitalization.layoutlmv3_png2txt._chunk_all_lines')
    @patch('app.document_digitalization.layoutlmv3_png2txt._make_chunk_objects')
    @patch('app.document_digitalization.layoutlmv3_png2txt._PROCESSOR')
    @patch('app.document_digitalization.layoutlmv3_png2txt._MODEL')
    def test_layoutlm_image_to_text_without_bbox(self, mock_model, mock_processor, mock_make_chunk_objects, 
                                               mock_chunk_all_lines, mock_compute_threshold, 
                                               mock_bucket_tokens, mock_flatten_tokens, mock_load_image):
        """Test layoutlm_image_to_text without bounding boxes."""
        # Set up mocks
        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        
        mock_flatten_tokens.return_value = self.mock_token_data
        mock_bucket_tokens.return_value = [self.mock_token_data]
        mock_compute_threshold.return_value = 20
        mock_chunk_all_lines.return_value = [self.mock_token_data]
        mock_make_chunk_objects.return_value = self.mock_chunk_objects
        
        # Call the function
        result = layoutlm_image_to_text(self.sample_png, include_bbox=False)
        
        # Verify the result
        self.assertIsInstance(result, str)
        
        # Check if the result contains expected formatted text with y-coordinates
        self.assertIn('[y=', result)  # Contains y-coordinates
        self.assertIn('"Invoice Number"', result)
        self.assertIn('"12345"', result)
        self.assertIn('"Customer ACME Corp"', result)
        
        # Verify that the necessary functions were called
        mock_load_image.assert_called_once_with(self.sample_png)
        mock_processor.assert_called_once()
        mock_flatten_tokens.assert_called_once()
        mock_bucket_tokens.assert_called_once()
        mock_compute_threshold.assert_called_once()
        mock_chunk_all_lines.assert_called_once()
        mock_make_chunk_objects.assert_called_once()
    
    @patch('app.document_digitalization.layoutlmv3_png2txt._load_image')
    @patch('app.document_digitalization.layoutlmv3_png2txt._flatten_tokens')
    @patch('app.document_digitalization.layoutlmv3_png2txt._bucket_tokens_by_line')
    @patch('app.document_digitalization.layoutlmv3_png2txt._compute_dynamic_threshold')
    @patch('app.document_digitalization.layoutlmv3_png2txt._chunk_all_lines')
    @patch('app.document_digitalization.layoutlmv3_png2txt._make_chunk_objects')
    @patch('app.document_digitalization.layoutlmv3_png2txt._PROCESSOR')
    @patch('app.document_digitalization.layoutlmv3_png2txt._MODEL')
    def test_layoutlm_image_to_text_with_bbox(self, mock_model, mock_processor, mock_make_chunk_objects, 
                                            mock_chunk_all_lines, mock_compute_threshold, 
                                            mock_bucket_tokens, mock_flatten_tokens, mock_load_image):
        """Test layoutlm_image_to_text with bounding boxes."""
        # Set up mocks
        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        
        mock_flatten_tokens.return_value = self.mock_token_data
        mock_bucket_tokens.return_value = [self.mock_token_data]
        mock_compute_threshold.return_value = 20
        mock_chunk_all_lines.return_value = [self.mock_token_data]
        mock_make_chunk_objects.return_value = self.mock_chunk_objects
        
        # Call the function
        result = layoutlm_image_to_text(self.sample_png, include_bbox=True)
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertEqual(len(result), 3)  # Three text chunks
        
        # Verify the structure of the items
        for item in result:
            self.assertIn('text', item)
            self.assertIn('bbox', item)
            self.assertIsInstance(item['text'], str)
            self.assertIsInstance(item['bbox'], list)
            self.assertEqual(len(item['bbox']), 4)  # bbox should be [x0, y0, x1, y1]
        
        # Check if contents are as expected
        self.assertEqual(result[0]['text'], "Invoice Number")
        self.assertEqual(result[1]['text'], "12345")
        self.assertEqual(result[2]['text'], "Customer ACME Corp")
        
        # Verify bounding box coordinates are reasonable
        self.assertEqual(result[0]['bbox'], [100, 50, 300, 70])
        self.assertEqual(result[1]['bbox'], [310, 50, 400, 70])
        self.assertEqual(result[2]['bbox'], [100, 100, 350, 120])
        
        # Verify that the necessary functions were called
        mock_load_image.assert_called_once_with(self.sample_png)
        mock_processor.assert_called_once()
        mock_flatten_tokens.assert_called_once()
        mock_bucket_tokens.assert_called_once()
        mock_compute_threshold.assert_called_once()
        mock_chunk_all_lines.assert_called_once()
        mock_make_chunk_objects.assert_called_once()
    
    def test_real_layoutlm_execution(self):
        """Test real execution of layoutlm_image_to_text with a real PNG file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_LAYOUTLM_TESTS'):
            self.skipTest("Skipping real LayoutLM test. Set RUN_REAL_LAYOUTLM_TESTS=1 to enable.")
        
        # Test without bbox
        text_result = layoutlm_image_to_text(self.sample_png, include_bbox=False)
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)
        self.assertIn('[y=', text_result)
        
        # Test with bbox
        bbox_result = layoutlm_image_to_text(self.sample_png, include_bbox=True)
        self.assertIsInstance(bbox_result, list)
        self.assertGreater(len(bbox_result), 0)
        for item in bbox_result:
            self.assertIn('text', item)
            self.assertIn('bbox', item)
            self.assertIsInstance(item['text'], str)
            self.assertIsInstance(item['bbox'], list)
            self.assertEqual(len(item['bbox']), 4)


if __name__ == "__main__":
    unittest.main() 