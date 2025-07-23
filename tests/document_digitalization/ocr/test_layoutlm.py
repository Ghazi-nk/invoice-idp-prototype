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

from app.ocr.layoutlmv3_png2txt import layoutlm_image_to_text


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
    
    @patch('app.document_digitalization.layoutlmv3_png2txt.Image')
    @patch('app.document_digitalization.layoutlmv3_png2txt._PROCESSOR')
    def test_layoutlm_image_to_text(self, mock_processor, mock_image_class):
        """Test layoutlm_image_to_text with plain text output."""
        # Set up mocks
        mock_image_instance = MagicMock()
        mock_image_class.open.return_value = mock_image_instance
        mock_image_instance.convert.return_value = mock_image_instance
        
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        
        # Mock the necessary parts to return our test data
        mock_processor.tokenizer = MagicMock()
        
        # Set up mock inputs to include token IDs and bounding boxes
        mock_inputs = {
            "input_ids": [[101, 1996, 2256, 102]],  # Mock token IDs
            "bbox": [[
                [0, 0, 0, 0],  # CLS token
                [100, 50, 200, 70],  # "Invoice"
                [210, 50, 300, 70],  # "Number" 
                [0, 0, 0, 0]   # SEP token
            ]]
        }
        mock_processor.return_value = mock_inputs
        
        # Patch the token conversion
        with patch('app.document_digitalization.layoutlmv3_png2txt._PROCESSOR.tokenizer.convert_ids_to_tokens') as mock_convert:
            mock_convert.side_effect = lambda x: {
                101: "[CLS]",
                1996: "Invoice",
                2256: "Number",
                102: "[SEP]"
            }.get(x, "")
            
            # Set up special tokens
            mock_processor.tokenizer.all_special_tokens = ["[CLS]", "[SEP]"]
            
            # Call the function
            result = layoutlm_image_to_text(self.sample_png)
        
        # Verify the result
        self.assertIsInstance(result, str)
        
        # Check that the image was processed correctly
        mock_image_class.open.assert_called_once_with(self.sample_png)
        mock_image_instance.convert.assert_called_once_with("RGB")
        
        # Verify processor was called correctly
        mock_processor.assert_called_once()
        
    def test_real_layoutlm_execution(self):
        """Test real execution of layoutlm_image_to_text with a real PNG file."""
        # Skip if not running in a real environment
        if not os.environ.get('RUN_REAL_LAYOUTLM_TESTS'):
            self.skipTest("Skipping real LayoutLM test. Set RUN_REAL_LAYOUTLM_TESTS=1 to enable.")
        
        # Test with plain text output
        text_result = layoutlm_image_to_text(self.sample_png)
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)


if __name__ == "__main__":
    unittest.main() 