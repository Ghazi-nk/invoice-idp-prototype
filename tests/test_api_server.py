import unittest
import base64
from unittest.mock import patch

from fastapi.testclient import TestClient

# The FastAPI app instance to be tested
from app.api_server import app


class TestApiServer(unittest.TestCase):
    """
    Unit tests for the Invoice Extraction API server.
    """

    def setUp(self):
        """Set up the test client and sample data before each test."""
        self.client = TestClient(app)
        # A simple, valid base64 encoded string for testing ("test-pdf")
        self.sample_b64_pdf = base64.b64encode(b"test-pdf").decode('utf-8')

    # --- Tests for /api/v1/extract ---

    @patch('app.api_server.extract_invoice_fields_from_pdf')
    @patch('app.api_server.get_available_engines', return_value=['tesseract', 'doctr'])
    @patch('app.api_server.save_base64_to_temp_pdf')
    def test_extract_data_success(self, mock_save_pdf, mock_get_engines, mock_extract):
        """Should successfully extract data from a PDF."""
        # Arrange
        mock_save_pdf.return_value.__enter__.return_value = "/tmp/fake.pdf"
        mock_extract.return_value = {"total_amount": 123.45, "vendor_name": "TestCorp"}

        # Act
        response = self.client.post(
            "/api/v1/extract",
            json={"invoice_base64": self.sample_b64_pdf, "engine": "tesseract"}
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"total_amount": 123.45, "vendor_name": "TestCorp"})
        mock_extract.assert_called_once_with(pdf_path="/tmp/fake.pdf", engine="tesseract", clean=True)

    @patch('app.api_server.extract_invoice_fields_from_pdf')
    @patch('app.api_server.get_available_engines', return_value=['tesseract'])
    @patch('app.api_server.save_base64_to_temp_pdf')
    def test_extract_data_invalid_engine_fallback(self, mock_save_pdf, mock_get_engines, mock_extract):
        """Should fall back to the default engine if an invalid one is requested."""
        # Arrange
        mock_save_pdf.return_value.__enter__.return_value = "/tmp/fake.pdf"

        # Act
        self.client.post(
            "/api/v1/extract",
            json={"invoice_base64": self.sample_b64_pdf, "engine": "invalid_engine"}
        )

        # Assert
        # Check that the call was made with the default engine, 'tesseract'
        mock_extract.assert_called_once_with(pdf_path="/tmp/fake.pdf", engine="tesseract", clean=True)

    # --- Tests for /api/v1/ocr ---

    @patch('app.api_server.ocr_pdf')
    @patch('app.api_server.get_available_engines', return_value=['tesseract'])
    @patch('app.api_server.save_base64_to_temp_pdf')
    def test_get_ocr_text_success(self, mock_save_pdf, mock_get_engines, mock_ocr):
        """Should successfully return raw OCR text."""
        # Arrange
        mock_save_pdf.return_value.__enter__.return_value = "/tmp/fake.pdf"
        mock_ocr.return_value = ["Page 1 text.", "Page 2 text."]

        # Act
        response = self.client.post(
            "/api/v1/ocr",
            json={"invoice_base64": self.sample_b64_pdf, "engine": "tesseract"}
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["Page 1 text.", "Page 2 text."])
        mock_ocr.assert_called_once_with(pdf_path="/tmp/fake.pdf", engine="tesseract")

    # --- Tests for /api/v1/to-images ---

    @patch('app.api_server.os.remove')
    @patch('app.api_server.os.path.exists', return_value=True)
    @patch('app.api_server.encode_image_to_base64', return_value="encoded_image_string")
    @patch('app.api_server.pdf_to_png_with_pymupdf')
    @patch('app.api_server.save_base64_to_temp_pdf')
    def test_pdf_to_images_base64_success(self, mock_save_pdf, mock_pdf_to_png, mock_encode, mock_exists, mock_remove):
        """Should convert a PDF to a list of base64 encoded images."""
        # Arrange
        mock_save_pdf.return_value.__enter__.return_value = "/tmp/fake.pdf"
        mock_pdf_to_png.return_value = ["/tmp/page1.png", "/tmp/page2.png"]

        # Act
        response = self.client.post("/api/v1/to-images", json={"invoice_base64": self.sample_b64_pdf})

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"images": ["encoded_image_string", "encoded_image_string"]})
        self.assertEqual(mock_encode.call_count, 2)
        self.assertEqual(mock_remove.call_count, 2)  # Check that files are cleaned up

    # --- Tests for /api/v1/to-pdf ---

    def test_base64_to_pdf_file_success(self):
        """Should convert a base64 string to a downloadable PDF file."""
        # Act
        response = self.client.post("/api/v1/to-pdf", json={"invoice_base64": self.sample_b64_pdf})

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"test-pdf")
        # FIX: Changed 'media-type' to the correct header key 'content-type'
        self.assertEqual(response.headers['content-type'], "application/pdf")
        self.assertIn("attachment; filename=invoice.pdf", response.headers['content-disposition'])

    def test_base64_to_pdf_file_invalid_string(self):
        """Should return a 400 error for an invalid base64 string."""
        # Act
        response = self.client.post("/api/v1/to-pdf", json={"invoice_base64": "this is not base64"})

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid base64 string provided", response.json()['detail'])

    # --- Test for Error Handling ---

    @patch('app.api_server.save_base64_to_temp_pdf')
    def test_generic_server_error(self, mock_save_pdf):
        """Should return a 500 error for unexpected exceptions."""
        # Arrange: Make one of the mocked functions raise a generic Exception
        mock_save_pdf.side_effect = Exception("Something went very wrong")

        # Act
        response = self.client.post(
            "/api/v1/extract",
            json={"invoice_base64": self.sample_b64_pdf}
        )

        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("An unexpected server error occurred", response.json()['detail'])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)