
import base64
import json
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY

# Der zu testende Code
# Annahme: Der Code befindet sich in einer Datei namens pdf_utils.py
from app.ocr import pdf_utils

# Definieren einer Dummy-Konstante, damit der Test ohne die echte config.py läuft
pdf_utils.TMP_DIR = "/tmp"


class TestPdfUtils(unittest.TestCase):

    # --- Tests für save_base64_to_temp_pdf ---

    @patch('utils.pdf_utils.os.remove')
    @patch('utils.pdf_utils.os.path.exists', return_value=True)
    @patch('utils.pdf_utils.tempfile.NamedTemporaryFile')
    @patch('utils.pdf_utils.base64.b64decode')
    def test_save_base64_to_temp_pdf_success(self, mock_b64decode, mock_tempfile, mock_exists, mock_remove):
        """
        Testet das erfolgreiche Dekodieren und Speichern von Base64 in einer temporären Datei.
        """
        # --- Arrange ---
        mock_file_handle = MagicMock()
        mock_file_context = MagicMock()
        mock_file_context.__enter__.return_value = mock_file_handle
        mock_file_handle.name = "/tmp/fake_temp_file.pdf"
        mock_tempfile.return_value = mock_file_context

        base64_string = "dGVzdA=="  # "test" in base64
        decoded_data = b"test"
        mock_b64decode.return_value = decoded_data

        # --- Act & Assert ---
        with pdf_utils.save_base64_to_temp_pdf(base64_string) as temp_path:
            mock_b64decode.assert_called_once_with(base64_string)
            mock_tempfile.assert_called_once_with(suffix=".pdf", delete=False)
            mock_file_handle.write.assert_called_once_with(decoded_data)
            self.assertEqual(temp_path, "/tmp/fake_temp_file.pdf")

        mock_exists.assert_called_once_with("/tmp/fake_temp_file.pdf")
        mock_remove.assert_called_once_with("/tmp/fake_temp_file.pdf")

    # --- Tests für encode_image_to_base64 ---

    def test_encode_image_to_base64(self):
        """
        Testet die korrekte Base64-Kodierung von Bilddaten.
        """
        # --- Arrange ---
        fake_image_path = "/fake/image.png"
        image_content = b"\x89PNG\r\n\x1a\n"  # Minimaler PNG-Header
        m_open = mock_open(read_data=image_content)

        # --- Act ---
        with patch("builtins.open", m_open):
            result = pdf_utils.encode_image_to_base64(fake_image_path)

        # --- Assert ---
        m_open.assert_called_once_with(fake_image_path, "rb")
        expected_base64 = base64.b64encode(image_content).decode('utf-8')
        self.assertEqual(result, expected_base64)

    # --- Tests für pdf_to_png_with_pymupdf ---

    @patch('utils.pdf_utils.fitz.open')
    def test_pdf_to_png_with_pymupdf_success(self, mock_fitz_open):
        """
        Testet die erfolgreiche Konvertierung von PDF zu PNG mit PyMuPDF.
        """
        # --- Arrange ---
        mock_pixmap = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_pixmap.return_value = mock_pixmap
        mock_page2 = MagicMock()
        mock_page2.get_pixmap.return_value = mock_pixmap

        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc

        pdf_path = "/fake/doc.pdf"

        # --- Act ---
        result_paths = pdf_utils.pdf_to_png_with_pymupdf(pdf_path, zoom=2.0)

        # --- Assert ---
        mock_fitz_open.assert_called_once_with(pdf_path)

        self.assertEqual(mock_page1.get_pixmap.call_count, 1)
        self.assertEqual(mock_page2.get_pixmap.call_count, 1)

        mock_page1.get_pixmap.assert_called_with(matrix=ANY, alpha=False)
        matrix_arg = mock_page1.get_pixmap.call_args[1]['matrix']
        self.assertEqual(matrix_arg.a, 2.0)
        self.assertEqual(matrix_arg.d, 2.0)

        self.assertEqual(mock_pixmap.save.call_count, 2)

        # Baue die erwarteten Pfade OS-unabhängig zusammen
        expected_path1 = os.path.join("/tmp", "doc_page1.png")
        expected_path2 = os.path.join("/tmp", "doc_page2.png")

        # Prüfe die Aufrufe mit den korrekt erstellten Pfaden
        mock_pixmap.save.assert_has_calls([
            unittest.mock.call(expected_path1),
            unittest.mock.call(expected_path2),
        ])

        # Prüfe das Ergebnis mit den korrekt erstellten Pfaden
        self.assertEqual(result_paths, [expected_path1, expected_path2])

    @patch('utils.pdf_utils.fitz.open')
    def test_pdf_to_png_with_pymupdf_empty_pdf(self, mock_fitz_open):
        """
        Testet, ob ein Fehler ausgelöst wird, wenn das PDF keine Seiten hat.
        """
        # --- Arrange ---
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_fitz_open.return_value = mock_doc

        # --- Act & Assert ---
        with self.assertRaisesRegex(RuntimeError, "Keine Seiten in PDF"):
            pdf_utils.pdf_to_png_with_pymupdf("/fake/empty.pdf")

    # --- Tests für extract_text_if_searchable ---

    @patch('utils.pdf_utils.fitz.open')
    def test_extract_text_if_searchable_with_text(self, mock_fitz_open):
        """
        Testet die Extraktion, wenn das PDF durchsuchbaren Text enthält.
        """
        # --- Arrange ---
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Text von Seite 1."
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Text von Seite 2."

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc

        m_open = mock_open()

        # --- Act ---
        with patch("builtins.open", m_open):
            result = pdf_utils.extract_text_if_searchable("/fake/searchable.pdf")

        # --- Assert ---
        mock_page1.get_text.assert_called_once()
        mock_page2.get_text.assert_called_once()

        # Baue den erwarteten Pfad OS-unabhängig zusammen
        expected_text_path = os.path.join("/tmp", "searchable.txt")

        # Prüfe den Aufruf mit dem korrekt erstellten Pfad
        m_open.assert_called_once_with(expected_text_path, "w", encoding="utf-8")

        expected_text = "Text von Seite 1.\nText von Seite 2."
        m_open().write.assert_called_once_with(expected_text)

        self.assertEqual(result, json.dumps(expected_text, ensure_ascii=False))

    @patch('utils.pdf_utils.fitz.open')
    def test_extract_text_if_searchable_image_pdf(self, mock_fitz_open):
        """
        Testet das Verhalten, wenn das PDF keinen extrahierbaren Text hat (Bild-PDF).
        """
        # --- Arrange ---
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # Kein Text gefunden
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        m_open = mock_open()

        # --- Act ---
        with patch("builtins.open", m_open):
            result = pdf_utils.extract_text_if_searchable("/fake/image.pdf")

        # --- Assert ---
        mock_page.get_text.assert_called_once()
        m_open.assert_not_called()
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)