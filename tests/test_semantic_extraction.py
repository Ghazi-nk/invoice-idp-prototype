import unittest
from unittest.mock import patch, MagicMock

# Der zu testende Code
# Annahme: Der Code befindet sich in einer Datei namens semantic_extraction.py
import app.semantic_extraction as llm_utils


class TestExtractFirstCompleteJson(unittest.TestCase):
    """
    Testet die Hilfsfunktion, die JSON aus einem Text extrahiert.
    Diese Funktion hat keine externen Abhängigkeiten und benötigt kein Mocking.
    """

    def test_extracts_with_json_tags(self):
        """Sollte den Inhalt aus <json_output> Tags priorisieren."""
        text = 'Einleitung... <json_output>{"key": "value"}</json_output> ...Nachwort'
        expected = '{"key": "value"}'
        self.assertEqual(llm_utils._extract_first_complete_json(text), expected)

    def test_extracts_with_tags_and_newlines(self):
        """Sollte auch mit Tags über mehrere Zeilen funktionieren."""
        text = 'Hello\n<json_output>\n{\n  "id": 123\n}\n</json_output>\nWorld'
        expected = '{\n  "id": 123\n}'
        self.assertEqual(llm_utils._extract_first_complete_json(text), expected)

    def test_fallback_to_brace_matching(self):
        """Sollte auf die Klammersuche zurückfallen, wenn keine Tags da sind."""
        text = 'Hier ist der Output: {"a": {"b": "c"}} und noch mehr Text.'
        expected = '{"a": {"b": "c"}}'
        self.assertEqual(llm_utils._extract_first_complete_json(text), expected)

    def test_handles_braces_in_strings(self):
        """Sollte mit Klammern innerhalb von JSON-Strings korrekt umgehen."""
        text = '{"message": "Ein Text {mit Klammern}"}'
        self.assertEqual(llm_utils._extract_first_complete_json(text), text)

    def test_returns_none_for_incomplete_json(self):
        """Sollte None zurückgeben, wenn das JSON unvollständig ist."""
        text = '{"key": "value"'
        self.assertIsNone(llm_utils._extract_first_complete_json(text))

    def test_returns_none_for_no_json(self):
        """Sollte None zurückgeben, wenn kein JSON-Objekt im Text ist."""
        text = "Das ist nur einfacher Text ohne JSON."
        self.assertIsNone(llm_utils._extract_first_complete_json(text))


class TestOllamaExtractInvoiceFields(unittest.TestCase):
    """
    Testet die Hauptfunktion zur API-Kommunikation.
    Hier wird intensiv gemockt.
    """

    # Mock für `requests.post` und `pathlib.Path`
    # Der `create=True` Parameter bei `patch` erlaubt das Mocken von Objekten,
    # die innerhalb der Funktion importiert werden (z.B. requests in deinem Code)
    @patch('app.semantic_extraction.requests.post')
    @patch('app.semantic_extraction.Path')
    def test_successful_extraction(self, mock_path, mock_post):
        """Testet den erfolgreichen Ende-zu-Ende-Durchlauf."""
        # --- Arrange: Lege das Verhalten der Mocks fest ---

        # 1. Mock für das Lesen der Prompt-Dateien
        mock_system_prompt = "Du bist ein Rechnungs-Extraktor."
        mock_user_prompt = "Extrahiere die Felder."

        # Simuliere, dass `read_text` je nach aufgerufener Datei etwas anderes zurückgibt
        def mock_read_text(encoding="utf-8"):
            path_str = str(mock_path.call_args[0][0]) if mock_path.call_args else ""
            if llm_utils.SYSTEM_PROMPT_FILE in path_str:
                return mock_system_prompt
            elif llm_utils.USER_PROMPT_FILE in path_str:
                return mock_user_prompt
            return "default_prompt"
        
        mock_path.return_value.read_text.side_effect = mock_read_text

        # 2. Mock für die API-Antwort von requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        api_response_content = 'Some text... <json_output>{"invoice_id": "123"}</json_output>'
        mock_response.json.return_value = {
            "message": {"content": api_response_content}
        }
        mock_post.return_value = mock_response

        # --- Act: Rufe die zu testende Funktion auf ---
        ocr_pages = ["Text von Seite 1"]
        result = llm_utils.ollama_extract_invoice_fields(ocr_pages)

        # --- Assert: Überprüfe, ob alles wie erwartet abgelaufen ist ---

        # 1. Wurde `requests.post` korrekt aufgerufen?
        mock_post.assert_called_once()
        # Hole die Argumente, mit denen post aufgerufen wurde
        args, kwargs = mock_post.call_args
        # Überprüfe den Body der Anfrage
        request_body = kwargs['json']

        self.assertEqual(request_body['model'], llm_utils.OLLAMA_MODEL)
        self.assertEqual(len(request_body['messages']), 3)  # System + 1 Page + User
        self.assertEqual(request_body['messages'][0]['role'], 'system')
        self.assertEqual(request_body['messages'][0]['content'], mock_system_prompt)
        self.assertIn("Text von Seite 1", request_body['messages'][1]['content'])
        self.assertEqual(request_body['messages'][2]['content'], mock_user_prompt)

        # 2. Ist das Endergebnis das geparste JSON?
        self.assertEqual(result, {"invoice_id": "123"})

    def test_raises_error_on_empty_ocr_pages(self):
        """Sollte einen ValueError bei leerer Input-Liste auslösen."""
        with self.assertRaisesRegex(ValueError, "Input ocr_pages list cannot be empty"):
            llm_utils.ollama_extract_invoice_fields([])

    @patch('app.semantic_extraction.Path')
    def test_raises_error_if_prompt_file_not_found(self, mock_path):
        """Sollte einen RuntimeError auslösen, wenn eine Prompt-Datei fehlt."""
        # Simuliere, dass beim Lesen der Datei ein Fehler auftritt
        mock_path.return_value.read_text.side_effect = FileNotFoundError("Datei nicht da")

        with self.assertRaisesRegex(RuntimeError, "Could not find a required prompt file"):
            llm_utils.ollama_extract_invoice_fields(["some text"])

    @patch('app.semantic_extraction.requests.post')
    @patch('app.semantic_extraction.Path')
    def test_raises_error_on_api_failure(self, mock_path, mock_post):
        """Sollte einen RuntimeError bei einem API-Fehler (Status != 200) auslösen."""
        # Arrange
        mock_path.return_value.read_text.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Act & Assert
        with self.assertRaisesRegex(RuntimeError, "Ollama API Error: 500 – Internal Server Error"):
            llm_utils.ollama_extract_invoice_fields(["some text"])

    @patch('app.semantic_extraction.requests.post')
    @patch('app.semantic_extraction.Path')
    def test_raises_error_when_no_json_in_response(self, mock_path, mock_post):
        """Sollte einen ValueError auslösen, wenn die API-Antwort kein JSON enthält."""
        # Arrange
        mock_path.return_value.read_text.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hier ist leider kein JSON."}
        }
        mock_post.return_value = mock_response

        # Act & Assert
        with self.assertRaisesRegex(ValueError, "Could not find a complete JSON object"):
            llm_utils.ollama_extract_invoice_fields(["some text"])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)