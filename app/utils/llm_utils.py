import json
import re
from pathlib import Path
from typing import List, Dict

import requests

# Assume these are defined in your config file
from utils.config import CHAT_ENDPOINT, OLLAMA_MODEL

# --- Constants and Setup ---
BASE_DIR = Path(__file__).resolve().parents[2]
PROMPT_FILE = BASE_DIR / "resources" / "Prompts" / "prompt.txt"
PROMPT_TXT = PROMPT_FILE.read_text(encoding="utf-8")


def ollama_extract_invoice_fields(ocr_pages: List[str]) -> Dict:
    """
    Extracts invoice fields by sending OCR text to an LLM, page by page, in a single chat session.

    Args:
        ocr_pages: A list of strings, where each string is the OCR text of one page.

    Returns:
        A dictionary containing the extracted invoice fields.
    """
    # 1. Split the master prompt into system instructions and the final user command.
    try:
        system_prompt, user_command_template = PROMPT_TXT.split("## Nutzer")
    except ValueError:
        raise ValueError("Prompt file is missing the '## Nutzer' separator.")

    # The actual command to generate the JSON.
    final_user_command = user_command_template.split("Hier ist der OCR-Text der neuen Rechnung:")[0].strip()

    # 2. Build the initial message list with the system prompt.
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    # 3. Add each page's text as a separate user message.
    if not ocr_pages:
        raise ValueError("Input ocr_pages list cannot be empty.")

    num_pages = len(ocr_pages)
    for i, page_text in enumerate(ocr_pages):
        # Frame each page's text with its page number to give the model clear context.
        message_content = f"Here is the text for Page {i + 1} of {num_pages}:\n\n---\n{page_text}\n---"
        messages.append({"role": "user", "content": message_content})

    # 4. Add the final user command to trigger the JSON generation.
    messages.append({"role": "user", "content": final_user_command})

    # 5. Send the complete conversation to the chat endpoint.
    body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}

    requests.packages.urllib3.disable_warnings()
    print(f"Sending {len(messages)} messages ({num_pages} pages) to the chat model...")
    resp = requests.post(CHAT_ENDPOINT, json=body, verify=False, timeout=600)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API Error: {resp.status_code} – {resp.text}")

    # 6. Extract the JSON from the final assistant message.
    try:
        raw_content = resp.json().get("message", {}).get("content", "")
    except (AttributeError, KeyError):
        raise ValueError("Ollama chat response is not in the expected format.")

    print(f"Ollama response: {raw_content}")
    json_chunk = _extract_first_complete_json(raw_content)

    if not json_chunk:
        raise ValueError("Could not find a complete JSON object in Ollama response:\n" + raw_content[:500])

    # The regex patch can still be useful as models might add extra braces.
    json_chunk = re.sub(r'^\{\{(.*)\}\}$', r'{\1}', json_chunk, count=1, flags=re.DOTALL)

    try:
        return json.loads(json_chunk)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON still malformed after processing: {e}\nChunk:\n{json_chunk!r}")


def _extract_first_complete_json(text: str) -> str | None:
    """
    (This helper function remains unchanged)
    Return the first brace-balanced JSON object found in *text*,
    or None if no complete object exists.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth, in_string, escape = 0, False, False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: idx + 1]
    return None