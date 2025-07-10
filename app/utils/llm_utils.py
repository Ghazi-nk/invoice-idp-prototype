import json
import re
from pathlib import Path
from typing import List, Dict

import requests

# --- Configuration for prompt files ---
from utils.config import (
    CHAT_ENDPOINT,
    OLLAMA_MODEL,
    SYSTEM_PROMPT_FILE,
    USER_PROMPT_OCR_FILE
)


def ollama_extract_invoice_fields(ocr_pages: List[str]) -> Dict:
    """
    Extracts invoice fields by sending OCR text to an LLM, using a prompt strategy
    based on the type of OCR data provided.
    """
    # 1. Load the appropriate prompt files based on the engine type
    try:
        system_prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
        user_prompt = Path(USER_PROMPT_OCR_FILE).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find a required prompt file: {e}")

    # 2. Build the initial message list with the system prompt.
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    # 3. Add each page's text as a separate user message.
    if not ocr_pages:
        raise ValueError("Input ocr_pages list cannot be empty.")

    num_pages = len(ocr_pages)
    for i, page_text in enumerate(ocr_pages):
        message_content = f"Here is the text for Page {i + 1} of {num_pages}:\n\n---\n{page_text}\n---"
        messages.append({"role": "user", "content": message_content})

    # 4. Add the final user prompt to trigger the JSON generation.
    messages.append({"role": "user", "content": user_prompt.strip()})

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

    # The {{...}} regex patch is no longer the primary method, but can serve as a fallback.
    json_chunk = re.sub(r'^\{\{(.*)\}\}$', r'{\1}', json_chunk, count=1, flags=re.DOTALL)

    try:
        return json.loads(json_chunk)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON still malformed after processing: {e}\nChunk:\n{json_chunk!r}")


def _extract_first_complete_json(text: str) -> str | None:
    """
    Extracts the first complete JSON object from a string.
    It first tries to find content within <json_output> tags,
    then falls back to finding the first brace-balanced object.
    """
    # --- NEW LOGIC: Prioritize finding content within <json_output> tags ---
    try:
        # Using regex to find content between the start and end tags, across multiple lines
        tag_match = re.search(r'<json_output>(.*?)</json_output>', text, re.DOTALL)
        if tag_match:
            # If tags are found, we work only with the content inside them
            text = tag_match.group(1).strip()
    except re.error:
        pass  # In case of a regex error, fall back to the old method

    # --- Fallback Logic: Find the first brace-balanced object ---
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