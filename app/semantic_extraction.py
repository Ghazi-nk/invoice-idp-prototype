import json
import re
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import requests

# --- Configuration for prompt files ---
from app.config import (
    CHAT_ENDPOINT,
    OLLAMA_MODEL,
    SYSTEM_PROMPT_FILE,
    USER_PROMPT_FILE,
    PDF_QUERY_SYSTEM_PROMPT,
    PDF_QUERY_USER_PROMPT
)


def ollama_extract_invoice_fields(ocr_pages: List[str]) -> Tuple[Dict, float]:
    """
    Extracts invoice fields by sending OCR text to an LLM.
    
    Args:
        ocr_pages: List of strings containing OCR text from each page
    
    Returns:
        A tuple containing (extracted_fields, ollama_duration)
    """
    # Load standard prompts for regular text extraction
    try:
        if SYSTEM_PROMPT_FILE is None or USER_PROMPT_FILE is None:
            raise FileNotFoundError("Required prompt files are not set in environment variables.")
        
        system_prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
        user_prompt = Path(USER_PROMPT_FILE).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find a required prompt file: {e}")

    # Build the initial message list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    # Add each page's text as a separate user message
    if not ocr_pages:
        raise ValueError("Input ocr_pages list cannot be empty.")

    num_pages = len(ocr_pages)
    for i, page_text in enumerate(ocr_pages):
        message_content = f"Here is the text for Page {i + 1} of {num_pages}:\n\n---\n{page_text}\n---"
        messages.append({"role": "user", "content": message_content})

    # Add the final user prompt to trigger the JSON generation
    messages.append({"role": "user", "content": user_prompt.strip()})

    # Send the complete conversation to the chat endpoint
    body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}

    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    print(f"Sending {len(messages)} messages ({num_pages} pages) to the chat model...")
    ollama_start_time = time.perf_counter()
    resp = requests.post(CHAT_ENDPOINT, json=body, verify=False, timeout=600)
    ollama_duration = time.perf_counter() - ollama_start_time

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API Error: {resp.status_code} – {resp.text}")

    # Extract the JSON from the final assistant message
    try:
        raw_content = resp.json().get("message", {}).get("content", "")
    except (AttributeError, KeyError):
        raise ValueError("Ollama chat response is not in the expected format.")

    print(f"Ollama response: {raw_content}")
    json_chunk = _extract_first_complete_json(raw_content)

    if not json_chunk:
        raise ValueError("Could not find a complete JSON object in Ollama response:\n" + raw_content[:500])

    # The {{...}} regex patch is no longer the primary method, but can serve as a fallback
    json_chunk = re.sub(r'^\{\{(.*)\}\}$', r'{\1}', json_chunk, count=1, flags=re.DOTALL)

    try:
        return json.loads(json_chunk), ollama_duration
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON still malformed after processing: {e}\nChunk:\n{json_chunk!r}")


def ollama_process_with_custom_prompt(ocr_pages: List[str], prompt: str) -> str:
    """
    Processes OCR text with a custom prompt and returns the raw Ollama response.
    
    Args:
        ocr_pages: List of strings containing OCR text from each page
        prompt: The custom prompt to send to Ollama
        
    Returns:
        The raw string response from Ollama
    """
    # Load the system prompt for PDF querying
    try:
        system_prompt = Path(PDF_QUERY_SYSTEM_PROMPT).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        # Fall back to a generic system prompt if the file is missing
        system_prompt = "You are a helpful assistant for document processing."
        print(f"Warning: Could not find PDF query system prompt file: {e}")
    
    # Build the message list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    # Add each page's text as a separate user message
    num_pages = len(ocr_pages)
    for i, page_text in enumerate(ocr_pages):
        message_content = f"Here is the text for Page {i + 1} of {num_pages}:\n\n---\n{page_text}\n---"
        messages.append({"role": "user", "content": message_content})

    # Load the user prompt template if available
    try:
        user_prompt_template = Path(PDF_QUERY_USER_PROMPT).read_text(encoding="utf-8")
        # Replace the placeholder with the user's custom prompt
        user_prompt = user_prompt_template.replace("[Hier den OCR-Rohtext einfügen]", "")
        # Add the custom prompt at the end
        final_prompt = f"{user_prompt.strip()}\n\n{prompt}"
    except FileNotFoundError as e:
        # Fall back to just using the custom prompt if the file is missing
        final_prompt = prompt
        print(f"Warning: Could not find PDF query user prompt file: {e}")

    # Add the final user prompt
    messages.append({"role": "user", "content": final_prompt})

    # Send the complete conversation to the chat endpoint
    body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}

    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    resp = requests.post(CHAT_ENDPOINT, json=body, verify=False, timeout=600)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API Error: {resp.status_code} – {resp.text}")

    # Extract the raw content from the response
    try:
        raw_content = resp.json().get("message", {}).get("content", "")
        return raw_content
    except (AttributeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Ollama chat response is not in the expected format: {e}")


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