import json
import requests
from pathlib import Path
from config import GENERATE_ENDPOINT, OLLAMA_MODEL

BASE_DIR = Path(__file__).resolve().parents[2]
PROMPT_FILE = BASE_DIR / "resources" / "Prompts" / "prompt.txt"
PROMPT_TEMPLATE = PROMPT_FILE.read_text(encoding="utf-8")


def ask_ollama_for_invoice_fields(ocr_text: str) -> dict:
    # Zusammensetzen des Prompts korrekt mit Triple-Quotes
    prompt = f"""
{PROMPT_TEMPLATE}
{ocr_text}
"""

    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    # SSL-Warnings deaktivieren
    requests.packages.urllib3.disable_warnings()
    resp = requests.post(GENERATE_ENDPOINT, json=body, verify=False, timeout=600)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API: {resp.status_code} - {resp.text}")

    raw = resp.json().get("response", "")

    # JSON-Auszug aus der Antwort
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"Kein JSON in Antwort: {raw}")

    return json.loads(raw[start : end + 1])
