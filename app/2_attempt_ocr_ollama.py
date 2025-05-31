import os
import sys
import json
import subprocess
import requests
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv

# ------------------ CONFIGURATION (from environment) ------------------
load_dotenv()
# 1) Base URL for the Ollama API (requires HTW VPN connection)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
if not OLLAMA_BASE_URL:
    print("Error: OLLAMA_BASE_URL is not set in the environment.", file=sys.stderr)
    sys.exit(1)

# 2) Which Ollama model to use for invoice-field extraction
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
if not OLLAMA_MODEL:
    print("Error: OLLAMA_MODEL is not set in the environment.", file=sys.stderr)
    sys.exit(1)

# 3) Full endpoint for text completion (constructed automatically)
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# 4) Directory layout (input, output, temp). Must be set in env.
INPUT_DIR  = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
TMP_DIR    = os.getenv("TMP_DIR")
if not INPUT_DIR or not OUTPUT_DIR or not TMP_DIR:
    print("Error: INPUT_DIR, OUTPUT_DIR, and TMP_DIR must all be set in the environment.", file=sys.stderr)
    sys.exit(1)

# 5) Tesseract command (must be installed and on your PATH, or set full path here)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if not TESSERACT_CMD:
    print("Error: TESSERACT_CMD is not set in the environment.", file=sys.stderr)
    sys.exit(1)

# -----------------------------------------------------

def pdf_to_png(pdf_path: str, dpi: int = 300) -> str:
    """
    Convert the first page of a PDF into a high-resolution RGB PNG.
    Returns the file path of the generated PNG.
    """
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to image: {e}")

    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")

    img: Image.Image = pages[0].convert("RGB")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    png_path = os.path.join(TMP_DIR, f"{base_name}_page1.png")
    img.save(png_path, format="PNG")
    return png_path

def ocr_png_to_text(png_path: str) -> str:
    """
    Run Tesseract OCR (German language) on the PNG and return the extracted text.
    Writes an output TXT file to TMP_DIR.
    """
    txt_base = os.path.splitext(os.path.basename(png_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{txt_base}.txt")

    cmd = [
        TESSERACT_CMD,
        png_path,
        os.path.splitext(txt_path)[0],  # Tesseract automatically appends .txt
        "-l", "deu"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")

    if not os.path.isfile(txt_path):
        raise RuntimeError(f"OCR text file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def ask_ollama_for_invoice_fields(ocr_text: str) -> dict:
    """
    Send the OCR text to Ollama's /api/generate endpoint with an extraction prompt.
    Returns a Python dict containing the seven invoice fields.
    """
    prompt = f"""
IMPORTANT: Do NOT output any chain of thought, <think> tags, or extra explanation.
Return ONLY one valid JSON object and nothing else.

Hier ist der OCR-Text einer deutschen Rechnung:
\"\"\"
{ocr_text}
\"\"\"

Extrahiere bitte exakt diese sieben Felder und gib NUR gültiges JSON zurück (keine zusätzlichen Erklärungen):

- Rechnungsnummer
- Rechnungsdatum
- Verkäufer
- Käufer
- Leistungsbeschreibung (als JSON-Array, z.B. [\"Position 1\", \"Position 2\", …])
- Gesamtbetrag
- Umsatzsteuer

Antwortformat (genau ein JSON-Objekt):
{{
  "Rechnungsnummer": "<hier>",
  "Rechnungsdatum":  "<hier>",
  "Verkäufer":       "<hier>",
  "Käufer":          "<hier>",
  "Leistungsbeschreibung": ["<hier>", "<hier>", …],
  "Gesamtbetrag":    "<hier>",
  "Umsatzsteuer":    "<hier>"
}}
"""

    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    # Suppress SSL warnings (VPN handles certificate)
    requests.packages.urllib3.disable_warnings()
    try:
        resp = requests.post(GENERATE_ENDPOINT, json=body, verify=False, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ollama endpoint: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API returned {resp.status_code}: {resp.text}")

    resp_json = resp.json()

    # Ollama returns a top-level "response" field containing the generated text
    raw_text = resp_json.get("response", "")
    if not raw_text or raw_text.strip() == "":
        raise RuntimeError("Ollama returned empty text for invoice fields.")

    # Extract the JSON substring (from first '{' to last '}')
    start = raw_text.find("{")
    end   = raw_text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"Could not find JSON in LLM response:\n{raw_text}")

    json_str = raw_text[start : end + 1]
    try:
        invoice_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Ollama response:\n{e}\nRaw:\n{json_str}")

    return invoice_data

def process_single_pdf(pdf_path: str) -> dict | None:
    """
    Complete pipeline for one PDF:
      1) PDF -> PNG
      2) PNG -> OCR text
      3) OCR text -> Ollama LLM -> JSON fields
    Returns the dict of extracted fields, or None if any step fails.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[Info] Processing '{base_name}.pdf' ...")

    try:
        png_path = pdf_to_png(pdf_path, dpi=300)
    except Exception as e:
        print(f"[Error] PDF->PNG failed for '{base_name}': {e}")
        return None

    try:
        ocr_text = ocr_png_to_text(png_path)
        # Uncomment for debugging the first 200 chars of OCR:
        # print(f"[Debug] OCR text for '{base_name}':\n{ocr_text[:200]!r}...\n")
    except Exception as e:
        print(f"[Error] OCR failed for '{base_name}': {e}")
        return None

    try:
        invoice_data = ask_ollama_for_invoice_fields(ocr_text)
    except Exception as e:
        print(f"[Error] LLM extraction failed for '{base_name}': {e}")
        return None

    return invoice_data

def main():
    # Ensure directories exist (INPUT_DIR, OUTPUT_DIR, TMP_DIR)
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR,    exist_ok=True)

    # Process all PDF files in INPUT_DIR
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        out_json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        # Skip if JSON already exists
        if os.path.isfile(out_json_path):
            print(f"[Info] '{out_json_path}' already exists. Skipping.")
            continue

        invoice_fields = process_single_pdf(pdf_path)
        if invoice_fields is None:
            print(f"[Warning] Could not extract fields for '{fname}'.")
            continue

        # Save the extracted fields to output/<invoice_basename>.json
        try:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(invoice_fields, f, ensure_ascii=False, indent=4)
            print(f"[Info] Saved: {out_json_path}")
        except Exception as e:
            print(f"[Error] Failed to write JSON for '{fname}': {e}")
            continue

    print("→ All PDFs processed.")

if __name__ == "__main__":
    main()
