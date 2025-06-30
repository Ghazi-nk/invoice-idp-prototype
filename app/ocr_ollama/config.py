import os
import sys
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SAMPLE_PDF_PATH = os.getenv("SAMPLE_PDF_PATH")
SAMPLE_PNG_PATH = os.getenv("SAMPLE_PNG_PATH")
SAMPLE_TXT_PATH = os.getenv("SAMPLE_TXT_PATH")

if not OLLAMA_BASE_URL:
    print("Error: OLLAMA_BASE_URL is not set.", file=sys.stderr)
    sys.exit(1)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
if not OLLAMA_MODEL:
    print("Error: OLLAMA_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

INPUT_DIR  = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
TMP_DIR    = os.getenv("TMP_DIR")
if not all([INPUT_DIR, OUTPUT_DIR, TMP_DIR]):
    print("Error: INPUT_DIR, OUTPUT_DIR und TMP_DIR müssen gesetzt sein.", file=sys.stderr)
    sys.exit(1)

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if not TESSERACT_CMD:
    print("Error: TESSERACT_CMD ist nicht gesetzt.", file=sys.stderr)
    sys.exit(1)
