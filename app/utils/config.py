import os
import sys
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SAMPLE_PDF_PATH = os.getenv("SAMPLE_PDF_PATH")
SAMPLE_PNG_PATH = os.getenv("SAMPLE_PNG_PATH")

SAMPLE_M_PDF_PATH = os.getenv("SAMPLE_M_PDF_PATH")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

INPUT_DIR  = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
TMP_DIR    = os.getenv("TMP_DIR")

TESSERACT_CMD = os.getenv("TESSERACT_CMD")

INVOICES_DIR = os.getenv("INVOICES_DIR")
GROUND_TRUTH_DIR = os.getenv("GROUND_TRUTH_DIR")
REPORTS_DIR = os.getenv("REPORTS_DIR")
SAMPLE_FOLDER = os.getenv("SAMPLE_FOLDER")


config_vars = {
    "SAMPLE_PDF_PATH": SAMPLE_PDF_PATH,
    "SAMPLE_PNG_PATH": SAMPLE_PNG_PATH,
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_MODEL": OLLAMA_MODEL,
    "INPUT_DIR": INPUT_DIR,
    "OUTPUT_DIR": OUTPUT_DIR,
    "TMP_DIR": TMP_DIR,
    "TESSERACT_CMD": TESSERACT_CMD
}
for var, value in config_vars.items():
    if not value:
        print(f"Error: {var} is not set.", file=sys.stderr)
        sys.exit(1)
